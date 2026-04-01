package train

import (
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/lora"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// TrainConfig holds hyper-parameters and paths for LoRA fine-tuning.
type TrainConfig struct {
	Rank        int
	Alpha       float32
	LR          float32
	Momentum    float32
	WeightDecay float32
	Epochs      int
	OutputDir   string
	DataPath    string
}

// Trainer orchestrates the LoRA fine-tuning loop: forward, backward, SGD update.
type Trainer struct {
	config  TrainConfig
	model   *runtime.ModelRuntime
	tok     *tokenizer.Tokenizer
	adapter *lora.Adapter
	gpu     *lora.GPUAdapter
	grads   *GradientBuffers
	b       backend.Backend
	train   backend.TrainingOps
}

// NewTrainer initialises LoRA weights, uploads them to the GPU, and allocates
// gradient buffers. The returned Trainer is ready for Train().
func NewTrainer(cfg TrainConfig, model *runtime.ModelRuntime, tok *tokenizer.Tokenizer) (*Trainer, error) {
	b := model.Backend()
	training, ok := b.(backend.TrainingOps)
	if !ok {
		return nil, fmt.Errorf("backend does not support training ops")
	}

	mc := model.Config()
	numLayers := mc.NumHiddenLayers
	hiddenSize := mc.HiddenSize
	headDim := mc.EffectiveHeadDim()
	qDim := mc.NumAttentionHeads * headDim
	vDim := mc.NumKeyValueHeads * headDim

	adapterCfg := lora.AdapterConfig{
		Rank:          cfg.Rank,
		Alpha:         cfg.Alpha,
		TargetModules: []string{"q_proj", "v_proj"},
	}
	adapter := InitAdapter(adapterCfg, numLayers, hiddenSize, qDim, vDim)

	gpu, err := lora.UploadToGPU(adapter, b.(lora.Allocator))
	if err != nil {
		return nil, fmt.Errorf("upload adapter to GPU: %w", err)
	}

	// Attach the adapter so the training forward pass applies LoRA deltas.
	model.AttachLoRA(gpu)

	grads := AllocGradients(b, gpu, numLayers, hiddenSize, qDim, vDim)

	return &Trainer{
		config:  cfg,
		model:   model,
		tok:     tok,
		adapter: adapter,
		gpu:     gpu,
		grads:   grads,
		b:       b,
		train:   training,
	}, nil
}

// Train runs the training loop over the given examples for the configured
// number of epochs, printing the loss after each step and saving a final
// checkpoint. It handles Ctrl-C gracefully by saving a checkpoint before exit.
func (t *Trainer) Train(examples []Example) error {
	mc := t.model.Config()
	hiddenSize := mc.HiddenSize
	vocabSize := mc.VocabSize
	headDim := mc.EffectiveHeadDim()
	numLayers := mc.NumHiddenLayers
	qDim := mc.NumAttentionHeads * headDim
	vDim := mc.NumKeyValueHeads * headDim

	// Set up Ctrl-C handler.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)

	interrupted := false

	for epoch := 0; epoch < t.config.Epochs; epoch++ {
		// Shuffle examples each epoch.
		perm := rand.Perm(len(examples))
		shuffled := make([]Example, len(examples))
		for i, j := range perm {
			shuffled[i] = examples[j]
		}

		for step, ex := range shuffled {
			// Non-blocking check for Ctrl-C.
			select {
			case <-sigCh:
				interrupted = true
			default:
			}
			if interrupted {
				fmt.Printf("\n[interrupted] saving checkpoint before exit...\n")
				if err := t.saveCheckpoint(); err != nil {
					return fmt.Errorf("save checkpoint on interrupt: %w", err)
				}
				fmt.Printf("[interrupted] checkpoint saved to %s\n", t.config.OutputDir)
				return nil
			}

			// Tokenize the example.
			tokens, promptLen, err := t.tokenizeExample(ex)
			if err != nil {
				return fmt.Errorf("epoch %d step %d: tokenize: %w", epoch, step, err)
			}
			if len(tokens) < 2 {
				continue // Need at least 2 tokens for a target.
			}

			seqLen := len(tokens)

			// Build loss mask.
			mask := BuildLossMask(tokens, ex.Format, promptLen)

			// Build targets: tokens shifted left by one position.
			targets := make([]int32, seqLen)
			for i := 0; i < seqLen-1; i++ {
				targets[i] = tokens[i+1]
			}
			targets[seqLen-1] = 0 // Last position has no target.

			// Convert int32 tokens to int for TrainingForward.
			intTokens := make([]int, seqLen)
			for i, tok := range tokens {
				intTokens[i] = int(tok)
			}

			// Forward pass (must happen before uploading targets/mask because
			// TrainingForward calls ResetPool which recycles pool buffers).
			logits, savedPerLayer, finalNormInput, err := t.model.TrainingForward(intTokens)
			if err != nil {
				return fmt.Errorf("epoch %d step %d: forward: %w", epoch, step, err)
			}

			// Upload targets (int32) and mask (float32) to GPU AFTER forward
			// pass to avoid ResetPool recycling these buffers.
			targetsGPU := t.b.Alloc(seqLen * 4)
			targetsBytes := int32SliceToBytes(targets)
			t.b.ToDevice(targetsGPU, targetsBytes)

			maskGPU := t.b.Alloc(seqLen * 4)
			maskBytes := float32SliceToBytes(mask)
			t.b.ToDevice(maskGPU, maskBytes)

			// Zero gradients.
			ZeroGradients(t.train, t.grads, t.gpu, numLayers, hiddenSize, qDim, vDim)

			// Backward pass.
			loss := Backward(
				t.b, t.train,
				logits, targetsGPU, maskGPU,
				seqLen, vocabSize, hiddenSize,
				t.model, savedPerLayer, finalNormInput,
				t.gpu, t.grads,
			)

			// SGD update.
			t.sgdUpdate(numLayers, hiddenSize, qDim, vDim)

			// Free saved activations.
			for _, sa := range savedPerLayer {
				sa.Free(t.b)
			}

			fmt.Printf("epoch %d/%d  step %d/%d  loss=%.4f\n",
				epoch+1, t.config.Epochs, step+1, len(shuffled), loss)
		}
	}

	// Save final checkpoint.
	if err := t.saveCheckpoint(); err != nil {
		return fmt.Errorf("save final checkpoint: %w", err)
	}
	fmt.Printf("checkpoint saved to %s\n", t.config.OutputDir)
	return nil
}

// sgdUpdate applies plain SGD with weight decay to all LoRA parameter matrices.
func (t *Trainer) sgdUpdate(numLayers, hiddenSize, qDim, vDim int) {
	rank := t.gpu.Rank
	lr := t.config.LR
	wd := t.config.WeightDecay

	for i := 0; i < numLayers; i++ {
		la := t.gpu.GetLayer(i)
		if la == nil {
			continue
		}
		if la.HasQ {
			t.train.SGDUpdate(la.QA, t.grads.DQA[i], lr, wd, rank*hiddenSize)
			t.train.SGDUpdate(la.QB, t.grads.DQB[i], lr, wd, qDim*rank)
		}
		if la.HasV {
			t.train.SGDUpdate(la.VA, t.grads.DVA[i], lr, wd, rank*hiddenSize)
			t.train.SGDUpdate(la.VB, t.grads.DVB[i], lr, wd, vDim*rank)
		}
	}
}

// saveCheckpoint downloads GPU weights back to the CPU adapter and writes
// the checkpoint in HuggingFace PEFT format.
func (t *Trainer) saveCheckpoint() error {
	mc := t.model.Config()
	numLayers := mc.NumHiddenLayers
	headDim := mc.EffectiveHeadDim()
	hiddenSize := mc.HiddenSize
	rank := t.gpu.Rank

	qDim := mc.NumAttentionHeads * headDim
	vDim := mc.NumKeyValueHeads * headDim

	t.b.Sync()

	for i := 0; i < numLayers; i++ {
		la := t.gpu.GetLayer(i)
		if la == nil {
			continue
		}
		layer := &t.adapter.Layers[i]

		if la.HasQ {
			layer.QA = downloadF32(t.b, la.QA, rank*hiddenSize)
			layer.QB = downloadF32(t.b, la.QB, qDim*rank)
		}
		if la.HasV {
			layer.VA = downloadF32(t.b, la.VA, rank*hiddenSize)
			layer.VB = downloadF32(t.b, la.VB, vDim*rank)
		}
	}

	return lora.SaveAdapter(t.adapter, t.config.OutputDir)
}

// tokenizeExample encodes an Example into token IDs and returns the prompt
// length (for prompt/completion format, this is the number of prompt tokens;
// for text format it is 0).
func (t *Trainer) tokenizeExample(ex Example) ([]int32, int, error) {
	var promptLen int

	switch ex.Format {
	case FormatText:
		ids, err := t.tok.Encode(ex.Text)
		if err != nil {
			return nil, 0, err
		}
		tokens := intsToInt32(ids)
		return tokens, 0, nil

	case FormatPromptCompletion:
		promptIDs, err := t.tok.Encode(ex.Prompt)
		if err != nil {
			return nil, 0, fmt.Errorf("encode prompt: %w", err)
		}
		completionIDs, err := t.tok.Encode(ex.Completion)
		if err != nil {
			return nil, 0, fmt.Errorf("encode completion: %w", err)
		}
		promptLen = len(promptIDs)
		allIDs := append(promptIDs, completionIDs...)
		tokens := intsToInt32(allIDs)
		return tokens, promptLen, nil

	default:
		return nil, 0, fmt.Errorf("unknown data format: %d", ex.Format)
	}
}

// downloadF32 reads n float32 values from a GPU buffer back to the host.
func downloadF32(b backend.Backend, ptr tensor.DevicePtr, n int) []float32 {
	buf := make([]byte, n*4)
	b.ToHost(buf, ptr)
	b.Sync()
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(buf[i*4]) | uint32(buf[i*4+1])<<8 |
			uint32(buf[i*4+2])<<16 | uint32(buf[i*4+3])<<24
		out[i] = *(*float32)(unsafe.Pointer(&bits))
	}
	return out
}

// int32SliceToBytes converts []int32 to little-endian []byte.
func int32SliceToBytes(data []int32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
}

// float32SliceToBytes converts []float32 to little-endian []byte.
func float32SliceToBytes(data []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
}

// intsToInt32 converts []int to []int32.
func intsToInt32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}
