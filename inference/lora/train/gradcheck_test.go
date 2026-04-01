//go:build metal && darwin && cgo

package train

import (
	"math"
	"os"
	"testing"

	"vexel/inference/backend"
	"vexel/inference/backend/metal"
	"vexel/inference/lora"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// TestGradCheck verifies backward pass gradients against numerical estimates.
func TestGradCheck(t *testing.T) {
	modelPath := os.Getenv("VEXEL_TEST_MODEL")
	if modelPath == "" {
		candidates := []string{
			"/Users/qeetbastudio/projects/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
		}
		for _, p := range candidates {
			if _, err := os.Stat(p); err == nil {
				modelPath = p
				break
			}
		}
	}
	if modelPath == "" {
		t.Skip("No test model available (set VEXEL_TEST_MODEL)")
	}

	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer gpuBackend.Close()

	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("Open GGUF: %v", err)
	}
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	memCtx := memory.NewInferenceContext(tensor.Metal)
	totalScratch := modelCfg.TotalArenaBytes(256)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	model, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, modelCfg)
	if err != nil {
		t.Fatalf("NewModelRuntime: %v", err)
	}
	if err := model.LoadWeightsF32(modelPath); err != nil {
		t.Fatalf("LoadWeightsF32: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		t.Fatalf("CopyWeightsToDevice: %v", err)
	}

	mc := model.Config()
	hiddenSize := mc.HiddenSize
	headDim := mc.EffectiveHeadDim()
	qDim := mc.NumAttentionHeads * headDim
	vDim := mc.NumKeyValueHeads * headDim
	vocabSize := mc.VocabSize
	numLayers := mc.NumHiddenLayers

	rank := 2
	adapterCfg := lora.AdapterConfig{
		Rank:          rank,
		Alpha:         float32(rank),
		TargetModules: []string{"q_proj", "v_proj"},
	}
	cpuAdapter := InitAdapter(adapterCfg, numLayers, hiddenSize, qDim, vDim)
	gpu, err := lora.UploadToGPU(cpuAdapter, gpuBackend)
	if err != nil {
		t.Fatalf("UploadToGPU: %v", err)
	}
	model.AttachLoRA(gpu)

	var training backend.TrainingOps = gpuBackend
	grads := AllocGradients(gpuBackend, gpu, numLayers, hiddenSize, qDim, vDim)

	tokens := []int{1, 2, 3, 4, 5}
	seqLen := len(tokens)
	mask := BuildLossMask(make([]int32, seqLen), FormatText, 0)
	targets := make([]int32, seqLen)
	for i := 0; i < seqLen-1; i++ {
		targets[i] = int32(tokens[i+1])
	}

	// Helper: compute loss for current LoRA weights
	computeLoss := func() float32 {
		logits, saved, _, err := model.TrainingForward(tokens)
		if err != nil {
			t.Fatalf("TrainingForward: %v", err)
		}
		defer func() {
			for _, sa := range saved {
				sa.Free(gpuBackend)
			}
		}()

		targetsGPU := gpuBackend.Alloc(seqLen * 4)
		gpuBackend.ToDevice(targetsGPU, int32SliceToBytes(targets))
		maskGPU := gpuBackend.Alloc(seqLen * 4)
		gpuBackend.ToDevice(maskGPU, float32SliceToBytes(mask))

		dLogits := gpuBackend.Alloc(seqLen * vocabSize * 4)
		var rawLoss float32
		training.CrossEntropyLossForwardBackward(logits, targetsGPU, maskGPU, dLogits, &rawLoss, seqLen, vocabSize)
		gpuBackend.Sync()

		numMasked := 0
		for _, m := range mask {
			if m != 0 {
				numMasked++
			}
		}
		if numMasked == 0 {
			numMasked = 1
		}
		return rawLoss / float32(numMasked)
	}

	// Compute analytical gradients via full backward
	logits, savedPerLayer, finalNormInput, err := model.TrainingForward(tokens)
	if err != nil {
		t.Fatalf("TrainingForward: %v", err)
	}
	targetsGPU := gpuBackend.Alloc(seqLen * 4)
	gpuBackend.ToDevice(targetsGPU, int32SliceToBytes(targets))
	maskGPU := gpuBackend.Alloc(seqLen * 4)
	gpuBackend.ToDevice(maskGPU, float32SliceToBytes(mask))

	ZeroGradients(training, grads, gpu, numLayers, hiddenSize, qDim, vDim)
	avgLoss := backwardFullDebug(gpuBackend, training, logits, targetsGPU, maskGPU,
		seqLen, vocabSize, hiddenSize, model, savedPerLayer, finalNormInput, gpu, grads, t)
	gpuBackend.Sync()

	t.Logf("Analytical backward loss: %.4f", avgLoss)

	for _, sa := range savedPerLayer {
		sa.Free(gpuBackend)
	}

	// Check gradients for QA[0], QB[0], VA[0], VB[0] (first layer, first few elements)
	type gradCheck struct {
		name string
		wPtr tensor.DevicePtr
		gPtr tensor.DevicePtr
		size int
	}
	lastLayer := numLayers - 1
	checks := []gradCheck{
		{"QB[last]", gpu.Layers[lastLayer].QB, grads.DQB[lastLayer], qDim * rank},
		{"VB[last]", gpu.Layers[lastLayer].VB, grads.DVB[lastLayer], vDim * rank},
	}

	eps := float32(5e-3)
	anyFailed := false

	for _, chk := range checks {
		gradData := downloadF32(gpuBackend, chk.gPtr, chk.size)
		wData := downloadF32(gpuBackend, chk.wPtr, chk.size)

		// Check elements with largest analytical gradients (most informative)
		type idxVal struct {
			idx int
			val float32
		}
		var sortedByMag []idxVal
		for i, v := range gradData {
			if math.Abs(float64(v)) > 1e-4 {
				sortedByMag = append(sortedByMag, idxVal{i, v})
			}
		}
		// Take top 5 by magnitude
		for i := 0; i < len(sortedByMag); i++ {
			for j := i + 1; j < len(sortedByMag); j++ {
				if math.Abs(float64(sortedByMag[j].val)) > math.Abs(float64(sortedByMag[i].val)) {
					sortedByMag[i], sortedByMag[j] = sortedByMag[j], sortedByMag[i]
				}
			}
		}
		if len(sortedByMag) > 5 {
			sortedByMag = sortedByMag[:5]
		}
		checkIndices := make([]int, len(sortedByMag))
		for i, sv := range sortedByMag {
			checkIndices[i] = sv.idx
		}
		for _, checkIdx := range checkIndices {
			if checkIdx >= chk.size {
				continue
			}
			analyticalGrad := gradData[checkIdx]
			origVal := wData[checkIdx]

			wData[checkIdx] = origVal + eps
			gpuBackend.ToDevice(chk.wPtr, float32SliceToBytes(wData))
			lossPlus := computeLoss()

			wData[checkIdx] = origVal - eps
			gpuBackend.ToDevice(chk.wPtr, float32SliceToBytes(wData))
			lossMinus := computeLoss()

			wData[checkIdx] = origVal
			gpuBackend.ToDevice(chk.wPtr, float32SliceToBytes(wData))

			numericalGrad := (lossPlus - lossMinus) / (2 * eps)

			relErr := math.Abs(float64(analyticalGrad-numericalGrad)) /
				(math.Abs(float64(analyticalGrad)) + math.Abs(float64(numericalGrad)) + 1e-8)

			status := "PASS"
			if relErr > 0.2 {
				status = "FAIL"
				anyFailed = true
			}

			t.Logf("[%s] %s[%d]: anal=%.6e num=%.6e relErr=%.4f",
				status, chk.name, checkIdx, analyticalGrad, numericalGrad, relErr)
		}
	}

	if anyFailed {
		t.Error("One or more gradient checks failed (relErr > 0.2)")
	}
}

// gpuNorm computes L2 norm of a GPU buffer (downloads and computes on CPU).
func gpuNorm(b backend.Backend, ptr tensor.DevicePtr, n int) float64 {
	data := downloadF32(b, ptr, n)
	var sum float64
	for _, v := range data {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum)
}

// backwardFullDebug is backwardFull with logging of gradient magnitudes.
func backwardFullDebug(
	b backend.Backend,
	training backend.TrainingOps,
	logits tensor.DevicePtr,
	targets tensor.DevicePtr,
	mask tensor.DevicePtr,
	seqLen, vocabSize, hiddenSize int,
	model *runtime.ModelRuntime,
	savedPerLayer []*runtime.SavedActivations,
	finalNormInput tensor.DevicePtr,
	adapter *lora.GPUAdapter,
	grads *GradientBuffers,
	t *testing.T,
) float32 {
	cfg := model.Config()
	numHeads := cfg.NumAttentionHeads
	numKVHeads := cfg.NumKeyValueHeads
	headDim := cfg.EffectiveHeadDim()
	intermediateSize := cfg.IntermediateSize
	numLayers := cfg.NumHiddenLayers
	eps := float32(cfg.RMSNormEPS)
	rank := adapter.Rank
	loraScale := adapter.Scale
	qDim := numHeads * headDim
	vDim := numKVHeads * headDim

	dLogits := b.Alloc(seqLen * vocabSize * 4)
	var rawLoss float32
	training.CrossEntropyLossForwardBackward(logits, targets, mask, dLogits, &rawLoss, seqLen, vocabSize)
	b.Sync()
	numMasked := countMasked(b, mask, seqLen)
	if numMasked == 0 { numMasked = 1 }
	avgLoss := rawLoss / float32(numMasked)
	if scaler, ok := b.(backend.ScaleOps); ok {
		scaler.ScaleBuffer(dLogits, 1.0/float32(numMasked), seqLen*vocabSize)
	}
	t.Logf("  |dLogits| = %.6f", gpuNorm(b, dLogits, seqLen*vocabSize))

	dFinalNormOut := b.Alloc(seqLen * hiddenSize * 4)
	if !model.OutputHead.DevicePtr().IsNil() {
		b.MatMul(dLogits, model.OutputHead.DevicePtr(), dFinalNormOut, seqLen, hiddenSize, vocabSize)
	}
	t.Logf("  |dFinalNormOut| = %.6f", gpuNorm(b, dFinalNormOut, seqLen*hiddenSize))

	dResidual := b.Alloc(seqLen * hiddenSize * 4)
	if !model.FinalNorm.DevicePtr().IsNil() {
		training.RMSNormBackward(dFinalNormOut, finalNormInput, model.FinalNorm.DevicePtr(), dResidual, seqLen, hiddenSize, eps)
	}
	t.Logf("  |dResidual after final norm| = %.6f", gpuNorm(b, dResidual, seqLen*hiddenSize))

	for li := numLayers - 1; li >= 0; li-- {
		saved := savedPerLayer[li]
		layer := model.Layer(li)
		if layer == nil || saved == nil { continue }

		dAct := b.Alloc(seqLen * intermediateSize * 4)
		if !layer.W2.DevicePtr().IsNil() {
			b.MatMul(dResidual, layer.W2.DevicePtr(), dAct, seqLen, intermediateSize, hiddenSize)
		}
		dGate := b.Alloc(seqLen * intermediateSize * 4)
		dUp := b.Alloc(seqLen * intermediateSize * 4)
		training.SiLUMulBackward(dAct, saved.Gate, saved.Up, dGate, dUp, seqLen*intermediateSize)
		dFFNInput := b.Alloc(seqLen * hiddenSize * 4)
		training.Zero(dFFNInput, seqLen*hiddenSize)
		if !layer.W1.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dGate, layer.W1.DevicePtr(), tmp, seqLen, hiddenSize, intermediateSize)
			b.Add(dFFNInput, tmp, dFFNInput, seqLen*hiddenSize)
		}
		if !layer.W3.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dUp, layer.W3.DevicePtr(), tmp, seqLen, hiddenSize, intermediateSize)
			b.Add(dFFNInput, tmp, dFFNInput, seqLen*hiddenSize)
		}
		ffnNormInput := recomputeFFNNormInput(b, saved, layer, seqLen, hiddenSize, qDim)
		dFFNNorm := b.Alloc(seqLen * hiddenSize * 4)
		if !layer.FFNNorm.DevicePtr().IsNil() {
			training.RMSNormBackward(dFFNInput, ffnNormInput, layer.FFNNorm.DevicePtr(), dFFNNorm, seqLen, hiddenSize, eps)
		}
		b.Add(dResidual, dFFNNorm, dResidual, seqLen*hiddenSize)

		dAttnOut := b.Alloc(seqLen * qDim * 4)
		if !layer.Wo.DevicePtr().IsNil() {
			b.MatMul(dResidual, layer.Wo.DevicePtr(), dAttnOut, seqLen, qDim, hiddenSize)
		}
		dQ := b.Alloc(seqLen * qDim * 4)
		dK := b.Alloc(seqLen * numKVHeads * headDim * 4)
		dV := b.Alloc(seqLen * vDim * 4)
		attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
		attnWeights := computeAttnWeights(b, saved.Q, saved.K, seqLen, numHeads, numKVHeads, headDim, attnScale)
		training.SDPABackward(dAttnOut, saved.Q, saved.K, saved.V, attnWeights, dQ, dK, dV, seqLen, headDim, numHeads, numKVHeads)
		training.RoPEBackward(dQ, dK, headDim, numHeads, numKVHeads, seqLen, 0, layer.RoPEDim, layer.RoPETheta, layer.RoPENeox)

		if li == 0 {
			t.Logf("  Layer %d: |dResidual|=%.6f |dAttnOut|=%.6f |dQ|=%.6f |dK|=%.6f |dV|=%.6f",
				li,
				gpuNorm(b, dResidual, seqLen*hiddenSize),
				gpuNorm(b, dAttnOut, seqLen*qDim),
				gpuNorm(b, dQ, seqLen*qDim),
				gpuNorm(b, dK, seqLen*numKVHeads*headDim),
				gpuNorm(b, dV, seqLen*vDim))
			// Also check attention weights
			t.Logf("  Layer %d: |attnWeights|=%.6f |Q|=%.6f |K|=%.6f |V|=%.6f",
				li,
				gpuNorm(b, attnWeights, numHeads*seqLen*seqLen),
				gpuNorm(b, saved.Q, seqLen*qDim),
				gpuNorm(b, saved.K, seqLen*numKVHeads*headDim),
				gpuNorm(b, saved.V, seqLen*vDim))
		}

		la := adapter.GetLayer(li)
		if la != nil {
			normOut := saved.NormOut
			loraGrads(b, training, la, normOut, dQ, dV, grads, li,
				seqLen, hiddenSize, qDim, vDim, rank, loraScale)
		}

		dNormOut := b.Alloc(seqLen * hiddenSize * 4)
		training.Zero(dNormOut, seqLen*hiddenSize)
		if !layer.Wq.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dQ, layer.Wq.DevicePtr(), tmp, seqLen, hiddenSize, qDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}
		if !layer.Wk.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dK, layer.Wk.DevicePtr(), tmp, seqLen, hiddenSize, numKVHeads*headDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}
		if !layer.Wv.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dV, layer.Wv.DevicePtr(), tmp, seqLen, hiddenSize, vDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}
		dAttnNorm := b.Alloc(seqLen * hiddenSize * 4)
		if !layer.AttnNorm.DevicePtr().IsNil() {
			training.RMSNormBackward(dNormOut, saved.Residual, layer.AttnNorm.DevicePtr(), dAttnNorm, seqLen, hiddenSize, eps)
		}
		b.Add(dResidual, dAttnNorm, dResidual, seqLen*hiddenSize)
	}

	return avgLoss
}

// backwardFull is the full backward pass without the early return.
func backwardFull(
	b backend.Backend,
	training backend.TrainingOps,
	logits tensor.DevicePtr,
	targets tensor.DevicePtr,
	mask tensor.DevicePtr,
	seqLen, vocabSize, hiddenSize int,
	model *runtime.ModelRuntime,
	savedPerLayer []*runtime.SavedActivations,
	finalNormInput tensor.DevicePtr,
	adapter *lora.GPUAdapter,
	grads *GradientBuffers,
) float32 {
	cfg := model.Config()
	numHeads := cfg.NumAttentionHeads
	numKVHeads := cfg.NumKeyValueHeads
	headDim := cfg.EffectiveHeadDim()
	intermediateSize := cfg.IntermediateSize
	numLayers := cfg.NumHiddenLayers
	eps := float32(cfg.RMSNormEPS)
	rank := adapter.Rank
	loraScale := adapter.Scale

	qDim := numHeads * headDim
	vDim := numKVHeads * headDim

	dLogits := b.Alloc(seqLen * vocabSize * 4)
	var rawLoss float32
	training.CrossEntropyLossForwardBackward(logits, targets, mask, dLogits, &rawLoss, seqLen, vocabSize)
	b.Sync()

	numMasked := countMasked(b, mask, seqLen)
	if numMasked == 0 {
		numMasked = 1
	}
	avgLoss := rawLoss / float32(numMasked)

	if scaler, ok := b.(backend.ScaleOps); ok {
		scaler.ScaleBuffer(dLogits, 1.0/float32(numMasked), seqLen*vocabSize)
	}

	// Phase 2: output head + final norm
	dFinalNormOut := b.Alloc(seqLen * hiddenSize * 4)
	if !model.OutputHead.DevicePtr().IsNil() {
		b.MatMul(dLogits, model.OutputHead.DevicePtr(), dFinalNormOut, seqLen, hiddenSize, vocabSize)
	}

	dResidual := b.Alloc(seqLen * hiddenSize * 4)
	if !model.FinalNorm.DevicePtr().IsNil() {
		training.RMSNormBackward(dFinalNormOut, finalNormInput, model.FinalNorm.DevicePtr(), dResidual, seqLen, hiddenSize, eps)
	}

	// Phase 3: per-layer
	for li := numLayers - 1; li >= 0; li-- {
		saved := savedPerLayer[li]
		layer := model.Layer(li)
		if layer == nil || saved == nil {
			continue
		}

		// FFN backward
		dAct := b.Alloc(seqLen * intermediateSize * 4)
		if !layer.W2.DevicePtr().IsNil() {
			b.MatMul(dResidual, layer.W2.DevicePtr(), dAct, seqLen, intermediateSize, hiddenSize)
		}

		dGate := b.Alloc(seqLen * intermediateSize * 4)
		dUp := b.Alloc(seqLen * intermediateSize * 4)
		training.SiLUMulBackward(dAct, saved.Gate, saved.Up, dGate, dUp, seqLen*intermediateSize)

		dFFNInput := b.Alloc(seqLen * hiddenSize * 4)
		training.Zero(dFFNInput, seqLen*hiddenSize)
		if !layer.W1.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dGate, layer.W1.DevicePtr(), tmp, seqLen, hiddenSize, intermediateSize)
			b.Add(dFFNInput, tmp, dFFNInput, seqLen*hiddenSize)
		}
		if !layer.W3.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dUp, layer.W3.DevicePtr(), tmp, seqLen, hiddenSize, intermediateSize)
			b.Add(dFFNInput, tmp, dFFNInput, seqLen*hiddenSize)
		}

		ffnNormInput := recomputeFFNNormInput(b, saved, layer, seqLen, hiddenSize, qDim)

		dFFNNorm := b.Alloc(seqLen * hiddenSize * 4)
		if !layer.FFNNorm.DevicePtr().IsNil() {
			training.RMSNormBackward(dFFNInput, ffnNormInput, layer.FFNNorm.DevicePtr(), dFFNNorm, seqLen, hiddenSize, eps)
		}
		b.Add(dResidual, dFFNNorm, dResidual, seqLen*hiddenSize)

		// Attention backward
		dAttnOut := b.Alloc(seqLen * qDim * 4)
		if !layer.Wo.DevicePtr().IsNil() {
			b.MatMul(dResidual, layer.Wo.DevicePtr(), dAttnOut, seqLen, qDim, hiddenSize)
		}

		dQ := b.Alloc(seqLen * qDim * 4)
		dK := b.Alloc(seqLen * numKVHeads * headDim * 4)
		dV := b.Alloc(seqLen * vDim * 4)

		attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
		attnWeights := computeAttnWeights(b, saved.Q, saved.K, seqLen, numHeads, numKVHeads, headDim, attnScale)
		training.SDPABackward(dAttnOut, saved.Q, saved.K, saved.V, attnWeights, dQ, dK, dV, seqLen, headDim, numHeads, numKVHeads)
		training.RoPEBackward(dQ, dK, headDim, numHeads, numKVHeads, seqLen, 0, layer.RoPEDim, layer.RoPETheta, layer.RoPENeox)

		la := adapter.GetLayer(li)
		if la != nil {
			normOut := saved.NormOut
			loraGrads(b, training, la, normOut, dQ, dV, grads, li,
				seqLen, hiddenSize, qDim, vDim, rank, loraScale)
		}

		dNormOut := b.Alloc(seqLen * hiddenSize * 4)
		training.Zero(dNormOut, seqLen*hiddenSize)
		if !layer.Wq.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dQ, layer.Wq.DevicePtr(), tmp, seqLen, hiddenSize, qDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}
		if !layer.Wk.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dK, layer.Wk.DevicePtr(), tmp, seqLen, hiddenSize, numKVHeads*headDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}
		if !layer.Wv.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dV, layer.Wv.DevicePtr(), tmp, seqLen, hiddenSize, vDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}

		dAttnNorm := b.Alloc(seqLen * hiddenSize * 4)
		if !layer.AttnNorm.DevicePtr().IsNil() {
			training.RMSNormBackward(dNormOut, saved.Residual, layer.AttnNorm.DevicePtr(), dAttnNorm, seqLen, hiddenSize, eps)
		}
		b.Add(dResidual, dAttnNorm, dResidual, seqLen*hiddenSize)
	}

	return avgLoss
}
