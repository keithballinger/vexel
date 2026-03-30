package scheduler

import (
	"context"
	"fmt"
	"os"
	"time"
	"unsafe"

	"vexel/inference/medusa"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// MedusaConfig configures Medusa-enabled scheduling.
type MedusaConfig struct {
	// Enable online training of Medusa heads during inference
	EnableOnlineTraining bool

	// UseGPUTraining enables GPU-accelerated training (requires Metal backend)
	UseGPUTraining bool

	// Path to pre-trained Medusa heads (optional)
	HeadsPath string

	// Number of Medusa heads (future token predictions)
	NumHeads int

	// UseTreeVerification enables tree-based candidate verification.
	// When enabled, top-k candidates per head form a candidate tree,
	// and multiple paths are evaluated to maximize accepted tokens.
	UseTreeVerification bool

	// TreeTopK is the branching factor for the candidate tree.
	// Each head contributes its top-k predictions as children at that level.
	TreeTopK int

	// TreeMaxNodes limits the total number of nodes in the candidate tree
	// to prevent combinatorial explosion with many heads and high top-k.
	TreeMaxNodes int

	// Training configuration
	TrainingConfig medusa.OnlineConfig
}

// DefaultMedusaConfig returns reasonable defaults for Medusa scheduling.
func DefaultMedusaConfig() MedusaConfig {
	return MedusaConfig{
		EnableOnlineTraining: true,
		NumHeads:             4,
		UseTreeVerification:  true,
		TreeTopK:             3,
		TreeMaxNodes:         64,
		TrainingConfig:       medusa.DefaultOnlineConfig(),
	}
}

// MedusaScheduler wraps Scheduler with Medusa speculative decoding support.
type MedusaScheduler struct {
	*Scheduler

	trainer      medusa.Trainer
	medusaConfig MedusaConfig

	// Ring buffers for collecting (hidden, token) pairs during training
	// We need to track historical hidden states to pair with future tokens
	recentTokens  []int
	recentHiddens [][]float32
	maxRecent     int

	// Cached hidden state from last decode (for Medusa head inference)
	lastHidden []float32

	// Speculative decoding metrics
	specMetrics SpeculativeMetrics

	// Adaptive speculation: starts OFF, enables when acceptance proves positive.
	// Tracks the last N verification steps' acceptance counts.
	recentAccepted []int // ring buffer of accepted counts per step
	recentIdx      int   // write position in ring buffer
	recentFull     bool  // true once the buffer has wrapped
	probeCounter   int   // counts normal decode steps since last probe
}

// NewMedusaScheduler creates a Medusa-enabled scheduler.
func NewMedusaScheduler(
	rt *runtime.ModelRuntime,
	tok *tokenizer.Tokenizer,
	config Config,
	medusaConfig MedusaConfig,
) (*MedusaScheduler, error) {
	base, err := NewScheduler(rt, tok, config)
	if err != nil {
		return nil, err
	}

	ms := &MedusaScheduler{
		Scheduler:      base,
		medusaConfig:   medusaConfig,
		recentTokens:   make([]int, 0, medusaConfig.NumHeads+1),
		recentHiddens:  make([][]float32, 0, medusaConfig.NumHeads+1),
		maxRecent:      medusaConfig.NumHeads + 1,
		recentAccepted: make([]int, 4), // track last 4 speculation steps
	}

	// Initialize trainer
	if medusaConfig.EnableOnlineTraining {
		hiddenSize := rt.Config().HiddenSize
		vocabSize := rt.Config().VocabSize

		// Try to load pre-trained heads
		if medusaConfig.HeadsPath != "" {
			heads, err := medusa.Load(medusaConfig.HeadsPath)
			if err == nil {
				// Use GPU trainer for pre-trained heads (fast Forward)
				if medusaConfig.UseGPUTraining && gpuTrainingAvailable() {
					if b := rt.Backend(); b != nil {
						ms.trainer = medusa.NewGPUOnlineTrainerFromHeads(heads, medusaConfig.TrainingConfig, b)
						fmt.Printf("Loaded Medusa heads from %s → GPU (phase: Hot)\n", medusaConfig.HeadsPath)
					}
				}
				// Fall back to CPU trainer
				if ms.trainer == nil {
					ms.trainer = medusa.NewOnlineTrainerWithHeads(heads, medusaConfig.TrainingConfig)
					fmt.Printf("Loaded Medusa heads from %s → CPU (phase: %s)\n",
						medusaConfig.HeadsPath, ms.trainer.Phase())
				}
			} else {
				fmt.Printf("Warning: could not load Medusa heads from %s: %v\n",
					medusaConfig.HeadsPath, err)
			}
		}

		// Create new trainer if we don't have pre-trained heads
		if ms.trainer == nil {
			// Get lm_head weights for initializing Medusa heads
			// This dramatically improves speculation since heads start knowing how to predict tokens
			lmHeadWeights := rt.GetOutputHeadWeightsF32()
			if lmHeadWeights != nil {
				// Check for variation in weights
				var minW, maxW float32 = lmHeadWeights[0], lmHeadWeights[0]
				var sumW float64
				for _, w := range lmHeadWeights[:1000] { // Sample first 1000
					if w < minW {
						minW = w
					}
					if w > maxW {
						maxW = w
					}
					sumW += float64(w)
				}
				fmt.Printf("Initializing Medusa heads from lm_head weights (%d elements, min=%.4f, max=%.4f, mean=%.6f)\n",
					len(lmHeadWeights), minW, maxW, sumW/1000)
			}

			// Try GPU training if requested and available
			if medusaConfig.UseGPUTraining && gpuTrainingAvailable() {
				// Get backend from runtime for GPU training
				if b := rt.Backend(); b != nil {
					ms.trainer = createGPUTrainer(hiddenSize, vocabSize, medusaConfig.TrainingConfig, b, lmHeadWeights)
					if ms.trainer != nil {
						fmt.Println("Using GPU-accelerated Medusa training")
					}
				}
			}

			// CPU training is not supported - it doesn't use lm_head initialization
			// and produces poor results. GPU training is required.
			if ms.trainer == nil {
				return nil, fmt.Errorf("CPU Medusa training is not supported; GPU training required (use -gpu-training=true or ensure Metal backend is available)")
			}
		}
	}

	return ms, nil
}

// Start begins the scheduler and trainer loops.
func (ms *MedusaScheduler) Start(ctx context.Context) {
	// Start online trainer if enabled
	if ms.trainer != nil {
		ms.trainer.Start(ctx)
	}
}

// Stop stops the scheduler and trainer.
func (ms *MedusaScheduler) Stop() {
	if ms.trainer != nil {
		ms.trainer.Stop()
	}
}

// BaseScheduler returns the underlying Scheduler for use with components
// that require a *Scheduler (e.g., the HTTP server).
func (ms *MedusaScheduler) BaseScheduler() *Scheduler {
	return ms.Scheduler
}

// Run starts the Medusa scheduler's main loop.
func (ms *MedusaScheduler) Run(ctx context.Context) error {
	ms.Start(ctx)
	defer ms.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		if err := ms.step(ctx); err != nil {
			return err
		}

		if ms.SequenceCount() > 0 {
			continue
		}

		done := make(chan struct{})
		go func() {
			ms.cond.L.Lock()
			ms.cond.Wait()
			ms.cond.L.Unlock()
			close(done)
		}()

		select {
		case <-ctx.Done():
			ms.cond.Broadcast()
			return nil
		case <-done:
		}
	}
}

// step performs a single scheduling iteration with Medusa support.
func (ms *MedusaScheduler) step(ctx context.Context) error {
	ready := ms.collectReady()
	ms.metrics.ActiveSequences = len(ms.sequences)

	batch := ms.formBatches(ready)
	if len(batch) == 0 {
		return nil
	}

	// Hold GPU lock for the entire step to prevent concurrent Metal access
	// with the background training goroutine.
	if ms.trainer != nil {
		ms.trainer.GPULock()
		defer ms.trainer.GPUUnlock()
	}

	// Adaptive speculation: starts OFF, turns ON once heads prove useful.
	// Every 32 normal decode steps, run one speculation probe to test acceptance.
	// Once the probe window shows avg acceptance >= 0.5, speculation stays on.
	headsReady := ms.trainer != nil && ms.trainer.IsHot()

	if headsReady && ms.speculationEnabled() {
		// Speculation has proven useful — use it
		if ms.medusaConfig.UseTreeVerification {
			return ms.runTreeMedusaDecodeStep(ctx, batch)
		}
		return ms.runMedusaDecodeStep(ctx, batch)
	}

	if headsReady && ms.lastHidden != nil {
		// Heads are ready but speculation not yet proven — probe periodically.
		// The probe runs the heads' Forward to check if their top predictions
		// match the base model's output, without running verification or
		// committing any tokens. This avoids injecting bad tokens from probes.
		ms.probeCounter++
		if ms.probeCounter >= 8 {
			ms.probeCounter = 0
			ms.probeSpeculation(batch)
			if os.Getenv("MEDUSA_DEBUG") != "" {
				fmt.Printf("[Medusa] probe: speculationEnabled=%v, recentFull=%v\n",
					ms.speculationEnabled(), ms.recentFull)
			}
		}
	}

	// Normal decode with training data collection
	return ms.runDecodeStepWithTraining(ctx, batch)
}

// runDecodeStepWithTraining runs decode and collects training samples.
func (ms *MedusaScheduler) runDecodeStepWithTraining(ctx context.Context, batch []*Sequence) error {
	if len(batch) == 0 {
		return nil
	}

	useGPUCache := ms.runtime.GPUKVCache() != nil
	usePagedCache := ms.runtime.PagedKVCache() != nil

	// Process each sequence - check if any need prefill
	for _, seq := range batch {
		if seq.State() == StatePending && len(seq.PromptTokens()) > 0 && !seq.IsPrefillComplete() {
			if useGPUCache {
				if err := ms.runGPUPrefill(seq); err != nil {
					return err
				}
				continue
			} else if usePagedCache {
				if err := ms.runBatchedPrefill(seq); err != nil {
					return err
				}
				continue
			}
		}
	}

	// Filter to sequences that need decode
	decodeSeqs := make([]*Sequence, 0, len(batch))
	for _, seq := range batch {
		if seq.State() == StateDecoding || (seq.State() == StatePending && seq.IsPrefillComplete()) {
			decodeSeqs = append(decodeSeqs, seq)
		}
	}

	if len(decodeSeqs) == 0 {
		return nil
	}

	// Process one sequence at a time for training data collection
	for _, seq := range decodeSeqs {
		token, pos, hasMore := seq.NextInputToken()
		if !hasMore {
			token = 1
			pos = 0
		}

		var logits tensor.Tensor
		var hidden []float32
		var err error

		startTime := time.Now()

		if useGPUCache && ms.trainer != nil {
			// Use hidden-state-capturing decode for training
			logits, hidden, err = ms.runtime.DecodeWithGPUKVAndHidden([]int{token}, pos)
		} else if useGPUCache {
			logits, err = ms.runtime.DecodeWithGPUKV([]int{token}, pos)
		} else if usePagedCache && ms.trainer != nil {
			logits, hidden, err = ms.runtime.DecodeWithPagedKVAndHidden([]int{token}, seq.KVSeqID(), pos)
		} else if usePagedCache {
			logits, err = ms.runtime.DecodeWithPagedKV([]int{token}, seq.KVSeqID(), pos)
		} else {
			inputs := runtime.NewBatchRuntimeInputsWithPos([]int{token}, []int{pos}, nil)
			logits, err = ms.runtime.DecodeStep(inputs)
		}

		decodeTime := time.Since(startTime)

		if err != nil {
			return err
		}

		ms.metrics.TotalTokens++
		ms.metrics.DecodeTokens++
		ms.metrics.DecodeTime += decodeTime

		// Sample token
		vocabSize := ms.runtime.Config().VocabSize
		logitsData := ms.getLogitsOnCPU(logits, vocabSize)

		if logitsData == nil {
			seq.PushToken("?")
			continue
		}

		seq.AdvancePosition()
		if seq.State() == StatePending {
			seq.SetState(StateDecoding)
		}

		tokenID := ms.sampler.Sample(logitsData)
		seq.AddGeneratedToken(tokenID)

		// Collect training sample if we have hidden state
		if hidden != nil && ms.trainer != nil {
			ms.collectTrainingSample(hidden, tokenID)
		}

		// Cache hidden state for potential speculation next step
		ms.lastHidden = hidden

		// Check for EOS
		eosToken := 2
		if ms.tokenizer != nil {
			eosToken = ms.tokenizer.EOS()
		}
		if tokenID == eosToken {
			seq.SetState(StateFinished)
			seq.Close()
			continue
		}

		// Check max tokens
		if seq.ReachedMaxTokens(ms.config.MaxTokens) {
			seq.SetState(StateFinished)
			seq.Close()
			continue
		}

		// Decode and push token
		var text string
		if ms.tokenizer != nil {
			text, _ = ms.tokenizer.Decode([]int{tokenID})
		} else {
			text = fmt.Sprintf(" %d", tokenID)
		}
		seq.PushToken(text)
	}

	return nil
}

// collectTrainingSample adds a sample to the trainer's buffer.
// We maintain sliding windows of (hidden, token) pairs.
// When we have enough history, we can pair hidden state h0 with future tokens [t1, t2, t3, t4].
func (ms *MedusaScheduler) collectTrainingSample(hidden []float32, tokenID int) {
	// Make a copy of hidden state since it may be reused
	hiddenCopy := make([]float32, len(hidden))
	copy(hiddenCopy, hidden)

	// Add current (hidden, token) pair to history
	ms.recentTokens = append(ms.recentTokens, tokenID)
	ms.recentHiddens = append(ms.recentHiddens, hiddenCopy)

	// If we have enough history, create training samples
	// When we have [(h0,t0), (h1,t1), (h2,t2), (h3,t3), (h4,t4)]:
	// - hidden state h0 produced logits for predicting t0
	// - head 0 should predict t0 (same as base model)
	// - head 1 should predict t1, head 2 -> t2, head 3 -> t3
	// So targets are [t0, t1, t2, t3], NOT [t1, t2, t3, t4]!
	if len(ms.recentTokens) >= ms.medusaConfig.NumHeads {
		// Use the OLDEST hidden state (h0) with its future tokens [t0, t1, t2, t3]
		oldestHidden := ms.recentHiddens[0]
		futureTokens := ms.recentTokens[0:ms.medusaConfig.NumHeads]

		ms.trainer.AddSample(oldestHidden, futureTokens, len(ms.recentTokens))

		// Slide window - remove oldest entry
		ms.recentTokens = ms.recentTokens[1:]
		ms.recentHiddens = ms.recentHiddens[1:]
	}
}

// runMedusaDecodeStep uses Medusa heads for speculative decoding.
func (ms *MedusaScheduler) runMedusaDecodeStep(ctx context.Context, batch []*Sequence) error {
	if len(batch) == 0 {
		return nil
	}

	// Only support single sequence for speculation
	if len(batch) > 1 {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	seq := batch[0]
	useGPUCache := ms.runtime.GPUKVCache() != nil
	usePagedCache := ms.runtime.PagedKVCache() != nil

	// Handle prefill first
	if seq.State() == StatePending && len(seq.PromptTokens()) > 0 && !seq.IsPrefillComplete() {
		// Reset acceptance window for new sequence — give speculation a fresh chance
		// Reset acceptance window for the new sequence.
		ms.recentIdx = 0
		ms.recentFull = false

		if useGPUCache {
			if err := ms.runGPUPrefill(seq); err != nil {
				return err
			}
		} else if usePagedCache {
			if err := ms.runBatchedPrefill(seq); err != nil {
				return err
			}
		}
		// After prefill, fall back to normal decode to get initial hidden state
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	// Check if sequence is ready for decode
	if seq.State() != StateDecoding && !(seq.State() == StatePending && seq.IsPrefillComplete()) {
		return nil
	}

	// If we don't have a cached hidden state, do a normal decode first
	if ms.lastHidden == nil {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	// Get current input token and position
	inputToken, pos, hasMore := seq.NextInputToken()
	if !hasMore {
		inputToken = 1
		pos = 0
	}

	// Step 1: Generate draft tokens using Medusa heads
	heads := ms.trainer.Heads()
	numHeads := heads.GetNumHeads()
	draftTokens := make([]int, numHeads)

	for i := 0; i < numHeads; i++ {
		logits := heads.Forward(i, ms.lastHidden)
		if logits != nil {
			draftTokens[i] = argmaxFloat32(logits)
		}
	}

	ms.specMetrics.DraftTokensGenerated += numHeads

	// Step 2: Build verification sequence: [input_token, draft_0, draft_1, ..., draft_K-1]
	verifyTokens := make([]int, 1+numHeads)
	verifyTokens[0] = inputToken
	copy(verifyTokens[1:], draftTokens)

	// Step 3: Run target model on all tokens at once
	// First, save the current KV cache position for potential rollback
	cache := ms.runtime.GPUKVCache()
	pagedCache := ms.runtime.PagedKVCache()
	startPos := pos
	if cache != nil {
		startPos = cache.SeqLen()
	}

	startTime := time.Now()
	var allLogits tensor.Tensor
	var hiddenStates [][]float32
	var err error
	if usePagedCache {
		allLogits, hiddenStates, err = ms.runtime.VerifySpeculativeWithPagedKVAndHidden(verifyTokens, seq.KVSeqID(), startPos)
	} else {
		allLogits, hiddenStates, err = ms.runtime.VerifySpeculativeWithHidden(verifyTokens, startPos)
	}
	verifyTime := time.Since(startTime)

	if err != nil {
		// Fall back to standard decode on error
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	ms.specMetrics.VerificationSteps++
	ms.specMetrics.VerifyTime += verifyTime

	// Step 4: Get logits and verify each draft token
	vocabSize := ms.runtime.Config().VocabSize
	allLogitsData := ms.getLogitsOnCPU(allLogits, (1+numHeads)*vocabSize)

	if allLogitsData == nil {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	// Hidden states: hiddenStates[i] is the pre-norm hidden state for token i
	// hiddenStates[0] -> input token
	// hiddenStates[1] -> draft_0
	// etc.

	// Step 5: Accept tokens until first rejection
	// logits[i] predicts token at position i+1
	// So logits[0] predicts draft_0, logits[1] predicts draft_1, etc.
	numAccepted := 0
	var finalToken int

	// Debug: print draft vs target for first few tokens
	debugSpec := os.Getenv("MEDUSA_DEBUG") != ""
	if debugSpec {
		// Also print a few values from lastHidden to verify it's changing
		hiddenSample := ms.lastHidden[0]
		if len(ms.lastHidden) > 100 {
			hiddenSample = ms.lastHidden[100]
		}
		// Compare head 0's prediction with what the base model would predict
		// by running logits[0] argmax (which should match head 0 if FC2 = lm_head)
		logits0 := allLogitsData[0:vocabSize]
		basePred := argmaxFloat32(logits0)
		fmt.Printf("[Spec] inputToken=%d, head0=%d, basePred=%d, hidden[100]=%.4f\n",
			inputToken, draftTokens[0], basePred, hiddenSample)
	}

	// IMPORTANT: Medusa head i predicts position +i relative to lastHidden.
	// Since lastHidden is from position P-1:
	// - head 0 predicts position P (which should equal inputToken)
	// - head 1 predicts position P+1 (should match logits[0])
	// - head 2 predicts position P+2 (should match logits[1])
	// - etc.
	// So we compare draftTokens[i+1] with logits[i] for i >= 0.
	// First, verify that head 0's prediction matches inputToken.
	if draftTokens[0] == inputToken {
		// Head 0 correctly predicted inputToken
		numAccepted++
		ms.specMetrics.DraftTokensAccepted++
		if debugSpec {
			fmt.Printf("[Spec] head 0: draft=%d matches inputToken=%d\n", draftTokens[0], inputToken)
		}

		// Now check remaining heads against verification logits
		for i := 0; i < numHeads-1; i++ {
			targetLogits := allLogitsData[i*vocabSize : (i+1)*vocabSize]
			targetToken := ms.sampler.Sample(targetLogits)

			if debugSpec {
				fmt.Printf("[Spec] head %d: draft=%d, target=%d, match=%v\n", i+1, draftTokens[i+1], targetToken, draftTokens[i+1] == targetToken)
			}

			// Check if draft[i+1] matches target (logits[i] predicts position P+1+i)
			if draftTokens[i+1] == targetToken {
				numAccepted++
				ms.specMetrics.DraftTokensAccepted++
			} else {
				// Rejection: use target's token as the correction
				finalToken = targetToken
				break
			}
		}

		// If all drafts accepted, sample one more token from the last position
		if numAccepted == numHeads {
			lastLogits := allLogitsData[(numHeads-1)*vocabSize : numHeads*vocabSize]
			finalToken = ms.sampler.Sample(lastLogits)
		}
	} else {
		if debugSpec {
			fmt.Printf("[Spec] head 0: draft=%d != inputToken=%d (reject)\n", draftTokens[0], inputToken)
		}
		// Head 0 didn't match inputToken, so we can't use any drafts
		// Use the token from standard decode
		targetLogits := allLogitsData[0:vocabSize]
		finalToken = ms.sampler.Sample(targetLogits)
	}

	ms.recordAcceptance(numAccepted)

	// Step 6: Truncate KV cache to only keep accepted tokens
	// The verification added (1 + numHeads) tokens to cache
	// We want to keep (1 + numAccepted) tokens (input + accepted drafts)
	// But actually, we need to rollback to startPos + (1 + numAccepted)
	newCacheLen := startPos + 1 + numAccepted
	if cache != nil {
		cache.Truncate(newCacheLen)
	}
	if pagedCache != nil && seq.KVSeqID() != 0 {
		pagedCache.TruncateSequence(seq.KVSeqID(), newCacheLen)
	}
	if gpuPool := ms.runtime.GetGPUBlockPool(); gpuPool != nil {
		gpuPool.TruncateSequence(seq.KVSeqID(), newCacheLen)
	}

	// Step 7: Output all accepted tokens + final token
	acceptedTokens := make([]int, 0, numAccepted+1)

	// First, output the accepted draft tokens
	for i := 0; i < numAccepted; i++ {
		acceptedTokens = append(acceptedTokens, draftTokens[i])
	}

	// Add the final token (correction or bonus token)
	acceptedTokens = append(acceptedTokens, finalToken)

	// Update metrics
	ms.metrics.TotalTokens += len(acceptedTokens)
	ms.metrics.DecodeTokens += len(acceptedTokens)

	// Process each accepted token
	eosToken := 2
	if ms.tokenizer != nil {
		eosToken = ms.tokenizer.EOS()
	}

	for _, tokenID := range acceptedTokens {
		seq.AdvancePosition()
		if seq.State() == StatePending {
			seq.SetState(StateDecoding)
		}

		seq.AddGeneratedToken(tokenID)

		// Check for EOS
		if tokenID == eosToken {
			seq.SetState(StateFinished)
			seq.Close()
			ms.lastHidden = nil
			return nil
		}

		// Check max tokens
		if seq.ReachedMaxTokens(ms.config.MaxTokens) {
			seq.SetState(StateFinished)
			seq.Close()
			ms.lastHidden = nil
			return nil
		}

		// Decode and push token
		var text string
		if ms.tokenizer != nil {
			text, _ = ms.tokenizer.Decode([]int{tokenID})
		} else {
			text = fmt.Sprintf(" %d", tokenID)
		}
		seq.PushToken(text)
	}

	// Step 8: Use hidden states from VerifySpeculativeWithHidden
	// The hidden state at position numAccepted is what we need for next speculation
	// (position 0 = input token, position 1 = draft_0, etc.)
	// After accepting numAccepted tokens + final token, the relevant hidden state is at index numAccepted
	// because that's the position that produced the final token's logits

	if len(hiddenStates) > numAccepted && hiddenStates[numAccepted] != nil {
		ms.lastHidden = hiddenStates[numAccepted]

		// Collect training samples for ALL positions we have hidden states for
		// Each hidden state at position i predicts the token at position i+1
		if ms.trainer != nil {
			for i := 0; i < len(hiddenStates) && i < len(verifyTokens); i++ {
				// Build future tokens starting from position i+1
				futureStart := i + 1
				if futureStart < len(verifyTokens) {
					// The token at position i+1 is what this hidden state should predict
					// For training, we want the actual tokens that follow
					actualToken := verifyTokens[futureStart]
					ms.collectTrainingSample(hiddenStates[i], actualToken)
				}
			}
		}
	} else {
		ms.lastHidden = nil
	}

	return nil
}

// runTreeMedusaDecodeStep uses tree-based candidate verification.
// It builds a candidate tree from top-k Medusa head predictions,
// extracts all paths sorted by confidence, and verifies the best
// path against the target model. On partial rejection, alternate
// paths are evaluated to maximize accepted tokens.
func (ms *MedusaScheduler) runTreeMedusaDecodeStep(ctx context.Context, batch []*Sequence) error {
	if len(batch) == 0 {
		return nil
	}

	// Only support single sequence for speculation
	if len(batch) > 1 {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	seq := batch[0]
	useGPUCache := ms.runtime.GPUKVCache() != nil
	usePagedCache := ms.runtime.PagedKVCache() != nil

	// Handle prefill first
	if seq.State() == StatePending && len(seq.PromptTokens()) > 0 && !seq.IsPrefillComplete() {
		// Reset acceptance window for the new sequence.
		ms.recentIdx = 0
		ms.recentFull = false

		if useGPUCache {
			if err := ms.runGPUPrefill(seq); err != nil {
				return err
			}
		} else if usePagedCache {
			if err := ms.runBatchedPrefill(seq); err != nil {
				return err
			}
		}
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	if seq.State() != StateDecoding && !(seq.State() == StatePending && seq.IsPrefillComplete()) {
		return nil
	}

	if ms.lastHidden == nil {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	inputToken, pos, hasMore := seq.NextInputToken()
	if !hasMore {
		inputToken = 1
		pos = 0
	}

	// Step 1: Get logits from all heads
	heads := ms.trainer.Heads()
	headLogits := heads.ForwardAll(ms.lastHidden)

	// Step 2: Build candidate tree
	tree := medusa.BuildCandidateTree(headLogits, ms.medusaConfig.TreeTopK, ms.medusaConfig.TreeMaxNodes)
	if tree == nil {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	// Step 3: Extract all paths sorted by confidence
	paths := tree.Paths()
	if len(paths) == 0 {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	// Step 4: Use the best path as draft sequence for verification
	bestPath := paths[0]
	numDraft := len(bestPath.Tokens)
	ms.specMetrics.DraftTokensGenerated += numDraft

	// Build verification sequence: [input_token, draft_0, draft_1, ...]
	verifyTokens := make([]int, 1+numDraft)
	verifyTokens[0] = inputToken
	copy(verifyTokens[1:], bestPath.Tokens)

	// Step 5: Run target model verification
	cache := ms.runtime.GPUKVCache()
	pagedCache := ms.runtime.PagedKVCache()
	startPos := pos
	if cache != nil {
		startPos = cache.SeqLen()
	}

	startTime := time.Now()
	var allLogits tensor.Tensor
	var hiddenStates [][]float32
	var err error
	if usePagedCache {
		allLogits, hiddenStates, err = ms.runtime.VerifySpeculativeWithPagedKVAndHidden(verifyTokens, seq.KVSeqID(), startPos)
	} else {
		allLogits, hiddenStates, err = ms.runtime.VerifySpeculativeWithHidden(verifyTokens, startPos)
	}
	verifyTime := time.Since(startTime)

	if err != nil {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	ms.specMetrics.VerificationSteps++
	ms.specMetrics.VerifyTime += verifyTime

	// Step 6: Extract verification logits
	vocabSize := ms.runtime.Config().VocabSize
	allLogitsData := ms.getLogitsOnCPU(allLogits, (1+numDraft)*vocabSize)
	if allLogitsData == nil {
		return ms.runDecodeStepWithTraining(ctx, batch)
	}

	// Step 7: Evaluate all tree paths against verification logits to find
	// the path with the most accepted tokens.
	bestIdx, numAccepted, finalToken := selectBestTreePath(paths, allLogitsData, vocabSize)

	ms.specMetrics.DraftTokensAccepted += numAccepted
	ms.recordAcceptance(numAccepted)

	debugSpec := os.Getenv("MEDUSA_DEBUG") != ""
	if debugSpec {
		fmt.Printf("[TreeSpec] best_path=%v, selected_path=%v, accepted=%d, final=%d, total_paths=%d\n",
			bestPath.Tokens, paths[bestIdx].Tokens, numAccepted, finalToken, len(paths))
	}

	// Step 8: Truncate KV cache
	newCacheLen := startPos + 1 + numAccepted
	if cache != nil {
		cache.Truncate(newCacheLen)
	}
	if pagedCache != nil && seq.KVSeqID() != 0 {
		pagedCache.TruncateSequence(seq.KVSeqID(), newCacheLen)
	}
	if gpuPool := ms.runtime.GetGPUBlockPool(); gpuPool != nil {
		gpuPool.TruncateSequence(seq.KVSeqID(), newCacheLen)
	}

	// Step 9: Output accepted tokens + final token
	selectedPath := paths[bestIdx]
	acceptedTokens := make([]int, 0, numAccepted+1)
	for i := 0; i < numAccepted && i < len(selectedPath.Tokens); i++ {
		acceptedTokens = append(acceptedTokens, selectedPath.Tokens[i])
	}
	acceptedTokens = append(acceptedTokens, finalToken)

	ms.metrics.TotalTokens += len(acceptedTokens)
	ms.metrics.DecodeTokens += len(acceptedTokens)

	eosToken := 2
	if ms.tokenizer != nil {
		eosToken = ms.tokenizer.EOS()
	}

	for _, tokenID := range acceptedTokens {
		seq.AdvancePosition()
		if seq.State() == StatePending {
			seq.SetState(StateDecoding)
		}

		seq.AddGeneratedToken(tokenID)

		if tokenID == eosToken {
			seq.SetState(StateFinished)
			seq.Close()
			ms.lastHidden = nil
			return nil
		}

		if seq.ReachedMaxTokens(ms.config.MaxTokens) {
			seq.SetState(StateFinished)
			seq.Close()
			ms.lastHidden = nil
			return nil
		}

		var text string
		if ms.tokenizer != nil {
			text, _ = ms.tokenizer.Decode([]int{tokenID})
		} else {
			text = fmt.Sprintf(" %d", tokenID)
		}
		seq.PushToken(text)
	}

	// Step 10: Update hidden state for next speculation
	if len(hiddenStates) > numAccepted && hiddenStates[numAccepted] != nil {
		ms.lastHidden = hiddenStates[numAccepted]

		if ms.trainer != nil {
			for i := 0; i < len(hiddenStates) && i < len(verifyTokens); i++ {
				futureStart := i + 1
				if futureStart < len(verifyTokens) {
					actualToken := verifyTokens[futureStart]
					ms.collectTrainingSample(hiddenStates[i], actualToken)
				}
			}
		}
	} else {
		ms.lastHidden = nil
	}

	return nil
}

// selectBestTreePath evaluates candidate paths against verification logits
// from the target model. It returns the index of the path with the most
// accepted tokens, the acceptance count, and the final (correction or bonus) token.
//
// The verifyLogits are produced by running the best path through the target model,
// so at each position, the logits are conditioned on the prefix of the best path.
// For paths that share a common prefix with the best path up to the divergence point,
// the logits at positions within the shared prefix are still valid.
func selectBestTreePath(paths []medusa.CandidatePath, verifyLogits []float32, vocabSize int) (bestIdx int, accepted int, finalToken int) {
	if len(paths) == 0 || len(verifyLogits) == 0 || vocabSize <= 0 {
		return 0, 0, 0
	}

	numPositions := len(verifyLogits) / vocabSize

	// First, evaluate the best path (index 0) - this is the path we actually verified
	bestAccepted := 0
	bestFinal := 0

	if len(paths[0].Tokens) > 0 {
		for i, draftToken := range paths[0].Tokens {
			if i >= numPositions {
				break
			}
			posLogits := verifyLogits[i*vocabSize : (i+1)*vocabSize]
			targetToken := argmaxFloat32(posLogits)
			if draftToken == targetToken {
				bestAccepted++
			} else {
				bestFinal = targetToken
				break
			}
		}

		// If all tokens accepted, sample bonus token
		if bestAccepted == len(paths[0].Tokens) && bestAccepted < numPositions {
			posLogits := verifyLogits[bestAccepted*vocabSize : (bestAccepted+1)*vocabSize]
			bestFinal = argmaxFloat32(posLogits)
		}
	}

	// Only use the best path. Alternate paths cannot be safely accepted because
	// the KV cache contains the best path's tokens at every position — accepting
	// tokens from a different path creates a mismatch between the cached KV state
	// and the output token sequence, producing corrupt output.
	if bestAccepted == 0 && numPositions > 0 && bestFinal == 0 {
		posLogits := verifyLogits[0:vocabSize]
		bestFinal = argmaxFloat32(posLogits)
	}

	return 0, bestAccepted, bestFinal
}

// recordAcceptance records how many draft tokens were accepted in a speculation step.
func (ms *MedusaScheduler) recordAcceptance(accepted int) {
	ms.recentAccepted[ms.recentIdx] = accepted
	ms.recentIdx = (ms.recentIdx + 1) % len(ms.recentAccepted)
	if ms.recentIdx == 0 {
		ms.recentFull = true
	}
}

// probeSpeculation checks if Medusa heads can predict the next token correctly
// without running full verification. It compares head 0's top prediction against
// the last generated token (which the base model already produced).
func (ms *MedusaScheduler) probeSpeculation(batch []*Sequence) {
	if len(batch) == 0 || ms.lastHidden == nil || ms.trainer == nil {
		return
	}

	seq := batch[0]
	genTokens := seq.GeneratedTokens()
	if len(genTokens) < 2 {
		return
	}

	// Head 0 predicts the NEXT token from lastHidden.
	// lastHidden was set after the previous decode step.
	// The token that was actually generated is genTokens[len-1].
	// But head 0 from that hidden state would predict the token AFTER the previous step,
	// which IS genTokens[len-1] (the most recent token).
	heads := ms.trainer.Heads()
	logits := heads.Forward(0, ms.lastHidden)
	if logits == nil {
		return
	}

	predicted := argmaxFloat32(logits)
	actual := genTokens[len(genTokens)-1]

	accepted := 0
	if predicted == actual {
		accepted = 1
	}
	if os.Getenv("MEDUSA_DEBUG") != "" && ms.probeCounter == 0 {
		fmt.Printf("[Medusa] probe: predicted=%d actual=%d match=%v\n", predicted, actual, predicted == actual)
	}
	ms.recordAcceptance(accepted)
}

// speculationEnabled returns true if recent speculation acceptance rate is
// high enough to justify the verification overhead. Speculation starts OFF
// and only activates after the rolling window shows average acceptance >= 0.5
// tokens per step. This prevents garbage output from barely-trained heads.
//
// The window is populated by periodic probe steps (see step()) that run
// speculation once every probeInterval steps to check if heads have improved.
func (ms *MedusaScheduler) speculationEnabled() bool {
	if !ms.recentFull {
		return false // not enough probe data yet
	}
	total := 0
	for _, a := range ms.recentAccepted {
		total += a
	}
	avg := float64(total) / float64(len(ms.recentAccepted))
	return avg >= 0.5
}

// argmaxFloat32 returns the index of the maximum value in the slice.
func argmaxFloat32(values []float32) int {
	if len(values) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := values[0]
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// MedusaMetrics returns Medusa-specific metrics.
type MedusaMetrics struct {
	Phase            string
	SamplesCollected int64
	TrainingSteps    int64
	CurrentLoss      float32
	HeadAccuracies   []float32

	// Speculative decoding metrics
	DraftTokensGenerated int
	DraftTokensAccepted  int
	AcceptanceRate       float64
	EffectiveSpeedup     float64
}

// MedusaMetrics returns current Medusa training and speculation metrics.
func (ms *MedusaScheduler) MedusaMetrics() MedusaMetrics {
	result := MedusaMetrics{
		DraftTokensGenerated: ms.specMetrics.DraftTokensGenerated,
		DraftTokensAccepted:  ms.specMetrics.DraftTokensAccepted,
		AcceptanceRate:       ms.specMetrics.AcceptanceRate(),
		EffectiveSpeedup:     ms.specMetrics.Speedup(),
	}

	if ms.trainer == nil {
		result.Phase = "disabled"
		return result
	}

	m := ms.trainer.Metrics()
	result.Phase = m.Phase.String()
	result.SamplesCollected = m.SamplesCollected
	result.TrainingSteps = m.TrainingSteps
	result.CurrentLoss = m.CurrentLoss
	result.HeadAccuracies = m.HeadAccuracies

	return result
}

// SaveHeads saves the trained Medusa heads to a file.
// Acquires GPU lock to ensure consistent read of weight buffers.
func (ms *MedusaScheduler) SaveHeads(path string) error {
	if ms.trainer == nil {
		return fmt.Errorf("no trainer initialized")
	}
	ms.trainer.GPULock()
	defer ms.trainer.GPUUnlock()
	return ms.trainer.SaveHeads(path)
}

// Trainer returns the online trainer (for inspection/debugging).
func (ms *MedusaScheduler) Trainer() medusa.Trainer {
	return ms.trainer
}

// ForceHot forces the trainer into Hot phase for testing speculation.
func (ms *MedusaScheduler) ForceHot() {
	if ms.trainer != nil {
		ms.trainer.ForceHot()
	}
}

// getLogitsOnCPU returns logits as a float32 slice on CPU.
// NOTE: Caller must ensure GPU work is complete (e.g., Decode functions call Sync).
func (ms *MedusaScheduler) getLogitsOnCPU(logits tensor.Tensor, numElements int) []float32 {
	ptr := logits.DevicePtr()
	if ptr.IsNil() {
		return nil
	}

	// If already on CPU, use unsafe slice directly
	if ptr.Location() == tensor.CPU {
		return tensor.ToFloat32Slice(logits)
	}

	// GPU: need to copy to host
	backend := ms.runtime.Backend()
	if backend == nil {
		return nil
	}

	// Allocate host buffer as float32 directly to avoid byte-to-float conversion
	result := make([]float32, numElements)
	hostData := unsafe.Slice((*byte)(unsafe.Pointer(&result[0])), numElements*4)
	// NOTE: Sync is NOT needed here - the Decode functions already call Sync()
	// before returning, ensuring GPU work is complete.
	backend.ToHost(hostData, ptr)

	return result
}
