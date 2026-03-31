package scheduler

import (
	"context"
	"fmt"
	"time"

	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
)

// SpeculativeScheduler wraps Scheduler with speculative decoding support
// using a separate draft model for token speculation.
type SpeculativeScheduler struct {
	*Scheduler

	draftRuntime *runtime.ModelRuntime
	decoder      *SpeculativeDecoder
	specConfig   SpeculativeConfig
	specMetrics  SpeculativeMetrics
	adaptive     *AdaptiveDraftLength
}

// NewSpeculativeScheduler creates a speculative scheduler using a separate draft model.
// The draft model should be smaller/faster than the target model.
func NewSpeculativeScheduler(
	target *runtime.ModelRuntime,
	draft *runtime.ModelRuntime,
	tok *tokenizer.Tokenizer,
	config Config,
	specConfig SpeculativeConfig,
) (*SpeculativeScheduler, error) {
	base, err := NewScheduler(target, tok, config)
	if err != nil {
		return nil, err
	}

	s := sampler.New(config.SamplerConfig, 42)

	adaptiveCfg := AdaptiveConfig{
		InitialDraftTokens: specConfig.NumDraftTokens,
		MinDraftTokens:     1,
		MaxDraftTokens:     8,
		WindowSize:         10,
		IncreaseThreshold:  0.80,
		DecreaseThreshold:  0.40,
	}

	ss := &SpeculativeScheduler{
		Scheduler:    base,
		draftRuntime: draft,
		specConfig:   specConfig,
		decoder:      NewSpeculativeDecoder(target, draft, s, specConfig),
		adaptive:     NewAdaptiveDraftLength(adaptiveCfg),
	}

	return ss, nil
}

// Run starts the speculative scheduler's main loop.
func (ss *SpeculativeScheduler) Run(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		if err := ss.step(ctx); err != nil {
			return err
		}

		// If sequences remain, loop immediately
		if ss.SequenceCount() > 0 {
			continue
		}

		// No sequences — wait for wakeup from AddSequence
		done := make(chan struct{})
		go func() {
			ss.cond.L.Lock()
			ss.cond.Wait()
			ss.cond.L.Unlock()
			close(done)
		}()

		select {
		case <-ctx.Done():
			ss.cond.Broadcast()
			return nil
		case <-done:
		}
	}
}

// step performs a single scheduling iteration with speculative decoding.
func (ss *SpeculativeScheduler) step(ctx context.Context) error {
	ready := ss.collectReady()

	ss.mu.Lock()
	ss.metrics.ActiveSequences = len(ss.sequences)
	ss.mu.Unlock()

	batch := ss.formBatches(ready)
	if len(batch) == 0 {
		return nil
	}

	// Only use speculative decoding for single-sequence GPU KV cache path
	if len(batch) == 1 && ss.runtime.GPUKVCache() != nil {
		return ss.runSpeculativeDecodeStep(ctx, batch)
	}

	// Fall back to standard decode for multi-sequence or non-GPU paths
	return ss.runDecodeStep(ctx, batch)
}

// runSpeculativeDecodeStep uses the draft model to speculate tokens,
// then verifies with the target model.
func (ss *SpeculativeScheduler) runSpeculativeDecodeStep(ctx context.Context, batch []*Sequence) error {
	seq := batch[0]

	// Handle prefill first
	if seq.State() == StatePending && len(seq.PromptTokens()) > 0 && !seq.IsPrefillComplete() {
		if err := ss.runGPUPrefill(seq); err != nil {
			return err
		}
		// Also prefill draft model
		if err := ss.prefillDraftModel(seq); err != nil {
			return err
		}
		return nil
	}

	// Check if sequence is ready for decode
	if seq.State() != StateDecoding && !(seq.State() == StatePending && seq.IsPrefillComplete()) {
		return nil
	}

	// Get current input token and position
	inputToken, pos, hasMore := seq.NextInputToken()
	if !hasMore {
		inputToken = 1
		pos = 0
	}

	startTime := time.Now()

	// Step 1: Adapt draft token count based on recent acceptance rate
	ss.decoder.config.NumDraftTokens = ss.adaptive.NumDraftTokens()

	// Step 2: Generate draft tokens from draft model
	draftTokens, draftProbs, err := ss.decoder.GenerateDraftTokensFrom(inputToken, pos)
	if err != nil {
		// Fall back to standard decode on error
		return ss.runDecodeStep(ctx, batch)
	}

	if len(draftTokens) == 0 {
		// No draft tokens generated, fall back to standard single-token decode
		return ss.runDecodeStep(ctx, batch)
	}

	// Step 3: Verify draft tokens with target model
	numAccepted, finalToken, _, err := ss.decoder.VerifyDraftTokens(pos, inputToken, draftTokens, draftProbs)
	if err != nil {
		return ss.runDecodeStep(ctx, batch)
	}

	// Step 4: Record acceptance for adaptive draft length tuning
	ss.adaptive.RecordStep(numAccepted, len(draftTokens))

	// Step 5: Truncate target KV cache to keep only accepted tokens
	// Verification added (1 + len(draftTokens)) entries to cache.
	// We want to keep entries up to pos + 1 + numAccepted.
	if cache := ss.runtime.GPUKVCache(); cache != nil {
		newLen := pos + 1 + numAccepted
		cache.Truncate(newLen)
	}

	// Step 6: Truncate draft model KV cache too
	// Draft generated draftTokens entries starting at pos.
	// We want to keep entries up to pos + numAccepted.
	if cache := ss.draftRuntime.GPUKVCache(); cache != nil {
		newLen := pos + numAccepted
		cache.Truncate(newLen)
	}

	decodeTime := time.Since(startTime)

	// Step 7: Output accepted draft tokens + final token
	acceptedTokens := make([]int, 0, numAccepted+1)
	for i := 0; i < numAccepted; i++ {
		acceptedTokens = append(acceptedTokens, draftTokens[i])
	}
	acceptedTokens = append(acceptedTokens, finalToken)

	// Update speculative metrics
	ss.specMetrics = ss.decoder.Metrics()

	// Process each accepted token into the sequence
	for _, tokenID := range acceptedTokens {
		seq.AdvancePosition()
		if seq.State() == StatePending {
			seq.SetState(StateDecoding)
		}

		seq.AddGeneratedToken(tokenID)

		// Check for EOS or extra stop tokens
		if ss.Scheduler.isStopToken(tokenID, seq) {
			seq.SetState(StateFinished)
			seq.Close()
			break
		}

		// Check max tokens
		if seq.ReachedMaxTokens(ss.config.MaxTokens) {
			seq.SetState(StateFinished)
			seq.Close()
			break
		}

		// Decode and push token
		var text string
		if ss.tokenizer != nil {
			text, _ = ss.tokenizer.Decode([]int{tokenID})
		} else {
			text = fmt.Sprintf(" %d", tokenID)
		}
		seq.PushToken(text)
	}

	// Update metrics
	ss.mu.Lock()
	ss.metrics.TotalTokens += len(acceptedTokens)
	ss.metrics.DecodeTokens += len(acceptedTokens)
	ss.metrics.DecodeTime += decodeTime
	ss.mu.Unlock()

	return nil
}

// prefillDraftModel runs prefill on the draft model using the sequence's prompt tokens.
func (ss *SpeculativeScheduler) prefillDraftModel(seq *Sequence) error {
	promptTokens := seq.PromptTokens()
	if len(promptTokens) == 0 {
		return nil
	}

	// Reset draft model cache
	if cache := ss.draftRuntime.GPUKVCache(); cache != nil {
		cache.Reset()
	}

	// Run prefill on draft model
	_, err := ss.draftRuntime.DecodeWithGPUKV(promptTokens, 0)
	return err
}

// SpecMetrics returns the current speculative decoding metrics.
func (ss *SpeculativeScheduler) SpecMetrics() SpeculativeMetrics {
	return ss.specMetrics
}
