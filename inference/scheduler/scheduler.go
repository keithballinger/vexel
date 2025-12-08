package scheduler

import (
	"context"
	"fmt"
	"time"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// Config holds configuration for the scheduler.
type Config struct {
	MaxBatchSize  int
	MaxSequences  int
	MaxTokens     int            // Max tokens to generate per sequence (0 = unlimited)
	SamplerConfig sampler.Config // Sampling parameters
}

// SchedulerMetrics holds performance indicators.
type SchedulerMetrics struct {
	ActiveSequences int
	CompletedSequences int
	TotalTokens int
}

// Scheduler manages the execution of sequences.
type Scheduler struct {
	runtime   *runtime.ModelRuntime
	tokenizer *tokenizer.Tokenizer
	sampler   *sampler.Sampler
	config    Config
	sequences map[SequenceID]*Sequence
	metrics   SchedulerMetrics
}

// NewScheduler creates a new Scheduler instance.
func NewScheduler(rt *runtime.ModelRuntime, tok *tokenizer.Tokenizer, config Config) (*Scheduler, error) {
	if rt == nil {
		return nil, fmt.Errorf("runtime cannot be nil")
	}

	// Create sampler with config (use time-based seed for randomness)
	s := sampler.New(config.SamplerConfig, time.Now().UnixNano())

	return &Scheduler{
		runtime:   rt,
		tokenizer: tok,
		sampler:   s,
		config:    config,
		sequences: make(map[SequenceID]*Sequence),
	}, nil
}

// Run starts the scheduler's main loop.
// It blocks until the context is canceled or a fatal error occurs.
func (s *Scheduler) Run(ctx context.Context) error {
	// Simple ticker for now, maybe event-driven later
	ticker := time.NewTicker(1 * time.Millisecond) // aggressive poll
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := s.step(ctx); err != nil {
				return err
			}
		}
	}
}

// step performs a single scheduling iteration.
func (s *Scheduler) step(ctx context.Context) error {
	// 1. Collect ready sequences
	ready := s.collectReady()
	
	// Update Active Sequences metric
	s.metrics.ActiveSequences = len(s.sequences)
	
	// 2. Form batch
	batch := s.formBatches(ready)
	if len(batch) == 0 {
		return nil
	}
	
	// 3. Run DecodeStep
	if err := s.runDecodeStep(ctx, batch); err != nil {
		return err
	}

	return nil
}

// collectReady identifies sequences that are eligible for execution.
func (s *Scheduler) collectReady() []*Sequence {
	ready := make([]*Sequence, 0)
	for _, seq := range s.sequences {
		switch seq.State() {
		case StatePending, StatePrefill, StateDecoding:
			ready = append(ready, seq)
		}
	}
	return ready
}

// formBatches selects a subset of ready sequences to run in the next step.
func (s *Scheduler) formBatches(ready []*Sequence) []*Sequence {
	if len(ready) == 0 {
		return nil
	}
	
	if len(ready) <= s.config.MaxBatchSize {
		return ready
	}
	
	return ready[:s.config.MaxBatchSize]
}

// runDecodeStep orchestrates the model execution for a batch of sequences.
func (s *Scheduler) runDecodeStep(ctx context.Context, batch []*Sequence) error {
	if len(batch) == 0 {
		return nil
	}

	// Check if we have paged KV cache
	usePagedCache := s.runtime.PagedKVCache() != nil

	// Prepare inputs for the runtime
	tokens := make([]int, len(batch))
	positions := make([]int, len(batch))
	seqIDs := make([]int64, len(batch))

	for i, seq := range batch {
		token, pos, hasMore := seq.NextInputToken()
		if !hasMore {
			// No token available - use BOS as fallback
			token = 1
			pos = 0
		}
		tokens[i] = token
		positions[i] = pos
		seqIDs[i] = seq.KVSeqID()
	}

	var logits tensor.Tensor
	var err error

	if usePagedCache {
		// Use paged KV cache path
		inputs := runtime.NewBatchRuntimeInputsFull(tokens, positions, seqIDs)
		logits, err = s.runtime.DecodeStepWithPagedKV(inputs)
	} else {
		// Legacy path without paged cache
		inputs := runtime.NewBatchRuntimeInputsWithPos(tokens, positions, nil)
		logits, err = s.runtime.DecodeStep(inputs)
	}

	if err != nil {
		return err
	}

	// Update metrics
	s.metrics.TotalTokens += len(batch)
	
	// Sample and Decode
	// Logits: [Batch, Vocab]
	// We need raw access.
	logitsData := tensor.ToFloat32Slice(logits)
	vocabSize := s.runtime.Config().VocabSize

	for i, seq := range batch {
		// Advance position after processing this token
		seq.AdvancePosition()

		// Update state based on prefill completion
		if seq.State() == StatePending {
			if seq.IsPrefillComplete() {
				seq.SetState(StateDecoding)
			} else {
				seq.SetState(StatePrefill)
			}
		} else if seq.State() == StatePrefill && seq.IsPrefillComplete() {
			seq.SetState(StateDecoding)
		}

		// Only sample and output on the last prompt token or during decode
		if !seq.IsPrefillComplete() {
			// Still prefilling - don't sample yet
			continue
		}

		// Extract logits for this sequence
		start := i * vocabSize
		end := start + vocabSize

		// Safety check
		if logitsData == nil || end > len(logitsData) {
			// Fallback if tensor is invalid (e.g. mock runtime returning empty)
			seq.PushToken("?")
			continue
		}

		seqLogits := logitsData[start:end]

		// Sample using configured sampler
		tokenID := s.sampler.Sample(seqLogits)

		// Track the generated token
		seq.AddGeneratedToken(tokenID)

		// Check for EOS token
		eosToken := 2 // Default Llama EOS
		if s.tokenizer != nil {
			eosToken = s.tokenizer.EOS()
		}
		if tokenID == eosToken {
			seq.SetState(StateFinished)
			seq.Close()
			continue
		}

		// Check for max tokens
		if s.config.MaxTokens > 0 && len(seq.GeneratedTokens()) >= s.config.MaxTokens {
			seq.SetState(StateFinished)
			seq.Close()
			continue
		}

		// Decode
		// If tokenizer is nil (e.g. benchmark/test), push ID as string
		var text string
		if s.tokenizer != nil {
			text, _ = s.tokenizer.Decode([]int{tokenID})
		} else {
			text = fmt.Sprintf(" %d", tokenID)
		}

		seq.PushToken(text)
	}

	return nil
}

// AddSequence registers a new sequence with the scheduler.
// If a tokenizer is available, encodes the prompt into tokens.
func (s *Scheduler) AddSequence(seq *Sequence) {
	// Encode prompt if tokenizer available and prompt not yet encoded
	if s.tokenizer != nil && len(seq.PromptTokens()) == 0 && seq.prompt != "" {
		tokens, err := s.tokenizer.Encode(seq.prompt)
		if err == nil {
			seq.SetPromptTokens(tokens)
		}
	}

	// Create sequence in PagedKVCache if available
	if cache := s.runtime.PagedKVCache(); cache != nil {
		kvSeqID := cache.CreateSequence()
		seq.SetKVSeqID(kvSeqID)
	}

	s.sequences[seq.ID()] = seq
}

// RemoveSequence removes a sequence and cleans up its KV cache.
func (s *Scheduler) RemoveSequence(id SequenceID) {
	seq, ok := s.sequences[id]
	if !ok {
		return
	}

	// Clean up KV cache
	if cache := s.runtime.PagedKVCache(); cache != nil && seq.KVSeqID() != 0 {
		cache.DeleteSequence(seq.KVSeqID())
	}

	delete(s.sequences, id)
	s.metrics.CompletedSequences++
}

// SequenceCount returns the number of active sequences.
func (s *Scheduler) SequenceCount() int {
	return len(s.sequences)
}

// Metrics returns the current performance metrics.
func (s *Scheduler) Metrics() SchedulerMetrics {
	return s.metrics
}