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
	MaxBatchSize int
	MaxSequences int
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
	config    Config
	sequences map[SequenceID]*Sequence
	metrics   SchedulerMetrics
}

// NewScheduler creates a new Scheduler instance.
func NewScheduler(rt *runtime.ModelRuntime, tok *tokenizer.Tokenizer, config Config) (*Scheduler, error) {
	if rt == nil {
		return nil, fmt.Errorf("runtime cannot be nil")
	}
	
	return &Scheduler{
		runtime:   rt,
		tokenizer: tok,
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
		if seq.State() == StatePending || seq.State() == StateDecoding {
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

	// Prepare inputs for the runtime
	// In a real implementation, we'd extract tokens/cache IDs from the sequences.
	inputs := runtime.BatchRuntimeInputs{
		// ... map batch to inputs ...
	}

	// Execute model
	// Note: runtime.DecodeStep signature is (inputs BatchRuntimeInputs) (tensor.Tensor, error)
	// We ignore the output tensor for now as we aren't processing logits yet.
	logits, err := s.runtime.DecodeStep(inputs)
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
		if seq.State() == StatePending {
			seq.SetState(StateDecoding)
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
		
		// Sample
		tokenID := sampler.Argmax(seqLogits)
		
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
func (s *Scheduler) AddSequence(seq *Sequence) {
	s.sequences[seq.ID()] = seq
}

// SequenceCount returns the number of active sequences.
func (s *Scheduler) SequenceCount() int {
	return len(s.sequences)
}

// Metrics returns the current performance metrics.
func (s *Scheduler) Metrics() SchedulerMetrics {
	return s.metrics
}