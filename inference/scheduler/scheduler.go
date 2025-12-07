package scheduler

import (
	"context"
	"fmt"
	"time"
	"vexel/inference/runtime"
)

// Config holds configuration for the scheduler.
type Config struct {
	MaxBatchSize int
	MaxSequences int
}

// Scheduler manages the execution of sequences.
type Scheduler struct {
	runtime   *runtime.ModelRuntime
	config    Config
	sequences map[SequenceID]*Sequence
}

// NewScheduler creates a new Scheduler instance.
func NewScheduler(rt *runtime.ModelRuntime, config Config) (*Scheduler, error) {
	if rt == nil {
		return nil, fmt.Errorf("runtime cannot be nil")
	}
	
	return &Scheduler{
		runtime:   rt,
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
	// ready := s.collectReady()
	
	// 2. Form batch
	// batch := s.formBatches(ready)
	
	// 3. Run DecodeStep (TODO)
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
// It respects MaxBatchSize and scheduling policy (currently simple FIFO/priority).
func (s *Scheduler) formBatches(ready []*Sequence) []*Sequence {
	if len(ready) == 0 {
		return nil
	}

	// Basic implementation: Take up to MaxBatchSize
	// Priority: Decoding > Pending (Simple heuristic to finish tasks faster)
	// For now, we just truncate.
	
	if len(ready) <= s.config.MaxBatchSize {
		return ready
	}
	
	return ready[:s.config.MaxBatchSize]
}