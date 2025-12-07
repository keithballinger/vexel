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
	runtime *runtime.ModelRuntime
	config  Config
	// sequences map[SequenceID]*Sequence // To be added
}

// NewScheduler creates a new Scheduler instance.
func NewScheduler(rt *runtime.ModelRuntime, config Config) (*Scheduler, error) {
	if rt == nil {
		return nil, fmt.Errorf("runtime cannot be nil")
	}
	
	return &Scheduler{
		runtime: rt,
		config:  config,
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
	// 1. Collect ready sequences (TODO)
	// 2. Form batch (TODO)
	// 3. Run DecodeStep (TODO)
	return nil
}