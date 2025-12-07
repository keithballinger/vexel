package scheduler

import (
	"fmt"
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
