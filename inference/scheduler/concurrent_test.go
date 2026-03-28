package scheduler

import (
	"sync"
	"sync/atomic"
	"testing"

	"vexel/inference/pkg/sampler"
)

// newTestScheduler creates a minimal Scheduler for unit tests that don't need
// a real ModelRuntime. It bypasses NewScheduler (which requires a non-nil runtime)
// by constructing the struct directly.
func newTestScheduler(cfg Config) *Scheduler {
	s := sampler.New(cfg.SamplerConfig, 42)
	return &Scheduler{
		sampler:   s,
		config:    cfg,
		sequences: make(map[SequenceID]*Sequence),
		signal:    make(chan struct{}, 1),
	}
}

// addSequenceNoRuntime registers a sequence without touching the runtime
// (no tokenizer encoding, no KV cache creation). This mirrors the core
// registration logic of AddSequence for testing purposes.
func (s *Scheduler) addSequenceNoRuntime(seq *Sequence) {
	s.mu.Lock()
	s.sequences[seq.ID()] = seq
	s.mu.Unlock()
	s.wakeUp()
}

func TestConcurrentAddSequenceAllRegistered(t *testing.T) {
	cfg := Config{
		MaxBatchSize:  4,
		MaxSequences:  100,
		MaxTokens:     8,
		SamplerConfig: sampler.DefaultConfig(),
	}
	sched := newTestScheduler(cfg)

	// Add 50 sequences concurrently
	numSeqs := 50
	var wg sync.WaitGroup
	var added int32

	for i := 0; i < numSeqs; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			seq := NewSequence(SequenceID(id+1), "test")
			sched.addSequenceNoRuntime(seq)
			atomic.AddInt32(&added, 1)
		}(i)
	}

	wg.Wait()

	if count := sched.SequenceCount(); count != numSeqs {
		t.Errorf("expected %d sequences registered, got %d", numSeqs, count)
	}

	if int(atomic.LoadInt32(&added)) != numSeqs {
		t.Errorf("expected %d goroutines to complete, got %d", numSeqs, added)
	}
}

func TestRapidAddSequenceAllRegistered(t *testing.T) {
	cfg := Config{
		MaxBatchSize:  8,
		MaxSequences:  200,
		MaxTokens:     4,
		SamplerConfig: sampler.DefaultConfig(),
	}
	sched := newTestScheduler(cfg)

	// Rapidly add 100 sequences sequentially
	for i := 0; i < 100; i++ {
		seq := NewSequence(SequenceID(i+1), "hello")
		sched.addSequenceNoRuntime(seq)
	}

	if count := sched.SequenceCount(); count != 100 {
		t.Errorf("expected 100 sequences, got %d", count)
	}
}
