package scheduler

import (
	"testing"
	"vexel/inference/runtime"
)

func TestFormBatches(t *testing.T) {
	// Setup with MaxBatchSize = 2
	cfg := Config{
		MaxBatchSize: 2,
		MaxSequences: 16,
	}
	rt := &runtime.ModelRuntime{}
	sched, _ := NewScheduler(rt, cfg)

	// Create 3 sequences
	s1 := NewSequence(1, "1")
	s1.SetState(StatePending)
	
	s2 := NewSequence(2, "2")
	s2.SetState(StateDecoding)
	
	s3 := NewSequence(3, "3")
	s3.SetState(StateDecoding)

	ready := []*Sequence{s1, s2, s3}

	// Test formBatches
	// Since MaxBatchSize is 2, we expect a batch of size 2.
	// We might also expect it to prioritize decoding over prefill, or FIFO.
	// For now, let's just check the size constraint.
	
	batch := sched.formBatches(ready)

	if len(batch) > cfg.MaxBatchSize {
		t.Errorf("Expected batch size <= %d, got %d", cfg.MaxBatchSize, len(batch))
	}

	if len(batch) == 0 {
		t.Error("Expected non-empty batch")
	}

	// Ensure the returned sequences are from the ready list
	for _, seq := range batch {
		found := false
		for _, r := range ready {
			if r.ID() == seq.ID() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Batch contains sequence %d which was not in ready list", seq.ID())
		}
	}
}
