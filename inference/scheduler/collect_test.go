package scheduler

import (
	"testing"
	"vexel/inference/runtime"
)

func TestCollectReady(t *testing.T) {
	// Setup
	cfg := Config{
		MaxBatchSize: 8,
		MaxSequences: 16,
	}
	rt := &runtime.ModelRuntime{}
	sched, _ := NewScheduler(rt, nil, cfg)

	// Create sequences
	s1 := NewSequence(1, "Prompt 1") // Pending
	s2 := NewSequence(2, "Prompt 2")
	s2.SetState(StateDecoding) // Decoding
	s3 := NewSequence(3, "Prompt 3")
	s3.SetState(StateFinished) // Finished

	// Add to scheduler directly (simulating admission)
	if sched.sequences == nil {
		sched.sequences = make(map[SequenceID]*Sequence)
	}
	sched.sequences[s1.ID()] = s1
	sched.sequences[s2.ID()] = s2
	sched.sequences[s3.ID()] = s3

	// Test collectReady
	ready := sched.collectReady()

	// Expect s1 (Pending) and s2 (Decoding) to be ready. s3 (Finished) should be ignored.
	if len(ready) != 2 {
		t.Errorf("Expected 2 ready sequences, got %d", len(ready))
	}

	foundS1 := false
	foundS2 := false

	for _, s := range ready {
		if s.ID() == s1.ID() {
			foundS1 = true
		}
		if s.ID() == s2.ID() {
			foundS2 = true
		}
	}

	if !foundS1 {
		t.Error("Sequence 1 (Pending) missing from ready list")
	}
	if !foundS2 {
		t.Error("Sequence 2 (Decoding) missing from ready list")
	}
}
