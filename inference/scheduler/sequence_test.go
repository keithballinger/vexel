package scheduler_test

import (
	"testing"
	"vexel/inference/scheduler"
)

func TestSequenceState(t *testing.T) {
	// 1. Initialization
	seqID := scheduler.SequenceID(101)
	seq := scheduler.NewSequence(seqID, "What is the capital of France?")

	if seq.ID() != seqID {
		t.Errorf("expected ID %v, got %v", seqID, seq.ID())
	}

	if seq.State() != scheduler.StatePending {
		t.Errorf("expected initial state Pending, got %v", seq.State())
	}

	// 2. Transition to Prefill
	seq.SetState(scheduler.StatePrefill)
	if seq.State() != scheduler.StatePrefill {
		t.Errorf("expected state Prefill, got %v", seq.State())
	}

	// 3. Transition to Decoding
	seq.SetState(scheduler.StateDecoding)
	if seq.State() != scheduler.StateDecoding {
		t.Errorf("expected state Decoding, got %v", seq.State())
	}

	// 4. Transition to Finished
	seq.SetState(scheduler.StateFinished)
	if seq.State() != scheduler.StateFinished {
		t.Errorf("expected state Finished, got %v", seq.State())
	}
}
