package scheduler_test

import (
	"testing"
	"vexel/inference/scheduler"
)

func TestSequenceStreaming(t *testing.T) {
	seq := scheduler.NewSequence(1, "Test")
	
	// Channel should be initialized
	if seq.TokenChan() == nil {
		t.Error("Sequence TokenChan is nil")
	}
	
	// Simulate scheduler pushing a token
	go func() {
		seq.PushToken("Hello")
		seq.Close()
	}()
	
	// Consume
	token := <-seq.TokenChan()
	if token != "Hello" {
		t.Errorf("Expected 'Hello', got '%s'", token)
	}
	
	// Check close
	_, ok := <-seq.TokenChan()
	if ok {
		t.Error("Channel should be closed")
	}
}
