package runtime_test

import (
	"testing"
	"vexel/inference/kv"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestBatchRuntimeInputs(t *testing.T) {
	// Setup
	inputTokens := []int{101, 102, 103}
	// Create mock handles for 3 sequences
	handles := []*kv.SeqKVHandle{
		kv.NewSeqKVHandle([]kv.PageIndex{1}),
		kv.NewSeqKVHandle([]kv.PageIndex{2}),
		kv.NewSeqKVHandle([]kv.PageIndex{3}),
	}
	
	// Create Inputs
	inputs := runtime.NewBatchRuntimeInputs(inputTokens, handles)

	// Verify
	if len(inputs.Tokens()) != 3 {
		t.Errorf("Expected 3 tokens, got %d", len(inputs.Tokens()))
	}
	if len(inputs.KVHandles()) != 3 {
		t.Errorf("Expected 3 handles, got %d", len(inputs.KVHandles()))
	}

	// Verify tensor conversion (conceptually)
	// The runtime will eventually convert []int to a tensor on the device
	// Here we just check the container holds the raw data correctly
	if inputs.Tokens()[0] != 101 {
		t.Error("Token data mismatch")
	}
}
