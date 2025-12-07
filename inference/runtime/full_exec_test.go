package runtime_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/kv"
	"vexel/inference/memory"
	"vexel/inference/runtime"
)

func TestDecodeStepExecution(t *testing.T) {
	// Setup
	cfg := runtime.Llama3_8B()
	cfg.NumHiddenLayers = 1 // Reduce layers for faster test
	cfg.HiddenSize = 128    // Small hidden size
	cfg.NumAttentionHeads = 4
	cfg.NumKeyValueHeads = 2
	cfg.IntermediateSize = 384
	cfg.VocabSize = 256

	b := cpu.NewBackend()
	ctx := &memory.InferenceContext{} // Mock/Empty
	cache := &kv.KVCache{}            // Mock/Empty
	
	rt, err := runtime.NewModelRuntime(b, ctx, cache, cfg)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	// Create inputs
	// We need 1 token input
	// Since we don't have tokenization, we just need the runtime to not crash.
	// Runtime assumes inputs are prepared? Or does DecodeStep take tokens?
	// Currently DecodeStep takes BatchRuntimeInputs.
	
	inputs := runtime.BatchRuntimeInputs{
		// TODO: Add fields for input tokens/positions if not present
	}

	// This test asserts that DecodeStep actually runs through the layers.
	// Since we can't easily spy on the backend calls without a mock backend,
	// we will rely on it running without error and producing a tensor of correct shape (if we return one).
	// Currently DecodeStep returns (Tensor, error).
	
	out, err := rt.DecodeStep(inputs)
	if err != nil {
		t.Fatalf("DecodeStep failed: %v", err)
	}
	
	// Check output shape?
	// If output is logits, it should be [Batch, Vocab].
	// Our mock currently returns empty tensor.
	// Once implemented, we expect non-empty.
	
	// For now, failure is that it returns empty tensor/nil error from the MOCK.
	// We want to force the implementation of the logic.
	// The implementation should return a tensor with shape related to VocabSize.
	
	// Note: The previous "Mock success" just returned empty tensor.
	// This test will PASS if I don't change anything, which is bad TDD.
	// I need to assert something that the MOCK doesn't do.
	// E.g. "Output tensor should have size > 0"
	
	// Currently `tensor` package struct might not expose size directly public?
	// It has `Shape`.
	
	if out.Shape().Rank() != 2 {
		t.Errorf("Expected output tensor to have Rank 2 [Batch, Vocab], got %d", out.Shape().Rank())
	}
}
