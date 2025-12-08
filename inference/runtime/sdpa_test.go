package runtime_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// SpyBackendExtended includes Matmul counting
// We reuse SpyBackend from spy_test.go if accessible (same package), 
// but it was defined in a _test file, so it might not be visible if packages differ.
// Both are package runtime_test. So SpyBackend should be visible.

func TestSDPA(t *testing.T) {
	// Setup Spy
	spy := &SpyBackend{Backend: cpu.NewBackend()}
	block := runtime.NewBlockRuntime(spy)

	// Mock Q, K, V
	// Seq=2, HeadDim=4
	// Q, K, V should be [Seq, HeadDim]
	// In GQA/MHA, it's [Seq, Heads, HeadDim].
	// Our BlockRuntime Execute currently assumes flattened?
	// Let's assume Execute handles the internal calls.
	
	// We want to verify that Execute calls Matmul for Scores (Q*K) and Output (Score*V).
	// Since Execute is one big function, we test it end-to-end.
	
	input := tensor.NewTensor(tensor.NewShape(1, 4), tensor.Float32, tensor.NewDevicePtr(tensor.CPU, 123))
	scratch := tensor.NewTensor(tensor.NewShape(100), tensor.Float32, tensor.NewDevicePtr(tensor.CPU, 456))
	
	block.Execute(input, scratch, nil, 0, 0)
	
	// QKV Proj (3 Matmuls) + O Proj (1 Matmul) + MLP (3 Matmuls) = 7
	// Plus SDPA involves Q*K^T and Attn*V.
	// Are these separate backend calls?
	// If SDPA is implemented via Matmul, we expect MORE Matmul calls.
	// Q*K^T -> [Seq, Seq]
	// Attn*V -> [Seq, Dim]
	
	// Current implementation:
	// Q, K, V proj are Matmuls.
	// RoPE.
	// SDPA... ?
	// O proj is Matmul.
	
	// If SDPA is implemented using primitive Matmuls, we expect +2 Matmuls per head?
	// Or we might implement `backend.SDPA(...)` if we want a fused kernel.
	// The plan says "Implement SDPA with causal masking".
	// If it's a kernel, we need to add it to Interface.
	// If it's logic in BlockRuntime composed of Matmuls, we test logic.
	
	// Given the complexity of Attention (Transpose, Reshape, Mask, Softmax, Matmul), 
	// typically naive implementations compose primitives.
	// But optimizing it usually means a Fused SDPA kernel (FlashAttention).
	
	// Let's assume for CPU we compose primitives.
	// We need Softmax (already added).
	
	// This test asserts that Softmax IS called during execution (part of SDPA).
	if spy.SoftmaxCalls == 0 {
		t.Error("Expected Softmax calls for SDPA, got 0")
	}
}
