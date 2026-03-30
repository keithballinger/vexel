package runtime_test

import (
	"testing"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestMemoryPlan(t *testing.T) {
	cfg := runtime.Llama3_8B()

	// Plan for:
	// - Batch Size: 64
	// - Context Length: 2048
	// - Weights: Q4_0
	// - KV Cache: Standard (BF16)

	plan := cfg.MemoryPlan(64, 2048, tensor.Q4_0)

	// Weights: ~4GB (Q4)
	if plan.Weights < 3_500_000_000 {
		t.Errorf("Expected >3.5GB weights, got %d", plan.Weights)
	}

	// KV Cache:
	// 1 token = 131072 bytes (from previous test)
	// 64 seqs * 2048 context = 131,072 tokens
	// Total = 131072 * 131072 = ~17 GB!
	// Wait, previous test was 1 seq * 1024 context = 134MB.
	// So 64 * 2 = 128x more.
	// 134MB * 128 = ~17 GB.

	if plan.KV < 15_000_000_000 {
		t.Errorf("Expected >15GB KV cache, got %d", plan.KV)
	}

	// Scratch:
	// Batch 64.
	// Previous test Batch 128 was >32MB. So Batch 64 should be >16MB.
	if plan.Scratch < 16_000_000 {
		t.Errorf("Expected >16MB scratch, got %d", plan.Scratch)
	}

	// Total
	expectedTotal := plan.Weights + plan.KV + plan.Scratch
	if plan.Total != expectedTotal {
		t.Errorf("Total %d does not match sum %d", plan.Total, expectedTotal)
	}
}
