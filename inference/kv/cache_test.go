package kv_test

import (
	"testing"
	"vexel/inference/kv"
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

func TestKVCache(t *testing.T) {
	// Setup dependencies
	loc := tensor.CPU
	ctx := memory.NewInferenceContext(loc)
	// Add a KV arena
	ctx.AddArena(memory.KV, 1024*1024) // 1MB

	dtype := tensor.Float16
	headDim := 64
	blockLen := 16
	cfg := kv.NewKVConfig(dtype, headDim, blockLen)

	// Create Cache
	cache, err := kv.NewKVCache(ctx, cfg, 100) // Max 100 blocks
	if err != nil {
		t.Fatalf("NewKVCache failed: %v", err)
	}

	if cache.Config().BlockBytes() != cfg.BlockBytes() {
		t.Error("Cache config mismatch")
	}

	// Test page allocation
	// We expect the cache to manage a list of free blocks
	// Ideally, we'd test allocating a sequence, but for this first pass,
	// we just want to ensure the struct exists and compiles with these methods.
	if cache.FreeBlocks() != 100 {
		t.Errorf("Expected 100 free blocks, got %d", cache.FreeBlocks())
	}
}
