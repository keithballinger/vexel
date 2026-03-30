package kv

import (
	"testing"
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

func TestKVCachePaging(t *testing.T) {
	// Setup mock context
	ctx := memory.NewInferenceContext(tensor.CPU)
	headDim := 128
	blockLen := 16
	maxBlocks := 10
	config := NewKVConfig(tensor.Float32, headDim, blockLen)

	// Pre-calculate size and add arena
	totalBytes := config.BlockBytes() * maxBlocks
	ctx.AddArena(memory.KV, totalBytes)

	// Initialize cache
	cache, err := NewKVCache(ctx, config, maxBlocks)
	if err != nil {
		t.Fatalf("Failed to create KVCache: %v", err)
	}

	if cache.FreeBlocks() != maxBlocks {
		t.Errorf("Expected %d free blocks, got %d", maxBlocks, cache.FreeBlocks())
	}

	// 1. Allocate blocks for a sequence
	seqLen := 40 // Should take 3 blocks (16+16+8)
	blocks, err := cache.AllocateBlocks(seqLen)
	if err != nil {
		t.Fatalf("Failed to allocate blocks: %v", err)
	}

	if len(blocks) != 3 {
		t.Errorf("Expected 3 blocks for seqLen 40, got %d", len(blocks))
	}

	if cache.FreeBlocks() != maxBlocks-3 {
		t.Errorf("Expected %d free blocks remaining, got %d", maxBlocks-3, cache.FreeBlocks())
	}

	// 2. Map virtual positions to physical block pointers
	// (This will be used by the scheduler to build the page table for the GPU)
	// For now, let's verify we can get the base pointers for these blocks
	for i, blockIdx := range blocks {
		ptr := cache.GetBlockPtr(blockIdx)
		if ptr.IsNil() {
			t.Errorf("Block %d (index %d) pointer is nil", i, blockIdx)
		}
	}

	// 3. Free blocks
	cache.FreeBlocksList(blocks)
	if cache.FreeBlocks() != maxBlocks {
		t.Errorf("Expected %d free blocks after freeing, got %d", maxBlocks, cache.FreeBlocks())
	}
}
