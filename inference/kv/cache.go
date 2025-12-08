package kv

import (
	"fmt"
	"vexel/inference/memory"
	"vexel/inference/tensor"
)

// KVCache manages the paged memory for Key-Value states.
type KVCache struct {
	config    KVConfig
	basePtr   tensor.DevicePtr
	maxBlocks int
	freeStack []int // Stack of free block indices
}

// NewKVCache initializes a new KV cache.
// It pre-allocates memory for 'maxBlocks' from the provided context's KV arena.
func NewKVCache(ctx *memory.InferenceContext, config KVConfig, maxBlocks int) (*KVCache, error) {
	blockBytes := config.BlockBytes()
	totalBytes := blockBytes * maxBlocks

	arena := ctx.GetArena(memory.KV)
	if arena == nil {
		return nil, fmt.Errorf("KV arena not initialized in context")
	}

	// Allocate the entire pool
	ptr, err := arena.Alloc(totalBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate KV cache: %v", err)
	}

	// Initialize free stack with all block indices [0, maxBlocks-1]
	// We push in reverse order so that we pop 0 first (optional optimization for locality)
	freeStack := make([]int, maxBlocks)
	for i := 0; i < maxBlocks; i++ {
		freeStack[i] = maxBlocks - 1 - i
	}

	return &KVCache{
		config:    config,
		basePtr:   ptr,
		maxBlocks: maxBlocks,
		freeStack: freeStack,
	}, nil
}

// Config returns the cache configuration.
func (c *KVCache) Config() KVConfig {
	return c.config
}

// FreeBlocks returns the number of available blocks.
func (c *KVCache) FreeBlocks() int {
	return len(c.freeStack)
}

// GetView returns a pointer to the KV storage for a given layer.
// Simplified: Assumes contiguous for MVP.
// Returns K and V pointers.
func (c *KVCache) GetView(layer int, pos int) (tensor.DevicePtr, tensor.DevicePtr) {
	// TODO: Implement paging lookup table.
	// For MVP, return nil to signify "no cache" (generating first token).
	// Or return offset into basePtr if we assume linear mapping per seq?
	return tensor.DevicePtr{}, tensor.DevicePtr{}
}