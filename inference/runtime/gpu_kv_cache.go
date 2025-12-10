package runtime

import (
	"sync"

	"vexel/inference/backend"
	"vexel/inference/tensor"
)

// GPUKVCache is a simple contiguous KV cache that lives entirely on GPU.
// For single-sequence inference, this is more efficient than paging through CPU.
type GPUKVCache struct {
	mu sync.Mutex

	backend    backend.Backend
	numLayers  int
	numKVHeads int
	headDim    int
	maxSeqLen  int

	// Per-layer K and V buffers: [maxSeqLen, numKVHeads, headDim]
	kBuffers []tensor.DevicePtr
	vBuffers []tensor.DevicePtr

	// Current sequence length (same for all layers in single-sequence case)
	seqLen int
}

// NewGPUKVCache creates a new GPU-resident KV cache.
func NewGPUKVCache(b backend.Backend, numLayers, numKVHeads, headDim, maxSeqLen int) *GPUKVCache {
	cache := &GPUKVCache{
		backend:    b,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		maxSeqLen:  maxSeqLen,
		kBuffers:   make([]tensor.DevicePtr, numLayers),
		vBuffers:   make([]tensor.DevicePtr, numLayers),
	}

	// Allocate GPU buffers for each layer
	bufferSize := maxSeqLen * numKVHeads * headDim * 4 // float32

	for i := 0; i < numLayers; i++ {
		cache.kBuffers[i] = b.Alloc(bufferSize)
		cache.vBuffers[i] = b.Alloc(bufferSize)
	}

	return cache
}

// Reset clears the cache for a new sequence.
func (c *GPUKVCache) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.seqLen = 0
}

// SeqLen returns the current sequence length.
func (c *GPUKVCache) SeqLen() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.seqLen
}

// AppendKV appends new K/V data to the cache for a specific layer.
// kPtr, vPtr should contain [newTokens, numKVHeads, headDim] data.
// Returns the GPU pointers to the full K/V cache for this layer.
func (c *GPUKVCache) AppendKV(layerIdx int, kPtr, vPtr tensor.DevicePtr, newTokens int) (fullK, fullV tensor.DevicePtr, fullSeqLen int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Calculate byte offset for current position
	tokenSize := c.numKVHeads * c.headDim * 4 // bytes per token
	offset := c.seqLen * tokenSize
	copySize := newTokens * tokenSize

	// Copy new K/V into cache at the right position using GPU-to-GPU copy
	if copier, ok := c.backend.(backend.BufferCopier); ok {
		copier.CopyBuffer(kPtr, 0, c.kBuffers[layerIdx], offset, copySize)
		copier.CopyBuffer(vPtr, 0, c.vBuffers[layerIdx], offset, copySize)
	}
	// Note: No sync needed - Metal command queue serializes work in submission order

	// Update sequence length (only after last layer to maintain consistency)
	if layerIdx == c.numLayers-1 {
		c.seqLen += newTokens
	}

	// Return full buffers and the sequence length including new tokens
	return c.kBuffers[layerIdx], c.vBuffers[layerIdx], c.seqLen + newTokens
}

// GetKV returns the K/V cache buffers for a specific layer.
func (c *GPUKVCache) GetKV(layerIdx int) (k, v tensor.DevicePtr, seqLen int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.kBuffers[layerIdx], c.vBuffers[layerIdx], c.seqLen
}

// Free releases all GPU buffers.
func (c *GPUKVCache) Free() {
	for i := 0; i < c.numLayers; i++ {
		c.backend.Free(c.kBuffers[i])
		c.backend.Free(c.vBuffers[i])
	}
}
