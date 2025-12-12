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
	useFP16    bool // Use FP16 storage for 2x memory savings

	// Per-layer K and V buffers: [maxSeqLen, numKVHeads, headDim]
	kBuffers []tensor.DevicePtr
	vBuffers []tensor.DevicePtr

	// Current sequence length (same for all layers in single-sequence case)
	seqLen int
}

// NewGPUKVCache creates a new GPU-resident KV cache (FP32).
func NewGPUKVCache(b backend.Backend, numLayers, numKVHeads, headDim, maxSeqLen int) *GPUKVCache {
	return NewGPUKVCacheWithPrecision(b, numLayers, numKVHeads, headDim, maxSeqLen, false)
}

// NewGPUKVCacheFP16 creates a new GPU-resident KV cache with FP16 storage.
// Provides 2x memory savings with slightly reduced precision.
func NewGPUKVCacheFP16(b backend.Backend, numLayers, numKVHeads, headDim, maxSeqLen int) *GPUKVCache {
	return NewGPUKVCacheWithPrecision(b, numLayers, numKVHeads, headDim, maxSeqLen, true)
}

// NewGPUKVCacheWithPrecision creates a new GPU-resident KV cache.
// useFP16: if true, stores K/V in FP16 format for 2x memory savings.
func NewGPUKVCacheWithPrecision(b backend.Backend, numLayers, numKVHeads, headDim, maxSeqLen int, useFP16 bool) *GPUKVCache {
	cache := &GPUKVCache{
		backend:    b,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		maxSeqLen:  maxSeqLen,
		useFP16:    useFP16,
		kBuffers:   make([]tensor.DevicePtr, numLayers),
		vBuffers:   make([]tensor.DevicePtr, numLayers),
	}

	// Allocate GPU buffers for each layer
	bytesPerElement := 4 // float32
	if useFP16 {
		bytesPerElement = 2 // float16
	}
	bufferSize := maxSeqLen * numKVHeads * headDim * bytesPerElement

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

// UseFP16 returns true if this cache stores K/V in FP16 format.
func (c *GPUKVCache) UseFP16() bool {
	return c.useFP16
}

// AppendKV appends new K/V data to the cache for a specific layer.
// kPtr, vPtr should contain [newTokens, numKVHeads, headDim] data.
// For FP32 cache: expects FP32 input. For FP16 cache: expects FP16 input.
// Returns the GPU pointers to the full K/V cache for this layer.
func (c *GPUKVCache) AppendKV(layerIdx int, kPtr, vPtr tensor.DevicePtr, newTokens int) (fullK, fullV tensor.DevicePtr, fullSeqLen int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Calculate byte offset for current position
	bytesPerElement := 4 // float32
	if c.useFP16 {
		bytesPerElement = 2 // float16
	}
	tokenSize := c.numKVHeads * c.headDim * bytesPerElement
	offset := c.seqLen * tokenSize
	copySize := newTokens * tokenSize

	// Copy new K/V into cache at the right position using GPU-to-GPU copy
	if copier, ok := c.backend.(backend.BufferCopier); ok {
		copier.CopyBuffer(kPtr, 0, c.kBuffers[layerIdx], offset, copySize)
		copier.CopyBuffer(vPtr, 0, c.vBuffers[layerIdx], offset, copySize)
	}
	// Note: No sync needed - Metal command queue serializes work in submission order

	// Calculate the sequence length including the new tokens BEFORE updating
	fullSeqLen = c.seqLen + newTokens

	// Update sequence length (only after last layer to maintain consistency)
	if layerIdx == c.numLayers-1 {
		c.seqLen += newTokens
	}

	// Return full buffers and the sequence length including new tokens
	return c.kBuffers[layerIdx], c.vBuffers[layerIdx], fullSeqLen
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
