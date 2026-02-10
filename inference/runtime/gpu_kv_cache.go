package runtime

import (
	"sync"

	"vexel/inference/backend"
	"vexel/inference/tensor"
)

// KVPrecision represents the storage format for KV cache.
type KVPrecision int

const (
	KVPrecisionFP32 KVPrecision = iota // 4 bytes per element
	KVPrecisionFP16                     // 2 bytes per element (2x savings)
	KVPrecisionQ8_0                     // 34 bytes per 32 elements (4x savings)
)

// Q8_0 format constants
const (
	Q8BlockSize     = 32 // Elements per Q8_0 block
	Q8BytesPerBlock = 34 // 2 byte f16 scale + 32 int8 values
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
	useFP16    bool        // Deprecated: use precision instead
	precision  KVPrecision // Storage precision

	// Per-layer K and V buffers: [numKVHeads, maxSeqLen, headDim]
	// This layout makes decode SDPA reads contiguous when iterating over sequence positions.
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

// NewGPUKVCacheQ8_0 creates a new GPU-resident KV cache with Q8_0 quantization.
// Provides 4x memory savings with minimal accuracy loss.
// Requires backend to support Q8_0Ops interface.
func NewGPUKVCacheQ8_0(b backend.Backend, numLayers, numKVHeads, headDim, maxSeqLen int) *GPUKVCache {
	return newGPUKVCacheWithKVPrecision(b, numLayers, numKVHeads, headDim, maxSeqLen, KVPrecisionQ8_0)
}

// NewGPUKVCacheWithPrecision creates a new GPU-resident KV cache.
// useFP16: if true, stores K/V in FP16 format for 2x memory savings.
func NewGPUKVCacheWithPrecision(b backend.Backend, numLayers, numKVHeads, headDim, maxSeqLen int, useFP16 bool) *GPUKVCache {
	precision := KVPrecisionFP32
	if useFP16 {
		precision = KVPrecisionFP16
	}
	return newGPUKVCacheWithKVPrecision(b, numLayers, numKVHeads, headDim, maxSeqLen, precision)
}

// newGPUKVCacheWithKVPrecision creates a GPU-resident KV cache with specified precision.
func newGPUKVCacheWithKVPrecision(b backend.Backend, numLayers, numKVHeads, headDim, maxSeqLen int, precision KVPrecision) *GPUKVCache {
	cache := &GPUKVCache{
		backend:    b,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		maxSeqLen:  maxSeqLen,
		useFP16:    precision == KVPrecisionFP16,
		precision:  precision,
		kBuffers:   make([]tensor.DevicePtr, numLayers),
		vBuffers:   make([]tensor.DevicePtr, numLayers),
	}

	// Calculate buffer size based on precision
	numElements := maxSeqLen * numKVHeads * headDim
	var bufferSize int
	switch precision {
	case KVPrecisionFP32:
		bufferSize = numElements * 4 // 4 bytes per float32
	case KVPrecisionFP16:
		bufferSize = numElements * 2 // 2 bytes per float16
	case KVPrecisionQ8_0:
		// Q8_0: 34 bytes per 32 elements
		numBlocks := (numElements + Q8BlockSize - 1) / Q8BlockSize
		bufferSize = numBlocks * Q8BytesPerBlock
	}

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
// Deprecated: use Precision() instead.
func (c *GPUKVCache) UseFP16() bool {
	return c.precision == KVPrecisionFP16
}

// UseQ8_0 returns true if this cache stores K/V in Q8_0 format.
func (c *GPUKVCache) UseQ8_0() bool {
	return c.precision == KVPrecisionQ8_0
}

// Precision returns the storage precision of this cache.
func (c *GPUKVCache) Precision() KVPrecision {
	return c.precision
}

// KVHeadStride returns the stride between KV heads in elements (maxSeqLen * headDim).
// This is needed for the head-major [numKVHeads, maxSeqLen, headDim] layout.
func (c *GPUKVCache) KVHeadStride() int {
	return c.maxSeqLen * c.headDim
}

// AppendKV appends new K/V data to the cache for a specific layer.
// kPtr, vPtr should contain [newTokens, numKVHeads, headDim] data (input layout).
// Cache stores data as [numKVHeads, maxSeqLen, headDim] for decode-friendly access.
// For FP32 cache: expects FP32 input. For FP16 cache: expects FP16 input.
// Returns the GPU pointers to the full K/V cache for this layer.
func (c *GPUKVCache) AppendKV(layerIdx int, kPtr, vPtr tensor.DevicePtr, newTokens int) (fullK, fullV tensor.DevicePtr, fullSeqLen int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Input layout: [newTokens, numKVHeads, headDim]
	// Cache layout: [numKVHeads, maxSeqLen, headDim]

	// Try FP16 scatter kernel first (single dispatch, most efficient)
	if c.useFP16 {
		if fp16Ops, ok := c.backend.(backend.FP16Ops); ok {
			// Use GPU scatter kernel - single dispatch for K and V
			fp16Ops.ScatterKVF16(kPtr, c.kBuffers[layerIdx], newTokens, c.numKVHeads, c.headDim, c.maxSeqLen, c.seqLen)
			fp16Ops.ScatterKVF16(vPtr, c.vBuffers[layerIdx], newTokens, c.numKVHeads, c.headDim, c.maxSeqLen, c.seqLen)

			fullSeqLen = c.seqLen + newTokens
			if layerIdx == c.numLayers-1 {
				c.seqLen += newTokens
			}
			return c.kBuffers[layerIdx], c.vBuffers[layerIdx], fullSeqLen
		}
	} else {
		// Try F32 scatter kernel
		if scatter, ok := c.backend.(backend.KVScatter); ok {
			scatter.ScatterKV(kPtr, c.kBuffers[layerIdx], newTokens, c.numKVHeads, c.headDim, c.maxSeqLen, c.seqLen)
			scatter.ScatterKV(vPtr, c.vBuffers[layerIdx], newTokens, c.numKVHeads, c.headDim, c.maxSeqLen, c.seqLen)

			fullSeqLen = c.seqLen + newTokens
			if layerIdx == c.numLayers-1 {
				c.seqLen += newTokens
			}
			return c.kBuffers[layerIdx], c.vBuffers[layerIdx], fullSeqLen
		}
	}

	// Fall back to blit copies (less efficient but works for all cases)
	bytesPerElement := 4 // float32
	if c.useFP16 {
		bytesPerElement = 2 // float16
	}

	copier, hasCopier := c.backend.(backend.BufferCopier)
	if !hasCopier {
		// No copier available, can't append
		return c.kBuffers[layerIdx], c.vBuffers[layerIdx], c.seqLen
	}

	// For each token t and head h:
	//   src offset: t * numKVHeads * headDim + h * headDim
	//   dst offset: h * maxSeqLen * headDim + (seqLen + t) * headDim

	headSize := c.headDim * bytesPerElement           // bytes per head per position
	srcHeadStride := c.numKVHeads * c.headDim         // elements between same head in different tokens
	dstHeadStride := c.maxSeqLen * c.headDim          // elements between heads in cache (in elements)
	dstHeadStrideBytes := dstHeadStride * bytesPerElement

	for t := 0; t < newTokens; t++ {
		for h := 0; h < c.numKVHeads; h++ {
			// Source: token t, head h
			srcOffset := (t*srcHeadStride + h*c.headDim) * bytesPerElement
			// Dest: head h, position (seqLen + t)
			dstOffset := h*dstHeadStrideBytes + (c.seqLen+t)*headSize

			copier.CopyBufferBatched(kPtr, srcOffset, c.kBuffers[layerIdx], dstOffset, headSize)
			copier.CopyBufferBatched(vPtr, srcOffset, c.vBuffers[layerIdx], dstOffset, headSize)
		}
	}

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

// Truncate rolls back the KV cache to a specific sequence length.
// This is used for speculative decoding when draft tokens are rejected.
// The cache buffers are not cleared - we just update the sequence pointer.
func (c *GPUKVCache) Truncate(newSeqLen int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if newSeqLen < 0 {
		newSeqLen = 0
	}
	if newSeqLen > c.seqLen {
		newSeqLen = c.seqLen
	}
	c.seqLen = newSeqLen
}

// Free releases all GPU buffers.
func (c *GPUKVCache) Free() {
	for i := 0; i < c.numLayers; i++ {
		c.backend.Free(c.kBuffers[i])
		c.backend.Free(c.vBuffers[i])
	}
}
