package kv

import "vexel/inference/tensor"

// KVConfig defines the structure and layout of the KV cache.
type KVConfig struct {
	DType    tensor.DType
	HeadDim  int
	BlockLen int
}

// NewKVConfig creates a new KVConfig.
func NewKVConfig(dtype tensor.DType, headDim, blockLen int) KVConfig {
	return KVConfig{
		DType:    dtype,
		HeadDim:  headDim,
		BlockLen: blockLen,
	}
}

// BlockBytes returns the size of a single block in bytes.
// A block stores K and V data for 'BlockLen' tokens.
func (c KVConfig) BlockBytes() int {
	// Size = 2 (K+V) * BlockLen * HeadDim * SizeOf(DType)
	return 2 * c.BlockLen * c.HeadDim * c.DType.SizeBytes()
}
