package runtime

import (
	"vexel/inference/backend"
	"vexel/inference/kv"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/tensor"
)

// ModelRuntime manages the execution of a model.
type ModelRuntime struct {
	backend    backend.Backend
	ctx        *memory.InferenceContext
	cache      *kv.KVCache      // Legacy simple cache
	pagedCache *kv.PagedKVCache // Production paged cache
	gpuCache   *GPUKVCache      // GPU-resident KV cache for Metal/CUDA
	config     ModelConfig
	layers     []*BlockRuntime

	// Global weights (stored as DevicePtr for GPU execution)
	Embedding  tensor.Tensor
	FinalNorm  tensor.Tensor
	OutputHead tensor.Tensor

	// Keep mapped file alive
	mappedFile interface{ Close() error }

	// GGUF loader (kept open for potential streaming access)
	ggufLoader *gguf.TensorLoader

	// Keep converted weight slices alive to prevent GC
	keepAlive      [][]float32
	keepAliveBytes [][]byte // For raw quantized weight data
}

// NewModelRuntime initializes a new model runtime.
func NewModelRuntime(b backend.Backend, ctx *memory.InferenceContext, cache *kv.KVCache, config ModelConfig) (*ModelRuntime, error) {
	// Initialize layers with config for GQA support
	layers := make([]*BlockRuntime, config.NumHiddenLayers)
	for i := range layers {
		layers[i] = NewBlockRuntime(b, config)
	}

	return &ModelRuntime{
		backend: b,
		ctx:     ctx,
		cache:   cache,
		config:  config,
		layers:  layers,
	}, nil
}

// Config returns the model configuration.
func (m *ModelRuntime) Config() ModelConfig {
	return m.config
}

// Backend returns the underlying compute backend.
func (m *ModelRuntime) Backend() backend.Backend {
	return m.backend
}

// Layer returns the block runtime at the given index.
func (m *ModelRuntime) Layer(i int) *BlockRuntime {
	if i < 0 || i >= len(m.layers) {
		return nil
	}
	return m.layers[i]
}

// SetPagedKVCache sets the paged KV cache for production use.
func (m *ModelRuntime) SetPagedKVCache(cache *kv.PagedKVCache) {
	m.pagedCache = cache
}

// PagedKVCache returns the paged KV cache.
func (m *ModelRuntime) PagedKVCache() *kv.PagedKVCache {
	return m.pagedCache
}

// CreatePagedKVCache creates and sets a new paged KV cache based on model config.
func (m *ModelRuntime) CreatePagedKVCache(maxBlocks int) *kv.PagedKVCache {
	headDim := m.config.HiddenSize / m.config.NumAttentionHeads
	config := kv.PagedKVConfig{
		NumLayers:  m.config.NumHiddenLayers,
		NumKVHeads: m.config.NumKeyValueHeads,
		HeadDim:    headDim,
		BlockSize:  16, // Standard block size
		MaxBlocks:  maxBlocks,
	}
	cache := kv.NewPagedKVCache(config)
	m.pagedCache = cache
	return cache
}

// CreateGPUKVCache creates a GPU-resident KV cache for faster inference.
// This avoids CPU roundtrips for KV data during decode.
func (m *ModelRuntime) CreateGPUKVCache(maxSeqLen int) *GPUKVCache {
	headDim := m.config.HiddenSize / m.config.NumAttentionHeads
	cache := NewGPUKVCache(
		m.backend,
		m.config.NumHiddenLayers,
		m.config.NumKeyValueHeads,
		headDim,
		maxSeqLen,
	)
	m.gpuCache = cache
	return cache
}

// GPUKVCache returns the GPU KV cache if available.
func (m *ModelRuntime) GPUKVCache() *GPUKVCache {
	return m.gpuCache
}

// outputHeadMatMul performs logits = state @ OutputHead^T, using quantized kernel if available.
func (m *ModelRuntime) outputHeadMatMul(statePtr, logitsPtr tensor.DevicePtr, batchSize, vocabSize, hiddenSize int) {
	if m.OutputHead.DevicePtr().IsNil() {
		return
	}

	// Check if we can use quantized kernel
	if m.OutputHead.IsQuantized() {
		if quantBackend, ok := m.backend.(backend.QuantizedMatMul); ok {
			switch m.OutputHead.QuantProfile() {
			case tensor.Q6_K:
				quantBackend.MatMulQ6_K(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			case tensor.Q4_0:
				quantBackend.MatMulQ4_0(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			}
		}
	}

	// Fall back to F32 matmul
	m.backend.MatMulTransposed(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
}
