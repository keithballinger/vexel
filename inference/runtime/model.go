package runtime

import (
	"unsafe"

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
// Automatically uses FP16 storage if the backend supports it (2x memory savings).
func (m *ModelRuntime) CreateGPUKVCache(maxSeqLen int) *GPUKVCache {
	headDim := m.config.HiddenSize / m.config.NumAttentionHeads

	// Auto-enable FP16 if backend supports it
	// TEMPORARILY DISABLED - debugging garbled output
	useFP16 := false
	// if _, ok := m.backend.(backend.FP16Ops); ok {
	// 	useFP16 = true
	// }

	cache := NewGPUKVCacheWithPrecision(
		m.backend,
		m.config.NumHiddenLayers,
		m.config.NumKeyValueHeads,
		headDim,
		maxSeqLen,
		useFP16,
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
	// Note: Quantized matmul only supports M=1 (matvec), so we process one at a time for batched
	if m.OutputHead.IsQuantized() {
		if quantBackend, ok := m.backend.(backend.QuantizedMatMul); ok {
			profile := m.OutputHead.QuantProfile()

			// For batch operations, process one position at a time
			if batchSize > 1 && (profile == tensor.Q6_K || profile == tensor.Q4_0 || profile == tensor.Q4_K) {
				stateRowBytes := uintptr(hiddenSize * 4)
				logitsRowBytes := uintptr(vocabSize * 4)

				for i := 0; i < batchSize; i++ {
					rowStatePtr := tensor.DevicePtrOffset(statePtr, uintptr(i)*stateRowBytes)
					rowLogitsPtr := tensor.DevicePtrOffset(logitsPtr, uintptr(i)*logitsRowBytes)

					switch profile {
					case tensor.Q6_K:
						quantBackend.MatMulQ6_K(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					case tensor.Q4_0:
						quantBackend.MatMulQ4_0(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					case tensor.Q4_K:
						quantBackend.MatMulQ4_K(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					}
				}
				return
			}

			// Single position: use optimized kernel directly
			switch profile {
			case tensor.Q6_K:
				quantBackend.MatMulQ6_K(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			case tensor.Q4_0:
				quantBackend.MatMulQ4_0(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			case tensor.Q4_K:
				quantBackend.MatMulQ4_K(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			}
		}
	}

	// Fall back to F32 matmul
	m.backend.MatMulTransposed(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
}

// GetOutputHeadWeightsF32 returns the output head (lm_head) weights as F32.
// This is useful for initializing Medusa heads from the base model.
// Returns weights in [vocab_size, hidden_size] layout.
func (m *ModelRuntime) GetOutputHeadWeightsF32() []float32 {
	if m.OutputHead.DevicePtr().IsNil() {
		return nil
	}

	vocabSize := m.config.VocabSize
	hiddenSize := m.config.HiddenSize
	numElements := vocabSize * hiddenSize

	// Download from GPU
	var dataSize int
	if m.OutputHead.IsQuantized() {
		profile := m.OutputHead.QuantProfile()
		switch profile {
		case tensor.Q6_K:
			// Q6_K: 210 bytes per 256 elements
			numBlocks := (numElements + 255) / 256
			dataSize = numBlocks * 210
		case tensor.Q4_K:
			// Q4_K: 144 bytes per 256 elements
			numBlocks := (numElements + 255) / 256
			dataSize = numBlocks * 144
		case tensor.Q4_0:
			// Q4_0: 18 bytes per 32 elements
			numBlocks := (numElements + 31) / 32
			dataSize = numBlocks * 18
		default:
			// Unknown quantization, try F32
			dataSize = numElements * 4
		}
	} else {
		dataSize = numElements * 4
	}

	rawData := make([]byte, dataSize)
	m.backend.ToHost(rawData, m.OutputHead.DevicePtr())
	m.backend.Sync()

	// Dequantize if needed
	if m.OutputHead.IsQuantized() {
		profile := m.OutputHead.QuantProfile()
		var tensorType gguf.TensorType
		switch profile {
		case tensor.Q6_K:
			tensorType = gguf.TensorTypeQ6_K
		case tensor.Q4_K:
			tensorType = gguf.TensorTypeQ4_K
		case tensor.Q4_0:
			tensorType = gguf.TensorTypeQ4_0
		default:
			tensorType = gguf.TensorTypeF32
		}
		return gguf.Dequantize(rawData, tensorType, numElements)
	}

	// Already F32, convert bytes to float32
	result := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		bits := uint32(rawData[i*4]) | uint32(rawData[i*4+1])<<8 |
			uint32(rawData[i*4+2])<<16 | uint32(rawData[i*4+3])<<24
		result[i] = float32frombits(bits)
	}
	return result
}

// float32frombits converts a uint32 bit pattern to float32.
func float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}
