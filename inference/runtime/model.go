package runtime

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"sync"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/kv"
	"vexel/inference/lora"
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
	gpuPool    *GPUBlockPool    // GPU-resident paged block pool (Track 3)
	config     ModelConfig
	layers     []*BlockRuntime
	plan       *ExecutionPlan // Model-aware execution plan
	useScratch bool           // Whether scratch allocator is active (disables FP16 KV cache)

	// Global weights (stored as DevicePtr for GPU execution)
	Embedding     tensor.Tensor
	FinalNorm     tensor.Tensor
	FinalNormBias tensor.Tensor // For LayerNorm architectures (Phi)
	OutputHead    tensor.Tensor

	// Keep mapped file alive
	mappedFile interface{ Close() error }

	// GGUF loader (kept open for potential streaming access)
	ggufLoader *gguf.TensorLoader

	// Keep converted weight slices alive to prevent GC
	keepAlive      [][]float32
	keepAliveBytes [][]byte // For raw quantized weight data

	// Verbose enables detailed loading/config output
	verbose bool

	// loraAdapter holds GPU-resident LoRA weights attached via AttachLoRA.
	loraAdapter *lora.GPUAdapter
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

// Plan returns the execution plan for this model.
func (m *ModelRuntime) Plan() *ExecutionPlan {
	return m.plan
}

// BuildPlan creates and stores an execution plan based on model config.
// This should be called after loading the model and before inference.
func (m *ModelRuntime) BuildPlan(deviceMeta DeviceMeta, config *PlanConfig) {
	modelMeta := m.ModelMeta()
	m.plan = BuildExecutionPlan(modelMeta, deviceMeta, config)

	// Propagate plan to all blocks so they can use plan-based kernel selection
	for _, layer := range m.layers {
		layer.SetPlan(m.plan)
	}
}

// SetupRoPEFreqs creates and uploads the RoPE inverse frequency buffer to the device
// if the model uses learned RoPE frequencies (RoPEFreqScales in config). This buffer
// is shared across all layers. If no learned frequencies are configured, this is a no-op.
func (m *ModelRuntime) SetupRoPEFreqs() {
	if len(m.config.RoPEFreqScales) == 0 {
		return // No learned frequencies, use standard theta-based RoPE
	}

	headDim := m.config.EffectiveHeadDim()
	halfDim := headDim / 2

	if len(m.config.RoPEFreqScales) != halfDim {
		log.Printf("[WARNING] RoPEFreqScales length %d != expected %d (headDim/2), ignoring",
			len(m.config.RoPEFreqScales), halfDim)
		return
	}

	// Upload frequency buffer to device
	sizeBytes := halfDim * 4
	freqBuf := m.backend.Alloc(sizeBytes)
	if freqBuf.IsNil() {
		log.Printf("[WARNING] Failed to allocate %d bytes for RoPE freq buffer", sizeBytes)
		return
	}

	// Convert float32 slice to byte slice for ToDevice
	data := make([]byte, sizeBytes)
	for i, f := range m.config.RoPEFreqScales {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(f))
	}
	m.backend.ToDevice(freqBuf, data)

	// Set on all layers
	for _, layer := range m.layers {
		layer.SetRoPEFreqBuffer(freqBuf)
	}

	if m.verbose {
		log.Printf("[CONFIG] Loaded %d learned RoPE inverse frequencies", halfDim)
	}
}

// ModelMeta extracts model metadata from the runtime config.
func (m *ModelRuntime) ModelMeta() ModelMeta {
	headDim := m.config.EffectiveHeadDim()

	// Build quant format stats from loaded tensors
	quantFormats := make(map[string]int)
	for _, layer := range m.layers {
		if layer.Wq.IsQuantized() {
			quantFormats[layer.Wq.QuantProfile().String()]++
		}
	}

	// Infer model name from architecture
	name := "unknown"
	if m.ggufLoader != nil {
		// Try to get architecture from GGUF metadata
		name = fmt.Sprintf("%dL-%dH-%dKV", m.config.NumHiddenLayers, m.config.NumAttentionHeads, m.config.NumKeyValueHeads)
	}

	return ModelMeta{
		Name:             name,
		HiddenSize:       m.config.HiddenSize,
		IntermediateSize: m.config.IntermediateSize,
		NumLayers:        m.config.NumHiddenLayers,
		NumHeads:         m.config.NumAttentionHeads,
		NumKVHeads:       m.config.NumKeyValueHeads,
		HeadDim:          headDim,
		VocabSize:        m.config.VocabSize,
		MaxSeqLen:        m.config.MaxSeqLen,
		QuantFormats:     quantFormats,
		NormType:         m.config.NormType,
	}
}

// Backend returns the underlying compute backend.
func (m *ModelRuntime) Backend() backend.Backend {
	return m.backend
}

// SetUseScratch marks whether the scratch allocator is active.
// When active, FP16 KV cache is disabled (FP16 fused kernels lack offset support).
func (m *ModelRuntime) SetUseScratch(active bool) {
	m.useScratch = active
}

// SetVerbose enables or disables verbose loading/config output.
func (m *ModelRuntime) SetVerbose(v bool) {
	m.verbose = v
	if v {
		m.logConfigDetails()
	}
}

// logConfigDetails prints detailed model configuration information (only in verbose mode).
func (m *ModelRuntime) logConfigDetails() {
	c := m.config
	log.Printf("[CONFIG] Architecture=%s: NormType=%v, MLPType=%v, HasBias=%v, ParallelResidual=%v, RoPENeox=%v",
		"auto-detected", c.NormType, c.MLPType, c.HasBias, c.ParallelResidual, c.RoPENeox)
	if c.HeadDim > 0 {
		log.Printf("[CONFIG] HeadDim=%d (explicit, vs hiddenSize/numHeads=%d)", c.HeadDim, c.HiddenSize/c.NumAttentionHeads)
	}
	if c.AttentionLogitSoftCap > 0 {
		log.Printf("[CONFIG] AttentionLogitSoftCap=%.1f, FinalLogitSoftCap=%.1f", c.AttentionLogitSoftCap, c.FinalLogitSoftCap)
	}
	if c.HasPostNorms {
		log.Printf("[CONFIG] HasPostNorms=true (Gemma 2 post-attention and post-FFN norms)")
	}
}

// applyFinalNorm applies the final normalization (RMSNorm or LayerNorm based on config).
func (m *ModelRuntime) applyFinalNorm(xPtr, outPtr tensor.DevicePtr, rows, cols int) {
	if m.FinalNorm.DevicePtr().IsNil() {
		return
	}
	debug := os.Getenv("DEBUG_DECODE") == "1"
	if m.config.NormType == NormLayerNorm {
		if layerNormOps, ok := m.backend.(backend.LayerNormOps); ok {
			if debug {
				fmt.Printf("[DEBUG] applyFinalNorm: Using LayerNorm, bias=%v\n", !m.FinalNormBias.DevicePtr().IsNil())
			}
			layerNormOps.LayerNorm(xPtr, m.FinalNorm.DevicePtr(), m.FinalNormBias.DevicePtr(), outPtr, rows, cols, float32(m.config.RMSNormEPS))
			return
		}
	}
	// Fallback to RMSNorm
	if debug {
		fmt.Printf("[DEBUG] applyFinalNorm: Using RMSNorm\n")
	}
	m.backend.RMSNorm(xPtr, m.FinalNorm.DevicePtr(), outPtr, rows, cols, float32(m.config.RMSNormEPS))
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
	headDim := m.config.EffectiveHeadDim()
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
// Set VEXEL_KV_FP32=1 to force FP32 KV cache for testing.
func (m *ModelRuntime) CreateGPUKVCache(maxSeqLen int) *GPUKVCache {
	headDim := m.config.EffectiveHeadDim()

	// Auto-enable FP16 if backend supports it (2x memory bandwidth savings).
	// Disable FP16 when scratch allocator is active — the FP16 fused decode kernels
	// (FusedRMSNormQKV_F16, SDPAF16, F16InAdd) don't handle scratch buffer offsets,
	// causing identical attention output across all layers.
	useFP16 := false
	if _, ok := m.backend.(backend.FP16Ops); ok {
		useFP16 = true
	}
	// Disable FP16 when scratch is active (set by InitScratch before CreateGPUKVCache)
	if m.useScratch {
		useFP16 = false
	}
	// Allow forcing FP32 for testing (VEXEL_KV_FP32=1)
	if os.Getenv("VEXEL_KV_FP32") == "1" {
		useFP16 = false
	}

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

// CreateGPUBlockPool creates a GPU-resident paged block pool for paged KV batching.
// This eliminates CPU roundtrips in the paged KV path by keeping blocks on the GPU.
// Uses FP16 KV cache by default for 2x memory savings; set VEXEL_KV_FP32=1 to force FP32.
func (m *ModelRuntime) CreateGPUBlockPool(maxBlocksPerLayer int) *GPUBlockPool {
	pagedOps, ok := m.backend.(backend.PagedKVOps)
	if !ok {
		return nil
	}
	// Default to FP16 for paged KV unless VEXEL_KV_FP32=1 is set
	useFP16 := true
	if os.Getenv("VEXEL_KV_FP32") == "1" {
		useFP16 = false
	}
	headDim := m.config.EffectiveHeadDim()
	pool := NewGPUBlockPool(
		m.backend, pagedOps,
		m.config.NumHiddenLayers,
		m.config.NumKeyValueHeads,
		headDim,
		16, // standard block size
		maxBlocksPerLayer,
		useFP16,
	)
	m.gpuPool = pool
	return pool
}

// GPUBlockPool returns the GPU block pool if available.
func (m *ModelRuntime) GetGPUBlockPool() *GPUBlockPool {
	return m.gpuPool
}

var outputHeadDebugOnce sync.Once

// ApplyFinalLogitSoftCap applies tanh-based soft-capping to logits on CPU.
// Gemma 2 uses cap * tanh(logits / cap) with cap=30.0 on the final output logits.
// This prevents extreme logit values from dominating sampling.
// No-op if FinalLogitSoftCap is 0 (disabled).
func (m *ModelRuntime) ApplyFinalLogitSoftCap(logitsData []float32) {
	cap := m.config.FinalLogitSoftCap
	if cap <= 0 {
		return
	}
	invCap := 1.0 / float64(cap)
	for i := range logitsData {
		logitsData[i] = float32(float64(cap) * math.Tanh(float64(logitsData[i])*invCap))
	}
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
			if debugDecode {
				outputHeadDebugOnce.Do(func() {
					fmt.Printf("[DEBUG] OutputHead: IsQuantized=%v, Profile=%v, Backend implements QuantizedMatMul=%v\n",
						m.OutputHead.IsQuantized(), profile, ok)
				})
			}

			// For batch operations, process one position at a time
			if batchSize > 1 && (profile == tensor.Q6_K || profile == tensor.Q4_0 || profile == tensor.Q4_K || profile == tensor.Q5_K || profile == tensor.Q5_0 || profile == tensor.Q8_0) {
				stateRowBytes := uintptr(hiddenSize * 4)
				logitsRowBytes := uintptr(vocabSize * 4)

				for i := 0; i < batchSize; i++ {
					rowStatePtr := tensor.DevicePtrOffset(statePtr, uintptr(i)*stateRowBytes)
					rowLogitsPtr := tensor.DevicePtrOffset(logitsPtr, uintptr(i)*logitsRowBytes)

					switch profile {
					case tensor.Q6_K:
						quantBackend.MatMulQ6_K(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					case tensor.Q5_K:
						quantBackend.MatMulQ5_K(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					case tensor.Q4_0:
						quantBackend.MatMulQ4_0(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					case tensor.Q4_K:
						quantBackend.MatMulQ4_K(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					case tensor.Q5_0:
						quantBackend.MatMulQ5_0(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					case tensor.Q8_0:
						quantBackend.MatMulQ8_0(rowStatePtr, m.OutputHead.DevicePtr(), rowLogitsPtr, 1, vocabSize, hiddenSize)
					}
				}
				return
			}

			// Single position: use optimized kernel directly
			switch profile {
			case tensor.Q6_K:
				quantBackend.MatMulQ6_K(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			case tensor.Q5_K:
				quantBackend.MatMulQ5_K(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			case tensor.Q4_0:
				quantBackend.MatMulQ4_0(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			case tensor.Q4_K:
				quantBackend.MatMulQ4_K(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			case tensor.Q5_0:
				quantBackend.MatMulQ5_0(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
				return
			case tensor.Q8_0:
				quantBackend.MatMulQ8_0(statePtr, m.OutputHead.DevicePtr(), logitsPtr, batchSize, vocabSize, hiddenSize)
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
		case tensor.Q5_K:
			// Q5_K: 176 bytes per 256 elements
			numBlocks := (numElements + 255) / 256
			dataSize = numBlocks * 176
		case tensor.Q4_0:
			// Q4_0: 18 bytes per 32 elements
			numBlocks := (numElements + 31) / 32
			dataSize = numBlocks * 18
		case tensor.Q5_0:
			// Q5_0: 22 bytes per 32 elements
			numBlocks := (numElements + 31) / 32
			dataSize = numBlocks * 22
		case tensor.Q8_0:
			// Q8_0: 34 bytes per 32 elements
			numBlocks := (numElements + 31) / 32
			dataSize = numBlocks * 34
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
		case tensor.Q5_K:
			tensorType = gguf.TensorTypeQ5_K
		case tensor.Q4_0:
			tensorType = gguf.TensorTypeQ4_0
		case tensor.Q5_0:
			tensorType = gguf.TensorTypeQ5_0
		case tensor.Q8_0:
			tensorType = gguf.TensorTypeQ8_0
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

// AttachLoRA wires a GPU-resident LoRA adapter into the model's layer stack.
// The adapter's weights are injected into Q and V projections during forward
// passes until replaced or set to nil.
func (m *ModelRuntime) AttachLoRA(adapter *lora.GPUAdapter) {
	m.loraAdapter = adapter
	m.wireLoRA()
}

// wireLoRA propagates the current loraAdapter to each BlockRuntime so the
// forward pass can apply the LoRA deltas without an extra indirection.
func (m *ModelRuntime) wireLoRA() {
	if m.loraAdapter == nil {
		// Detach: clear all layer references.
		for _, layer := range m.layers {
			layer.loraLayer = nil
			layer.loraRank = 0
			layer.loraScale = 0
		}
		return
	}
	for i, layer := range m.layers {
		la := m.loraAdapter.GetLayer(i)
		if la != nil {
			layer.loraLayer = la
			layer.loraRank = m.loraAdapter.Rank
			layer.loraScale = m.loraAdapter.Scale
		}
	}
}
