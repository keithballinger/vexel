package runtime

import (
	"math"

	"vexel/inference/backend"
	"vexel/inference/kv"
	"vexel/inference/tensor"
)

// sqrt is a helper for float64 square root
func sqrt(x float64) float64 {
	return math.Sqrt(x)
}

// BlockRuntime represents a single transformer layer (Attention + MLP).
type BlockRuntime struct {
	backend backend.Backend

	// Config (needed for GQA)
	NumAttentionHeads int
	NumKeyValueHeads  int
	HeadDim           int
	HiddenSize        int
	IntermediateSize  int
	RoPETheta         float64
	RMSNormEPS        float64

	// Weights (stored as DevicePtr for GPU execution)
	AttnNorm   tensor.Tensor
	Wq, Wk, Wv tensor.Tensor
	Wo         tensor.Tensor

	FFNNorm    tensor.Tensor
	W1, W2, W3 tensor.Tensor // Gate, Down, Up
}

// NewBlockRuntime creates a new block runtime with config.
func NewBlockRuntime(b backend.Backend, config ModelConfig) *BlockRuntime {
	headDim := config.HiddenSize / config.NumAttentionHeads
	return &BlockRuntime{
		backend:           b,
		NumAttentionHeads: config.NumAttentionHeads,
		NumKeyValueHeads:  config.NumKeyValueHeads,
		HeadDim:           headDim,
		HiddenSize:        config.HiddenSize,
		IntermediateSize:  config.IntermediateSize,
		RoPETheta:         config.RoPETheta,
		RMSNormEPS:        config.RMSNormEPS,
	}
}

// Execute performs the forward pass for this block using DevicePtr operations.
// All tensors are expected to have valid DevicePtr (allocated on backend device).
// x: Input tensor [seqLen, Hidden] with DevicePtr
// scratch: Temporary buffer with DevicePtr (large enough for all intermediates)
// kvCache: Pointer to KV cache manager (unused in this path, for compatibility)
// layerIdx: Index of this layer (unused in this path)
// pos: Current token position (for RoPE)
func (b *BlockRuntime) Execute(x, scratch tensor.Tensor, kvCache *kv.KVCache, layerIdx, pos int) (tensor.Tensor, error) {
	xPtr := x.DevicePtr()
	scratchPtr := scratch.DevicePtr()

	if xPtr.IsNil() || scratchPtr.IsNil() {
		return x, nil
	}

	// Dimensions from config
	seqLen := x.Shape().NumElements() / b.HiddenSize
	hiddenSize := b.HiddenSize
	numHeads := b.NumAttentionHeads
	numKVHeads := b.NumKeyValueHeads
	headDim := b.HeadDim
	intermediateSize := b.IntermediateSize

	// Derived sizes (in float32 elements)
	qSize := seqLen * numHeads * headDim
	kvSize := seqLen * numKVHeads * headDim

	// Calculate sizes for intermediates
	normOutBytes := seqLen * hiddenSize * 4
	qBytes := qSize * 4
	kvBytes := kvSize * 4
	attnOutBytes := qSize * 4
	gateBytes := seqLen * intermediateSize * 4
	upBytes := seqLen * intermediateSize * 4

	// Allocate intermediate buffers
	// For GPU: allocate separate buffers (Metal doesn't support buffer+offset)
	// For CPU: use scratch with offsets for better memory locality
	var normOutPtr, qPtr, kPtr, vPtr, attnOutPtr, gatePtr, upPtr tensor.DevicePtr

	if scratchPtr.Location() == tensor.CPU {
		// CPU: use offset-based sub-allocation from scratch buffer
		offset := uintptr(0)
		normOutPtr = tensor.DevicePtrOffset(scratchPtr, offset)
		offset += uintptr(normOutBytes)
		qPtr = tensor.DevicePtrOffset(scratchPtr, offset)
		offset += uintptr(qBytes)
		kPtr = tensor.DevicePtrOffset(scratchPtr, offset)
		offset += uintptr(kvBytes)
		vPtr = tensor.DevicePtrOffset(scratchPtr, offset)
		offset += uintptr(kvBytes)
		attnOutPtr = tensor.DevicePtrOffset(scratchPtr, offset)
		offset += uintptr(attnOutBytes)
		gatePtr = tensor.DevicePtrOffset(scratchPtr, offset)
		offset += uintptr(gateBytes)
		upPtr = tensor.DevicePtrOffset(scratchPtr, offset)
		_ = upBytes // Used for allocation
	} else {
		// GPU: allocate separate buffers for each intermediate
		// Metal doesn't support buffer+offset addressing in kernels
		normOutPtr = b.backend.Alloc(normOutBytes)
		qPtr = b.backend.Alloc(qBytes)
		kPtr = b.backend.Alloc(kvBytes)
		vPtr = b.backend.Alloc(kvBytes)
		attnOutPtr = b.backend.Alloc(attnOutBytes)
		gatePtr = b.backend.Alloc(gateBytes)
		upPtr = b.backend.Alloc(upBytes)
	}

	// 1. RMSNorm (Attention)
	if !b.AttnNorm.DevicePtr().IsNil() {
		b.backend.RMSNorm(xPtr, b.AttnNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
	}

	// 2. Q/K/V Projections
	// Wq: [numHeads*headDim, hiddenSize] -> Q: [seqLen, numHeads*headDim]
	if !b.Wq.DevicePtr().IsNil() {
		qDim := b.Wq.Shape().Dims()[0]
		b.backend.MatMulTransposed(normOutPtr, b.Wq.DevicePtr(), qPtr, seqLen, qDim, hiddenSize)
	}

	// Wk: [numKVHeads*headDim, hiddenSize] -> K: [seqLen, numKVHeads*headDim]
	if !b.Wk.DevicePtr().IsNil() {
		kDim := b.Wk.Shape().Dims()[0]
		b.backend.MatMulTransposed(normOutPtr, b.Wk.DevicePtr(), kPtr, seqLen, kDim, hiddenSize)
	}

	// Wv: [numKVHeads*headDim, hiddenSize] -> V: [seqLen, numKVHeads*headDim]
	if !b.Wv.DevicePtr().IsNil() {
		vDim := b.Wv.Shape().Dims()[0]
		b.backend.MatMulTransposed(normOutPtr, b.Wv.DevicePtr(), vPtr, seqLen, vDim, hiddenSize)
	}

	// 3. RoPE - Apply to Q and K
	b.backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, pos, float32(b.RoPETheta))

	// 4. Attention
	scale := float32(1.0 / sqrt(float64(headDim)))
	if seqLen == 1 {
		// Decode: use SDPA kernel (for single token, K/V are the current token only)
		b.backend.SDPA(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
	} else {
		// Prefill: use SDPAPrefill kernel with causal masking
		b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
	}

	// 5. Output Projection and residual
	if !b.Wo.DevicePtr().IsNil() {
		oDim := b.Wo.Shape().Dims()[0]
		b.backend.MatMulTransposed(attnOutPtr, b.Wo.DevicePtr(), normOutPtr, seqLen, oDim, numHeads*headDim)
		// Add residual: x = x + normOut
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	}

	// 6. FFN RMSNorm
	if !b.FFNNorm.DevicePtr().IsNil() {
		b.backend.RMSNorm(xPtr, b.FFNNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
	}

	// 7. MLP: SwiGLU variant
	// Gate projection: gate = SiLU(normOut @ W1^T)
	if !b.W1.DevicePtr().IsNil() {
		w1Dim := b.W1.Shape().Dims()[0]
		b.backend.MatMulTransposed(normOutPtr, b.W1.DevicePtr(), gatePtr, seqLen, w1Dim, hiddenSize)
	}

	// Up projection: up = normOut @ W3^T
	if !b.W3.DevicePtr().IsNil() {
		w3Dim := b.W3.Shape().Dims()[0]
		b.backend.MatMulTransposed(normOutPtr, b.W3.DevicePtr(), upPtr, seqLen, w3Dim, hiddenSize)
	}

	// SiLU on gate
	b.backend.SiLU(gatePtr, gatePtr, seqLen*intermediateSize)

	// Multiply gate * up
	b.backend.Mul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)

	// Down projection and residual
	if !b.W2.DevicePtr().IsNil() {
		w2Dim := b.W2.Shape().Dims()[0]
		b.backend.MatMulTransposed(gatePtr, b.W2.DevicePtr(), normOutPtr, seqLen, w2Dim, intermediateSize)
		// Add residual: x = x + normOut
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	}

	return x, nil
}

// ExecuteWithPagedKV performs the forward pass using paged KV cache.
// This version uses DevicePtr for GPU execution.
// NOTE: The paged KV cache integration needs to be refactored for GPU operation.
// For now, this delegates to the basic Execute function.
// TODO: Implement GPU-native KV cache with DevicePtr storage.
func (b *BlockRuntime) ExecuteWithPagedKV(x, scratch tensor.Tensor, pagedCache *kv.PagedKVCache, seqID int64, layerIdx, startPos int) (tensor.Tensor, error) {
	// For now, delegate to Execute which uses DevicePtr operations
	// The KV cache integration needs separate refactoring for GPU
	return b.Execute(x, scratch, nil, layerIdx, startPos)
}