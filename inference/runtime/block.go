package runtime

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/kv"
	"vexel/inference/tensor"
)

// skipMidLayerSync controls whether to skip the mid-layer sync.
// Set VEXEL_SKIP_MID_SYNC=1 to test without the sync.
var skipMidLayerSync = os.Getenv("VEXEL_SKIP_MID_SYNC") == "1"


// debugBlockTensor prints stats about a tensor for block debugging
func (b *BlockRuntime) debugBlockTensor(name string, ptr tensor.DevicePtr, numElements int) {
	if !debugDecode || ptr.IsNil() || numElements == 0 {
		return
	}

	data := make([]byte, numElements*4)
	toHost, ok := b.backend.(interface {
		ToHost([]byte, tensor.DevicePtr)
		Sync()
	})
	if !ok {
		return
	}
	toHost.Sync() // Wait for GPU before reading
	toHost.ToHost(data, ptr)

	values := make([]float32, numElements)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}

	min, max, sum := values[0], values[0], float32(0)
	nanCount := 0
	for _, v := range values {
		if math.IsNaN(float64(v)) {
			nanCount++
		} else {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
			sum += v
		}
	}
	mean := sum / float32(numElements-nanCount)

	n := 8
	if n > len(values) {
		n = len(values)
	}
	fmt.Printf("[BLOCK] %s [%d]: min=%.4f max=%.4f mean=%.4f nan=%d first%d=%v\n",
		name, len(values), min, max, mean, nanCount, n, values[:n])
}

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

	// Cached interface check for quantized matmul
	quantMatMul backend.QuantizedMatMul

	// Cached interface for command buffer batching
	batcher backend.Batcher

	// Cached interface for FP16 operations
	fp16Ops backend.FP16Ops

	// Cached interface for Q8_0 operations
	q8Ops backend.Q8_0Ops

	// Cached interface for fused operations
	fusedOps backend.FusedOps

	// Execution plan (set by ModelRuntime.BuildPlan)
	plan *ExecutionPlan
}

// SetPlan sets the execution plan for this block.
func (b *BlockRuntime) SetPlan(plan *ExecutionPlan) {
	b.plan = plan
}

// NewBlockRuntime creates a new block runtime with config.
func NewBlockRuntime(b backend.Backend, config ModelConfig) *BlockRuntime {
	headDim := config.HiddenSize / config.NumAttentionHeads
	br := &BlockRuntime{
		backend:           b,
		NumAttentionHeads: config.NumAttentionHeads,
		NumKeyValueHeads:  config.NumKeyValueHeads,
		HeadDim:           headDim,
		HiddenSize:        config.HiddenSize,
		IntermediateSize:  config.IntermediateSize,
		RoPETheta:         config.RoPETheta,
		RMSNormEPS:        config.RMSNormEPS,
	}

	// Cache quantized matmul interface check
	br.quantMatMul, _ = b.(backend.QuantizedMatMul)

	// Cache batcher interface check
	br.batcher, _ = b.(backend.Batcher)

	// Cache FP16 operations interface check
	br.fp16Ops, _ = b.(backend.FP16Ops)

	// Cache Q8_0 operations interface check
	br.q8Ops, _ = b.(backend.Q8_0Ops)

	// Cache fused operations interface check
	br.fusedOps, _ = b.(backend.FusedOps)

	return br
}

// matMulTransposed performs C = A @ W^T, dispatching to quantized kernel if supported.
func (b *BlockRuntime) matMulTransposed(a tensor.DevicePtr, w tensor.Tensor, out tensor.DevicePtr, m, n, k int) {
	if w.IsQuantized() && b.quantMatMul != nil {
		switch w.QuantProfile() {
		case tensor.Q4_0:
			// Use GPU-native Q4_0 kernel
			b.quantMatMul.MatMulQ4_0(a, w.DevicePtr(), out, m, n, k)
			return
		case tensor.Q4_K:
			// Use GPU-native Q4_K kernel (only M=1 for now)
			b.quantMatMul.MatMulQ4_K(a, w.DevicePtr(), out, m, n, k)
			return
		case tensor.Q6_K:
			// Use GPU-native Q6_K kernel (only M=1 for now)
			b.quantMatMul.MatMulQ6_K(a, w.DevicePtr(), out, m, n, k)
			return
		}
	}
	// Fall back to F32 matmul
	b.backend.MatMulTransposed(a, w.DevicePtr(), out, m, n, k)
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

	// Debug: print input to layer (only for layer 0)
	if debugDecode && layerIdx == 0 {
		b.debugBlockTensor("L0 Input (x)", xPtr, seqLen*hiddenSize)
		b.debugBlockTensor("L0 AttnNorm weights", b.AttnNorm.DevicePtr(), hiddenSize)
	}

	// 1. RMSNorm (Attention)
	if !b.AttnNorm.DevicePtr().IsNil() {
		b.backend.RMSNorm(xPtr, b.AttnNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
	}
	b.backend.Sync()

	// Debug: print norm output for layer 0
	if debugDecode && layerIdx == 0 {
		b.debugBlockTensor("L0 RMSNorm out", normOutPtr, seqLen*hiddenSize)
	}

	// 2. Q/K/V Projections (using quantized matmul if available)
	// Wq: [numHeads*headDim, hiddenSize] -> Q: [seqLen, numHeads*headDim]
	if debugDecode && layerIdx == 0 {
		fmt.Printf("[BLOCK] L0 Wq: shape=%v quantized=%v profile=%v\n",
			b.Wq.Shape().Dims(), b.Wq.IsQuantized(), b.Wq.QuantProfile())
	}
	if !b.Wq.DevicePtr().IsNil() {
		qDim := b.Wq.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.Wq, qPtr, seqLen, qDim, hiddenSize)
	}
	b.backend.Sync()

	// Debug: print Q projection for layer 0
	if debugDecode && layerIdx == 0 {
		b.debugBlockTensor("L0 After Wq", qPtr, qSize)
	}

	// Wk: [numKVHeads*headDim, hiddenSize] -> K: [seqLen, numKVHeads*headDim]
	if !b.Wk.DevicePtr().IsNil() {
		kDim := b.Wk.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.Wk, kPtr, seqLen, kDim, hiddenSize)
	}

	// Wv: [numKVHeads*headDim, hiddenSize] -> V: [seqLen, numKVHeads*headDim]
	if !b.Wv.DevicePtr().IsNil() {
		vDim := b.Wv.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.Wv, vPtr, seqLen, vDim, hiddenSize)
	}

	// 3. RoPE - Apply to Q and K
	b.backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, pos, float32(b.RoPETheta))

	// 4. Attention
	scale := float32(1.0 / sqrt(float64(headDim)))
	if seqLen == 1 {
		// Decode: use SDPA kernel (for single token, K/V are the current token only)
		b.backend.SDPA(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
	} else {
		// Prefill: use SDPAPrefill kernel with causal masking (FA2 threshold handled in backend)
		b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
	}

	// 5. Output Projection and residual
	if !b.Wo.DevicePtr().IsNil() {
		oDim := b.Wo.Shape().Dims()[0]
		b.matMulTransposed(attnOutPtr, b.Wo, normOutPtr, seqLen, oDim, numHeads*headDim)
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
		b.matMulTransposed(normOutPtr, b.W1, gatePtr, seqLen, w1Dim, hiddenSize)
	}

	// Up projection: up = normOut @ W3^T
	if !b.W3.DevicePtr().IsNil() {
		w3Dim := b.W3.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.W3, upPtr, seqLen, w3Dim, hiddenSize)
	}

	// Fused SiLU+Mul: gate = silu(gate) * up
	b.backend.SiLUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)

	// Down projection and residual
	if !b.W2.DevicePtr().IsNil() {
		w2Dim := b.W2.Shape().Dims()[0]
		b.matMulTransposed(gatePtr, b.W2, normOutPtr, seqLen, w2Dim, intermediateSize)
		// Add residual: x = x + normOut
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	}

	return x, nil
}

// ExecuteWithPagedKV performs the forward pass using paged KV cache.
// This version stores K/V during prefill and retrieves them during decode.
func (b *BlockRuntime) ExecuteWithPagedKV(x, scratch tensor.Tensor, pagedCache *kv.PagedKVCache, seqID int64, layerIdx, startPos int) (tensor.Tensor, error) {
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
	var normOutPtr, qPtr, kPtr, vPtr, attnOutPtr, gatePtr, upPtr tensor.DevicePtr

	if scratchPtr.Location() == tensor.CPU {
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
		_ = upBytes
	} else {
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
	// No sync - operations serialized in Metal command queue

	// 2. Q/K/V Projections
	if !b.Wq.DevicePtr().IsNil() {
		qDim := b.Wq.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.Wq, qPtr, seqLen, qDim, hiddenSize)
	}
	// No sync - operations serialized in Metal command queue

	if !b.Wk.DevicePtr().IsNil() {
		kDim := b.Wk.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.Wk, kPtr, seqLen, kDim, hiddenSize)
	}

	if !b.Wv.DevicePtr().IsNil() {
		vDim := b.Wv.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.Wv, vPtr, seqLen, vDim, hiddenSize)
	}

	// 3. RoPE - Apply to Q and K
	b.backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, startPos, float32(b.RoPETheta))

	// 4. Store current K/V in cache and compute attention
	var fullKPtr, fullVPtr tensor.DevicePtr
	var fullSeqLen int

	if pagedCache != nil {
		// Copy K/V from device to CPU for cache storage
		kData := make([]float32, kvSize)
		vData := make([]float32, kvSize)

		if kPtr.Location() == tensor.CPU {
			kSlice := (*[1 << 28]float32)(unsafe.Pointer(kPtr.Addr()))[:kvSize:kvSize]
			vSlice := (*[1 << 28]float32)(unsafe.Pointer(vPtr.Addr()))[:kvSize:kvSize]
			copy(kData, kSlice)
			copy(vData, vSlice)
		} else {
			kBytes := make([]byte, kvSize*4)
			vBytes := make([]byte, kvSize*4)
			b.backend.Sync() // Ensure RoPE/projections complete before reading
			b.backend.ToHost(kBytes, kPtr)
			b.backend.ToHost(vBytes, vPtr)
			b.backend.Sync()
			kData = bytesToFloat32(kBytes)
			vData = bytesToFloat32(vBytes)
		}

		// Store in cache
		err := pagedCache.StoreKVBatch(seqID, layerIdx, startPos, kData, vData, seqLen)
		if err != nil {
			return x, fmt.Errorf("failed to store KV: %w", err)
		}

		// Get full K/V sequence from cache for attention
		currentPos := startPos + seqLen - 1
		fullK, fullV := pagedCache.GetKVSlice(seqID, layerIdx, currentPos)
		fullSeqLen = len(fullK) / (numKVHeads * headDim)

		// Allocate GPU buffers for full K/V and copy
		fullKBytes := len(fullK) * 4
		fullVBytes := len(fullV) * 4

		if scratchPtr.Location() == tensor.CPU {
			fullKPtr = tensor.NewDevicePtr(tensor.CPU, uintptr(unsafePointer(&fullK[0])))
			fullVPtr = tensor.NewDevicePtr(tensor.CPU, uintptr(unsafePointer(&fullV[0])))
		} else {
			fullKPtr = b.backend.Alloc(fullKBytes)
			fullVPtr = b.backend.Alloc(fullVBytes)
			b.backend.ToDevice(fullKPtr, float32ToBytes(fullK))
			b.backend.ToDevice(fullVPtr, float32ToBytes(fullV))
			b.backend.Sync()
		}
	} else {
		// No cache - use current K/V only (for prefill self-attention)
		fullKPtr = kPtr
		fullVPtr = vPtr
		fullSeqLen = seqLen
	}

	// 5. Attention: Q @ K^T -> softmax -> @ V
	scale := float32(1.0 / sqrt(float64(headDim)))

	if seqLen == 1 {
		// Decode: single query against full KV sequence
		// SDPA expects kvLen as the KV sequence length
		b.backend.SDPA(qPtr, fullKPtr, fullVPtr, attnOutPtr, fullSeqLen, numHeads, numKVHeads, headDim, scale)
	} else {
		// Prefill: use causal SDPAPrefill (FA2 threshold handled in backend)
		b.backend.SDPAPrefill(qPtr, fullKPtr, fullVPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
	}
	// No sync - SDPA must complete before Wo can read attnOut (serialized in queue)

	// 6. Output Projection and residual
	if !b.Wo.DevicePtr().IsNil() {
		oDim := b.Wo.Shape().Dims()[0]
		b.matMulTransposed(attnOutPtr, b.Wo, normOutPtr, seqLen, oDim, numHeads*headDim)
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	}
	// No sync - operations serialized in command queue

	// 7. FFN RMSNorm
	if !b.FFNNorm.DevicePtr().IsNil() {
		b.backend.RMSNorm(xPtr, b.FFNNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
	}
	// No sync - RMSNorm must complete before W1/W3 can read normOut (serialized in queue)

	// 8. MLP: SwiGLU variant
	if !b.W1.DevicePtr().IsNil() {
		w1Dim := b.W1.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.W1, gatePtr, seqLen, w1Dim, hiddenSize)
	}

	if !b.W3.DevicePtr().IsNil() {
		w3Dim := b.W3.Shape().Dims()[0]
		b.matMulTransposed(normOutPtr, b.W3, upPtr, seqLen, w3Dim, hiddenSize)
	}
	// No sync - W1/W3 must complete before SiLU can read gatePtr/upPtr (serialized)

	b.backend.SiLUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)
	// No sync - SiLUMul must complete before W2 can read gatePtr (serialized)

	if !b.W2.DevicePtr().IsNil() {
		w2Dim := b.W2.Shape().Dims()[0]
		b.matMulTransposed(gatePtr, b.W2, normOutPtr, seqLen, w2Dim, intermediateSize)
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	}
	// No sync - next layer's RMSNorm is serialized in queue

	return x, nil
}

// ExecuteWithGPUKV performs the forward pass using GPU-resident KV cache.
// This avoids CPU roundtrips for KV data during decode, providing significant speedup.
func (b *BlockRuntime) ExecuteWithGPUKV(x, scratch tensor.Tensor, gpuCache *GPUKVCache, layerIdx, startPos int) (tensor.Tensor, error) {
	xPtr := x.DevicePtr()
	scratchPtr := scratch.DevicePtr()

	if xPtr.IsNil() || scratchPtr.IsNil() {
		return x, nil
	}

	// Use command buffer batching if available (and not profiling - profiling needs sync points)
	// Note: Batching disabled - causes incorrect output due to Metal memory hazards
	useBatching := false // b.batcher != nil && !profiler.enabled
	if useBatching {
		b.batcher.BeginBatch()
		defer b.batcher.EndBatch()
	}

	// Profiling helper - only syncs and times when profiling is enabled
	profileOp := func(name string, op func()) {
		if profiler.enabled {
			b.backend.Sync()
			start := time.Now()
			op()
			b.backend.Sync()
			RecordOp(name, time.Since(start))
		} else {
			op()
		}
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

	// Allocate intermediate buffers from scratch space
	var normOutPtr, qPtr, kPtr, vPtr, attnOutPtr, gatePtr, upPtr tensor.DevicePtr

	if scratchPtr.Location() == tensor.CPU {
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
		_ = upBytes
	} else {
		normOutPtr = b.backend.Alloc(normOutBytes)
		qPtr = b.backend.Alloc(qBytes)
		kPtr = b.backend.Alloc(kvBytes)
		vPtr = b.backend.Alloc(kvBytes)
		attnOutPtr = b.backend.Alloc(attnOutBytes)
		gatePtr = b.backend.Alloc(gateBytes)
		upPtr = b.backend.Alloc(upBytes)
	}

	// FP16/Q8_0 KV cache support: allocate conversion buffers if needed
	useFP16KVCache := gpuCache != nil && gpuCache.UseFP16() && b.fp16Ops != nil
	useQ8KVCache := gpuCache != nil && gpuCache.UseQ8_0() && b.q8Ops != nil
	var kF16Ptr, vF16Ptr, qF16Ptr, attnOutF16Ptr tensor.DevicePtr
	var kQ8Ptr, vQ8Ptr tensor.DevicePtr
	if useFP16KVCache && scratchPtr.Location() != tensor.CPU {
		// Allocate FP16 buffers for K, V (for cache storage)
		kF16Ptr = b.backend.Alloc(kvSize * 2) // FP16 = 2 bytes per element
		vF16Ptr = b.backend.Alloc(kvSize * 2)
		// For decode AND prefill, we need FP16 Q and attention output if using F16 path
		qF16Ptr = b.backend.Alloc(qSize * 2)
		attnOutF16Ptr = b.backend.Alloc(qSize * 2)
	}
	if useQ8KVCache && scratchPtr.Location() != tensor.CPU {
		// Allocate Q8_0 buffers for K, V (for cache storage)
		// Q8_0: 34 bytes per 32 elements
		numBlocks := (kvSize + Q8BlockSize - 1) / Q8BlockSize
		q8Size := numBlocks * Q8BytesPerBlock
		kQ8Ptr = b.backend.Alloc(q8Size)
		vQ8Ptr = b.backend.Alloc(q8Size)
	}

	// 1. RMSNorm + Q/K/V Projections
	// Try to use fused RMSNorm+MatMul for Decode (seqLen=1) if weights are Q4_0
	canFuseAttn := seqLen == 1 && b.fusedOps != nil &&
		b.Wq.QuantProfile() == tensor.Q4_0 &&
		b.Wk.QuantProfile() == tensor.Q4_0 &&
		b.Wv.QuantProfile() == tensor.Q4_0

	// FP16 end-to-end path: output Q, K, V directly as FP16 to eliminate conversions
	// This saves 3 FP32→FP16 conversions per layer (K, V for cache, Q for SDPA)
	useFP16Path := canFuseAttn && useFP16KVCache

	if useFP16Path {
		// FP16 path: output directly to FP16 buffers, no conversion needed
		profileOp("FusedRMSNorm+QKV_F16", func() {
			// Q -> qF16Ptr (FP16)
			b.fusedOps.MatMulQ4_0_FusedRMSNormF16(xPtr, b.AttnNorm.DevicePtr(), b.Wq.DevicePtr(), qF16Ptr, 1, qSize, hiddenSize, float32(b.RMSNormEPS))
			// K -> kF16Ptr (FP16)
			b.fusedOps.MatMulQ4_0_FusedRMSNormF16(xPtr, b.AttnNorm.DevicePtr(), b.Wk.DevicePtr(), kF16Ptr, 1, kvSize, hiddenSize, float32(b.RMSNormEPS))
			// V -> vF16Ptr (FP16)
			b.fusedOps.MatMulQ4_0_FusedRMSNormF16(xPtr, b.AttnNorm.DevicePtr(), b.Wv.DevicePtr(), vF16Ptr, 1, kvSize, hiddenSize, float32(b.RMSNormEPS))
		})
	} else if canFuseAttn {
		profileOp("FusedRMSNorm+QKV", func() {
			// Q
			b.fusedOps.MatMulQ4_0_FusedRMSNorm(xPtr, b.AttnNorm.DevicePtr(), b.Wq.DevicePtr(), qPtr, 1, qSize, hiddenSize, float32(b.RMSNormEPS))
			// K
			b.fusedOps.MatMulQ4_0_FusedRMSNorm(xPtr, b.AttnNorm.DevicePtr(), b.Wk.DevicePtr(), kPtr, 1, kvSize, hiddenSize, float32(b.RMSNormEPS))
			// V
			b.fusedOps.MatMulQ4_0_FusedRMSNorm(xPtr, b.AttnNorm.DevicePtr(), b.Wv.DevicePtr(), vPtr, 1, kvSize, hiddenSize, float32(b.RMSNormEPS))
		})
	} else {
		// Standard path
		// Debug: check input before RMSNorm for prefill
		if debugDecode && layerIdx <= 1 {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d Input x (before RMSNorm)", layerIdx), xPtr, seqLen*hiddenSize)
		}
		profileOp("RMSNorm", func() {
			if !b.AttnNorm.DevicePtr().IsNil() {
				b.backend.RMSNorm(xPtr, b.AttnNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
			}
		})
		// Debug: check RMSNorm output for prefill
		if debugDecode && layerIdx <= 1 {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d Standard RMSNorm out (prefill)", layerIdx), normOutPtr, seqLen*hiddenSize)
			b.debugBlockTensor(fmt.Sprintf("L%d AttnNorm weights", layerIdx), b.AttnNorm.DevicePtr(), hiddenSize)
		}

		profileOp("Wq", func() {
			if !b.Wq.DevicePtr().IsNil() {
				qDim := b.Wq.Shape().Dims()[0]
				b.matMulTransposed(normOutPtr, b.Wq, qPtr, seqLen, qDim, hiddenSize)
			}
		})

		profileOp("Wk", func() {
			if !b.Wk.DevicePtr().IsNil() {
				kDim := b.Wk.Shape().Dims()[0]
				b.matMulTransposed(normOutPtr, b.Wk, kPtr, seqLen, kDim, hiddenSize)
			}
		})

		profileOp("Wv", func() {
			if !b.Wv.DevicePtr().IsNil() {
				vDim := b.Wv.Shape().Dims()[0]
				b.matMulTransposed(normOutPtr, b.Wv, vPtr, seqLen, vDim, hiddenSize)
			}
		})
	}

	// 3. RoPE - Apply to Q and K
	profileOp("RoPE", func() {
		if useFP16Path {
			// FP16 path: apply RoPE directly to FP16 Q and K
			b.fp16Ops.RoPEF16(qF16Ptr, kF16Ptr, headDim, numHeads, numKVHeads, seqLen, startPos, float32(b.RoPETheta))
		} else {
			b.backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, startPos, float32(b.RoPETheta))
		}
	})

	// 4. Append K/V to GPU cache and get pointers for SDPA
	// This uses GPU-to-GPU copy - no CPU roundtrip!
	// NOTE: Mid-layer sync is required for correct output. Without it, the encoder
	// ordering causes incorrect results. The sync commits the current batch and
	// starts a new one. Investigation ongoing - see docs/llama_cpp_kernel_analysis.md
	// Set VEXEL_SKIP_MID_SYNC=1 to test without the sync.
	if useBatching && !skipMidLayerSync {
		b.batcher.EndBatch()
		b.batcher.BeginBatch()
	}

	var fullKPtr, fullVPtr tensor.DevicePtr
	var fullSeqLen int
	profileOp("KVCache", func() {
		if useFP16Path {
			// FP16 path: K and V already in FP16 (from fused kernel), no conversion needed
			// AppendKV uses CopyBufferBatched which integrates with command batching
			fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kF16Ptr, vF16Ptr, seqLen)
		} else if useFP16KVCache {
			// Convert FP32 K/V to FP16 before storing in cache
			b.fp16Ops.ConvertF32ToF16(kPtr, kF16Ptr, kvSize)
			b.fp16Ops.ConvertF32ToF16(vPtr, vF16Ptr, kvSize)
			// AppendKV uses CopyBufferBatched which integrates with command batching
			// No explicit sync needed - blit and compute on same command buffer
			fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kF16Ptr, vF16Ptr, seqLen)
		} else if useQ8KVCache {
			// Quantize FP32 K/V to Q8_0 before storing in cache
			b.q8Ops.QuantizeF32ToQ8_0(kPtr, kQ8Ptr, kvSize)
			b.q8Ops.QuantizeF32ToQ8_0(vPtr, vQ8Ptr, kvSize)
			// AppendKV uses CopyBufferBatched - no explicit sync needed
			fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kQ8Ptr, vQ8Ptr, seqLen)
		} else {
			fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kPtr, vPtr, seqLen)
		}
	})

	// 5. Attention: Q @ K^T -> softmax -> @ V
	scale := float32(1.0 / sqrt(float64(headDim)))

	// Debug: dump Q, K, V before SDPA
	if debugDecode && layerIdx <= 2 {
		b.debugBlockTensor(fmt.Sprintf("L%d Q before SDPA", layerIdx), qPtr, qSize)
		if seqLen == 1 {
			b.debugBlockTensor(fmt.Sprintf("L%d fullK before SDPA", layerIdx), fullKPtr, fullSeqLen*numKVHeads*headDim)
			b.debugBlockTensor(fmt.Sprintf("L%d fullV before SDPA", layerIdx), fullVPtr, fullSeqLen*numKVHeads*headDim)
		} else {
			b.debugBlockTensor(fmt.Sprintf("L%d K before SDPA", layerIdx), kPtr, kvSize)
			b.debugBlockTensor(fmt.Sprintf("L%d V before SDPA", layerIdx), vPtr, kvSize)
		}
	}

	profileOp("SDPA", func() {
		if seqLen == 1 {
			// Decode: single query against full KV sequence
			if useFP16Path {
				// FP16 path: Q already in FP16 (from fused kernel), no Q conversion needed
				b.fp16Ops.SDPAF16(qF16Ptr, fullKPtr, fullVPtr, attnOutF16Ptr, fullSeqLen, numHeads, numKVHeads, headDim, scale, gpuCache.KVHeadStride())
				b.fp16Ops.ConvertF16ToF32(attnOutF16Ptr, attnOutPtr, qSize)
			} else if useFP16KVCache {
				// FP16 KV cache but Q is FP32: convert Q to FP16
				b.fp16Ops.ConvertF32ToF16(qPtr, qF16Ptr, qSize)
				b.fp16Ops.SDPAF16(qF16Ptr, fullKPtr, fullVPtr, attnOutF16Ptr, fullSeqLen, numHeads, numKVHeads, headDim, scale, gpuCache.KVHeadStride())
				b.fp16Ops.ConvertF16ToF32(attnOutF16Ptr, attnOutPtr, qSize)
			} else if useQ8KVCache {
				// Q8_0 path: use Q8_0 SDPA with FP32 Q and Q8_0 K/V
				b.q8Ops.SDPAQ8_0(qPtr, fullKPtr, fullVPtr, attnOutPtr, fullSeqLen, numHeads, numKVHeads, headDim, scale)
			} else {
				b.backend.SDPA(qPtr, fullKPtr, fullVPtr, attnOutPtr, fullSeqLen, numHeads, numKVHeads, headDim, scale)
			}
		} else {
			// Prefill: use causal SDPAPrefill
			if useFP16KVCache {
				// FP16 path: convert Q to FP16, use FA2 F16, convert output back to FP32
				// Note: kF16Ptr and vF16Ptr already contain the F16 converted K/V from step 4
				b.fp16Ops.ConvertF32ToF16(qPtr, qF16Ptr, qSize)
				b.fp16Ops.SDPAPrefillF16(qF16Ptr, kF16Ptr, vF16Ptr, attnOutF16Ptr, seqLen, numHeads, numKVHeads, headDim, scale)
				b.fp16Ops.ConvertF16ToF32(attnOutF16Ptr, attnOutPtr, qSize)
			} else {
				// FP32 path
				b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
			}
		}
	})

	// Debug: dump attention output after SDPA
	if debugDecode && layerIdx <= 1 {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d AttnOut after SDPA", layerIdx), attnOutPtr, qSize)
	}

	// 6. Output Projection and residual
	profileOp("Wo", func() {
		if !b.Wo.DevicePtr().IsNil() {
			oDim := b.Wo.Shape().Dims()[0]
			b.matMulTransposed(attnOutPtr, b.Wo, normOutPtr, seqLen, oDim, numHeads*headDim)
		}
	})
	if debugDecode && layerIdx <= 1 {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d After Wo (attn output)", layerIdx), normOutPtr, seqLen*hiddenSize)
		b.debugBlockTensor(fmt.Sprintf("L%d x before Add1", layerIdx), xPtr, seqLen*hiddenSize)
	}
	// 7. Add1 + FFN RMSNorm + MLP Projections
	// We fused Add+RMSNorm before, but now we want to fuse RMSNorm+MatMul.
	// We can't easily fuse Add+RMSNorm+MatMul.
	// Strategies:
	// A. Add (separate). Fused RMSNorm+MatMul (for Gate/Up).
	//    Cost: Read x, Write x (Add). Read x (Gate). Read x (Up). Total 4 ops.
	//    Old: Add (2). RMSNorm (2). Gate (1). Up (1). Total 6 ops.
	//    Savings: 2 ops (norm write/read).
	// B. Fused Add+RMSNorm. Standard MatMul.
	//    Cost: Add+RMSNorm (Read x, Read resid, Write norm). Gate (Read norm). Up (Read norm). Total 5 ops.
	//    Note: Add+RMSNorm writes `normOut`. Does it update `x`?
	//    Our `AddRMSNorm` kernel updates `x` in place AND writes `out`.
	//    So: Read x, Read resid, Write x, Write norm. Total 4 ops.
	//    Then Gate (Read norm), Up (Read norm). Total 6 ops.
	//    Strategy A seems better: Add (Read x, Read resid, Write x) -> 3 ops.
	//    Then Fused RMSNorm+MatMul (Read x, Read normW) x 2.
	//    Total traffic:
	//    A: Add (3*K). Fused (2*K). Fused (2*K). Total 7*K.
	//    B: Add+RMSNorm (4*K). MatMul (1.5*K). MatMul (1.5*K). Total 7*K.
	//    Tie?
	//    Wait, `MatMul` reads `W` too.
	//    Let's assume Fused RMSNorm+MatMul is better because it avoids allocating/writing `normOut`.
	//    Also `AddRMSNorm` logic in `metal_bridge` is `x = x + residual; out = RMSNorm(x)`.
	//    It writes both.
	//    If we use separate Add, then Fused RMSNorm+MatMul:
	//    Add: `x += residual`. (Read x, Read res, Write x).
	//    Fused: `gate = (RMS(x) @ W1)`. (Read x).
	//    Fused: `up = (RMS(x) @ W3)`. (Read x).
	//    So we skip writing `normOut`.

	// 7. Add1
	profileOp("Add1", func() {
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	})
	if debugDecode && layerIdx <= 1 {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d x after Add1", layerIdx), xPtr, seqLen*hiddenSize)
	}

	// 8. MLP: SwiGLU variant
	canFuseFFN := seqLen == 1 && b.fusedOps != nil &&
		b.W1.QuantProfile() == tensor.Q4_0 &&
		b.W3.QuantProfile() == tensor.Q4_0

	if canFuseFFN {
		profileOp("FusedRMSNorm+GateUp", func() {
			w1Dim := b.W1.Shape().Dims()[0]
			b.fusedOps.MatMulQ4_0_FusedRMSNorm(xPtr, b.FFNNorm.DevicePtr(), b.W1.DevicePtr(), gatePtr, 1, w1Dim, hiddenSize, float32(b.RMSNormEPS))
			w3Dim := b.W3.Shape().Dims()[0]
			b.fusedOps.MatMulQ4_0_FusedRMSNorm(xPtr, b.FFNNorm.DevicePtr(), b.W3.DevicePtr(), upPtr, 1, w3Dim, hiddenSize, float32(b.RMSNormEPS))
		})
	} else {
		// Standard path (RMSNorm then MatMul)
		profileOp("RMSNorm2", func() {
			if !b.FFNNorm.DevicePtr().IsNil() {
				b.backend.RMSNorm(xPtr, b.FFNNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
			}
		})

		profileOp("W1", func() {
			if !b.W1.DevicePtr().IsNil() {
				w1Dim := b.W1.Shape().Dims()[0]
				b.matMulTransposed(normOutPtr, b.W1, gatePtr, seqLen, w1Dim, hiddenSize)
			}
		})

		profileOp("W3", func() {
			if !b.W3.DevicePtr().IsNil() {
				w3Dim := b.W3.Shape().Dims()[0]
				b.matMulTransposed(normOutPtr, b.W3, upPtr, seqLen, w3Dim, hiddenSize)
			}
		})
	}

	profileOp("SiLUMul", func() {
		b.backend.SiLUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)
	})

	profileOp("W2", func() {
		if !b.W2.DevicePtr().IsNil() {
			w2Dim := b.W2.Shape().Dims()[0]
			b.matMulTransposed(gatePtr, b.W2, normOutPtr, seqLen, w2Dim, intermediateSize)
		}
	})
	if debugDecode && layerIdx <= 1 {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d After W2 (MLP output)", layerIdx), normOutPtr, seqLen*hiddenSize)
		b.debugBlockTensor(fmt.Sprintf("L%d x before Add2", layerIdx), xPtr, seqLen*hiddenSize)
	}
	profileOp("Add2", func() {
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	})
	if debugDecode && layerIdx <= 1 {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d x after Add2 (FINAL)", layerIdx), xPtr, seqLen*hiddenSize)
	}

	return x, nil
}

// unsafePointer returns an unsafe.Pointer from the first element
func unsafePointer(p *float32) unsafe.Pointer {
	return unsafe.Pointer(p)
}

// float32ToBytes converts []float32 to []byte
func float32ToBytes(data []float32) []byte {
	result := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(result[i*4:], math.Float32bits(v))
	}
	return result
}
