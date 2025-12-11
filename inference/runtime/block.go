package runtime

import (
	"encoding/binary"
	"fmt"
	"math"
	"time"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/kv"
	"vexel/inference/tensor"
)

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
	toHost.ToHost(data, ptr)
	toHost.Sync()

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
		// Prefill: use SDPAPrefill kernel with causal masking
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
		// Prefill: use causal SDPAPrefill
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
	// Note: Batching measured to have minimal impact, disabled for now
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

	// 1. RMSNorm (Attention)
	profileOp("RMSNorm", func() {
		if !b.AttnNorm.DevicePtr().IsNil() {
			b.backend.RMSNorm(xPtr, b.AttnNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
		}
	})

	// 2. Q/K/V Projections
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

	// 3. RoPE - Apply to Q and K
	profileOp("RoPE", func() {
		b.backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, startPos, float32(b.RoPETheta))
	})

	// 4. Append K/V to GPU cache and get pointers for SDPA
	// This uses GPU-to-GPU copy - no CPU roundtrip!
	// CRITICAL: Must sync before CopyBuffer when batching is enabled!
	// CopyBuffer creates its own command buffer, but the batched commands
	// (RMSNorm, Q/K/V projections, RoPE) haven't been committed yet.
	// Without this sync, CopyBuffer reads uninitialized data.
	if useBatching {
		b.batcher.EndBatch() // Commit and execute the batch
		b.batcher.BeginBatch() // Start a new batch for remaining ops
	}

	var fullKPtr, fullVPtr tensor.DevicePtr
	var fullSeqLen int
	profileOp("KVCache", func() {
		fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kPtr, vPtr, seqLen)
	})

	// 5. Attention: Q @ K^T -> softmax -> @ V
	scale := float32(1.0 / sqrt(float64(headDim)))

	// Debug: dump Q, K, V before SDPA
	if debugDecode && layerIdx <= 1 {
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
			b.backend.SDPA(qPtr, fullKPtr, fullVPtr, attnOutPtr, fullSeqLen, numHeads, numKVHeads, headDim, scale)
		} else {
			// Prefill: use causal SDPAPrefill with just current K/V
			b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
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
	profileOp("Add1", func() {
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	})
	if debugDecode && layerIdx <= 1 {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d x after Add1", layerIdx), xPtr, seqLen*hiddenSize)
	}

	// 7. FFN RMSNorm
	profileOp("RMSNorm2", func() {
		if !b.FFNNorm.DevicePtr().IsNil() {
			b.backend.RMSNorm(xPtr, b.FFNNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
		}
	})

	// 8. MLP: SwiGLU variant
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