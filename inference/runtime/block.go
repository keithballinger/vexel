package runtime

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"
	"unsafe"

	"vexel/inference/backend"
	"vexel/inference/debug"
	"vexel/inference/kv"
	"vexel/inference/tensor"
)

// skipMidLayerSync is no longer needed — memory barriers replace the mid-layer batch split.
// Kept for backward compatibility but unused.
var _ = os.Getenv("VEXEL_SKIP_MID_SYNC")

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
	var nanPositions []int
	for i, v := range values {
		if math.IsNaN(float64(v)) {
			nanCount++
			if len(nanPositions) < 5 {
				nanPositions = append(nanPositions, i)
			}
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
	if nanCount > 0 && len(nanPositions) > 0 {
		fmt.Printf("[BLOCK] %s [%d]: min=%.4f max=%.4f mean=%.4f nan=%d at_pos=%v first%d=%v\n",
			name, len(values), min, max, mean, nanCount, nanPositions, n, values[:n])
	} else {
		fmt.Printf("[BLOCK] %s [%d]: min=%.4f max=%.4f mean=%.4f nan=%d first%d=%v\n",
			name, len(values), min, max, mean, nanCount, n, values[:n])
	}
}

// sqrt is a helper for float64 square root
func sqrt(x float64) float64 {
	return math.Sqrt(x)
}

// debugBlockTensorAtOffset prints a few values starting from a specific offset.
func (b *BlockRuntime) debugBlockTensorAtOffset(name string, ptr tensor.DevicePtr, offsetElements, count int) {
	if !debugDecode || ptr.IsNil() {
		return
	}

	toHost, ok := b.backend.(interface {
		ToHost([]byte, tensor.DevicePtr)
		Sync()
	})
	if !ok {
		return
	}

	bytesToRead := (offsetElements + count) * 4
	data := make([]byte, bytesToRead)

	toHost.Sync()
	toHost.ToHost(data, ptr)

	vals := make([]float32, count)
	for i := 0; i < count; i++ {
		vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[(offsetElements+i)*4:]))
	}
	fmt.Printf("[DEBUG] %s at offset %d: %v\n", name, offsetElements, vals)
}

// debugKVCacheHeadMajor prints KV cache values for head 0 at different positions.
// This verifies that head-major layout is correctly populated.
// Layout: [numKVHeads, maxSeqLen, headDim]
// For head h, position p: offset = h * kvHeadStride + p * headDim
func (b *BlockRuntime) debugKVCacheHeadMajor(name string, ptr tensor.DevicePtr, kvHeadStride, fullSeqLen, headDim int) {
	if !debugDecode || ptr.IsNil() {
		return
	}

	toHost, ok := b.backend.(interface {
		ToHost([]byte, tensor.DevicePtr)
		Sync()
	})
	if !ok {
		return
	}

	// Read enough to cover head 0's first 3 positions (or less if fullSeqLen is smaller)
	positionsToRead := 3
	if fullSeqLen < positionsToRead {
		positionsToRead = fullSeqLen
	}
	bytesToRead := positionsToRead * headDim * 4
	data := make([]byte, bytesToRead)

	toHost.Sync()
	toHost.ToHost(data, ptr)

	fmt.Printf("[DEBUG] %s head-major (kvHeadStride=%d, fullSeqLen=%d, headDim=%d):\n", name, kvHeadStride, fullSeqLen, headDim)

	// For head 0, positions are contiguous starting at offset 0
	for p := 0; p < positionsToRead; p++ {
		offset := p * headDim // For head 0
		vals := make([]float32, 4)
		for i := 0; i < 4 && offset+i < len(data)/4; i++ {
			vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[(offset+i)*4:]))
		}
		fmt.Printf("  pos %d: [%.6f, %.6f, %.6f, %.6f]\n", p, vals[0], vals[1], vals[2], vals[3])
	}
}

// debugBlockValueAt prints the value at a specific position for debugging
func (b *BlockRuntime) debugBlockValueAt(name string, ptr tensor.DevicePtr, pos int) {
	if !debugDecode || ptr.IsNil() {
		return
	}
	data := make([]byte, 4)
	toHost, ok := b.backend.(interface {
		ToHostOffset([]byte, tensor.DevicePtr, int)
		Sync()
	})
	if !ok {
		// Fallback: read 4 bytes at offset
		toHost2, ok2 := b.backend.(interface {
			ToHost([]byte, tensor.DevicePtr)
		})
		if !ok2 {
			return
		}
		// Read the whole thing and extract - inefficient but works
		fullData := make([]byte, (pos+1)*4)
		toHost2.ToHost(fullData, ptr)
		copy(data, fullData[pos*4:])
	} else {
		toHost.Sync()
		toHost.ToHostOffset(data, ptr, pos*4)
	}
	val := math.Float32frombits(binary.LittleEndian.Uint32(data))
	fmt.Printf("[BLOCK] %s: value=%.6f\n", name, val)
}

// debugCapture captures tensor data to the debug harness if enabled.
// This provides structured, targetable debug output.
func (b *BlockRuntime) debugCapture(layer, position int, op, name string, ptr tensor.DevicePtr, numElements int) {
	if !debug.ShouldCapture(layer, position, op) {
		return
	}
	if ptr.IsNil() || numElements == 0 {
		return
	}

	toHost, ok := b.backend.(interface {
		ToHost([]byte, tensor.DevicePtr)
		Sync()
	})
	if !ok {
		return
	}

	toHost.Sync()
	data := make([]byte, numElements*4)
	toHost.ToHost(data, ptr)

	floats := make([]float32, numElements)
	for i := range floats {
		floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}

	debug.Capture(layer, position, op, name, floats)
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
	RoPEDim           int  // Dimensions to rotate (0 = full headDim). For partial RoPE like Phi-2.
	RoPENeox          bool // NEOX-style RoPE (split pairs: i, i+dim/2) vs LLaMA-style (interleaved: 2i, 2i+1)
	SlidingWindow       int                // Window size for sliding window attention (0 = infinite/full context)
	AttentionWindowType AttentionWindowType // Window pattern: Global, Sliding, or Alternating
	RMSNormEPS        float64

	// Architecture-specific config
	NormType         NormType // RMSNorm (LLaMA) vs LayerNorm (Phi)
	MLPType          MLPType  // SwiGLU (LLaMA) vs GELU (Phi)
	HasBias          bool     // Whether model has bias terms
	ParallelResidual bool     // True: x + attn(norm(x)) + mlp(norm(x)), False: serial

	// Weights (stored as DevicePtr for GPU execution)
	AttnNorm     tensor.Tensor
	Wq, Wk, Wv   tensor.Tensor
	Wqkv         tensor.Tensor // Combined QKV projection (optional, for Phi-2 optimization)
	Wo           tensor.Tensor

	FFNNorm      tensor.Tensor
	PostAttnNorm tensor.Tensor // Post-attention RMSNorm weight (Gemma 2)
	PostFFNNorm  tensor.Tensor // Post-FFN RMSNorm weight (Gemma 2)
	W1, W2, W3   tensor.Tensor // Gate, Down, Up (W3 not used for GELU MLP)
	W1W3         tensor.Tensor // Combined gate+up projection (optional, fused prefill path)

	// Optional bias tensors (for Phi, GPT-2, etc.)
	AttnNormBias tensor.Tensor // LayerNorm bias for attention
	FFNNormBias  tensor.Tensor // LayerNorm bias for FFN
	WqBias       tensor.Tensor // Q projection bias
	WkBias       tensor.Tensor // K projection bias
	WvBias       tensor.Tensor // V projection bias
	WqkvBias     tensor.Tensor // Combined QKV projection bias (optional, for Phi-2 optimization)
	WoBias       tensor.Tensor // Output projection bias
	W1Bias       tensor.Tensor // Gate/Up projection bias
	W2Bias       tensor.Tensor // Down projection bias

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

	// Cached interface for QKV deinterleave (fused QKV prefill path)
	qkvDeinterleaver backend.QKVDeinterleaver

	// Cached interface for gate_up deinterleave (fused gate_up prefill path)
	gateUpDeinterleaver backend.GateUpDeinterleaver

	// Cached interface for LayerNorm operations
	layerNormOps backend.LayerNormOps

	// Cached interface for GELU operations
	geluOps backend.GELUOps

	// Cached interface for bias operations
	biasOps backend.BiasOps

	// Cached interface for logit soft-capping in attention (Gemma 2)
	softCapOps backend.SoftCapAttentionOps

	// Cached interface for learned RoPE frequency scaling (Gemma 2)
	scaledRoPEOps backend.ScaledRoPEOps

	// Cached interface for scratch buffer sub-allocation (Metal GPU)
	scratchAlloc backend.ScratchAllocator

	// Pre-computed RoPE inverse frequency buffer on device ([headDim/2] float32).
	// Non-nil when using learned RoPE scaling (e.g. Gemma 2 with RoPEFreqScales).
	ropeFreqBuf tensor.DevicePtr

	// Gemma 2 attention config
	AttentionLogitSoftCap float32 // 0 = disabled, typically 30.0 for Gemma 2
	HasPostNorms          bool    // Apply post-norms after attn and MLP (before residual)

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
		RoPEDim:           config.RoPEDim,  // 0 = full headDim (LLaMA), otherwise partial (Phi-2)
		RoPENeox:          config.RoPENeox, // NEOX-style (Phi) vs LLaMA-style RoPE
		SlidingWindow:       config.SlidingWindow,
		AttentionWindowType: config.AttentionWindowType,
		RMSNormEPS:        config.RMSNormEPS,
		NormType:          config.NormType,
		MLPType:           config.MLPType,
		HasBias:           config.HasBias,
		ParallelResidual:  config.ParallelResidual,
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

	// Cache QKV deinterleave interface check
	br.qkvDeinterleaver, _ = b.(backend.QKVDeinterleaver)
	br.gateUpDeinterleaver, _ = b.(backend.GateUpDeinterleaver)

	// Cache LayerNorm, GELU, Bias, and SoftCap operations interface checks
	br.layerNormOps, _ = b.(backend.LayerNormOps)
	br.geluOps, _ = b.(backend.GELUOps)
	br.biasOps, _ = b.(backend.BiasOps)
	br.softCapOps, _ = b.(backend.SoftCapAttentionOps)
	br.scaledRoPEOps, _ = b.(backend.ScaledRoPEOps)
	br.scratchAlloc, _ = b.(backend.ScratchAllocator)

	// Gemma 2 attention config
	br.AttentionLogitSoftCap = config.AttentionLogitSoftCap
	br.HasPostNorms = config.HasPostNorms

	return br
}

// useSlidingWindow returns true if the given layer should use sliding window attention.
// For WindowSliding: all layers use sliding window.
// For WindowAlternating (Gemma 2): odd layers use sliding window, even layers use global.
// For WindowGlobal: no layers use sliding window.
func (b *BlockRuntime) useSlidingWindow(layerIdx int) bool {
	if b.SlidingWindow <= 0 {
		return false // No window configured
	}
	switch b.AttentionWindowType {
	case WindowSliding:
		return true
	case WindowAlternating:
		return layerIdx%2 == 1 // Odd layers use sliding window
	default:
		return false
	}
}

// effectiveKVLen computes the effective KV sequence length for attention,
// applying sliding window when appropriate based on layer index.
// totalKVLen is the total number of KV positions available.
// Returns the number of positions to attend to and the start position offset.
func (b *BlockRuntime) effectiveKVLen(layerIdx, totalKVLen int) (kvLen, startPos int) {
	if b.useSlidingWindow(layerIdx) && totalKVLen > b.SlidingWindow {
		return b.SlidingWindow, totalKVLen - b.SlidingWindow
	}
	return totalKVLen, 0
}

// SetRoPEFreqBuffer stores a pre-computed inverse frequency buffer for learned RoPE scaling.
// freqBuf has [headDim/2] float32 values on the device. When set, RoPE dispatch uses
// the scaled kernel instead of computing frequencies from theta.
func (b *BlockRuntime) SetRoPEFreqBuffer(freqBuf tensor.DevicePtr) {
	b.ropeFreqBuf = freqBuf
}

// applyRoPE dispatches to the appropriate RoPE kernel — learned-frequency (scaled) or standard.
func (b *BlockRuntime) applyRoPE(qPtr, kPtr tensor.DevicePtr, headDim, numHeads, numKVHeads, seqLen, startPos int) {
	if !b.ropeFreqBuf.IsNil() && b.scaledRoPEOps != nil {
		b.scaledRoPEOps.RoPEWithFreqs(qPtr, kPtr, b.ropeFreqBuf, headDim, numHeads, numKVHeads, seqLen, startPos, b.RoPENeox)
	} else {
		b.backend.RoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, startPos, b.RoPEDim, float32(b.RoPETheta), b.RoPENeox)
	}
}

// matMulTransposed performs C = A @ W^T, dispatching to quantized kernel if supported.
func (b *BlockRuntime) matMulTransposed(a tensor.DevicePtr, w tensor.Tensor, out tensor.DevicePtr, m, n, k int) {
	// Debug: print all FP32 matmul calls
	debugMatMul := os.Getenv("DEBUG_MATMUL") == "1"
	if debugMatMul && !w.IsQuantized() && m > 1 {
		fmt.Printf("[DEBUG] FP32 matMulTransposed: m=%d, n=%d, k=%d\n", m, n, k)
	}
	if w.IsQuantized() && b.quantMatMul != nil {
		switch w.QuantProfile() {
		case tensor.Q4_0:
			// Use GPU-native Q4_0 kernel
			b.quantMatMul.MatMulQ4_0(a, w.DevicePtr(), out, m, n, k)
			return
		case tensor.Q4_K:
			// Use GPU-native Q4_K kernel
			b.quantMatMul.MatMulQ4_K(a, w.DevicePtr(), out, m, n, k)
			return
		case tensor.Q6_K:
			// Use GPU-native Q6_K kernel (only M=1 for now)
			b.quantMatMul.MatMulQ6_K(a, w.DevicePtr(), out, m, n, k)
			return
		case tensor.Q5_K:
			// Use GPU-native Q5_K kernel (only M=1 for now)
			b.quantMatMul.MatMulQ5_K(a, w.DevicePtr(), out, m, n, k)
			return
		case tensor.Q8_0:
			// Use GPU-native Q8_0 kernel
			b.quantMatMul.MatMulQ8_0(a, w.DevicePtr(), out, m, n, k)
			return
		case tensor.BF16:
			// Use GPU-native BF16 kernel (converts BF16→F32 on the fly)
			b.quantMatMul.MatMulBF16(a, w.DevicePtr(), out, m, n, k)
			return
		default:
			// Warn about unsupported quantization profile - falling back to F32
			// but the data is still quantized which will produce garbage!
			debugDecode := os.Getenv("DEBUG_DECODE") == "1"
			if debugDecode {
				fmt.Printf("[WARN] matMulTransposed: unsupported quant profile %v, falling back to F32 which will read garbage!\n", w.QuantProfile())
			}
		}
	}
	// Fall back to F32 matmul
	b.backend.MatMulTransposed(a, w.DevicePtr(), out, m, n, k)
}

// matMulTransposedWithBias performs C = A @ W^T + bias if bias is provided.
func (b *BlockRuntime) matMulTransposedWithBias(a tensor.DevicePtr, w, bias tensor.Tensor, out tensor.DevicePtr, m, n, k int) {
	b.matMulTransposed(a, w, out, m, n, k)
	if !bias.DevicePtr().IsNil() && b.biasOps != nil {
		b.biasOps.AddBias(out, bias.DevicePtr(), out, m, n)
	}
}

// applyNorm applies the appropriate normalization (RMSNorm or LayerNorm) based on config.
func (b *BlockRuntime) applyNorm(x, weight, bias, out tensor.DevicePtr, rows, cols int) {
	debugDecode := os.Getenv("DEBUG_DECODE") == "1"
	if b.NormType == NormLayerNorm && b.layerNormOps != nil {
		if debugDecode {
			fmt.Printf("[DEBUG] Using LayerNorm (NormType=%v, layerNormOps=%v)\n", b.NormType, b.layerNormOps != nil)
		}
		b.layerNormOps.LayerNorm(x, weight, bias, out, rows, cols, float32(b.RMSNormEPS))
	} else {
		if debugDecode {
			fmt.Printf("[DEBUG] Using RMSNorm (NormType=%v, layerNormOps=%v)\n", b.NormType, b.layerNormOps != nil)
		}
		// Default to RMSNorm (ignores bias)
		b.backend.RMSNorm(x, weight, out, rows, cols, float32(b.RMSNormEPS))
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

	// 1. Attention normalization (RMSNorm or LayerNorm)
	if !b.AttnNorm.DevicePtr().IsNil() {
		b.applyNorm(xPtr, b.AttnNorm.DevicePtr(), b.AttnNormBias.DevicePtr(), normOutPtr, seqLen, hiddenSize)
	}

	// 2. Q/K/V Projections (with bias support)
	if !b.Wq.DevicePtr().IsNil() {
		qDim := b.Wq.Shape().Dims()[0]
		b.matMulTransposedWithBias(normOutPtr, b.Wq, b.WqBias, qPtr, seqLen, qDim, hiddenSize)
	}
	if !b.Wk.DevicePtr().IsNil() {
		kDim := b.Wk.Shape().Dims()[0]
		b.matMulTransposedWithBias(normOutPtr, b.Wk, b.WkBias, kPtr, seqLen, kDim, hiddenSize)
	}
	if !b.Wv.DevicePtr().IsNil() {
		vDim := b.Wv.Shape().Dims()[0]
		b.matMulTransposedWithBias(normOutPtr, b.Wv, b.WvBias, vPtr, seqLen, vDim, hiddenSize)
	}

	// 3. RoPE - Apply to Q and K
	b.applyRoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, pos)

	// 4. Attention
	scale := float32(1.0 / sqrt(float64(headDim)))
	if seqLen == 1 {
		if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
			b.softCapOps.SDPASoftCap(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap, headDim)
		} else {
			b.backend.SDPA(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale, headDim)
		}
	} else {
		if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
			b.softCapOps.SDPAPrefillSoftCap(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap)
		} else {
			b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
		}
	}

	// Save normOut for parallel residual
	var attnResidualPtr tensor.DevicePtr
	if b.ParallelResidual {
		attnResidualPtr = upPtr
	} else {
		attnResidualPtr = normOutPtr
	}

	// 5. Output Projection with bias support
	if !b.Wo.DevicePtr().IsNil() {
		oDim := b.Wo.Shape().Dims()[0]
		b.matMulTransposedWithBias(attnOutPtr, b.Wo, b.WoBias, attnResidualPtr, seqLen, oDim, numHeads*headDim)
		// Post-attention norm (Gemma 2): norm after attn projection, before residual
		if b.HasPostNorms && !b.PostAttnNorm.DevicePtr().IsNil() {
			b.backend.RMSNorm(attnResidualPtr, b.PostAttnNorm.DevicePtr(), attnResidualPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
		}
		if !b.ParallelResidual {
			b.backend.Add(xPtr, attnResidualPtr, xPtr, seqLen*hiddenSize)
		}
	}

	// 6. FFN normalization
	if !b.ParallelResidual && !b.FFNNorm.DevicePtr().IsNil() {
		b.applyNorm(xPtr, b.FFNNorm.DevicePtr(), b.FFNNormBias.DevicePtr(), normOutPtr, seqLen, hiddenSize)
	}

	// 7. MLP
	if b.MLPType == MLPGELU {
		if !b.W1.DevicePtr().IsNil() {
			w1Dim := b.W1.Shape().Dims()[0]
			b.matMulTransposedWithBias(normOutPtr, b.W1, b.W1Bias, gatePtr, seqLen, w1Dim, hiddenSize)
		}
		if b.geluOps != nil {
			b.geluOps.GELU(gatePtr, gatePtr, seqLen*intermediateSize)
		}
		if !b.W2.DevicePtr().IsNil() {
			w2Dim := b.W2.Shape().Dims()[0]
			b.matMulTransposedWithBias(gatePtr, b.W2, b.W2Bias, normOutPtr, seqLen, w2Dim, intermediateSize)
		}
	} else {
		// Gated MLP: SwiGLU (LLaMA) or GeGLU (Gemma)
		if !b.W1.DevicePtr().IsNil() {
			w1Dim := b.W1.Shape().Dims()[0]
			b.matMulTransposed(normOutPtr, b.W1, gatePtr, seqLen, w1Dim, hiddenSize)
		}
		if !b.W3.DevicePtr().IsNil() {
			w3Dim := b.W3.Shape().Dims()[0]
			b.matMulTransposed(normOutPtr, b.W3, upPtr, seqLen, w3Dim, hiddenSize)
		}
		if b.MLPType == MLPGeGLU {
			if b.geluOps != nil {
				b.geluOps.GELUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)
			}
		} else {
			b.backend.SiLUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)
		}
		if !b.W2.DevicePtr().IsNil() {
			w2Dim := b.W2.Shape().Dims()[0]
			b.matMulTransposed(gatePtr, b.W2, normOutPtr, seqLen, w2Dim, intermediateSize)
		}
	}

	// 8. Post-FFN norm (Gemma 2): norm after MLP, before residual
	if b.HasPostNorms && !b.PostFFNNorm.DevicePtr().IsNil() {
		b.backend.RMSNorm(normOutPtr, b.PostFFNNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
	}

	// 9. Add residuals
	if b.ParallelResidual {
		b.backend.Add(xPtr, attnResidualPtr, xPtr, seqLen*hiddenSize)
		b.backend.Sync()
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	} else {
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	}

		return x, nil
	}
	
	// ExecuteWithPagedKV performs the forward pass using paged KV cache.
// This version stores K/V during prefill and retrieves them during decode.
// When gpuPool is non-nil and seqLen==1 (decode), K/V operations stay entirely
// on GPU using paged attention. Falls back to CPU path otherwise.
func (b *BlockRuntime) ExecuteWithPagedKV(x, scratch tensor.Tensor, pagedCache *kv.PagedKVCache, gpuPool *GPUBlockPool, seqID int64, layerIdx, startPos int) (tensor.Tensor, error) {
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

	// 1. Attention normalization (RMSNorm or LayerNorm depending on architecture)
	if !b.AttnNorm.DevicePtr().IsNil() {
		b.applyNorm(xPtr, b.AttnNorm.DevicePtr(), b.AttnNormBias.DevicePtr(), normOutPtr, seqLen, hiddenSize)
	}
	// No sync - operations serialized in Metal command queue

	// 2. Q/K/V Projections (with bias support for Phi)
	if !b.Wq.DevicePtr().IsNil() {
		qDim := b.Wq.Shape().Dims()[0]
		b.matMulTransposedWithBias(normOutPtr, b.Wq, b.WqBias, qPtr, seqLen, qDim, hiddenSize)
	}
	// No sync - operations serialized in Metal command queue

	if !b.Wk.DevicePtr().IsNil() {
		kDim := b.Wk.Shape().Dims()[0]
		b.matMulTransposedWithBias(normOutPtr, b.Wk, b.WkBias, kPtr, seqLen, kDim, hiddenSize)
	}

	if !b.Wv.DevicePtr().IsNil() {
		vDim := b.Wv.Shape().Dims()[0]
		b.matMulTransposedWithBias(normOutPtr, b.Wv, b.WvBias, vPtr, seqLen, vDim, hiddenSize)
	}

	// 3. RoPE - Apply to Q and K
	b.applyRoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, startPos)

	// 4. Store current K/V in cache and compute attention
	scale := float32(1.0 / sqrt(float64(headDim)))

	// GPU-native paged path: scatter K/V into GPU block pool and run paged SDPA.
	// Only used for decode (seqLen==1) where paged SDPA is available.
	if gpuPool != nil && seqLen == 1 {
		// Store K/V directly in GPU block pool (no CPU roundtrip)
		if err := gpuPool.StoreKV(layerIdx, seqID, startPos, kPtr, vPtr, seqLen); err != nil {
			return x, fmt.Errorf("gpu pool store: %w", err)
		}
		// Paged attention: reads K/V from block pool via block table
		if err := gpuPool.Attention(layerIdx, seqID, qPtr, attnOutPtr, numHeads, headDim, scale); err != nil {
			return x, fmt.Errorf("gpu pool attention: %w", err)
		}
	} else if gpuPool != nil && seqLen > 1 {
		// Prefill with GPU pool: store K/V for future decode, use contiguous for attention
		if err := gpuPool.StoreKV(layerIdx, seqID, startPos, kPtr, vPtr, seqLen); err != nil {
			return x, fmt.Errorf("gpu pool store prefill: %w", err)
		}
		// Self-attention over current tokens (contiguous, no cache needed)
		if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
			b.softCapOps.SDPAPrefillSoftCap(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap)
		} else {
			b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
		}
	} else if pagedCache != nil {
		// CPU paged path: GPU→CPU→GPU roundtrip (fallback)
		var fullKPtr, fullVPtr tensor.DevicePtr
		var fullSeqLen int

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
			b.backend.Sync()
			b.backend.ToHost(kBytes, kPtr)
			b.backend.ToHost(vBytes, vPtr)
			b.backend.Sync()
			kData = bytesToFloat32(kBytes)
			vData = bytesToFloat32(vBytes)
		}

		err := pagedCache.StoreKVBatch(seqID, layerIdx, startPos, kData, vData, seqLen)
		if err != nil {
			return x, fmt.Errorf("failed to store KV: %w", err)
		}

		currentPos := startPos + seqLen - 1
		attnStartPos := 0
		if b.useSlidingWindow(layerIdx) && currentPos >= b.SlidingWindow {
			attnStartPos = currentPos - b.SlidingWindow + 1
		}

		fullK, fullV := pagedCache.GetKVSlice(seqID, layerIdx, attnStartPos, currentPos)
		fullSeqLen = len(fullK) / (numKVHeads * headDim)

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

		if seqLen == 1 {
			if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
				b.softCapOps.SDPASoftCap(qPtr, fullKPtr, fullVPtr, attnOutPtr, fullSeqLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap, numKVHeads*headDim)
			} else {
				b.backend.SDPA(qPtr, fullKPtr, fullVPtr, attnOutPtr, fullSeqLen, numHeads, numKVHeads, headDim, scale, numKVHeads*headDim)
			}
		} else {
			if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
				b.softCapOps.SDPAPrefillSoftCap(qPtr, fullKPtr, fullVPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap)
			} else {
				b.backend.SDPAPrefill(qPtr, fullKPtr, fullVPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
			}
		}
	} else {
		// No cache - use current K/V only (for prefill self-attention)
		if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
			b.softCapOps.SDPAPrefillSoftCap(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap)
		} else {
			b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
		}
	}
	// No sync - SDPA must complete before Wo can read attnOut (serialized in queue)

	// Save normOut for parallel residual (Phi needs same normOut for both attn and MLP)
	// For parallel residual, we write Wo output to upPtr (unused in GELU path) to preserve normOut
	var attnResidualPtr tensor.DevicePtr
	if b.ParallelResidual {
		attnResidualPtr = upPtr // Use upPtr for Wo output (not needed for GELU MLP)
	} else {
		attnResidualPtr = normOutPtr
	}

	// 6. Output Projection with bias support
	if !b.Wo.DevicePtr().IsNil() {
		oDim := b.Wo.Shape().Dims()[0]
		b.matMulTransposedWithBias(attnOutPtr, b.Wo, b.WoBias, attnResidualPtr, seqLen, oDim, numHeads*headDim)
		// Post-attention norm (Gemma 2): norm after attn projection, before residual
		if b.HasPostNorms && !b.PostAttnNorm.DevicePtr().IsNil() {
			b.backend.RMSNorm(attnResidualPtr, b.PostAttnNorm.DevicePtr(), attnResidualPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
		}
		// For serial residual, add attention output to x immediately
		if !b.ParallelResidual {
			b.backend.Add(xPtr, attnResidualPtr, xPtr, seqLen*hiddenSize)
		}
	}
	// No sync - operations serialized in command queue

	// 7. FFN normalization (RMSNorm or LayerNorm depending on architecture)
	// For parallel residual architectures (like Phi), FFN uses the same normOut as attention (no separate FFN norm)
	if !b.ParallelResidual && !b.FFNNorm.DevicePtr().IsNil() {
		// Serial residual: apply separate FFN norm
		b.applyNorm(xPtr, b.FFNNorm.DevicePtr(), b.FFNNormBias.DevicePtr(), normOutPtr, seqLen, hiddenSize)
	}
	// For parallel residual, normOutPtr already contains the shared attention normalization
	// No sync - normalization must complete before MLP can read normOut (serialized in queue)

	// 8. MLP (SwiGLU for LLaMA-style, GELU for Phi-style)
	// Debug all layers for decode (seqLen==1) to find NaN source
	debugThisLayer := debugDecode && (seqLen == 1 || layerIdx <= 1 || layerIdx >= 30)
	if b.MLPType == MLPGELU {
		// Phi-style GELU MLP: hidden = GELU(normOut @ W1 + bias1), out = hidden @ W2 + bias2
		if debugThisLayer {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d normOut before W1", layerIdx), normOutPtr, seqLen*hiddenSize)
		}
		if !b.W1.DevicePtr().IsNil() {
			w1Dim := b.W1.Shape().Dims()[0]
			b.matMulTransposedWithBias(normOutPtr, b.W1, b.W1Bias, gatePtr, seqLen, w1Dim, hiddenSize)
		}
		if debugThisLayer {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d gatePtr after W1", layerIdx), gatePtr, seqLen*intermediateSize)
		}
		// Apply GELU activation
		if geluOps, ok := b.backend.(backend.GELUOps); ok {
			geluOps.GELU(gatePtr, gatePtr, seqLen*intermediateSize)
		}
		if debugThisLayer {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d gatePtr after GELU", layerIdx), gatePtr, seqLen*intermediateSize)
		}
		if !b.W2.DevicePtr().IsNil() {
			w2Dim := b.W2.Shape().Dims()[0]
			if debugThisLayer {
				fmt.Printf("[DEBUG] L%d W2: Shape=%v, IsQuant=%v, Profile=%v\n", layerIdx, b.W2.Shape(), b.W2.IsQuantized(), b.W2.QuantProfile())
				fmt.Printf("[DEBUG] L%d W2 matmul: m=%d, n=%d (w2Dim), k=%d (intermediateSize)\n", layerIdx, seqLen, w2Dim, intermediateSize)
				// Check W2 weight values
				b.debugBlockTensor(fmt.Sprintf("L%d W2 weights first row", layerIdx), b.W2.DevicePtr(), 2560)
			}
			b.matMulTransposedWithBias(gatePtr, b.W2, b.W2Bias, normOutPtr, seqLen, w2Dim, intermediateSize)
		}
		if debugThisLayer {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d normOut after W2", layerIdx), normOutPtr, seqLen*hiddenSize)
		}
	} else {
		// Gated MLP: gate = activation(normOut @ W1), up = normOut @ W3, out = (gate * up) @ W2
		// SwiGLU (LLaMA): activation = SiLU; GeGLU (Gemma): activation = GELU
		if !b.W1.DevicePtr().IsNil() {
			w1Dim := b.W1.Shape().Dims()[0]
			b.matMulTransposed(normOutPtr, b.W1, gatePtr, seqLen, w1Dim, hiddenSize)
		}
		if !b.W3.DevicePtr().IsNil() {
			w3Dim := b.W3.Shape().Dims()[0]
			b.matMulTransposed(normOutPtr, b.W3, upPtr, seqLen, w3Dim, hiddenSize)
		}
		if b.MLPType == MLPGeGLU {
			if b.geluOps != nil {
				b.geluOps.GELUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)
			}
		} else {
			b.backend.SiLUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)
		}
		if !b.W2.DevicePtr().IsNil() {
			w2Dim := b.W2.Shape().Dims()[0]
			b.matMulTransposed(gatePtr, b.W2, normOutPtr, seqLen, w2Dim, intermediateSize)
		}
	}
	// No sync - operations serialized in command queue

	// 9. Post-FFN norm (Gemma 2): norm after MLP, before residual
	if b.HasPostNorms && !b.PostFFNNorm.DevicePtr().IsNil() {
		b.backend.RMSNorm(normOutPtr, b.PostFFNNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
	}

	// 10. Add residuals
	if b.ParallelResidual {
		// Parallel residual: x = x + attn(norm(x)) + mlp(norm(x))
		b.backend.Add(xPtr, attnResidualPtr, xPtr, seqLen*hiddenSize) // Add attention
		// Sync required: second Add reads x which was just written by first Add
		b.backend.Sync()
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)      // Add MLP
	} else {
		// Serial residual: x is already updated with attention, just add MLP
		b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
	}
	// No sync - next layer's normalization is serialized in queue

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
	// Batching encodes all operations per layer into one command buffer, reducing ~10
	// waitUntilCompleted calls per layer to just 1 (via EndBatch). Memory barriers
	// between dependent dispatches ensure correct scratch buffer visibility.
	useBatching := b.batcher != nil
	// Profiler disables batching to measure individual kernel times
	if os.Getenv("VEXEL_GPU_PROFILE") == "1" {
		useBatching = false
	}

	if useBatching {
		b.batcher.BeginBatch()
		defer b.batcher.EndBatch()
	}

	// barrier inserts a buffer-scope memory barrier when batching is active.
	// Required between dispatches that share scratch buffer data (write→read dependency).
	// No-op when batching is off (separate command buffers are automatically serialized).
	barrier := func() {
		if useBatching {
			b.batcher.MemoryBarrier()
		}
	}

	// Debug Layer 15 specifically at pos >= 4 where NaN appears
	debugL15 := debugDecode && layerIdx == 15 && startPos >= 4
	if debugL15 {
		fmt.Printf("\n[DEBUG] ===== L15 at pos=%d ENTER =====\n", startPos)
	}

	// Debug harness capture - structured output for targeted debugging
	capture := func(op, name string, ptr tensor.DevicePtr, size int) {
		b.debugCapture(layerIdx, startPos, op, name, ptr, size)
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

	// Debug harness: capture input
	capture("input", "x", xPtr, seqLen*hiddenSize)

	// Calculate sizes for intermediates
	normOutBytes := seqLen * hiddenSize * 4
	qBytes := qSize * 4
	kvBytes := kvSize * 4
	attnOutBytes := qSize * 4
	gateBytes := seqLen * intermediateSize * 4
	upBytes := seqLen * intermediateSize * 4
	fusedMLPTempBytes := seqLen * intermediateSize * 2 * 4 // For fused W1W3 output before deinterleave

	// Allocate intermediate buffers.
	// For CPU: sub-allocate from scratch buffer (pointer arithmetic handles offsets correctly).
	// For GPU: use ScratchAlloc (bump allocation from pre-allocated MTLBuffer) when available,
	// falling back to pool Alloc. Scratch-allocated DevicePtrs have non-zero offsets which
	// are transparently handled by auto-detecting offset-aware kernel dispatch.
	var normOutPtr, qPtr, kPtr, vPtr, attnOutPtr, gatePtr, upPtr, fusedMLPTempPtr tensor.DevicePtr

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
		offset += uintptr(upBytes)
		fusedMLPTempPtr = tensor.DevicePtrOffset(scratchPtr, offset)
		_ = fusedMLPTempBytes
	} else if b.scratchAlloc != nil {
		// Reset scratch allocator for this layer (reuses the same pre-allocated buffer).
		b.scratchAlloc.ScratchReset()
		normOutPtr = b.scratchAlloc.ScratchAlloc(normOutBytes)
		qPtr = b.scratchAlloc.ScratchAlloc(qBytes)
		kPtr = b.scratchAlloc.ScratchAlloc(kvBytes)
		vPtr = b.scratchAlloc.ScratchAlloc(kvBytes)
		attnOutPtr = b.scratchAlloc.ScratchAlloc(attnOutBytes)
		gatePtr = b.scratchAlloc.ScratchAlloc(gateBytes)
		upPtr = b.scratchAlloc.ScratchAlloc(upBytes)
		fusedMLPTempPtr = b.scratchAlloc.ScratchAlloc(fusedMLPTempBytes)
	} else {
		// Fallback: individual pool allocations (no scratch allocator available).
		normOutPtr = b.backend.Alloc(normOutBytes)
		qPtr = b.backend.Alloc(qBytes)
		kPtr = b.backend.Alloc(kvBytes)
		vPtr = b.backend.Alloc(kvBytes)
		attnOutPtr = b.backend.Alloc(attnOutBytes)
		gatePtr = b.backend.Alloc(gateBytes)
		upPtr = b.backend.Alloc(upBytes)
		fusedMLPTempPtr = b.backend.Alloc(fusedMLPTempBytes)
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
	// Note: Fused kernels assume RMSNorm, so disable for LayerNorm models (Phi, GPT-2)
	canFuseAttn := seqLen == 1 && b.fusedOps != nil &&
		b.NormType == NormRMSNorm && // Only use fused kernels for RMSNorm
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
		// Debug all layers for decode (seqLen==1) to find NaN source
		debugThisLayer := debugDecode && (seqLen == 1 || layerIdx <= 1 || layerIdx >= 30)
		if debugL15 {
			b.backend.Sync()
			b.debugBlockTensor("L15 Input x", xPtr, seqLen*hiddenSize)
		}
		if debugThisLayer {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d Input x (before RMSNorm)", layerIdx), xPtr, seqLen*hiddenSize)
		}
		profileOp("AttnNorm", func() {
			if !b.AttnNorm.DevicePtr().IsNil() {
				b.applyNorm(xPtr, b.AttnNorm.DevicePtr(), b.AttnNormBias.DevicePtr(), normOutPtr, seqLen, hiddenSize)
			}
		})
		// Debug L15 norm output
		if debugL15 {
			b.backend.Sync()
			b.debugBlockTensor("L15 After Norm", normOutPtr, seqLen*hiddenSize)
		}
		// Debug harness: capture norm output
		capture("norm", "normOut", normOutPtr, seqLen*hiddenSize)
		// Debug: check RMSNorm output for prefill
		if debugThisLayer {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d Standard Norm out (prefill)", layerIdx), normOutPtr, seqLen*hiddenSize)
			b.debugBlockTensor(fmt.Sprintf("L%d AttnNorm weights", layerIdx), b.AttnNorm.DevicePtr(), hiddenSize)
		}

		profileOp("Wqkv", func() {
			if !b.Wqkv.DevicePtr().IsNil() && seqLen > 1 && b.qkvDeinterleaver != nil {
				// Fused QKV prefill path: single matmul → temp → deinterleave
				// Uses gate+up scratch space as temp buffer (contiguous via ScratchAlloc).
				// gateBytes+upBytes = seqLen*intermediateSize*4*2 ≥ seqLen*qkvDim*4 for all models.
				qkvDim := b.Wqkv.Shape().Dims()[0]
				b.matMulTransposedWithBias(normOutPtr, b.Wqkv, b.WqkvBias, gatePtr, seqLen, qkvDim, hiddenSize)
				// Barrier: deinterleave reads gatePtr written by fused matmul
				barrier()
				qDim := numHeads * headDim
				kvDim := numKVHeads * headDim
				b.qkvDeinterleaver.DeinterleaveQKV(gatePtr, qPtr, kPtr, vPtr, seqLen, qDim, kvDim)
			} else if !b.Wqkv.DevicePtr().IsNil() {
				// Fused QKV decode path (seqLen==1): output lands directly in Q|K|V scratch
				qkvDim := b.Wqkv.Shape().Dims()[0]
				b.matMulTransposedWithBias(normOutPtr, b.Wqkv, b.WqkvBias, qPtr, seqLen, qkvDim, hiddenSize)
			} else {
				if !b.Wq.DevicePtr().IsNil() {
					qDim := b.Wq.Shape().Dims()[0]
					b.matMulTransposedWithBias(normOutPtr, b.Wq, b.WqBias, qPtr, seqLen, qDim, hiddenSize)
				}
				if !b.Wk.DevicePtr().IsNil() {
					kDim := b.Wk.Shape().Dims()[0]
					b.matMulTransposedWithBias(normOutPtr, b.Wk, b.WkBias, kPtr, seqLen, kDim, hiddenSize)
				}
				if !b.Wv.DevicePtr().IsNil() {
					vDim := b.Wv.Shape().Dims()[0]
					b.matMulTransposedWithBias(normOutPtr, b.Wv, b.WvBias, vPtr, seqLen, vDim, hiddenSize)
				}
			}
		})
	}

	// Debug L15 Q/K/V after projection
	if debugL15 {
		b.backend.Sync()
		b.debugBlockTensor("L15 Q after proj", qPtr, qSize)
		b.debugBlockTensor("L15 K after proj", kPtr, kvSize)
		b.debugBlockTensor("L15 V after proj", vPtr, kvSize)
	}
	// Debug harness: capture Q/K/V after projection
	capture("qkv", "Q", qPtr, qSize)
	capture("qkv", "K", kPtr, kvSize)
	capture("qkv", "V", vPtr, kvSize)

	// Debug: check K and V right after projection, before RoPE and cache
	if debugDecode && (layerIdx <= 1 || layerIdx >= 30) {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d K after projection (before RoPE)", layerIdx), kPtr, kvSize)
		b.debugBlockTensor(fmt.Sprintf("L%d V after projection (before RoPE)", layerIdx), vPtr, kvSize)
		// Debug: for prefill with 2+ tokens, print token 1's V values
		if seqLen > 1 && layerIdx == 0 {
			b.debugBlockTensorAtOffset(fmt.Sprintf("L%d V token 1 (before cache)", layerIdx), vPtr, numKVHeads*headDim, 4)
			b.debugBlockTensorAtOffset(fmt.Sprintf("L%d K token 1 (before cache)", layerIdx), kPtr, numKVHeads*headDim, 4)
			// Also check normOut to see if the input to V projection is different for token 1
			b.debugBlockTensorAtOffset(fmt.Sprintf("L%d normOut token 1 (V input)", layerIdx), normOutPtr, hiddenSize, 4)
		}
	}

	// 3. RoPE - Apply to Q and K
	// Barrier: RoPE reads qPtr, kPtr written by QKV projections above (scratch buffer)
	barrier()
	profileOp("RoPE", func() {
		if useFP16Path {
			// FP16 path: apply RoPE directly to FP16 Q and K
			b.fp16Ops.RoPEF16(qF16Ptr, kF16Ptr, headDim, numHeads, numKVHeads, seqLen, startPos, b.RoPEDim, float32(b.RoPETheta), b.RoPENeox)
		} else {
			b.applyRoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, startPos)
		}
	})

	// Debug L15 after RoPE
	if debugL15 {
		b.backend.Sync()
		b.debugBlockTensor("L15 Q after RoPE", qPtr, qSize)
		b.debugBlockTensor("L15 K after RoPE", kPtr, kvSize)
	}
	// Debug harness: capture after RoPE
	capture("rope", "Q", qPtr, qSize)
	capture("rope", "K", kPtr, kvSize)

	// Debug: check K after RoPE, before cache
	if debugDecode && (layerIdx <= 1 || layerIdx >= 30) {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d K after RoPE (before cache)", layerIdx), kPtr, kvSize)
	}

	// 4. Append K/V to GPU cache and get pointers for SDPA
	// Barrier: KV cache reads kPtr, vPtr written by RoPE (scratch buffer)
	barrier()

	var fullKPtr, fullVPtr tensor.DevicePtr
	var fullSeqLen int
	profileOp("KVCache", func() {
		if useFP16Path {
			// FP16 path: K and V already in FP16 (from fused kernel), no conversion needed
			// AppendKV uses CopyBufferBatched which integrates with command batching
			fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kF16Ptr, vF16Ptr, tensor.Float16, seqLen)
		} else if useFP16KVCache {
			// Explicitly convert F32->F16 here so we have dense F16 buffers for SDPAPrefillF16
			// This is required because SDPAPrefillF16 needs [seqLen, numKVHeads, headDim] in F16
			b.fp16Ops.ConvertF32ToF16(kPtr, kF16Ptr, kvSize)
			b.fp16Ops.ConvertF32ToF16(vPtr, vF16Ptr, kvSize)
			
			// Use F16 scatter (since we now have F16 source)
			fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kF16Ptr, vF16Ptr, tensor.Float16, seqLen)
		} else if useQ8KVCache {
			// Quantize FP32 K/V to Q8_0 before storing in cache
			b.q8Ops.QuantizeF32ToQ8_0(kPtr, kQ8Ptr, kvSize)
			b.q8Ops.QuantizeF32ToQ8_0(vPtr, vQ8Ptr, kvSize)
			// AppendKV uses CopyBufferBatched - no explicit sync needed
			// Q8 path handles its own types implicitly via buffer size
			fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kQ8Ptr, vQ8Ptr, tensor.Uint8, seqLen)
		} else {
			fullKPtr, fullVPtr, fullSeqLen = gpuCache.AppendKV(layerIdx, kPtr, vPtr, tensor.Float32, seqLen)
		}
	})

	// Debug: check KV cache fullSeqLen
	if debugDecode && (layerIdx <= 1 || layerIdx >= 30) {
		fmt.Printf("[DEBUG] L%d KV cache: seqLen=%d, fullSeqLen=%d, kvSize=%d\n", layerIdx, seqLen, fullSeqLen, kvSize)
	}

	// Debug L15 KV cache
	if debugL15 {
		fmt.Printf("[DEBUG] L15 KV cache: fullSeqLen=%d, kvStride=%d\n", fullSeqLen, gpuCache.KVHeadStride())
		b.backend.Sync()
		b.debugBlockTensor("L15 fullK before SDPA", fullKPtr, fullSeqLen*numKVHeads*headDim)
		b.debugBlockTensor("L15 fullV before SDPA", fullVPtr, fullSeqLen*numKVHeads*headDim)
	}
	// Debug harness: capture KV cache
	// NOTE: KV cache is FP16 when useFP16KVCache=true, so values will appear corrupt
	// if read as FP32. Use debugBlockTensorF16 for proper reading.
	if !useFP16KVCache && !useFP16Path {
		capture("kv", "fullK", fullKPtr, fullSeqLen*numKVHeads*headDim)
		capture("kv", "fullV", fullVPtr, fullSeqLen*numKVHeads*headDim)
	}

	// 5. Attention: Q @ K^T -> softmax -> @ V
	scale := float32(1.0 / sqrt(float64(headDim)))

	// Debug: dump Q, K, V before SDPA
	debugThisLayerSDPA := debugDecode && (layerIdx <= 2 || layerIdx >= 30)
	if debugThisLayerSDPA {
		b.debugBlockTensor(fmt.Sprintf("L%d Q before SDPA", layerIdx), qPtr, qSize)
		if seqLen == 1 {
			b.debugBlockTensor(fmt.Sprintf("L%d fullK before SDPA", layerIdx), fullKPtr, fullSeqLen*numKVHeads*headDim)
			b.debugBlockTensor(fmt.Sprintf("L%d fullV before SDPA", layerIdx), fullVPtr, fullSeqLen*numKVHeads*headDim)
			// Debug: verify KV cache head-major layout has different values at different positions
			if debugDecode && layerIdx == 0 {
				b.debugKVCacheHeadMajor("L0 fullK", fullKPtr, gpuCache.KVHeadStride(), fullSeqLen, headDim)
				b.debugKVCacheHeadMajor("L0 fullV", fullVPtr, gpuCache.KVHeadStride(), fullSeqLen, headDim)
			}
		} else {
			b.debugBlockTensor(fmt.Sprintf("L%d K before SDPA", layerIdx), kPtr, kvSize)
			b.debugBlockTensor(fmt.Sprintf("L%d V before SDPA", layerIdx), vPtr, kvSize)
		}
	}

	profileOp("SDPA", func() {
		if seqLen == 1 {
			// Decode: single query against full KV sequence
			if debugDecode && layerIdx == 0 {
				fmt.Printf("[DEBUG] SDPA path: useFP16Path=%v useFP16KVCache=%v useQ8KVCache=%v -> ", useFP16Path, useFP16KVCache, useQ8KVCache)
			}
			// Debug Q and KV cache for layer 20 where NaN appears
			if debugDecode && layerIdx == 20 {
				b.backend.Sync()
				b.debugBlockTensor(fmt.Sprintf("L%d Q before SDPA", layerIdx), qPtr, qSize)
				fmt.Printf("[DEBUG] L20 SDPA params: fullSeqLen=%d, numHeads=%d, numKVHeads=%d, headDim=%d, scale=%f, kvStride=%d\n",
					fullSeqLen, numHeads, numKVHeads, headDim, scale, gpuCache.KVHeadStride())
			}
			if useFP16Path {
				if debugDecode && layerIdx == 0 {
					fmt.Printf("FP16Path\n")
				}
				// FP16 path: Q already in FP16 (from fused kernel), no Q conversion needed
				b.fp16Ops.SDPAF16(qF16Ptr, fullKPtr, fullVPtr, attnOutF16Ptr, fullSeqLen, numHeads, numKVHeads, headDim, scale, gpuCache.KVHeadStride())
				b.fp16Ops.ConvertF16ToF32(attnOutF16Ptr, attnOutPtr, qSize)
			} else if useFP16KVCache {
				if debugDecode && layerIdx == 0 {
					fmt.Printf("FP16KVCache\n")
				}
				// FP16 KV cache but Q is FP32: convert Q to FP16
				b.fp16Ops.ConvertF32ToF16(qPtr, qF16Ptr, qSize)
				b.fp16Ops.SDPAF16(qF16Ptr, fullKPtr, fullVPtr, attnOutF16Ptr, fullSeqLen, numHeads, numKVHeads, headDim, scale, gpuCache.KVHeadStride())
				b.fp16Ops.ConvertF16ToF32(attnOutF16Ptr, attnOutPtr, qSize)
			} else if useQ8KVCache {
				if debugDecode && layerIdx == 0 {
					fmt.Printf("Q8KVCache\n")
				}
				// Q8_0 path: use Q8_0 SDPA with FP32 Q and Q8_0 K/V
				b.q8Ops.SDPAQ8_0(qPtr, fullKPtr, fullVPtr, attnOutPtr, fullSeqLen, numHeads, numKVHeads, headDim, scale)
			} else {
				if debugDecode && layerIdx == 0 {
					fmt.Printf("FP32 (stride=%d)\n", gpuCache.KVHeadStride())
				}
				// Compute effective KV length (may be reduced by sliding window)
				effKVLen, kvStartPos := b.effectiveKVLen(layerIdx, fullSeqLen)
				sdpaKPtr, sdpaVPtr := fullKPtr, fullVPtr
				if kvStartPos > 0 {
					// Offset K/V pointers within head-major layout.
					// Layout: [numKVHeads, maxSeqLen, headDim]
					// Offsetting by startPos*headDim skips old positions in each head.
					offsetBytes := uintptr(kvStartPos * headDim * 4) // 4 bytes per float32
					sdpaKPtr = tensor.DevicePtrOffset(fullKPtr, offsetBytes)
					sdpaVPtr = tensor.DevicePtrOffset(fullVPtr, offsetBytes)
				}
				if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
					b.softCapOps.SDPASoftCap(qPtr, sdpaKPtr, sdpaVPtr, attnOutPtr, effKVLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap, gpuCache.KVHeadStride())
				} else {
					b.backend.SDPA(qPtr, sdpaKPtr, sdpaVPtr, attnOutPtr, effKVLen, numHeads, numKVHeads, headDim, scale, gpuCache.KVHeadStride())
				}
			}
		} else {
			// Prefill: use causal SDPAPrefill
			// Always use FP32 path for prefill — FA2v2 kernel is dramatically faster
			// than the FP16 FA2v1 path (11x at seq128), and kPtr/vPtr still hold
			// the original FP32 K/V in scratch memory even when KV cache is FP16.
			if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
				b.softCapOps.SDPAPrefillSoftCap(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap)
			} else {
				b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
			}
		}
	})

	// Debug L15 after SDPA
	if debugL15 {
		b.backend.Sync()
		b.debugBlockTensor("L15 AttnOut after SDPA", attnOutPtr, qSize)
	}
	// Debug harness: capture SDPA output
	capture("sdpa", "attnOut", attnOutPtr, qSize)

	// Debug: dump attention output after SDPA
	// Debug all layers for decode (seqLen==1) to find NaN source
	debugThisLayerPost := debugDecode && (seqLen == 1 || layerIdx <= 1 || layerIdx >= 30)
	if debugThisLayerPost {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d AttnOut after SDPA", layerIdx), attnOutPtr, qSize)
	}

	// 6. Output Projection
	// For parallel residual (Phi): Wo writes to upPtr (not used in GELU path) to preserve normOutPtr for MLP
	// For serial residual (LLaMA): Wo writes to normOutPtr, then we add to x
	woOutputPtr := normOutPtr
	if b.ParallelResidual {
		// Can't reuse attnOutPtr since Wo reads from it
		// Use upPtr instead - it's not used in the GELU MLP path (only SwiGLU uses it)
		// upPtr capacity is intermediateSize which is larger than hiddenSize
		woOutputPtr = upPtr
	}

	// Barrier: Wo reads attnOutPtr written by SDPA (scratch buffer)
	barrier()
	profileOp("Wo", func() {
		if !b.Wo.DevicePtr().IsNil() {
			oDim := b.Wo.Shape().Dims()[0]
			b.matMulTransposedWithBias(attnOutPtr, b.Wo, b.WoBias, woOutputPtr, seqLen, oDim, numHeads*headDim)
		}
		// Post-attention norm (Gemma 2): norm after attn projection, before residual
		if b.HasPostNorms && !b.PostAttnNorm.DevicePtr().IsNil() {
			b.backend.RMSNorm(woOutputPtr, b.PostAttnNorm.DevicePtr(), woOutputPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
		}
	})
	// Debug L15 after Wo
	if debugL15 {
		b.backend.Sync()
		b.debugBlockTensor("L15 After Wo", woOutputPtr, seqLen*hiddenSize)
	}
	// Debug harness: capture Wo output
	capture("wo", "woOut", woOutputPtr, seqLen*hiddenSize)
	if debugThisLayerPost {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d After Wo (attn output)", layerIdx), woOutputPtr, seqLen*hiddenSize)
		b.debugBlockTensor(fmt.Sprintf("L%d x before Add1", layerIdx), xPtr, seqLen*hiddenSize)
	}

	// 7. Add1 (serial residual only)
	// For parallel residual, we skip Add1 and do a combined add at the end
	// For SwiGLU with RMSNorm + fusedOps: defer Add1 to fuse with RMSNorm2
	// via AddRMSNorm kernel (saves 1 dispatch/layer = 32 dispatches total)
	// Barrier: Add1/AddRMSNorm reads normOutPtr written by Wo (scratch buffer)
	barrier()
	fuseAddRMSNorm := b.plan == nil || b.plan.Fusion.FuseAddRMSNorm
	canFuseAdd1Norm := fuseAddRMSNorm && !b.ParallelResidual && b.MLPType != MLPGELU &&
		b.NormType == NormRMSNorm && b.fusedOps != nil
	if !b.ParallelResidual && !canFuseAdd1Norm {
		profileOp("Add1", func() {
			b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
		})
		if debugThisLayerPost {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d x after Add1", layerIdx), xPtr, seqLen*hiddenSize)
		}
	}

	// 8. MLP: Architecture-dependent (SwiGLU for LLaMA, GELU for Phi)
	if b.MLPType == MLPGELU {
		// GELU MLP (Phi, GPT-2): hidden = GELU(x @ W1 + bias), out = hidden @ W2 + bias
		// For parallel residual: skip FFNNorm since we use the shared normOutPtr from AttnNorm
		if !b.ParallelResidual {
			profileOp("FFNNorm", func() {
				if !b.FFNNorm.DevicePtr().IsNil() {
					b.applyNorm(xPtr, b.FFNNorm.DevicePtr(), b.FFNNormBias.DevicePtr(), normOutPtr, seqLen, hiddenSize)
				}
			})
		}

		// Debug: check normOutPtr before W1 for GELU MLP
		if debugThisLayerPost {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d normOutPtr before W1", layerIdx), normOutPtr, seqLen*hiddenSize)
		}

		// Barrier: W1 reads normOutPtr written by FFNNorm (or shared from AttnNorm)
		barrier()
		profileOp("W1", func() {
			if !b.W1.DevicePtr().IsNil() {
				w1Dim := b.W1.Shape().Dims()[0]
				b.matMulTransposedWithBias(normOutPtr, b.W1, b.W1Bias, gatePtr, seqLen, w1Dim, hiddenSize)
			}
		})

		// Debug: check gatePtr after W1
		if debugThisLayerPost {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d gatePtr after W1", layerIdx), gatePtr, seqLen*intermediateSize)
			// Check per-token W1 output for layer 1
			if layerIdx == 1 && seqLen > 1 {
				b.debugBlockTensor(fmt.Sprintf("L%d W1 output token0", layerIdx), gatePtr, intermediateSize)
				b.debugBlockTensorAtOffset(fmt.Sprintf("L%d W1 output token1[0:8]", layerIdx), gatePtr, intermediateSize, 8)
				// Find the position of max value in token0's W1 output
				toHost, ok := b.backend.(interface {
					ToHost([]byte, tensor.DevicePtr)
				})
				if ok {
					data := make([]byte, intermediateSize*4)
					toHost.ToHost(data, gatePtr)
					maxVal := float32(-1e10)
					maxPos := 0
					for i := 0; i < intermediateSize; i++ {
						v := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
						if v > maxVal {
							maxVal = v
							maxPos = i
						}
					}
					fmt.Printf("[DEBUG] L1 W1 token0 max=%.4f at position %d\n", maxVal, maxPos)
				}
			}
			// Check position 8941 which produces NaN after GELU
			if seqLen > 1 && layerIdx >= 30 {
				b.debugBlockValueAt(fmt.Sprintf("L%d gatePtr[8941] BEFORE GELU", layerIdx), gatePtr, 8941)
			}
		}

		// Barrier: GELU reads gatePtr written by W1
		barrier()
		profileOp("GELU", func() {
			if b.geluOps != nil {
				b.geluOps.GELU(gatePtr, gatePtr, seqLen*intermediateSize)
			}
		})

		// Debug: check gatePtr after GELU
		if debugThisLayerPost {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d gatePtr after GELU", layerIdx), gatePtr, seqLen*intermediateSize)
			// Check position 8941 after GELU to confirm the NaN
			if seqLen > 1 && layerIdx >= 30 {
				b.debugBlockValueAt(fmt.Sprintf("L%d gatePtr[8941] AFTER GELU", layerIdx), gatePtr, 8941)
			}
		}

		// Debug W2 before matmul
		if debugThisLayerPost {
			b.backend.Sync()
			fmt.Printf("[DEBUG] L%d W2: Shape=%v, IsQuant=%v, Profile=%v, Ptr=0x%x\n", layerIdx, b.W2.Shape(), b.W2.IsQuantized(), b.W2.QuantProfile(), b.W2.DevicePtr().Addr())
			w2Dim := b.W2.Shape().Dims()[0]
			fmt.Printf("[DEBUG] L%d W2 matmul: m=%d, n=%d (w2Dim), k=%d (intermediateSize)\n", layerIdx, seqLen, w2Dim, intermediateSize)
			b.debugBlockTensor(fmt.Sprintf("L%d W2 weights row0", layerIdx), b.W2.DevicePtr(), 256)
			if !b.W2Bias.DevicePtr().IsNil() {
				b.debugBlockTensor(fmt.Sprintf("L%d W2Bias", layerIdx), b.W2Bias.DevicePtr(), 256)
			}
			// Check row 743 of W2 (where the explosion happens)
			if layerIdx == 1 {
				// Read entire W2 to verify structure (ToHost ignores offset for GPU!)
				toHost, ok := b.backend.(interface {
					ToHost([]byte, tensor.DevicePtr)
				})
				if ok {
					// Read full W2 buffer - [2560, 10240] = 26214400 elements
					totalW2Elements := 2560 * intermediateSize
					w2AllData := make([]byte, totalW2Elements*4)
					toHost.ToHost(w2AllData, b.W2.DevicePtr())

					// Check values at row 0 vs row 1 vs row 743
					row0Start := 0 * intermediateSize
					row1Start := 1 * intermediateSize
					row743Start := 743 * intermediateSize

					fmt.Printf("[DEBUG] L1 W2 row0[0:8]: ")
					for i := 0; i < 8; i++ {
						fmt.Printf("%.4f ", math.Float32frombits(binary.LittleEndian.Uint32(w2AllData[(row0Start+i)*4:])))
					}
					fmt.Println()
					fmt.Printf("[DEBUG] L1 W2 row1[0:8]: ")
					for i := 0; i < 8; i++ {
						fmt.Printf("%.4f ", math.Float32frombits(binary.LittleEndian.Uint32(w2AllData[(row1Start+i)*4:])))
					}
					fmt.Println()
					fmt.Printf("[DEBUG] L1 W2 row743[0:8]: ")
					for i := 0; i < 8; i++ {
						fmt.Printf("%.4f ", math.Float32frombits(binary.LittleEndian.Uint32(w2AllData[(row743Start+i)*4:])))
					}
					fmt.Println()

					// Also read gatePtr for token 0
					geluData := make([]byte, intermediateSize*4)
					toHost.ToHost(geluData, gatePtr)

					// Read bias (need to read all to get correct offset)
					biasAllData := make([]byte, hiddenSize*4)
					toHost.ToHost(biasAllData, b.W2Bias.DevicePtr())
					bias743 := math.Float32frombits(binary.LittleEndian.Uint32(biasAllData[743*4:]))

					// Now compute CORRECT dot product for output[0, 743]
					var sum743 float32
					var geluSum, w2Sum, geluMin, geluMax, w2Min, w2Max float32
					geluMin, w2Min = 1e10, 1e10
					geluMax, w2Max = -1e10, -1e10
					for k := 0; k < intermediateSize; k++ {
						gelu := math.Float32frombits(binary.LittleEndian.Uint32(geluData[k*4:]))
						w2 := math.Float32frombits(binary.LittleEndian.Uint32(w2AllData[(row743Start+k)*4:]))
						sum743 += gelu * w2
						geluSum += gelu
						w2Sum += w2
						if gelu < geluMin {
							geluMin = gelu
						}
						if gelu > geluMax {
							geluMax = gelu
						}
						if w2 < w2Min {
							w2Min = w2
						}
						if w2 > w2Max {
							w2Max = w2
						}
					}
					sum743 += bias743
					geluMean := geluSum / float32(intermediateSize)
					w2Mean := w2Sum / float32(intermediateSize)
					fmt.Printf("[DEBUG] L1 GELU token0: min=%.4f max=%.4f mean=%.4f\n", geluMin, geluMax, geluMean)
					fmt.Printf("[DEBUG] L1 W2 row743: min=%.4f max=%.4f mean=%.4f\n", w2Min, w2Max, w2Mean)
					fmt.Printf("[DEBUG] L1 Correct dot product for output[0,743]: %.4f (bias=%.4f)\n", sum743, bias743)

					// Compare with row 0 output
					var sum0 float32
					for k := 0; k < intermediateSize; k++ {
						gelu := math.Float32frombits(binary.LittleEndian.Uint32(geluData[k*4:]))
						w2 := math.Float32frombits(binary.LittleEndian.Uint32(w2AllData[(row0Start+k)*4:]))
						sum0 += gelu * w2
					}
					bias0 := math.Float32frombits(binary.LittleEndian.Uint32(biasAllData[0*4:]))
					sum0 += bias0
					fmt.Printf("[DEBUG] L1 Correct dot product for output[0,0]: %.4f (bias=%.4f)\n", sum0, bias0)

					// Check overall W2 statistics
					var w2AllMin, w2AllMax float32 = 1e10, -1e10
					var w2AllSum float32
					totalElements := 2560 * intermediateSize
					for i := 0; i < totalElements; i++ {
						v := math.Float32frombits(binary.LittleEndian.Uint32(w2AllData[i*4:]))
						if v < w2AllMin {
							w2AllMin = v
						}
						if v > w2AllMax {
							w2AllMax = v
						}
						w2AllSum += v
					}
					fmt.Printf("[DEBUG] L1 W2 overall: min=%.4f max=%.4f mean=%.6f\n", w2AllMin, w2AllMax, w2AllSum/float32(totalElements))
				}
			}
		}
		// Barrier: W2 reads gatePtr written by GELU
		barrier()
		profileOp("W2", func() {
			if !b.W2.DevicePtr().IsNil() {
				w2Dim := b.W2.Shape().Dims()[0]
				b.matMulTransposedWithBias(gatePtr, b.W2, b.W2Bias, normOutPtr, seqLen, w2Dim, intermediateSize)
			}
		})
		// Debug: check where the explosion happens
		if debugThisLayerPost && seqLen > 1 {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d W2 output token0", layerIdx), normOutPtr, hiddenSize)
			b.debugBlockTensorAtOffset(fmt.Sprintf("L%d W2 output token1[0:8]", layerIdx), normOutPtr, hiddenSize, 8)
			// Find where the min value is in token0
			if layerIdx == 1 {
				// Read all token0 output and find min position
				data := make([]byte, hiddenSize*4)
				toHost, ok := b.backend.(interface {
					ToHost([]byte, tensor.DevicePtr)
				})
				if ok {
					toHost.ToHost(data, normOutPtr)
					minVal := float32(0)
					minIdx := 0
					maxVal := float32(0)
					maxIdx := 0
					for i := 0; i < hiddenSize; i++ {
						v := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
						if v < minVal || i == 0 {
							minVal = v
							minIdx = i
						}
						if v > maxVal || i == 0 {
							maxVal = v
							maxIdx = i
						}
					}
					fmt.Printf("[DEBUG] L1 W2 token0: min=%.4f at idx=%d, max=%.4f at idx=%d\n", minVal, minIdx, maxVal, maxIdx)
					// Print values around the min position
					if minIdx > 0 && minIdx < hiddenSize-1 {
						b.debugBlockTensorAtOffset(fmt.Sprintf("L1 W2 token0[%d:%d]", minIdx-2, minIdx+6), normOutPtr, minIdx-2, 8)
					}
				}
			}
		}
	} else {
		// Gated MLP: SwiGLU (LLaMA/Mistral) or GeGLU (Gemma)
		// Fused MLP for FFN (only for RMSNorm + Q4_0 SwiGLU models, decode only)
		fuseMLP := b.plan == nil || b.plan.Fusion.FuseMLP
		canFuseFFN := fuseMLP && seqLen == 1 && b.fusedOps != nil &&
			b.NormType == NormRMSNorm &&
			b.MLPType == MLPSwiGLU && // Fused kernel uses SiLU, not GELU
			b.W1.QuantProfile() == tensor.Q4_0 &&
			b.W3.QuantProfile() == tensor.Q4_0

		if canFuseFFN {
			// Fully fused MLP: (Add1+RMSNorm) + FusedMLP
			// With AddRMSNorm: 2 dispatches instead of 4 (Add1 + RMSNorm + MatMul×2 + SiLUMul)
			w1Dim := b.W1.Shape().Dims()[0]
			if canFuseAdd1Norm {
				// Fused Add1+RMSNorm: x += attn_output, normOut = RMSNorm(x)
				profileOp("AddRMSNorm", func() {
					b.fusedOps.AddRMSNorm(xPtr, normOutPtr, b.FFNNorm.DevicePtr(), normOutPtr, 1, hiddenSize, float32(b.RMSNormEPS))
				})
			} else {
				profileOp("RMSNorm2", func() {
					b.backend.RMSNorm(xPtr, b.FFNNorm.DevicePtr(), normOutPtr, 1, hiddenSize, float32(b.RMSNormEPS))
				})
			}
			// Barrier: FusedMLP reads normOutPtr written by AddRMSNorm/RMSNorm2
			barrier()
			profileOp("FusedMLP", func() {
				b.fusedOps.MatMulQ4_0_FusedMLP(normOutPtr, b.W1.DevicePtr(), b.W3.DevicePtr(), gatePtr, 1, w1Dim, hiddenSize)
			})
		} else {
			// Standard path: separate Add1 already happened above (or fused here)
			if canFuseAdd1Norm {
				// Fused Add1+RMSNorm: x += attn_output, normOut = RMSNorm(x)
				profileOp("AddRMSNorm", func() {
					b.fusedOps.AddRMSNorm(xPtr, normOutPtr, b.FFNNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
				})
			} else {
				profileOp("RMSNorm2", func() {
					if !b.FFNNorm.DevicePtr().IsNil() {
						b.applyNorm(xPtr, b.FFNNorm.DevicePtr(), b.FFNNormBias.DevicePtr(), normOutPtr, seqLen, hiddenSize)
					}
				})
			}
			if debugThisLayerPost {
				b.backend.Sync()
				b.debugBlockTensor(fmt.Sprintf("L%d FFN normOut (after RMSNorm2)", layerIdx), normOutPtr, seqLen*hiddenSize)
			}

			// Barrier: W1/W3 read normOutPtr written by AddRMSNorm/RMSNorm2
			barrier()
			if !b.W1W3.DevicePtr().IsNil() && seqLen > 1 && b.gateUpDeinterleaver != nil {
				// Fused gate_up prefill path: single matmul → temp → deinterleave
				w1w3Dim := b.W1W3.Shape().Dims()[0]
				profileOp("W1W3", func() {
					b.matMulTransposed(normOutPtr, b.W1W3, fusedMLPTempPtr, seqLen, w1w3Dim, hiddenSize)
				})
				// Barrier: deinterleave reads fusedMLPTempPtr written by fused matmul
				barrier()
				profileOp("DeinterleaveMLP", func() {
					w1Dim := b.W1.Shape().Dims()[0]
					w3Dim := b.W3.Shape().Dims()[0]
					b.gateUpDeinterleaver.Deinterleave2Way(fusedMLPTempPtr, gatePtr, upPtr, seqLen, w1Dim, w3Dim)
				})
			} else {
				profileOp("W1", func() {
					if !b.W1.DevicePtr().IsNil() {
						w1Dim := b.W1.Shape().Dims()[0]
						b.matMulTransposed(normOutPtr, b.W1, gatePtr, seqLen, w1Dim, hiddenSize)
					}
				})
				if debugThisLayerPost {
					b.backend.Sync()
					b.debugBlockTensor(fmt.Sprintf("L%d gate after W1", layerIdx), gatePtr, seqLen*intermediateSize)
				}

				profileOp("W3", func() {
					if !b.W3.DevicePtr().IsNil() {
						w3Dim := b.W3.Shape().Dims()[0]
						b.matMulTransposed(normOutPtr, b.W3, upPtr, seqLen, w3Dim, hiddenSize)
					}
				})
				if debugThisLayerPost {
					b.backend.Sync()
					b.debugBlockTensor(fmt.Sprintf("L%d up after W3", layerIdx), upPtr, seqLen*intermediateSize)
				}
			}

			// Barrier: SiLUMul/GELUMul reads gatePtr, upPtr written by W1/W3 or deinterleave
			barrier()
			if b.MLPType == MLPGeGLU {
				profileOp("GELUMul", func() {
					if b.geluOps != nil {
						b.geluOps.GELUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)
					}
				})
			} else {
				profileOp("SiLUMul", func() {
					b.backend.SiLUMul(gatePtr, upPtr, gatePtr, seqLen*intermediateSize)
				})
			}
		}
		if debugThisLayerPost {
			b.backend.Sync()
			b.debugBlockTensor(fmt.Sprintf("L%d gate after activation*Mul", layerIdx), gatePtr, seqLen*intermediateSize)
		}

		// Barrier: W2 reads gatePtr written by FusedMLP or SiLUMul/GELUMul
		barrier()
		profileOp("W2", func() {
			if !b.W2.DevicePtr().IsNil() {
				w2Dim := b.W2.Shape().Dims()[0]
				b.matMulTransposed(gatePtr, b.W2, normOutPtr, seqLen, w2Dim, intermediateSize)
			}
		})
	}
	if debugThisLayerPost {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d After W2 (MLP output)", layerIdx), normOutPtr, seqLen*hiddenSize)
		b.debugBlockTensor(fmt.Sprintf("L%d x before Add2", layerIdx), xPtr, seqLen*hiddenSize)
	}
	// Debug harness: capture MLP output
	capture("mlp", "mlpOut", normOutPtr, seqLen*hiddenSize)

	// 9. Post-FFN norm (Gemma 2): norm after MLP, before residual
	if b.HasPostNorms && !b.PostFFNNorm.DevicePtr().IsNil() {
		b.backend.RMSNorm(normOutPtr, b.PostFFNNorm.DevicePtr(), normOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
	}

	// 10. Final residual add
	// Barrier: Add2 reads normOutPtr written by W2 (scratch buffer)
	barrier()
	// For parallel residual (Phi): x = x + attn_output + mlp_output (combined add)
	// For serial residual (LLaMA): x = x + mlp_output (attn was already added in Add1)
	if b.ParallelResidual {
		// Combined add: x = x + attnOutput + mlpOutput
		// attnOutput is in woOutputPtr (which is attnOutPtr for parallel residual)
		// mlpOutput is in normOutPtr
		profileOp("Add2_Parallel", func() {
			b.backend.Add(xPtr, woOutputPtr, xPtr, seqLen*hiddenSize) // x += attn_output
			// Barrier required: second Add reads x which was just written by first Add.
			// Without this barrier, race condition causes NaN (observed at L12 pos=6 for Phi-2).
			barrier()
			b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize) // x += mlp_output
		})
	} else {
		profileOp("Add2", func() {
			b.backend.Add(xPtr, normOutPtr, xPtr, seqLen*hiddenSize)
		})
	}
	if debugThisLayerPost {
		b.backend.Sync()
		b.debugBlockTensor(fmt.Sprintf("L%d x after Add2 (FINAL)", layerIdx), xPtr, seqLen*hiddenSize)
	}
	// Debug L15 final output
	if debugL15 {
		b.backend.Sync()
		b.debugBlockTensor("L15 x after Add2 (FINAL)", xPtr, seqLen*hiddenSize)
		fmt.Printf("[DEBUG] ===== L15 at pos=%d EXIT =====\n\n", startPos)
	}
	// Debug harness: capture final output
	capture("output", "x", xPtr, seqLen*hiddenSize)

	// No per-layer sync needed: Metal command queues guarantee FIFO execution.
	// All dispatches go to the same queue, so layer N's writes complete before
	// layer N+1's reads begin. Scratch buffer reuse is safe for the same reason.
	// The final sync point is in DecodeWithGPUKV after all layers + logits computation.

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
