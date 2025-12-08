package runtime

import (
	"math"

	"vexel/inference/backend/cpu"
	"vexel/inference/ir"
	"vexel/inference/kv"
	"vexel/inference/tensor"
)

// sqrt is a helper for float64 square root
func sqrt(x float64) float64 {
	return math.Sqrt(x)
}

// BlockRuntime represents a single transformer layer (Attention + MLP).
type BlockRuntime struct {
	backend cpu.Backend
	graph   *ir.BlockIR

	// Config (needed for GQA)
	NumAttentionHeads int
	NumKeyValueHeads  int
	HeadDim           int
	HiddenSize        int
	IntermediateSize  int
	RoPETheta         float64
	RMSNormEPS        float64

	// Weights
	AttnNorm   tensor.Tensor
	Wq, Wk, Wv tensor.Tensor
	Wo         tensor.Tensor

	FFNNorm    tensor.Tensor
	W1, W2, W3 tensor.Tensor // Gate, Down, Up
}

// NewBlockRuntime creates a new block runtime with config.
func NewBlockRuntime(backend cpu.Backend, config ModelConfig) *BlockRuntime {
	headDim := config.HiddenSize / config.NumAttentionHeads
	return &BlockRuntime{
		backend:           backend,
		NumAttentionHeads: config.NumAttentionHeads,
		NumKeyValueHeads:  config.NumKeyValueHeads,
		HeadDim:           headDim,
		HiddenSize:        config.HiddenSize,
		IntermediateSize:  config.IntermediateSize,
		RoPETheta:         config.RoPETheta,
		RMSNormEPS:        config.RMSNormEPS,
	}
}

// Execute performs the forward pass for this block.
// x: Input tensor [Batch*Seq, Hidden]
// scratch: Temporary buffer
// kvCache: Pointer to KV cache manager
// layerIdx: Index of this layer
// pos: Current token position (for RoPE and Cache)
func (b *BlockRuntime) Execute(x, scratch tensor.Tensor, kvCache *kv.KVCache, layerIdx, pos int) (tensor.Tensor, error) {
	xData := tensor.ToFloat32Slice(x)
	scratchData := tensor.ToFloat32Slice(scratch)

	if xData == nil || scratchData == nil {
		return x, nil
	}

	// Dimensions from config
	seqLen := x.Shape().NumElements() / b.HiddenSize // Batch*Seq (usually 1 for decode)
	hiddenSize := b.HiddenSize
	numHeads := b.NumAttentionHeads
	numKVHeads := b.NumKeyValueHeads
	headDim := b.HeadDim
	intermediateSize := b.IntermediateSize

	// Derived sizes
	qSize := seqLen * numHeads * headDim     // = seqLen * hiddenSize for standard
	kvSize := seqLen * numKVHeads * headDim  // Smaller for GQA
	headsPerKV := numHeads / numKVHeads      // How many Q heads share each KV head

	// Scratch layout:
	// [normOut][Q][K][V][scores][attnOut][gate][up]
	offset := 0

	// 1. RMSNorm (Attention)
	normOut := scratchData[offset : offset+seqLen*hiddenSize]
	offset += seqLen * hiddenSize

	wNorm := tensor.ToFloat32Slice(b.AttnNorm)
	if wNorm != nil {
		b.backend.RMSNorm(xData, wNorm, normOut, seqLen, hiddenSize, float32(b.RMSNormEPS))
	} else {
		copy(normOut, xData) // Pass through if no weights
	}

	// 2. Q/K/V Projections with correct dimensions
	qOut := scratchData[offset : offset+qSize]
	offset += qSize

	kOut := scratchData[offset : offset+kvSize]
	offset += kvSize

	vOut := scratchData[offset : offset+kvSize]
	offset += kvSize

	// Wq: [numHeads*headDim, hiddenSize] -> Q: [seqLen, numHeads*headDim]
	if !b.Wq.DevicePtr().IsNil() {
		wQ := tensor.ToFloat32Slice(b.Wq)
		qDim := b.Wq.Shape().Dims()[0] // numHeads * headDim
		b.backend.MatmulTransposeB(normOut, wQ, qOut, seqLen, qDim, hiddenSize)
	}

	// Wk: [numKVHeads*headDim, hiddenSize] -> K: [seqLen, numKVHeads*headDim]
	if !b.Wk.DevicePtr().IsNil() {
		wK := tensor.ToFloat32Slice(b.Wk)
		kDim := b.Wk.Shape().Dims()[0] // numKVHeads * headDim
		b.backend.MatmulTransposeB(normOut, wK, kOut, seqLen, kDim, hiddenSize)
	}

	// Wv: [numKVHeads*headDim, hiddenSize] -> V: [seqLen, numKVHeads*headDim]
	if !b.Wv.DevicePtr().IsNil() {
		wV := tensor.ToFloat32Slice(b.Wv)
		vDim := b.Wv.Shape().Dims()[0] // numKVHeads * headDim
		b.backend.MatmulTransposeB(normOut, wV, vOut, seqLen, vDim, hiddenSize)
	}

	// 3. RoPE - Apply to Q and K
	// Q is [seqLen, numHeads, headDim] flattened
	// K is [seqLen, numKVHeads, headDim] flattened
	// RoPE operates on each head vector independently, all heads at same seqPos use same RoPE
	b.backend.RoPE(qOut, nil, headDim, numHeads, seqLen, pos, float32(b.RoPETheta))
	b.backend.RoPE(kOut, nil, headDim, numKVHeads, seqLen, pos, float32(b.RoPETheta))

	// 4. Multi-Head Attention with GQA
	// For each query head, find its corresponding KV head and compute attention
	// attnOut: [seqLen, numHeads, headDim]
	attnOut := scratchData[offset : offset+qSize]
	offset += qSize

	// Per-head attention scores buffer (for single sequence: seqLen x seqLen per head)
	scoresSize := seqLen * seqLen
	scores := scratchData[offset : offset+scoresSize]
	offset += scoresSize

	// Scale factor for attention
	scale := float32(1.0 / sqrt(float64(headDim)))

	// Process each query head
	for h := 0; h < numHeads; h++ {
		// Which KV head does this query head use?
		kvHead := h / headsPerKV

		// Get pointers to this head's Q, K, V
		qHead := qOut[h*headDim : (h+1)*headDim]           // [headDim] for seqLen=1
		kHead := kOut[kvHead*headDim : (kvHead+1)*headDim] // [headDim]
		vHead := vOut[kvHead*headDim : (kvHead+1)*headDim] // [headDim]

		// For seqLen=1 (decode), attention is just: score = Q dot K, out = score * V
		// For seqLen>1 (prefill), we need proper matrix attention
		if seqLen == 1 {
			// Dot product Q . K
			var score float32
			for d := 0; d < headDim; d++ {
				score += qHead[d] * kHead[d]
			}
			score *= scale
			// Softmax of single element = 1.0
			// Output = V
			outHead := attnOut[h*headDim : (h+1)*headDim]
			for d := 0; d < headDim; d++ {
				outHead[d] = vHead[d]
			}
		} else {
			// Full attention for prefill: Q [seqLen, headDim], K [seqLen, headDim]
			// This is more complex - compute QK^T, mask, softmax, multiply by V
			// For now, simplified version without full causal masking
			for i := 0; i < seqLen; i++ {
				qRow := qOut[i*numHeads*headDim+h*headDim : i*numHeads*headDim+(h+1)*headDim]
				for j := 0; j < seqLen; j++ {
					kRow := kOut[j*numKVHeads*headDim+kvHead*headDim : j*numKVHeads*headDim+(kvHead+1)*headDim]
					var dot float32
					for d := 0; d < headDim; d++ {
						dot += qRow[d] * kRow[d]
					}
					scores[i*seqLen+j] = dot * scale
				}
			}

			// Apply causal mask (optional, set future to -inf)
			for i := 0; i < seqLen; i++ {
				for j := i + 1; j < seqLen; j++ {
					scores[i*seqLen+j] = -1e9
				}
			}

			// Softmax each row
			b.backend.Softmax(scores, scores, seqLen, seqLen)

			// Multiply by V: [seqLen, seqLen] x [seqLen, headDim] -> [seqLen, headDim]
			for i := 0; i < seqLen; i++ {
				outRow := attnOut[i*numHeads*headDim+h*headDim : i*numHeads*headDim+(h+1)*headDim]
				for d := 0; d < headDim; d++ {
					var sum float32
					for j := 0; j < seqLen; j++ {
						vVal := vOut[j*numKVHeads*headDim+kvHead*headDim+d]
						sum += scores[i*seqLen+j] * vVal
					}
					outRow[d] = sum
				}
			}
		}
	}

	// 5. Output Projection: attnOut [seqLen, numHeads*headDim] -> [seqLen, hiddenSize]
	// Then add residual to x
	if !b.Wo.DevicePtr().IsNil() {
		wO := tensor.ToFloat32Slice(b.Wo)
		oDim := b.Wo.Shape().Dims()[0] // hiddenSize
		// Project and store in normOut temporarily (we can reuse it)
		b.backend.MatmulTransposeB(attnOut, wO, normOut, seqLen, oDim, numHeads*headDim)
		// Add residual: x = x + attn_out
		for i := 0; i < seqLen*hiddenSize; i++ {
			xData[i] += normOut[i]
		}
	}

	// 6. FFN RMSNorm
	wFFN := tensor.ToFloat32Slice(b.FFNNorm)
	if wFFN != nil {
		b.backend.RMSNorm(xData, wFFN, normOut, seqLen, hiddenSize, float32(b.RMSNormEPS))
	} else {
		copy(normOut, xData)
	}

	// 7. MLP: SwiGLU variant
	// gate = SiLU(x @ W1^T)
	// up = x @ W3^T
	// out = (gate * up) @ W2^T

	// Allocate intermediate buffers
	gateOut := scratchData[offset : offset+seqLen*intermediateSize]
	offset += seqLen * intermediateSize

	upOut := scratchData[offset : offset+seqLen*intermediateSize]
	offset += seqLen * intermediateSize

	// Gate projection
	if !b.W1.DevicePtr().IsNil() {
		w1 := tensor.ToFloat32Slice(b.W1)
		w1Dim := b.W1.Shape().Dims()[0] // intermediateSize
		b.backend.MatmulTransposeB(normOut, w1, gateOut, seqLen, w1Dim, hiddenSize)
	}

	// Up projection
	if !b.W3.DevicePtr().IsNil() {
		w3 := tensor.ToFloat32Slice(b.W3)
		w3Dim := b.W3.Shape().Dims()[0] // intermediateSize
		b.backend.MatmulTransposeB(normOut, w3, upOut, seqLen, w3Dim, hiddenSize)
	}

	// SiLU on gate, then multiply with up
	b.backend.SiLU(gateOut, gateOut, seqLen*intermediateSize)
	for i := 0; i < seqLen*intermediateSize; i++ {
		gateOut[i] *= upOut[i]
	}

	// Down projection and residual
	if !b.W2.DevicePtr().IsNil() {
		w2 := tensor.ToFloat32Slice(b.W2)
		w2Dim := b.W2.Shape().Dims()[0] // hiddenSize
		// Project to hidden size, store in normOut
		b.backend.MatmulTransposeB(gateOut, w2, normOut, seqLen, w2Dim, intermediateSize)
		// Add residual
		for i := 0; i < seqLen*hiddenSize; i++ {
			xData[i] += normOut[i]
		}
	}

	return x, nil
}