package runtime

import (
	"fmt"
	"math"

	"vexel/inference/backend"
	"vexel/inference/tensor"
)

// SavedActivations holds per-layer intermediate values from the training forward
// pass. These are needed by the backward pass for gradient computation.
// All buffers are allocated with AllocPermanent so they survive across the
// forward/backward boundary (pool allocations would be recycled).
type SavedActivations struct {
	// NormOut is the RMSNorm output before Q/K/V projection [seqLen, hiddenSize].
	NormOut tensor.DevicePtr
	// Q is the query tensor after RoPE [seqLen, numHeads*headDim].
	Q tensor.DevicePtr
	// K is the key tensor after RoPE [seqLen, numKVHeads*headDim].
	K tensor.DevicePtr
	// V is the value tensor after projection (before attention) [seqLen, numKVHeads*headDim].
	V tensor.DevicePtr
	// AttnOut is the attention output before Wo projection [seqLen, numHeads*headDim].
	AttnOut tensor.DevicePtr
	// Gate holds the FFN gate pre-activation (before SiLU) [seqLen, intermediateSize].
	Gate tensor.DevicePtr
	// Up holds the FFN up projection output [seqLen, intermediateSize].
	Up tensor.DevicePtr
	// FFNNormOut is the RMSNorm output before FFN [seqLen, hiddenSize].
	FFNNormOut tensor.DevicePtr
	// Residual is the hidden state entering this layer (before any modification) [seqLen, hiddenSize].
	Residual tensor.DevicePtr
}

// Free releases all permanently allocated buffers in the saved activations.
func (sa *SavedActivations) Free(b backend.Backend) {
	if sa == nil {
		return
	}
	// Pool-allocated buffers: just nil the pointers. The buffers remain
	// in pool.inUse and will be recycled by ResetPool at the next step.
	// This avoids calling metal_release (which doesn't reliably free
	// memory on macOS) and instead reuses the same buffers each step.
	sa.NormOut = tensor.DevicePtr{}
	sa.Q = tensor.DevicePtr{}
	sa.K = tensor.DevicePtr{}
	sa.V = tensor.DevicePtr{}
	sa.AttnOut = tensor.DevicePtr{}
	sa.Gate = tensor.DevicePtr{}
	sa.Up = tensor.DevicePtr{}
	sa.FFNNormOut = tensor.DevicePtr{}
	sa.Residual = tensor.DevicePtr{}
}

// permAlloc allocates GPU memory for saved activations that must survive
// across the forward/backward boundary. Uses regular Alloc (pool-based).
// The TrainingForward method does NOT call ResetPool — the caller (Trainer)
// must manage pool resets carefully to avoid recycling saved activations
// before backward consumes them.
func permAlloc(b backend.Backend, bytes int) tensor.DevicePtr {
	return b.Alloc(bytes)
}

// TrainingForward runs the forward pass for a single transformer layer, saving
// intermediate activations needed for the backward pass. Unlike the inference
// forward pass (ExecuteWithGPUKV), this method:
//   - Processes the full sequence at once (no KV cache)
//   - Saves all intermediates to permanent allocations
//   - Uses SDPAPrefill for causal attention over the full sequence
//   - Does not use any fused/optimized kernel paths (clarity over speed)
//
// xPtr is the input hidden state [seqLen, hiddenSize] which is modified
// in-place with the residual additions. Returns the saved activations for
// this layer.
func (b *BlockRuntime) TrainingForward(xPtr tensor.DevicePtr, seqLen, layerIdx int) (*SavedActivations, error) {
	hiddenSize := b.HiddenSize
	numHeads := b.NumAttentionHeads
	numKVHeads := b.NumKeyValueHeads
	headDim := b.HeadDim
	intermediateSize := b.IntermediateSize

	qSize := seqLen * numHeads * headDim
	kvSize := seqLen * numKVHeads * headDim

	// Byte sizes for FP32 tensors.
	normOutBytes := seqLen * hiddenSize * 4
	qBytes := qSize * 4
	kvBytes := kvSize * 4
	attnOutBytes := qSize * 4
	gateBytes := seqLen * intermediateSize * 4
	upBytes := seqLen * intermediateSize * 4
	residualBytes := seqLen * hiddenSize * 4
	ffnNormBytes := seqLen * hiddenSize * 4

	sa := &SavedActivations{}

	// --- Save the input residual (clone xPtr) ---
	sa.Residual = permAlloc(b.backend, residualBytes)
	if copier, ok := b.backend.(backend.BufferCopier); ok {
		copier.CopyBuffer(xPtr, 0, sa.Residual, 0, residualBytes)
	} else {
		// Fallback: copy via host (slow but correct).
		tmp := make([]byte, residualBytes)
		b.backend.ToHost(tmp, xPtr)
		b.backend.ToDevice(sa.Residual, tmp)
	}

	// --- 1. Attention pre-norm ---
	sa.NormOut = permAlloc(b.backend, normOutBytes)
	if !b.AttnNorm.DevicePtr().IsNil() {
		b.applyNorm(xPtr, b.AttnNorm.DevicePtr(), b.AttnNormBias.DevicePtr(), sa.NormOut, seqLen, hiddenSize)
	}

	// --- 2. Q/K/V projections ---
	// Allocate temporary Q/K/V (will be overwritten by RoPE, then saved).
	qPtr := b.backend.Alloc(qBytes)
	kPtr := b.backend.Alloc(kvBytes)
	vPtr := b.backend.Alloc(kvBytes)

	if !b.Wqkv.DevicePtr().IsNil() && b.qkvDeinterleaver != nil {
		// Fused QKV path.
		qkvDim := b.Wqkv.Shape().Dims()[0]
		tmpBuf := b.backend.Alloc(seqLen * qkvDim * 4)
		b.matMulTransposedWithBias(sa.NormOut, b.Wqkv, b.WqkvBias, tmpBuf, seqLen, qkvDim, hiddenSize)
		qDim := numHeads * headDim
		kvDim := numKVHeads * headDim
		b.qkvDeinterleaver.DeinterleaveQKV(tmpBuf, qPtr, kPtr, vPtr, seqLen, qDim, kvDim)
	} else {
		if !b.Wq.DevicePtr().IsNil() {
			qDim := b.Wq.Shape().Dims()[0]
			b.matMulTransposedWithBias(sa.NormOut, b.Wq, b.WqBias, qPtr, seqLen, qDim, hiddenSize)
		}
		if b.loraLayer != nil && b.loraLayer.HasQ {
			b.applyLoRA(sa.NormOut, b.loraLayer.QA, b.loraLayer.QB, qPtr,
				seqLen, b.loraRank, hiddenSize, numHeads*headDim, b.loraScale)
		}
		if !b.Wk.DevicePtr().IsNil() {
			kDim := b.Wk.Shape().Dims()[0]
			b.matMulTransposedWithBias(sa.NormOut, b.Wk, b.WkBias, kPtr, seqLen, kDim, hiddenSize)
		}
		if b.loraLayer != nil && b.loraLayer.HasK {
			b.applyLoRA(sa.NormOut, b.loraLayer.KA, b.loraLayer.KB, kPtr,
				seqLen, b.loraRank, hiddenSize, numKVHeads*headDim, b.loraScale)
		}
		if !b.Wv.DevicePtr().IsNil() {
			vDim := b.Wv.Shape().Dims()[0]
			b.matMulTransposedWithBias(sa.NormOut, b.Wv, b.WvBias, vPtr, seqLen, vDim, hiddenSize)
		}
		if b.loraLayer != nil && b.loraLayer.HasV {
			b.applyLoRA(sa.NormOut, b.loraLayer.VA, b.loraLayer.VB, vPtr,
				seqLen, b.loraRank, hiddenSize, numKVHeads*headDim, b.loraScale)
		}
	}

	// Save V before RoPE (V is not rotated, so this is the final V).
	sa.V = permAlloc(b.backend, kvBytes)
	if copier, ok := b.backend.(backend.BufferCopier); ok {
		copier.CopyBuffer(vPtr, 0, sa.V, 0, kvBytes)
	} else {
		tmp := make([]byte, kvBytes)
		b.backend.ToHost(tmp, vPtr)
		b.backend.ToDevice(sa.V, tmp)
	}

	// --- 3. RoPE on Q and K ---
	b.applyRoPE(qPtr, kPtr, headDim, numHeads, numKVHeads, seqLen, 0)

	// Save Q and K after RoPE.
	sa.Q = permAlloc(b.backend, qBytes)
	sa.K = permAlloc(b.backend, kvBytes)
	if copier, ok := b.backend.(backend.BufferCopier); ok {
		copier.CopyBuffer(qPtr, 0, sa.Q, 0, qBytes)
		copier.CopyBuffer(kPtr, 0, sa.K, 0, kvBytes)
	} else {
		tmp := make([]byte, qBytes)
		b.backend.ToHost(tmp, qPtr)
		b.backend.ToDevice(sa.Q, tmp)
		tmp2 := make([]byte, kvBytes)
		b.backend.ToHost(tmp2, kPtr)
		b.backend.ToDevice(sa.K, tmp2)
	}

	// --- 4. SDPA (prefill, causal) ---
	attnOutPtr := b.backend.Alloc(attnOutBytes)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	if b.AttentionLogitSoftCap > 0 && b.softCapOps != nil {
		b.softCapOps.SDPAPrefillSoftCap(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale, b.AttentionLogitSoftCap)
	} else {
		b.backend.SDPAPrefill(qPtr, kPtr, vPtr, attnOutPtr, seqLen, numHeads, numKVHeads, headDim, scale)
	}

	// Save attention output.
	sa.AttnOut = permAlloc(b.backend, attnOutBytes)
	if copier, ok := b.backend.(backend.BufferCopier); ok {
		copier.CopyBuffer(attnOutPtr, 0, sa.AttnOut, 0, attnOutBytes)
	} else {
		tmp := make([]byte, attnOutBytes)
		b.backend.ToHost(tmp, attnOutPtr)
		b.backend.ToDevice(sa.AttnOut, tmp)
	}

	// --- 5. Output projection (Wo) ---
	woOutPtr := b.backend.Alloc(seqLen * hiddenSize * 4)
	if !b.Wo.DevicePtr().IsNil() {
		oDim := b.Wo.Shape().Dims()[0]
		b.matMulTransposedWithBias(attnOutPtr, b.Wo, b.WoBias, woOutPtr, seqLen, oDim, numHeads*headDim)
	}
	// LoRA O contribution: woOutPtr += scale * OB @ (OA @ attnOut)
	if b.loraLayer != nil && b.loraLayer.HasO {
		b.applyLoRA(attnOutPtr, b.loraLayer.OA, b.loraLayer.OB, woOutPtr,
			seqLen, b.loraRank, numHeads*headDim, hiddenSize, b.loraScale)
	}

	// Post-attention norm (Gemma 2).
	if b.HasPostNorms && !b.PostAttnNorm.DevicePtr().IsNil() {
		b.backend.RMSNorm(woOutPtr, b.PostAttnNorm.DevicePtr(), woOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
	}

	// --- 6. First residual add: x = x + woOut ---
	b.backend.Add(xPtr, woOutPtr, xPtr, seqLen*hiddenSize)

	// --- 7. FFN pre-norm ---
	sa.FFNNormOut = permAlloc(b.backend, ffnNormBytes)
	if !b.FFNNorm.DevicePtr().IsNil() {
		b.applyNorm(xPtr, b.FFNNorm.DevicePtr(), b.FFNNormBias.DevicePtr(), sa.FFNNormOut, seqLen, hiddenSize)
	}

	// --- 8. MLP ---
	if b.MLPType == MLPGELU {
		// GELU MLP (Phi, GPT-2): out = GELU(norm @ W1 + bias) @ W2 + bias
		gatePtr2 := b.backend.Alloc(gateBytes)
		if !b.W1.DevicePtr().IsNil() {
			w1Dim := b.W1.Shape().Dims()[0]
			b.matMulTransposedWithBias(sa.FFNNormOut, b.W1, b.W1Bias, gatePtr2, seqLen, w1Dim, hiddenSize)
		}
		// Save gate pre-activation for backward.
		sa.Gate = permAlloc(b.backend, gateBytes)
		if copier, ok := b.backend.(backend.BufferCopier); ok {
			copier.CopyBuffer(gatePtr2, 0, sa.Gate, 0, gateBytes)
		} else {
			tmp := make([]byte, gateBytes)
			b.backend.ToHost(tmp, gatePtr2)
			b.backend.ToDevice(sa.Gate, tmp)
		}
		// Apply GELU.
		if b.geluOps != nil {
			b.geluOps.GELU(gatePtr2, gatePtr2, seqLen*intermediateSize)
		}
		// Down projection.
		mlpOutPtr := b.backend.Alloc(seqLen * hiddenSize * 4)
		if !b.W2.DevicePtr().IsNil() {
			w2Dim := b.W2.Shape().Dims()[0]
			b.matMulTransposedWithBias(gatePtr2, b.W2, b.W2Bias, mlpOutPtr, seqLen, w2Dim, intermediateSize)
		}
		// Post-FFN norm (Gemma 2).
		if b.HasPostNorms && !b.PostFFNNorm.DevicePtr().IsNil() {
			b.backend.RMSNorm(mlpOutPtr, b.PostFFNNorm.DevicePtr(), mlpOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
		}
		// Second residual add.
		b.backend.Add(xPtr, mlpOutPtr, xPtr, seqLen*hiddenSize)
	} else {
		// Gated MLP: SwiGLU or GeGLU
		gatePtr2 := b.backend.Alloc(gateBytes)
		upPtr2 := b.backend.Alloc(upBytes)

		if !b.W1W3.DevicePtr().IsNil() && b.gateUpDeinterleaver != nil {
			// Fused gate_up path.
			w1w3Dim := b.W1W3.Shape().Dims()[0]
			tmpBuf := b.backend.Alloc(seqLen * w1w3Dim * 4)
			b.matMulTransposed(sa.FFNNormOut, b.W1W3, tmpBuf, seqLen, w1w3Dim, hiddenSize)
			w1Dim := b.W1.Shape().Dims()[0]
			w3Dim := b.W3.Shape().Dims()[0]
			b.gateUpDeinterleaver.Deinterleave2Way(tmpBuf, gatePtr2, upPtr2, seqLen, w1Dim, w3Dim)
		} else {
			if !b.W1.DevicePtr().IsNil() {
				w1Dim := b.W1.Shape().Dims()[0]
				b.matMulTransposed(sa.FFNNormOut, b.W1, gatePtr2, seqLen, w1Dim, hiddenSize)
			}
			if !b.W3.DevicePtr().IsNil() {
				w3Dim := b.W3.Shape().Dims()[0]
				b.matMulTransposed(sa.FFNNormOut, b.W3, upPtr2, seqLen, w3Dim, hiddenSize)
			}
		}

		// Save gate and up pre-activation for backward.
		sa.Gate = permAlloc(b.backend, gateBytes)
		sa.Up = permAlloc(b.backend, upBytes)
		if copier, ok := b.backend.(backend.BufferCopier); ok {
			copier.CopyBuffer(gatePtr2, 0, sa.Gate, 0, gateBytes)
			copier.CopyBuffer(upPtr2, 0, sa.Up, 0, upBytes)
		} else {
			tmp := make([]byte, gateBytes)
			b.backend.ToHost(tmp, gatePtr2)
			b.backend.ToDevice(sa.Gate, tmp)
			tmp2 := make([]byte, upBytes)
			b.backend.ToHost(tmp2, upPtr2)
			b.backend.ToDevice(sa.Up, tmp2)
		}

		// Apply activation: SiLUMul or GELUMul.
		if b.MLPType == MLPGeGLU {
			if b.geluOps != nil {
				b.geluOps.GELUMul(gatePtr2, upPtr2, gatePtr2, seqLen*intermediateSize)
			}
		} else {
			b.backend.SiLUMul(gatePtr2, upPtr2, gatePtr2, seqLen*intermediateSize)
		}

		// Down projection (W2).
		mlpOutPtr := b.backend.Alloc(seqLen * hiddenSize * 4)
		if !b.W2.DevicePtr().IsNil() {
			w2Dim := b.W2.Shape().Dims()[0]
			b.matMulTransposed(gatePtr2, b.W2, mlpOutPtr, seqLen, w2Dim, intermediateSize)
		}

		// Post-FFN norm (Gemma 2).
		if b.HasPostNorms && !b.PostFFNNorm.DevicePtr().IsNil() {
			b.backend.RMSNorm(mlpOutPtr, b.PostFFNNorm.DevicePtr(), mlpOutPtr, seqLen, hiddenSize, float32(b.RMSNormEPS))
		}

		// Second residual add.
		b.backend.Add(xPtr, mlpOutPtr, xPtr, seqLen*hiddenSize)
	}

	return sa, nil
}

// TrainingForward runs the full model forward pass for training, returning
// logits for every position in the sequence along with per-layer saved
// activations needed for the backward pass.
//
// tokens: input token IDs [seqLen]
// Returns:
//   - logits DevicePtr [seqLen, vocabSize] in FP32
//   - per-layer SavedActivations (one per transformer layer)
//   - the final hidden state DevicePtr [seqLen, hiddenSize] (needed for loss backward)
//   - error if any
func (m *ModelRuntime) TrainingForward(tokens []int) (tensor.DevicePtr, []*SavedActivations, tensor.DevicePtr, error) {
	if len(tokens) == 0 {
		return tensor.DevicePtr{}, nil, tensor.DevicePtr{}, fmt.Errorf("empty token sequence")
	}

	// NOTE: We do NOT call ResetPool here. Saved activations use pool-allocated
	// buffers that must survive until backward completes. The Trainer calls
	// ResetPool at the start of each training step (before TrainingForward).

	seqLen := len(tokens)
	hiddenSize := m.config.HiddenSize
	vocabSize := m.config.VocabSize

	// 1. Copy token IDs to device.
	tokenBytes := int32ToBytes(tokens)
	tokenPtr := m.backend.Alloc(len(tokenBytes))
	m.backend.ToDevice(tokenPtr, tokenBytes)

	// 2. Embedding lookup → statePtr [seqLen, hiddenSize].
	stateBytes := seqLen * hiddenSize * 4
	statePtr := permAlloc(m.backend, stateBytes)
	if !m.Embedding.DevicePtr().IsNil() {
		m.backend.Embedding(tokenPtr, seqLen, m.Embedding.DevicePtr(), statePtr, vocabSize, hiddenSize)
	}
	m.applyEmbeddingScale(statePtr, seqLen*hiddenSize)

	// 3. Run each transformer layer, collecting saved activations.
	allSaved := make([]*SavedActivations, len(m.layers))
	for i, layer := range m.layers {
		sa, err := layer.TrainingForward(statePtr, seqLen, i)
		if err != nil {
			// Free any already-saved activations on error.
			for j := 0; j < i; j++ {
				allSaved[j].Free(m.backend)
			}
			return tensor.DevicePtr{}, nil, tensor.DevicePtr{}, fmt.Errorf("training forward layer %d: %w", i, err)
		}
		allSaved[i] = sa
	}

	// Sync to ensure all layer work completes before final norm.
	m.backend.Sync()

	// 4. Final norm on the full sequence.
	normOutPtr := m.backend.Alloc(stateBytes)
	m.applyFinalNorm(statePtr, normOutPtr, seqLen, hiddenSize)

	// 5. Compute logits for ALL positions: logits = normOut @ OutputHead^T
	//    [seqLen, hiddenSize] @ [vocabSize, hiddenSize]^T → [seqLen, vocabSize]
	logitsBytes := seqLen * vocabSize * 4
	logitsPtr := m.backend.Alloc(logitsBytes)
	m.outputHeadMatMul(normOutPtr, logitsPtr, seqLen, vocabSize, hiddenSize)

	m.backend.Sync()

	return logitsPtr, allSaved, statePtr, nil
}
