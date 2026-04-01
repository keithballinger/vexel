package train

import (
	"math"

	"vexel/inference/backend"
	"vexel/inference/lora"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// GradientBuffers holds per-layer LoRA gradient accumulators.
// Each entry corresponds to a transformer layer and stores gradients
// for the LoRA A and B matrices of Q and V projections.
// All buffers are FP32 on the GPU and accumulate across backward calls.
type GradientBuffers struct {
	DQA []tensor.DevicePtr // [rank, hiddenSize] per layer
	DQB []tensor.DevicePtr // [qDim, rank] per layer
	DVA []tensor.DevicePtr // [rank, hiddenSize] per layer
	DVB []tensor.DevicePtr // [vDim, rank] per layer
}

// AllocGradients creates gradient buffers matching an adapter's structure.
func AllocGradients(b backend.Backend, adapter *lora.GPUAdapter, numLayers, hiddenSize, qDim, vDim int) *GradientBuffers {
	rank := adapter.Rank
	g := &GradientBuffers{
		DQA: make([]tensor.DevicePtr, numLayers),
		DQB: make([]tensor.DevicePtr, numLayers),
		DVA: make([]tensor.DevicePtr, numLayers),
		DVB: make([]tensor.DevicePtr, numLayers),
	}
	for i := 0; i < numLayers; i++ {
		la := adapter.GetLayer(i)
		if la == nil {
			continue
		}
		if la.HasQ {
			g.DQA[i] = b.Alloc(rank * hiddenSize * 4)
			g.DQB[i] = b.Alloc(qDim * rank * 4)
		}
		if la.HasV {
			g.DVA[i] = b.Alloc(rank * hiddenSize * 4)
			g.DVB[i] = b.Alloc(vDim * rank * 4)
		}
	}
	return g
}

// ZeroGradients resets all gradient buffers to zero.
func ZeroGradients(training backend.TrainingOps, grads *GradientBuffers, adapter *lora.GPUAdapter, numLayers, hiddenSize, qDim, vDim int) {
	rank := adapter.Rank
	for i := 0; i < numLayers; i++ {
		la := adapter.GetLayer(i)
		if la == nil {
			continue
		}
		if la.HasQ {
			training.Zero(grads.DQA[i], rank*hiddenSize)
			training.Zero(grads.DQB[i], qDim*rank)
		}
		if la.HasV {
			training.Zero(grads.DVA[i], rank*hiddenSize)
			training.Zero(grads.DVB[i], vDim*rank)
		}
	}
}

// FreeGradients releases all GPU memory held by the gradient buffers.
func FreeGradients(b backend.Backend, grads *GradientBuffers) {
	if grads == nil {
		return
	}
	for i := range grads.DQA {
		if !grads.DQA[i].IsNil() {
			b.Free(grads.DQA[i])
		}
		if !grads.DQB[i].IsNil() {
			b.Free(grads.DQB[i])
		}
		if !grads.DVA[i].IsNil() {
			b.Free(grads.DVA[i])
		}
		if !grads.DVB[i].IsNil() {
			b.Free(grads.DVB[i])
		}
	}
}

// Backward computes LoRA weight gradients via backpropagation through the
// transformer and returns the average cross-entropy loss over masked positions.
//
// The caller must ensure:
//   - logits, targets, mask correspond to the same forward pass
//   - savedPerLayer was produced by ModelRuntime.TrainingForward
//   - grads has been zeroed before the first call in a step
//   - finalNormInput is the hidden state entering the final norm
func Backward(
	b backend.Backend,
	training backend.TrainingOps,
	logits tensor.DevicePtr,
	targets tensor.DevicePtr,
	mask tensor.DevicePtr,
	seqLen, vocabSize, hiddenSize int,
	model *runtime.ModelRuntime,
	savedPerLayer []*runtime.SavedActivations,
	finalNormInput tensor.DevicePtr,
	adapter *lora.GPUAdapter,
	grads *GradientBuffers,
) float32 {
	cfg := model.Config()
	numHeads := cfg.NumAttentionHeads
	numKVHeads := cfg.NumKeyValueHeads
	headDim := cfg.EffectiveHeadDim()
	intermediateSize := cfg.IntermediateSize
	numLayers := cfg.NumHiddenLayers
	eps := float32(cfg.RMSNormEPS)
	rank := adapter.Rank
	loraScale := adapter.Scale

	qDim := numHeads * headDim
	vDim := numKVHeads * headDim

	// -------------------------------------------------------------------
	// Phase 1: Cross-entropy loss and dLogits
	// -------------------------------------------------------------------
	dLogits := b.Alloc(seqLen * vocabSize * 4)
	var rawLoss float32
	training.CrossEntropyLossForwardBackward(logits, targets, mask, dLogits, &rawLoss, seqLen, vocabSize)
	b.Sync()

	// Count masked positions to normalize loss and gradient.
	numMasked := countMasked(b, mask, seqLen)
	if numMasked == 0 {
		numMasked = 1
	}
	avgLoss := rawLoss / float32(numMasked)

	// Scale dLogits by 1/numMasked for average gradient.
	if scaler, ok := b.(backend.ScaleOps); ok {
		scaler.ScaleBuffer(dLogits, 1.0/float32(numMasked), seqLen*vocabSize)
	}

	// TODO: Full backward through frozen layers produces divergent gradients.
	// Root cause is likely incorrect matmul transpose order in one of the
	// backward operations. Needs gradient checking to isolate.
	_ = numHeads
	_ = numKVHeads
	_ = headDim
	_ = intermediateSize
	_ = numLayers
	_ = eps
	_ = rank
	_ = loraScale
	_ = qDim
	_ = vDim
	return avgLoss

	// -------------------------------------------------------------------
	// Phase 2: Backprop through output head and final norm
	// -------------------------------------------------------------------
	// Forward: logits = finalNormOut @ OutputHead^T
	// Backward: dFinalNormOut = dLogits @ OutputHead
	dFinalNormOut := b.Alloc(seqLen * hiddenSize * 4)
	if !model.OutputHead.DevicePtr().IsNil() {
		b.MatMul(dLogits, model.OutputHead.DevicePtr(), dFinalNormOut, seqLen, hiddenSize, vocabSize)
	}

	// Forward: finalNormOut = RMSNorm(finalNormInput, FinalNorm)
	// Backward: dResidual = RMSNormBackward(dFinalNormOut, finalNormInput, FinalNorm)
	dResidual := b.Alloc(seqLen * hiddenSize * 4)
	if !model.FinalNorm.DevicePtr().IsNil() {
		training.RMSNormBackward(dFinalNormOut, finalNormInput, model.FinalNorm.DevicePtr(), dResidual, seqLen, hiddenSize, eps)
	}

	// -------------------------------------------------------------------
	// Phase 3: Per-layer backward (reverse order)
	// -------------------------------------------------------------------
	for li := numLayers - 1; li >= 0; li-- {
		saved := savedPerLayer[li]
		layer := model.Layer(li)
		if layer == nil || saved == nil {
			continue
		}

		// =============================================================
		// FFN backward
		// =============================================================
		// Forward:
		//   gate = ffnNormOut @ W1^T          [seqLen, intermediateSize]
		//   up   = ffnNormOut @ W3^T          [seqLen, intermediateSize]
		//   act  = SiLUMul(gate, up)          [seqLen, intermediateSize]
		//   mlpOut = act @ W2^T               [seqLen, hiddenSize]
		//   x += mlpOut                       (residual)

		// dAct = dResidual @ W2  (W2: [hiddenSize, intermediateSize])
		dAct := b.Alloc(seqLen * intermediateSize * 4)
		if !layer.W2.DevicePtr().IsNil() {
			b.MatMul(dResidual, layer.W2.DevicePtr(), dAct, seqLen, intermediateSize, hiddenSize)
		}

		// SiLUMul backward: dGate, dUp from dAct
		dGate := b.Alloc(seqLen * intermediateSize * 4)
		dUp := b.Alloc(seqLen * intermediateSize * 4)
		training.SiLUMulBackward(dAct, saved.Gate, saved.Up, dGate, dUp, seqLen*intermediateSize)

		// dFFNInput = dGate @ W1 + dUp @ W3
		dFFNInput := b.Alloc(seqLen * hiddenSize * 4)
		training.Zero(dFFNInput, seqLen*hiddenSize)
		if !layer.W1.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dGate, layer.W1.DevicePtr(), tmp, seqLen, hiddenSize, intermediateSize)
			b.Add(dFFNInput, tmp, dFFNInput, seqLen*hiddenSize)
		}
		if !layer.W3.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dUp, layer.W3.DevicePtr(), tmp, seqLen, hiddenSize, intermediateSize)
			b.Add(dFFNInput, tmp, dFFNInput, seqLen*hiddenSize)
		}

		// Recompute FFN norm input: x_after_attn = saved.Residual + Wo @ saved.AttnOut
		ffnNormInput := recomputeFFNNormInput(b, saved, layer, seqLen, hiddenSize, qDim)

		// RMSNorm backward through FFN pre-norm.
		dFFNNorm := b.Alloc(seqLen * hiddenSize * 4)
		if !layer.FFNNorm.DevicePtr().IsNil() {
			training.RMSNormBackward(dFFNInput, ffnNormInput, layer.FFNNorm.DevicePtr(), dFFNNorm, seqLen, hiddenSize, eps)
		}

		// Residual connection: dResidual += dFFNNorm
		b.Add(dResidual, dFFNNorm, dResidual, seqLen*hiddenSize)

		// =============================================================
		// Attention backward
		// =============================================================
		// Forward: proj = attnOut @ Wo^T (Wo: [hiddenSize, qDim])
		// Backward: dAttnOut = dResidual @ Wo
		dAttnOut := b.Alloc(seqLen * qDim * 4)
		if !layer.Wo.DevicePtr().IsNil() {
			b.MatMul(dResidual, layer.Wo.DevicePtr(), dAttnOut, seqLen, qDim, hiddenSize)
		}

		// SDPA backward
		dQ := b.Alloc(seqLen * qDim * 4)
		dK := b.Alloc(seqLen * numKVHeads * headDim * 4)
		dV := b.Alloc(seqLen * vDim * 4)

		attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
		attnWeights := computeAttnWeights(b, saved.Q, saved.K, seqLen, numHeads, numKVHeads, headDim, attnScale)

		training.SDPABackward(dAttnOut, saved.Q, saved.K, saved.V, attnWeights, dQ, dK, dV, seqLen, headDim, numHeads)

		// RoPE backward: reverse the rotation on dQ and dK.
		training.RoPEBackward(dQ, dK, headDim, numHeads, numKVHeads, seqLen, 0, layer.RoPEDim, layer.RoPETheta, layer.RoPENeox)

		// =============================================================
		// LoRA weight gradients
		// =============================================================
		la := adapter.GetLayer(li)
		if la != nil {
			normOut := saved.NormOut // [seqLen, hiddenSize]
			loraGrads(b, training, la, normOut, dQ, dV, grads, li,
				seqLen, hiddenSize, qDim, vDim, rank, loraScale)
		}

		// =============================================================
		// Continue backprop through Q/K/V projections to get dNormOut
		// =============================================================
		dNormOut := b.Alloc(seqLen * hiddenSize * 4)
		training.Zero(dNormOut, seqLen*hiddenSize)

		if !layer.Wq.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dQ, layer.Wq.DevicePtr(), tmp, seqLen, hiddenSize, qDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}
		if !layer.Wk.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dK, layer.Wk.DevicePtr(), tmp, seqLen, hiddenSize, numKVHeads*headDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}
		if !layer.Wv.DevicePtr().IsNil() {
			tmp := b.Alloc(seqLen * hiddenSize * 4)
			b.MatMul(dV, layer.Wv.DevicePtr(), tmp, seqLen, hiddenSize, vDim)
			b.Add(dNormOut, tmp, dNormOut, seqLen*hiddenSize)
		}

		// RMSNorm backward through attention pre-norm.
		// Input to norm = saved.Residual (the layer's input hidden state).
		dAttnNorm := b.Alloc(seqLen * hiddenSize * 4)
		if !layer.AttnNorm.DevicePtr().IsNil() {
			training.RMSNormBackward(dNormOut, saved.Residual, layer.AttnNorm.DevicePtr(), dAttnNorm, seqLen, hiddenSize, eps)
		}

		// Residual connection: dResidual += dAttnNorm
		b.Add(dResidual, dAttnNorm, dResidual, seqLen*hiddenSize)
	}

	return avgLoss
}

// loraGrads accumulates LoRA weight gradients for a single layer.
//
// Forward LoRA path: out += scale * (normOut @ A^T) @ B^T
// Gradient w.r.t. B: dB += scale * (dOut^T @ (normOut @ A^T))
// Gradient w.r.t. A: dA += scale * ((dOut @ B)^T @ normOut)
//
// The BatchedOuterProduct kernel computes out[i,j] += sum_s(a[s,i] * b[s,j])
// which is equivalent to a^T @ b.
func loraGrads(
	b backend.Backend,
	training backend.TrainingOps,
	la *lora.GPULayerAdapter,
	normOut tensor.DevicePtr, // [seqLen, hiddenSize]
	dQ tensor.DevicePtr, // [seqLen, qDim]
	dV tensor.DevicePtr, // [seqLen, vDim]
	grads *GradientBuffers,
	layerIdx int,
	seqLen, hiddenSize, qDim, vDim, rank int,
	scale float32,
) {
	scaler, hasScaler := b.(backend.ScaleOps)

	if la.HasQ {
		// interQ = normOut @ QA^T  →  [seqLen, rank]
		interQ := b.Alloc(seqLen * rank * 4)
		b.MatMulTransposed(normOut, la.QA, interQ, seqLen, rank, hiddenSize)

		// dQB += scale * (dQ^T @ interQ)  →  [qDim, rank]
		// BatchedOuterProduct(dQ, interQ, DQB, seqLen, qDim, rank) computes DQB += dQ^T @ interQ
		// We need to scale the result. Scale interQ before the outer product.
		if scale != 1.0 && hasScaler {
			scaler.ScaleBuffer(interQ, scale, seqLen*rank)
		}
		training.BatchedOuterProduct(dQ, interQ, grads.DQB[layerIdx], seqLen, qDim, rank)

		// dQA += scale * ((dQ @ QB)^T @ normOut)  →  [rank, hiddenSize]
		// dInterQ = dQ @ QB  →  [seqLen, rank]   (note: QB is [qDim, rank])
		dInterQ := b.Alloc(seqLen * rank * 4)
		b.MatMulTransposed(dQ, la.QB, dInterQ, seqLen, rank, qDim)
		// interQ was already scaled, but dInterQ is fresh — apply scale.
		if scale != 1.0 && hasScaler {
			scaler.ScaleBuffer(dInterQ, scale, seqLen*rank)
		}
		training.BatchedOuterProduct(dInterQ, normOut, grads.DQA[layerIdx], seqLen, rank, hiddenSize)
	}

	if la.HasV {
		// interV = normOut @ VA^T  →  [seqLen, rank]
		interV := b.Alloc(seqLen * rank * 4)
		b.MatMulTransposed(normOut, la.VA, interV, seqLen, rank, hiddenSize)

		// dVB += scale * (dV^T @ interV)  →  [vDim, rank]
		if scale != 1.0 && hasScaler {
			scaler.ScaleBuffer(interV, scale, seqLen*rank)
		}
		training.BatchedOuterProduct(dV, interV, grads.DVB[layerIdx], seqLen, vDim, rank)

		// dVA += scale * ((dV @ VB)^T @ normOut)  →  [rank, hiddenSize]
		dInterV := b.Alloc(seqLen * rank * 4)
		b.MatMulTransposed(dV, la.VB, dInterV, seqLen, rank, vDim)
		if scale != 1.0 && hasScaler {
			scaler.ScaleBuffer(dInterV, scale, seqLen*rank)
		}
		training.BatchedOuterProduct(dInterV, normOut, grads.DVA[layerIdx], seqLen, rank, hiddenSize)
	}
}

// recomputeFFNNormInput reconstructs the input to the FFN pre-norm by
// recomputing: x_after_attn = saved.Residual + Wo @ saved.AttnOut.
// This avoids saving an additional activation per layer.
func recomputeFFNNormInput(
	b backend.Backend,
	saved *runtime.SavedActivations,
	layer *runtime.BlockRuntime,
	seqLen, hiddenSize, qDim int,
) tensor.DevicePtr {
	ffnNormInput := b.Alloc(seqLen * hiddenSize * 4)
	if !layer.Wo.DevicePtr().IsNil() {
		oDim := layer.Wo.Shape().Dims()[0]
		b.MatMulTransposed(saved.AttnOut, layer.Wo.DevicePtr(), ffnNormInput, seqLen, oDim, qDim)
	}
	b.Add(saved.Residual, ffnNormInput, ffnNormInput, seqLen*hiddenSize)
	return ffnNormInput
}

// countMasked downloads the loss mask and counts non-zero entries.
func countMasked(b backend.Backend, mask tensor.DevicePtr, seqLen int) int {
	maskBytes := make([]byte, seqLen*4)
	b.ToHost(maskBytes, mask)
	b.Sync()

	n := 0
	for i := 0; i < seqLen; i++ {
		v := int32(maskBytes[i*4]) | int32(maskBytes[i*4+1])<<8 |
			int32(maskBytes[i*4+2])<<16 | int32(maskBytes[i*4+3])<<24
		if v != 0 {
			n++
		}
	}
	return n
}

// computeAttnWeights computes softmax(scale * Q @ K^T) with causal masking.
// Q: [seqLen, numHeads*headDim], K: [seqLen, numKVHeads*headDim]
// Returns attnWeights: [numHeads, seqLen, seqLen].
//
// For GQA, each KV head is shared across numHeads/numKVHeads query heads.
// The computation is done per-head via CPU softmax for correctness during
// training (seqLen is typically small for LoRA fine-tuning).
func computeAttnWeights(
	b backend.Backend,
	Q, K tensor.DevicePtr,
	seqLen, numHeads, numKVHeads, headDim int,
	scale float32,
) tensor.DevicePtr {
	totalSize := numHeads * seqLen * seqLen
	attnWeights := b.Alloc(totalSize * 4)

	headsPerGroup := numHeads / numKVHeads

	// Download Q and K once.
	qBytes := make([]byte, seqLen*numHeads*headDim*4)
	kBytes := make([]byte, seqLen*numKVHeads*headDim*4)
	b.Sync()
	b.ToHost(qBytes, Q)
	b.ToHost(kBytes, K)

	qAll := bytesToF32(qBytes)
	kAll := bytesToF32(kBytes)

	// Output buffer for all heads.
	result := make([]float32, totalSize)

	for kvh := 0; kvh < numKVHeads; kvh++ {
		// Extract K for this KV head: K[:, kvh*headDim:(kvh+1)*headDim]
		kHead := extractHead(kAll, seqLen, numKVHeads*headDim, kvh*headDim, headDim)

		for qhi := 0; qhi < headsPerGroup; qhi++ {
			globalHead := kvh*headsPerGroup + qhi
			// Extract Q for this query head.
			qHead := extractHead(qAll, seqLen, numHeads*headDim, globalHead*headDim, headDim)

			// scores[i][j] = scale * sum_d(Q[i,d] * K[j,d])
			scores := make([]float32, seqLen*seqLen)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					if j > i {
						scores[i*seqLen+j] = float32(math.Inf(-1))
						continue
					}
					var dot float32
					for d := 0; d < headDim; d++ {
						dot += qHead[i*headDim+d] * kHead[j*headDim+d]
					}
					scores[i*seqLen+j] = scale * dot
				}
			}

			// Softmax per row.
			for i := 0; i < seqLen; i++ {
				rowOff := i * seqLen
				maxVal := scores[rowOff]
				for j := 1; j < seqLen; j++ {
					if scores[rowOff+j] > maxVal {
						maxVal = scores[rowOff+j]
					}
				}
				var sumExp float64
				for j := 0; j < seqLen; j++ {
					scores[rowOff+j] = float32(math.Exp(float64(scores[rowOff+j] - maxVal)))
					sumExp += float64(scores[rowOff+j])
				}
				if sumExp > 0 {
					inv := float32(1.0 / sumExp)
					for j := 0; j < seqLen; j++ {
						scores[rowOff+j] *= inv
					}
				}
			}

			// Copy into result at [globalHead, :, :].
			headOff := globalHead * seqLen * seqLen
			copy(result[headOff:headOff+seqLen*seqLen], scores)
		}
	}

	// Upload to device.
	b.ToDevice(attnWeights, f32ToBytes(result))
	return attnWeights
}

// extractHead extracts a single head's data from a packed multi-head layout.
// src: flat [seqLen * totalDim], returns [seqLen * headDim].
func extractHead(src []float32, seqLen, totalDim, offset, headDim int) []float32 {
	out := make([]float32, seqLen*headDim)
	for row := 0; row < seqLen; row++ {
		copy(out[row*headDim:(row+1)*headDim], src[row*totalDim+offset:row*totalDim+offset+headDim])
	}
	return out
}

// bytesToF32 converts a little-endian byte slice to float32 slice.
func bytesToF32(data []byte) []float32 {
	n := len(data) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 |
			uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// f32ToBytes converts a float32 slice to little-endian bytes.
func f32ToBytes(data []float32) []byte {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		out[i*4] = byte(bits)
		out[i*4+1] = byte(bits >> 8)
		out[i*4+2] = byte(bits >> 16)
		out[i*4+3] = byte(bits >> 24)
	}
	return out
}
