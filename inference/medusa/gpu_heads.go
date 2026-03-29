//go:build metal && darwin && cgo

// Package medusa implements Medusa speculative decoding with online training.
package medusa

import (
	"math"
	"math/rand"
	"sync"

	"vexel/inference/backend"
	"vexel/inference/tensor"
)

// GPUHead represents a single Medusa prediction head with weights on GPU.
type GPUHead struct {
	// FC1: [hiddenSize, hiddenSize] - first linear layer
	FC1 tensor.DevicePtr
	// FC2: [hiddenSize, vocabSize] - second linear layer
	FC2 tensor.DevicePtr

	// CPU copies for gradient computation (hybrid approach)
	FC1CPU []float32
	FC2CPU []float32

	// BypassFC1 skips FC1+ReLU, using FC2 directly on hidden state
	// Set true when initialized from lm_head to preserve the learned representation
	BypassFC1 bool
}

// GPUHeads manages multiple Medusa prediction heads with GPU acceleration.
type GPUHeads struct {
	mu sync.RWMutex

	NumHeads   int
	HiddenSize int
	VocabSize  int

	heads   []GPUHead
	backend backend.Backend

	// Scratch buffers for GPU training (lazy allocated)
	scratchAllocated bool
	scratchHidden    tensor.DevicePtr // [maxBatch, hiddenSize]
	scratchInter     tensor.DevicePtr // [maxBatch, hiddenSize]
	scratchPreRelu   tensor.DevicePtr // [maxBatch, hiddenSize] - pre-activation for SiLU backward
	scratchLogits    tensor.DevicePtr // [maxBatch, vocabSize]
	scratchDLogits   tensor.DevicePtr // [maxBatch, vocabSize]
	scratchDInter    tensor.DevicePtr // [maxBatch, hiddenSize]
	scratchGrad1     tensor.DevicePtr // [hiddenSize, hiddenSize]
	scratchGrad2     tensor.DevicePtr // [hiddenSize, vocabSize]
	maxBatchSize     int

	// Track whether GPU weights have been modified since last CPU sync
	weightsDirty bool

	// Permanent buffers for GPU-accelerated inference forward pass.
	// Allocated once and never recycled (survive ResetPool).
	fwdHidden tensor.DevicePtr // [1, hiddenSize]
	fwdInter  tensor.DevicePtr // [1, hiddenSize]
	fwdLogits tensor.DevicePtr // [1, vocabSize]
	fwdReady  bool
}

// NewGPUHeads creates GPU-accelerated Medusa heads.
func NewGPUHeads(numHeads, hiddenSize, vocabSize int, b backend.Backend) *GPUHeads {
	return NewGPUHeadsWithInit(numHeads, hiddenSize, vocabSize, b, nil)
}

// NewGPUHeadsWithInit creates GPUHeads with optional initialization from lm_head weights.
// If lmHeadWeights is provided (shape [vocab_size, hidden_size]), FC2 is initialized from it
// instead of random. This significantly improves speculation accuracy.
func NewGPUHeadsWithInit(numHeads, hiddenSize, vocabSize int, b backend.Backend, lmHeadWeights []float32) *GPUHeads {
	h := &GPUHeads{
		NumHeads:   numHeads,
		HiddenSize: hiddenSize,
		VocabSize:  vocabSize,
		heads:      make([]GPUHead, numHeads),
		backend:    b,
	}

	// Xavier initialization scale for FC1
	scale1 := float32(math.Sqrt(2.0 / float64(hiddenSize)))

	// Check if we have lm_head weights for FC2 initialization
	hasLMHead := len(lmHeadWeights) == vocabSize*hiddenSize

	for i := 0; i < numHeads; i++ {
		// Allocate GPU memory
		fc1Size := hiddenSize * hiddenSize * 4
		fc2Size := hiddenSize * vocabSize * 4

		h.heads[i].FC1 = b.Alloc(fc1Size)
		h.heads[i].FC2 = b.Alloc(fc2Size)

		// Initialize FC1 on CPU
		h.heads[i].FC1CPU = make([]float32, hiddenSize*hiddenSize)
		if hasLMHead {
			// When initializing from lm_head, bypass FC1 entirely
			// This allows hidden state to pass directly to FC2 (which is lm_head)
			h.heads[i].BypassFC1 = true
			// Still need identity FC1 for GPU training path
			for j := 0; j < hiddenSize; j++ {
				h.heads[i].FC1CPU[j*hiddenSize+j] = 1.0
			}
		} else {
			// Random initialization with near-identity for normal training
			for j := range h.heads[i].FC1CPU {
				h.heads[i].FC1CPU[j] = (rand.Float32()*2 - 1) * scale1 * 0.1 // Small init
			}
			for j := 0; j < hiddenSize; j++ {
				h.heads[i].FC1CPU[j*hiddenSize+j] += 0.9 // Near-identity on diagonal
			}
		}

		// Initialize FC2 on CPU
		h.heads[i].FC2CPU = make([]float32, hiddenSize*vocabSize)
		if hasLMHead {
			// Copy from lm_head weights (transposed: lm_head is [vocab, hidden], FC2 is [hidden, vocab])
			for hi := 0; hi < hiddenSize; hi++ {
				for vi := 0; vi < vocabSize; vi++ {
					// lmHeadWeights[vi * hiddenSize + hi] -> FC2CPU[hi * vocabSize + vi]
					h.heads[i].FC2CPU[hi*vocabSize+vi] = lmHeadWeights[vi*hiddenSize+hi]
				}
			}
		} else {
			// Random initialization
			scale2 := float32(math.Sqrt(2.0 / float64(hiddenSize)))
			for j := range h.heads[i].FC2CPU {
				h.heads[i].FC2CPU[j] = (rand.Float32()*2 - 1) * scale2
			}
		}

		// Copy to GPU
		h.copyWeightsToGPU(i)
	}

	return h
}

// copyWeightsToGPU copies head weights from CPU to GPU.
func (h *GPUHeads) copyWeightsToGPU(headIdx int) {
	head := &h.heads[headIdx]

	// Convert float32 to bytes and copy
	fc1Bytes := float32ToBytes(head.FC1CPU)
	fc2Bytes := float32ToBytes(head.FC2CPU)

	h.backend.ToDevice(head.FC1, fc1Bytes)
	h.backend.ToDevice(head.FC2, fc2Bytes)
}

// syncWeightsFromGPU copies weights from GPU to CPU for all heads.
// Call this before using Forward() after training.
func (h *GPUHeads) syncWeightsFromGPU() {
	if !h.weightsDirty {
		return
	}

	hiddenSize := h.HiddenSize
	vocabSize := h.VocabSize

	for i := range h.heads {
		head := &h.heads[i]

		fc1Bytes := make([]byte, hiddenSize*hiddenSize*4)
		fc2Bytes := make([]byte, hiddenSize*vocabSize*4)
		h.backend.ToHost(fc1Bytes, head.FC1)
		h.backend.ToHost(fc2Bytes, head.FC2)
		head.FC1CPU = bytesToFloat32(fc1Bytes)
		head.FC2CPU = bytesToFloat32(fc2Bytes)
	}

	h.backend.Sync()
	h.weightsDirty = false
}

// ensureFwdBuffers allocates permanent GPU buffers for inference forward pass.
func (h *GPUHeads) ensureFwdBuffers() {
	if h.fwdReady {
		return
	}
	type permanentAllocator interface {
		AllocPermanent(bytes int) tensor.DevicePtr
	}
	alloc := h.backend.Alloc
	if pa, ok := h.backend.(permanentAllocator); ok {
		alloc = pa.AllocPermanent
	}
	h.fwdHidden = alloc(h.HiddenSize * 4)
	h.fwdInter = alloc(h.HiddenSize * 4)
	h.fwdLogits = alloc(h.VocabSize * 4)
	h.fwdReady = true
}

// Forward computes logits for a single head on GPU.
// Caller must hold the GPU lock (via Trainer.GPULock) to prevent concurrent Metal access.
// hidden: [hiddenSize] on CPU
// Returns: logits [vocabSize] on CPU
func (h *GPUHeads) Forward(headIdx int, hidden []float32) []float32 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if headIdx < 0 || headIdx >= h.NumHeads {
		return nil
	}

	head := &h.heads[headIdx]
	hiddenSize := h.HiddenSize
	vocabSize := h.VocabSize

	h.ensureFwdBuffers()

	// Upload hidden state to GPU
	h.backend.ToDevice(h.fwdHidden, float32ToBytes(hidden))

	if head.BypassFC1 {
		// FC2 only: logits = hidden @ FC2  [1, hidden] @ [hidden, vocab] -> [1, vocab]
		h.backend.MatMul(h.fwdHidden, head.FC2, h.fwdLogits, 1, vocabSize, hiddenSize)
	} else {
		// FC1: intermediate = hidden @ FC1  [1, hidden] @ [hidden, hidden] -> [1, hidden]
		h.backend.MatMul(h.fwdHidden, head.FC1, h.fwdInter, 1, hiddenSize, hiddenSize)
		// SiLU in-place
		if trainOps, ok := h.backend.(backend.TrainingOps); ok {
			trainOps.SiLUInplace(h.fwdInter, hiddenSize)
		}
		// FC2: logits = intermediate @ FC2  [1, hidden] @ [hidden, vocab] -> [1, vocab]
		h.backend.MatMul(h.fwdInter, head.FC2, h.fwdLogits, 1, vocabSize, hiddenSize)
	}

	// Download logits to CPU
	logitsBytes := make([]byte, vocabSize*4)
	h.backend.Sync()
	h.backend.ToHost(logitsBytes, h.fwdLogits)
	h.backend.Sync()
	return bytesToFloat32(logitsBytes)
}

// ForwardAll computes logits for all heads on GPU.
func (h *GPUHeads) ForwardAll(hidden []float32) [][]float32 {
	results := make([][]float32, h.NumHeads)
	for i := 0; i < h.NumHeads; i++ {
		results[i] = h.Forward(i, hidden)
	}
	return results
}

// ForwardTopK returns top-k token predictions for all heads.
func (h *GPUHeads) ForwardTopK(hidden []float32, k int) [][]int {
	allLogits := h.ForwardAll(hidden)
	results := make([][]int, h.NumHeads)

	for i, logits := range allLogits {
		results[i] = topKIndices(logits, k)
	}
	return results
}

// topKIndices returns indices of k largest values.
func topKIndices(values []float32, k int) []int {
	if k > len(values) {
		k = len(values)
	}

	indices := make([]int, k)
	used := make([]bool, len(values))

	for i := 0; i < k; i++ {
		bestIdx := -1
		bestVal := float32(-math.MaxFloat32)
		for j, v := range values {
			if !used[j] && v > bestVal {
				bestVal = v
				bestIdx = j
			}
		}
		indices[i] = bestIdx
		if bestIdx >= 0 {
			used[bestIdx] = true
		}
	}

	return indices
}

// allocateScratch allocates GPU scratch buffers for training.
func (h *GPUHeads) allocateScratch(batchSize int) {
	if h.scratchAllocated && batchSize <= h.maxBatchSize {
		return
	}

	// Free old buffers if reallocating
	if h.scratchAllocated {
		h.freeScratch()
	}

	hiddenSize := h.HiddenSize
	vocabSize := h.VocabSize

	h.scratchHidden = h.backend.Alloc(batchSize * hiddenSize * 4)
	h.scratchInter = h.backend.Alloc(batchSize * hiddenSize * 4)
	h.scratchPreRelu = h.backend.Alloc(batchSize * hiddenSize * 4)
	h.scratchLogits = h.backend.Alloc(batchSize * vocabSize * 4)
	h.scratchDLogits = h.backend.Alloc(batchSize * vocabSize * 4)
	h.scratchDInter = h.backend.Alloc(batchSize * hiddenSize * 4)
	h.scratchGrad1 = h.backend.Alloc(hiddenSize * hiddenSize * 4)
	h.scratchGrad2 = h.backend.Alloc(hiddenSize * vocabSize * 4)

	h.maxBatchSize = batchSize
	h.scratchAllocated = true
}

// freeScratch releases GPU scratch buffers.
func (h *GPUHeads) freeScratch() {
	if !h.scratchAllocated {
		return
	}
	h.backend.Free(h.scratchHidden)
	h.backend.Free(h.scratchInter)
	h.backend.Free(h.scratchPreRelu)
	h.backend.Free(h.scratchLogits)
	h.backend.Free(h.scratchDLogits)
	h.backend.Free(h.scratchDInter)
	h.backend.Free(h.scratchGrad1)
	h.backend.Free(h.scratchGrad2)
	h.scratchAllocated = false
}

// TrainStep performs one training step with true GPU acceleration.
// Forward pass, backward pass, and weight update all run on GPU.
func (h *GPUHeads) TrainStep(samples []TrainingSample, lr float32) float32 {
	h.mu.Lock()
	defer h.mu.Unlock()

	if len(samples) == 0 {
		return 0
	}

	// Check if backend supports training operations
	trainOps, hasTrainOps := h.backend.(backend.TrainingOps)
	if !hasTrainOps {
		return h.trainStepCPU(samples, lr)
	}

	hiddenSize := h.HiddenSize
	vocabSize := h.VocabSize
	batchSize := len(samples)

	// Allocate scratch buffers if needed
	h.allocateScratch(batchSize)

	totalLoss := float32(0)

	// Train each head
	for headIdx := 0; headIdx < h.NumHeads; headIdx++ {
		head := &h.heads[headIdx]

		// Filter valid samples for this head
		validIndices := make([]int, 0, batchSize)
		targets := make([]int, 0, batchSize)
		for i, sample := range samples {
			if headIdx >= len(sample.FutureTokens) {
				continue
			}
			target := sample.FutureTokens[headIdx]
			if target < 0 || target >= vocabSize {
				continue
			}
			validIndices = append(validIndices, i)
			targets = append(targets, target)
		}

		if len(validIndices) == 0 {
			continue
		}

		validBatch := len(validIndices)

		// Prepare batched hidden states [validBatch, hiddenSize]
		batchedHidden := make([]float32, validBatch*hiddenSize)
		for i, idx := range validIndices {
			copy(batchedHidden[i*hiddenSize:], samples[idx].HiddenState)
		}

		// Upload batched hidden to GPU
		h.backend.ToDevice(h.scratchHidden, float32ToBytes(batchedHidden))

		// Forward pass on GPU
		// FC1: intermediate = hidden @ FC1 [validBatch, hidden] @ [hidden, hidden] -> [validBatch, hidden]
		h.backend.MatMul(h.scratchHidden, head.FC1, h.scratchInter, validBatch, hiddenSize, hiddenSize)

		// Save pre-activation for backward (copy to scratchPreRelu)
		preActBytes := make([]byte, validBatch*hiddenSize*4)
		h.backend.ToHost(preActBytes, h.scratchInter)
		h.backend.ToDevice(h.scratchPreRelu, preActBytes)

		// SiLU in-place
		trainOps.SiLUInplace(h.scratchInter, validBatch*hiddenSize)

		// FC2: logits = intermediate @ FC2 [validBatch, hidden] @ [hidden, vocab] -> [validBatch, vocab]
		h.backend.MatMul(h.scratchInter, head.FC2, h.scratchLogits, validBatch, vocabSize, hiddenSize)

		// Download logits for softmax + loss (CPU)
		logitsBytes := make([]byte, validBatch*vocabSize*4)
		h.backend.ToHost(logitsBytes, h.scratchLogits)
		h.backend.Sync()
		logitsBatch := bytesToFloat32(logitsBytes)

		// Compute softmax + loss + dLogits on CPU
		dLogitsBatch := make([]float32, validBatch*vocabSize)
		for i := 0; i < validBatch; i++ {
			logits := logitsBatch[i*vocabSize : (i+1)*vocabSize]
			probs := softmax(logits)
			loss := -float32(math.Log(float64(probs[targets[i]]) + 1e-10))
			totalLoss += loss

			// dLogits = probs - one_hot(target)
			copy(dLogitsBatch[i*vocabSize:], probs)
			dLogitsBatch[i*vocabSize+targets[i]] -= 1.0
		}

		// Clip gradients to prevent explosion (max norm per sample)
		const maxGradNorm = float32(1.0)
		for i := 0; i < validBatch; i++ {
			dLogits := dLogitsBatch[i*vocabSize : (i+1)*vocabSize]
			var norm float32
			for _, g := range dLogits {
				norm += g * g
			}
			norm = float32(math.Sqrt(float64(norm)))
			if norm > maxGradNorm {
				scale := maxGradNorm / norm
				for j := range dLogits {
					dLogits[j] *= scale
				}
			}
		}

		// Upload dLogits to GPU
		h.backend.ToDevice(h.scratchDLogits, float32ToBytes(dLogitsBatch))

		// Zero gradient buffers
		trainOps.Zero(h.scratchGrad1, hiddenSize*hiddenSize)
		trainOps.Zero(h.scratchGrad2, hiddenSize*vocabSize)

		// Backward pass on GPU
		// FC2 gradient: grad2[i,j] = sum_b(intermediate[b,i] * dLogits[b,j])
		trainOps.BatchedOuterProduct(h.scratchInter, h.scratchDLogits, h.scratchGrad2,
			validBatch, hiddenSize, vocabSize)

		// Backprop through FC2: dIntermediate = dLogits @ FC2^T
		// dLogits: [validBatch, vocab], FC2: [hidden, vocab], FC2^T: [vocab, hidden]
		// Result: [validBatch, hidden]
		h.backend.MatMulTransposed(h.scratchDLogits, head.FC2, h.scratchDInter,
			validBatch, hiddenSize, vocabSize)

		// SiLU backward: dSiLU/dx = sigmoid(x) * (1 + x*(1-sigmoid(x)))
		trainOps.SiLUBackward(h.scratchPreRelu, h.scratchDInter, validBatch*hiddenSize)

		// FC1 gradient: grad1[i,j] = sum_b(hidden[b,i] * dIntermediate[b,j])
		trainOps.BatchedOuterProduct(h.scratchHidden, h.scratchDInter, h.scratchGrad1,
			validBatch, hiddenSize, hiddenSize)

		// SGD update with weight decay on GPU: w = w*(1-lr*wd) - lr*grad
		// Moderate weight decay - too high destroys good initialization
		const weightDecay = float32(0.001)
		scaledLR := lr / float32(validBatch)
		trainOps.SGDUpdate(head.FC1, h.scratchGrad1, scaledLR, weightDecay, hiddenSize*hiddenSize)
		trainOps.SGDUpdate(head.FC2, h.scratchGrad2, scaledLR, weightDecay, hiddenSize*vocabSize)
	}

	// Mark weights as dirty - will sync to CPU when Forward() is called
	h.weightsDirty = true

	// Sync GPU to ensure all updates are complete
	h.backend.Sync()

	return totalLoss / float32(len(samples)*h.NumHeads)
}

// trainStepCPU is the fallback CPU-only training (slow but works on all backends).
func (h *GPUHeads) trainStepCPU(samples []TrainingSample, lr float32) float32 {
	totalLoss := float32(0)
	hiddenSize := h.HiddenSize
	vocabSize := h.VocabSize

	// Train each head
	for headIdx := 0; headIdx < h.NumHeads; headIdx++ {
		head := &h.heads[headIdx]

		// Accumulate gradients on CPU
		grad1 := make([]float32, hiddenSize*hiddenSize)
		grad2 := make([]float32, hiddenSize*vocabSize)
		validSamples := 0

		for _, sample := range samples {
			if headIdx >= len(sample.FutureTokens) {
				continue
			}
			targetToken := sample.FutureTokens[headIdx]
			if targetToken < 0 || targetToken >= vocabSize {
				continue
			}

			// Forward pass (using CPU weights for gradient computation)
			// FC1 forward with SiLU
			preAct := make([]float32, hiddenSize)
			intermediate := make([]float32, hiddenSize)
			for i := 0; i < hiddenSize; i++ {
				var sum float32
				for j := 0; j < hiddenSize; j++ {
					sum += sample.HiddenState[j] * head.FC1CPU[j*hiddenSize+i]
				}
				preAct[i] = sum
				intermediate[i] = silu(sum)
			}

			// FC2 forward
			logits := make([]float32, vocabSize)
			for i := 0; i < vocabSize; i++ {
				var sum float32
				for j := 0; j < hiddenSize; j++ {
					sum += intermediate[j] * head.FC2CPU[j*vocabSize+i]
				}
				logits[i] = sum
			}

			// Softmax + loss
			probs := softmax(logits)
			loss := -float32(math.Log(float64(probs[targetToken]) + 1e-10))
			totalLoss += loss

			// Backward: dL/dlogits = probs - one_hot(target)
			dLogits := make([]float32, vocabSize)
			copy(dLogits, probs)
			dLogits[targetToken] -= 1.0

			// Gradient for FC2: dFC2 = intermediate^T @ dLogits
			for i := 0; i < hiddenSize; i++ {
				for j := 0; j < vocabSize; j++ {
					grad2[i*vocabSize+j] += intermediate[i] * dLogits[j]
				}
			}

			// Backprop through FC2: dIntermediate = dLogits @ FC2^T
			dIntermediate := make([]float32, hiddenSize)
			for i := 0; i < hiddenSize; i++ {
				for j := 0; j < vocabSize; j++ {
					dIntermediate[i] += dLogits[j] * head.FC2CPU[i*vocabSize+j]
				}
			}

			// SiLU backward: dSiLU/dx = sigmoid(x) * (1 + x*(1-sigmoid(x)))
			for i := 0; i < hiddenSize; i++ {
				dIntermediate[i] *= siluDerivative(preAct[i])
			}

			// Gradient for FC1: dFC1 = hidden^T @ dIntermediate
			for i := 0; i < hiddenSize; i++ {
				for j := 0; j < hiddenSize; j++ {
					grad1[i*hiddenSize+j] += sample.HiddenState[i] * dIntermediate[j]
				}
			}

			validSamples++
		}

		if validSamples == 0 {
			continue
		}

		// Average gradients and apply update
		scale := lr / float32(validSamples)
		for i := range head.FC1CPU {
			head.FC1CPU[i] -= scale * grad1[i]
		}
		for i := range head.FC2CPU {
			head.FC2CPU[i] -= scale * grad2[i]
		}

		// Copy updated weights to GPU
		h.copyWeightsToGPU(headIdx)
	}

	return totalLoss / float32(len(samples)*h.NumHeads)
}

// Evaluate computes accuracy on a batch of samples.
func (h *GPUHeads) Evaluate(samples []TrainingSample) []float32 {
	accuracies := make([]float32, h.NumHeads)

	for headIdx := 0; headIdx < h.NumHeads; headIdx++ {
		correct := 0
		total := 0

		for _, sample := range samples {
			if headIdx >= len(sample.FutureTokens) {
				continue
			}
			targetToken := sample.FutureTokens[headIdx]

			logits := h.Forward(headIdx, sample.HiddenState)
			if logits == nil {
				continue
			}

			predToken := argmax(logits)
			if predToken == targetToken {
				correct++
			}
			total++
		}

		if total > 0 {
			accuracies[headIdx] = float32(correct) / float32(total)
		}
	}

	return accuracies
}

// ToCPUHeads converts GPU heads to CPU Heads format for saving.
// This reads all weight data from GPU memory back to the host.
func (g *GPUHeads) ToCPUHeads() *Heads {
	g.mu.RLock()
	defer g.mu.RUnlock()

	h := &Heads{
		NumHeads:   g.NumHeads,
		HiddenSize: g.HiddenSize,
		VocabSize:  g.VocabSize,
		heads:      make([]Head, g.NumHeads),
	}

	hiddenSize := g.HiddenSize
	vocabSize := g.VocabSize

	for i := range g.heads {
		head := &g.heads[i]

		// Read FC1 weights from GPU
		fc1Bytes := make([]byte, hiddenSize*hiddenSize*4)
		g.backend.ToHost(fc1Bytes, head.FC1)

		// Read FC2 weights from GPU
		fc2Bytes := make([]byte, hiddenSize*vocabSize*4)
		g.backend.ToHost(fc2Bytes, head.FC2)

		h.heads[i] = Head{
			FC1: bytesToFloat32(fc1Bytes),
			FC2: bytesToFloat32(fc2Bytes),
		}
	}

	g.backend.Sync()
	return h
}

// MemorySize returns approximate GPU memory usage in bytes.
func (h *GPUHeads) MemorySize() int64 {
	perHead := int64(h.HiddenSize*h.HiddenSize + h.HiddenSize*h.VocabSize)
	return int64(h.NumHeads) * perHead * 4
}

// Free releases GPU memory.
func (h *GPUHeads) Free() {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Free head weights
	for i := range h.heads {
		if !h.heads[i].FC1.IsNil() {
			h.backend.Free(h.heads[i].FC1)
		}
		if !h.heads[i].FC2.IsNil() {
			h.backend.Free(h.heads[i].FC2)
		}
	}

	// Free scratch buffers
	h.freeScratch()
}

// Helper functions

func float32ToBytes(f []float32) []byte {
	b := make([]byte, len(f)*4)
	for i, v := range f {
		bits := math.Float32bits(v)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}

func bytesToFloat32(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		f[i] = math.Float32frombits(bits)
	}
	return f
}
