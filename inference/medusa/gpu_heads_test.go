//go:build metal && darwin && cgo

package medusa

import (
	"math"
	"testing"

	"vexel/inference/backend"
	"vexel/inference/backend/metal"
)

func TestGPUForwardPass(t *testing.T) {
	// Test that GPU forward pass matches CPU forward pass
	metalBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}

	numHeads := 1
	hiddenSize := 64
	vocabSize := 100

	heads := NewGPUHeads(numHeads, hiddenSize, vocabSize, metalBackend)
	defer heads.Free()

	// Create test input
	hidden := make([]float32, hiddenSize)
	for i := range hidden {
		hidden[i] = float32(i) * 0.01
	}

	// CPU forward pass
	cpuLogits := heads.Forward(0, hidden)

	// GPU forward pass (using MatMul directly)
	// Upload hidden to GPU
	hiddenBytes := float32ToBytes(hidden)
	hiddenGPU := metalBackend.Alloc(hiddenSize * 4)
	metalBackend.ToDevice(hiddenGPU, hiddenBytes)

	// Verify hidden was uploaded correctly
	verifyHidden := make([]byte, hiddenSize*4)
	metalBackend.ToHost(verifyHidden, hiddenGPU)
	metalBackend.Sync()
	verifyHiddenF32 := bytesToFloat32(verifyHidden)
	t.Logf("Uploaded hidden[0:5]: %v", verifyHiddenF32[0:5])
	t.Logf("Original hidden[0:5]: %v", hidden[0:5])

	// Verify FC1 weights
	verifyFC1 := make([]byte, hiddenSize*hiddenSize*4)
	metalBackend.ToHost(verifyFC1, heads.heads[0].FC1)
	metalBackend.Sync()
	verifyFC1F32 := bytesToFloat32(verifyFC1)
	t.Logf("FC1 GPU[0:5]: %v", verifyFC1F32[0:5])
	t.Logf("FC1 CPU[0:5]: %v", heads.heads[0].FC1CPU[0:5])

	// Allocate output buffers
	interGPU := metalBackend.Alloc(hiddenSize * 4)
	logitsGPU := metalBackend.Alloc(vocabSize * 4)

	// FC1: intermediate = hidden @ FC1
	// MatMul(a, b, out, m, n, k) does C = A @ B where A is [M,K], B is [K,N], C is [M,N]
	// hidden is [1, hiddenSize], FC1 is [hiddenSize, hiddenSize], output is [1, hiddenSize]
	t.Logf("Calling MatMul with m=1, n=%d, k=%d", hiddenSize, hiddenSize)
	metalBackend.MatMul(hiddenGPU, heads.heads[0].FC1, interGPU, 1, hiddenSize, hiddenSize)

	// Sync to ensure the GPU operation completed
	metalBackend.Sync()

	// Download intermediate
	interBytes := make([]byte, hiddenSize*4)
	metalBackend.ToHost(interBytes, interGPU)
	metalBackend.Sync()
	gpuInter := bytesToFloat32(interBytes)

	// Compute CPU intermediate for comparison
	cpuInter := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		var sum float32
		for j := 0; j < hiddenSize; j++ {
			sum += hidden[j] * heads.heads[0].FC1CPU[j*hiddenSize+i]
		}
		cpuInter[i] = sum // before SiLU
	}

	// Compare intermediate (before SiLU)
	maxInterDiff := float32(0)
	for i := 0; i < hiddenSize; i++ {
		diff := float32(math.Abs(float64(cpuInter[i] - gpuInter[i])))
		if diff > maxInterDiff {
			maxInterDiff = diff
		}
	}
	t.Logf("Max intermediate diff (pre-SiLU): %e", maxInterDiff)

	if maxInterDiff > 1e-4 {
		// Print first few values for debugging
		t.Logf("CPU inter[0:5]: %v", cpuInter[0:5])
		t.Logf("GPU inter[0:5]: %v", gpuInter[0:5])
		t.Errorf("Intermediate values differ too much")
	}

	// Apply SiLU to GPU intermediate
	for i := range gpuInter {
		gpuInter[i] = silu(gpuInter[i])
	}

	// Upload SiLU'd intermediate and compute logits
	metalBackend.ToDevice(interGPU, float32ToBytes(gpuInter))
	metalBackend.MatMul(interGPU, heads.heads[0].FC2, logitsGPU, 1, vocabSize, hiddenSize)

	// Sync to ensure GPU operation completed
	metalBackend.Sync()

	// Download logits
	logitsBytes := make([]byte, vocabSize*4)
	metalBackend.ToHost(logitsBytes, logitsGPU)
	metalBackend.Sync()
	gpuLogits := bytesToFloat32(logitsBytes)

	// Compare logits
	maxLogitsDiff := float32(0)
	for i := 0; i < vocabSize; i++ {
		diff := float32(math.Abs(float64(cpuLogits[i] - gpuLogits[i])))
		if diff > maxLogitsDiff {
			maxLogitsDiff = diff
		}
	}
	t.Logf("Max logits diff: %e", maxLogitsDiff)

	if maxLogitsDiff > 1e-3 {
		t.Logf("CPU logits[0:5]: %v", cpuLogits[0:5])
		t.Logf("GPU logits[0:5]: %v", gpuLogits[0:5])
		t.Errorf("Logits differ too much")
	}

	metalBackend.Free(hiddenGPU)
	metalBackend.Free(interGPU)
	metalBackend.Free(logitsGPU)
}

func TestGPUTrainingGradients(t *testing.T) {
	// Create a small test case to verify GPU training matches CPU training
	b, err := metal.NewBackend(0) // device 0
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}

	// Original dimensions
	numHeads := 1
	hiddenSize := 64
	vocabSize := 100

	heads := NewGPUHeads(numHeads, hiddenSize, vocabSize, b)
	defer heads.Free()

	// Create a simple training sample
	hidden := make([]float32, hiddenSize)
	for i := range hidden {
		hidden[i] = float32(i) * 0.1
	}

	_ = []TrainingSample{
		{
			HiddenState:  hidden,
			FutureTokens: []int{5}, // target token
		},
	}

	// Get initial weights
	initialFC1 := make([]float32, len(heads.heads[0].FC1CPU))
	initialFC2 := make([]float32, len(heads.heads[0].FC2CPU))
	copy(initialFC1, heads.heads[0].FC1CPU)
	copy(initialFC2, heads.heads[0].FC2CPU)

	t.Logf("Initial FC1[0:5]: %v", initialFC1[0:5])
	t.Logf("Initial FC2[0:5]: %v", initialFC2[0:5])
	t.Logf("Hidden: %v", hidden)

	// ======== Manual CPU forward/backward for comparison ========
	head := &heads.heads[0]

	// FC1 forward (before ReLU)
	cpuInter := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		var sum float32
		for j := 0; j < hiddenSize; j++ {
			sum += hidden[j] * initialFC1[j*hiddenSize+i]
		}
		cpuInter[i] = sum
	}
	t.Logf("CPU inter (pre-ReLU): %v", cpuInter)

	// Save pre-SiLU for backward
	cpuPreAct := make([]float32, hiddenSize)
	copy(cpuPreAct, cpuInter)

	// SiLU activation
	for i := range cpuInter {
		cpuInter[i] = silu(cpuInter[i])
	}
	t.Logf("CPU inter (post-SiLU): %v", cpuInter)

	// FC2 forward
	cpuLogits := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		var sum float32
		for j := 0; j < hiddenSize; j++ {
			sum += cpuInter[j] * initialFC2[j*vocabSize+i]
		}
		cpuLogits[i] = sum
	}
	t.Logf("CPU logits: %v", cpuLogits)

	// Softmax + loss
	probs := softmax(cpuLogits)
	target := 5
	cpuLoss := -float32(math.Log(float64(probs[target]) + 1e-10))
	t.Logf("CPU loss: %f", cpuLoss)

	// dLogits = probs - one_hot(target)
	cpuDLogits := make([]float32, vocabSize)
	copy(cpuDLogits, probs)
	cpuDLogits[target] -= 1.0
	t.Logf("CPU dLogits: %v", cpuDLogits)

	// FC2 gradient: grad2[i,j] = inter[i] * dLogits[j]
	cpuGrad2 := make([]float32, hiddenSize*vocabSize)
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < vocabSize; j++ {
			cpuGrad2[i*vocabSize+j] = cpuInter[i] * cpuDLogits[j]
		}
	}
	t.Logf("CPU grad2[0:5]: %v", cpuGrad2[0:5])

	// Backprop through FC2: dInter = dLogits @ FC2^T
	cpuDInter := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < vocabSize; j++ {
			cpuDInter[i] += cpuDLogits[j] * initialFC2[i*vocabSize+j]
		}
	}
	t.Logf("CPU dInter (pre-SiLU backward): %v", cpuDInter)

	// SiLU backward: dSiLU/dx = sigmoid(x) * (1 + x*(1-sigmoid(x)))
	for i := 0; i < hiddenSize; i++ {
		cpuDInter[i] *= siluDerivative(cpuPreAct[i])
	}
	t.Logf("CPU dInter (post-SiLU backward): %v", cpuDInter)

	// FC1 gradient: grad1[i,j] = hidden[i] * dInter[j]
	cpuGrad1 := make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < hiddenSize; j++ {
			cpuGrad1[i*hiddenSize+j] = hidden[i] * cpuDInter[j]
		}
	}
	t.Logf("CPU grad1[0:5]: %v", cpuGrad1[0:5])

	// ======== GPU forward/backward ========
	// Allocate scratch buffers
	heads.allocateScratch(1)

	// Upload hidden
	b.ToDevice(heads.scratchHidden, float32ToBytes(hidden))

	// FC1: inter = hidden @ FC1
	b.MatMul(heads.scratchHidden, head.FC1, heads.scratchInter, 1, hiddenSize, hiddenSize)
	b.Sync()

	// Download GPU inter (pre-SiLU)
	gpuInterBytes := make([]byte, hiddenSize*4)
	b.ToHost(gpuInterBytes, heads.scratchInter)
	b.Sync()
	gpuInter := bytesToFloat32(gpuInterBytes)
	t.Logf("GPU inter (pre-SiLU): %v", gpuInter)

	// Compare pre-SiLU intermediate
	maxInterDiff := float32(0)
	for i := 0; i < hiddenSize; i++ {
		diff := float32(math.Abs(float64(cpuInter[i] - gpuInter[i])))
		if diff > maxInterDiff {
			maxInterDiff = diff
		}
	}
	t.Logf("Max inter diff (should use cpuPreAct): %e", maxInterDiff)

	// Compare to cpuPreAct since cpuInter has SiLU applied
	maxPreActDiff := float32(0)
	for i := 0; i < hiddenSize; i++ {
		diff := float32(math.Abs(float64(cpuPreAct[i] - gpuInter[i])))
		if diff > maxPreActDiff {
			maxPreActDiff = diff
		}
	}
	t.Logf("Max pre-SiLU diff: %e", maxPreActDiff)

	if maxPreActDiff > 1e-5 {
		t.Errorf("Pre-SiLU values differ: max diff = %e", maxPreActDiff)
	}

	// Save pre-activation to scratch
	b.ToDevice(heads.scratchPreRelu, gpuInterBytes)

	// Apply SiLU
	trainOps := heads.backend.(backend.TrainingOps)
	trainOps.SiLUInplace(heads.scratchInter, hiddenSize)
	b.Sync()

	// Download post-SiLU
	b.ToHost(gpuInterBytes, heads.scratchInter)
	b.Sync()
	gpuInterPostSiLU := bytesToFloat32(gpuInterBytes)
	t.Logf("GPU inter (post-SiLU): %v", gpuInterPostSiLU)

	// FC2: logits = inter @ FC2
	b.MatMul(heads.scratchInter, head.FC2, heads.scratchLogits, 1, vocabSize, hiddenSize)
	b.Sync()

	// Download logits
	gpuLogitsBytes := make([]byte, vocabSize*4)
	b.ToHost(gpuLogitsBytes, heads.scratchLogits)
	b.Sync()
	gpuLogits := bytesToFloat32(gpuLogitsBytes)
	t.Logf("GPU logits: %v", gpuLogits)

	// Compare logits
	maxLogitsDiff := float32(0)
	for i := 0; i < vocabSize; i++ {
		diff := float32(math.Abs(float64(cpuLogits[i] - gpuLogits[i])))
		if diff > maxLogitsDiff {
			maxLogitsDiff = diff
		}
	}
	t.Logf("Max logits diff: %e", maxLogitsDiff)

	if maxLogitsDiff > 1e-5 {
		t.Errorf("Logits differ: max diff = %e", maxLogitsDiff)
	}

	// Compute GPU loss and dLogits (same as CPU)
	gpuProbs := softmax(gpuLogits)
	gpuLoss := -float32(math.Log(float64(gpuProbs[target]) + 1e-10))
	t.Logf("GPU loss: %f", gpuLoss)

	gpuDLogits := make([]float32, vocabSize)
	copy(gpuDLogits, gpuProbs)
	gpuDLogits[target] -= 1.0
	t.Logf("GPU dLogits: %v", gpuDLogits)

	// Upload dLogits
	b.ToDevice(heads.scratchDLogits, float32ToBytes(gpuDLogits))

	// Zero gradient buffers
	trainOps.Zero(heads.scratchGrad1, hiddenSize*hiddenSize)
	trainOps.Zero(heads.scratchGrad2, hiddenSize*vocabSize)
	b.Sync()

	// FC2 gradient
	trainOps.BatchedOuterProduct(heads.scratchInter, heads.scratchDLogits, heads.scratchGrad2,
		1, hiddenSize, vocabSize)
	b.Sync()

	// Download grad2
	gpuGrad2Bytes := make([]byte, hiddenSize*vocabSize*4)
	b.ToHost(gpuGrad2Bytes, heads.scratchGrad2)
	b.Sync()
	gpuGrad2 := bytesToFloat32(gpuGrad2Bytes)
	t.Logf("GPU grad2[0:5]: %v", gpuGrad2[0:5])

	// Compare grad2
	maxGrad2Diff := float32(0)
	for i := 0; i < hiddenSize*vocabSize; i++ {
		diff := float32(math.Abs(float64(cpuGrad2[i] - gpuGrad2[i])))
		if diff > maxGrad2Diff {
			maxGrad2Diff = diff
		}
	}
	t.Logf("Max grad2 diff: %e", maxGrad2Diff)

	if maxGrad2Diff > 1e-5 {
		t.Errorf("Grad2 differs: max diff = %e", maxGrad2Diff)
	}

	// Backprop through FC2
	b.MatMulTransposed(heads.scratchDLogits, head.FC2, heads.scratchDInter, 1, hiddenSize, vocabSize)
	b.Sync()

	// Download dInter (pre-SiLU backward)
	gpuDInterBytes := make([]byte, hiddenSize*4)
	b.ToHost(gpuDInterBytes, heads.scratchDInter)
	b.Sync()
	gpuDInterPreSiLU := bytesToFloat32(gpuDInterBytes)
	t.Logf("GPU dInter (pre-SiLU backward): %v", gpuDInterPreSiLU)

	// Compare dInter pre-SiLU backward (before SiLU derivative)
	cpuDInterPreMask := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < vocabSize; j++ {
			cpuDInterPreMask[i] += cpuDLogits[j] * initialFC2[i*vocabSize+j]
		}
	}
	t.Logf("CPU dInter (pre-SiLU backward): %v", cpuDInterPreMask)

	maxDInterPreMaskDiff := float32(0)
	for i := 0; i < hiddenSize; i++ {
		diff := float32(math.Abs(float64(cpuDInterPreMask[i] - gpuDInterPreSiLU[i])))
		if diff > maxDInterPreMaskDiff {
			maxDInterPreMaskDiff = diff
		}
	}
	t.Logf("Max dInter (pre-SiLU backward) diff: %e", maxDInterPreMaskDiff)

	if maxDInterPreMaskDiff > 1e-5 {
		t.Errorf("dInter (pre-SiLU backward) differs: max diff = %e", maxDInterPreMaskDiff)
	}

	// SiLU backward
	trainOps.SiLUBackward(heads.scratchPreRelu, heads.scratchDInter, hiddenSize)
	b.Sync()

	// Download dInter (post-SiLU backward)
	b.ToHost(gpuDInterBytes, heads.scratchDInter)
	b.Sync()
	gpuDInterPostSiLU := bytesToFloat32(gpuDInterBytes)
	t.Logf("GPU dInter (post-SiLU backward): %v", gpuDInterPostSiLU)

	// Compare dInter post-SiLU backward
	maxDInterDiff := float32(0)
	for i := 0; i < hiddenSize; i++ {
		diff := float32(math.Abs(float64(cpuDInter[i] - gpuDInterPostSiLU[i])))
		if diff > maxDInterDiff {
			maxDInterDiff = diff
		}
	}
	t.Logf("Max dInter (post-SiLU) diff: %e", maxDInterDiff)

	if maxDInterDiff > 1e-5 {
		t.Errorf("dInter (post-SiLU) differs: max diff = %e", maxDInterDiff)
	}

	// FC1 gradient
	trainOps.BatchedOuterProduct(heads.scratchHidden, heads.scratchDInter, heads.scratchGrad1,
		1, hiddenSize, hiddenSize)
	b.Sync()

	// Download grad1
	gpuGrad1Bytes := make([]byte, hiddenSize*hiddenSize*4)
	b.ToHost(gpuGrad1Bytes, heads.scratchGrad1)
	b.Sync()
	gpuGrad1 := bytesToFloat32(gpuGrad1Bytes)
	t.Logf("GPU grad1[0:5]: %v", gpuGrad1[0:5])

	// Compare grad1
	maxGrad1Diff := float32(0)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		diff := float32(math.Abs(float64(cpuGrad1[i] - gpuGrad1[i])))
		if diff > maxGrad1Diff {
			maxGrad1Diff = diff
		}
	}
	t.Logf("Max grad1 diff: %e", maxGrad1Diff)

	if maxGrad1Diff > 1e-5 {
		t.Errorf("Grad1 differs: max diff = %e", maxGrad1Diff)
	}

	// ======== Test SGD update with weight decay ========
	lr := float32(0.01)
	weightDecay := float32(0.01) // L2 regularization

	// CPU SGD update with weight decay: w = w * (1 - lr * wd) - lr * grad
	decay := 1.0 - lr*weightDecay
	cpuFC1After := make([]float32, hiddenSize*hiddenSize)
	cpuFC2After := make([]float32, hiddenSize*vocabSize)
	copy(cpuFC1After, initialFC1)
	copy(cpuFC2After, initialFC2)
	for i := range cpuFC1After {
		cpuFC1After[i] = cpuFC1After[i]*decay - lr*cpuGrad1[i]
	}
	for i := range cpuFC2After {
		cpuFC2After[i] = cpuFC2After[i]*decay - lr*cpuGrad2[i]
	}
	t.Logf("CPU FC1 after[0:5]: %v", cpuFC1After[0:5])

	// GPU SGD update with weight decay
	trainOps.SGDUpdate(head.FC1, heads.scratchGrad1, lr, weightDecay, hiddenSize*hiddenSize)
	trainOps.SGDUpdate(head.FC2, heads.scratchGrad2, lr, weightDecay, hiddenSize*vocabSize)
	b.Sync()

	// Download updated weights
	gpuFC1Bytes := make([]byte, hiddenSize*hiddenSize*4)
	gpuFC2Bytes := make([]byte, hiddenSize*vocabSize*4)
	b.ToHost(gpuFC1Bytes, head.FC1)
	b.ToHost(gpuFC2Bytes, head.FC2)
	b.Sync()
	gpuFC1After := bytesToFloat32(gpuFC1Bytes)
	gpuFC2After := bytesToFloat32(gpuFC2Bytes)
	t.Logf("GPU FC1 after[0:5]: %v", gpuFC1After[0:5])

	// Compare FC1
	maxFC1Diff := float32(0)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		diff := float32(math.Abs(float64(cpuFC1After[i] - gpuFC1After[i])))
		if diff > maxFC1Diff {
			maxFC1Diff = diff
		}
	}
	t.Logf("Max FC1 diff after SGD: %e", maxFC1Diff)

	if maxFC1Diff > 1e-5 {
		t.Errorf("FC1 after SGD differs: max diff = %e", maxFC1Diff)
	}

	// Compare FC2
	maxFC2Diff := float32(0)
	for i := 0; i < hiddenSize*vocabSize; i++ {
		diff := float32(math.Abs(float64(cpuFC2After[i] - gpuFC2After[i])))
		if diff > maxFC2Diff {
			maxFC2Diff = diff
		}
	}
	t.Logf("Max FC2 diff after SGD: %e", maxFC2Diff)

	if maxFC2Diff > 1e-5 {
		t.Errorf("FC2 after SGD differs: max diff = %e", maxFC2Diff)
	}
}
