//go:build metal && darwin && cgo

package train

import (
	"math"
	"testing"

	"vexel/inference/backend"
	"vexel/inference/backend/metal"
)

// TestRoPEBackward verifies that RoPE forward then backward = identity.
// Since RoPE is an orthogonal rotation, backward(forward(x)) = x.
func TestRoPEBackwardIsInverse(t *testing.T) {
	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer gpuBackend.Close()
	training := backend.TrainingOps(gpuBackend)

	seqLen := 4
	headDim := 64
	numHeads := 4
	numKVHeads := 2
	ropeDim := headDim
	theta := float32(10000.0)
	ropeNeox := false

	// Create test data
	qData := make([]float32, seqLen*numHeads*headDim)
	kData := make([]float32, seqLen*numKVHeads*headDim)
	for i := range qData {
		qData[i] = 0.1 * float32(i%13-6)
	}
	for i := range kData {
		kData[i] = 0.2 * float32(i%7-3)
	}

	// Save original
	qOrig := make([]float32, len(qData))
	kOrig := make([]float32, len(kData))
	copy(qOrig, qData)
	copy(kOrig, kData)

	// Upload
	Q := gpuBackend.AllocPermanent(len(qData) * 4)
	K := gpuBackend.AllocPermanent(len(kData) * 4)
	gpuBackend.ToDevice(Q, float32SliceToBytes(qData))
	gpuBackend.ToDevice(K, float32SliceToBytes(kData))

	// Forward RoPE (in-place)
	gpuBackend.RoPE(Q, K, headDim, numHeads, numKVHeads, seqLen, 0, ropeDim, theta, ropeNeox)
	gpuBackend.Sync()

	// Read rotated
	qRotated := downloadF32(gpuBackend, Q, len(qData))

	// Verify rotation changed values
	var rotDiff float64
	for i := range qOrig {
		rotDiff += math.Abs(float64(qOrig[i] - qRotated[i]))
	}
	t.Logf("Q rotation magnitude: %.6f (should be >0)", rotDiff)
	if rotDiff < 0.01 {
		t.Error("RoPE forward didn't change Q values")
	}

	// Backward RoPE (in-place, should undo the rotation)
	training.RoPEBackward(Q, K, headDim, numHeads, numKVHeads, seqLen, 0, ropeDim, float64(theta), ropeNeox)
	gpuBackend.Sync()

	// Read back
	qAfter := downloadF32(gpuBackend, Q, len(qData))
	kAfter := downloadF32(gpuBackend, K, len(kData))

	// Compare against original
	var qMaxErr, kMaxErr float64
	for i := range qOrig {
		diff := math.Abs(float64(qOrig[i] - qAfter[i]))
		if diff > qMaxErr {
			qMaxErr = diff
		}
	}
	for i := range kOrig {
		diff := math.Abs(float64(kOrig[i] - kAfter[i]))
		if diff > kMaxErr {
			kMaxErr = diff
		}
	}

	t.Logf("RoPE forward+backward Q maxErr=%.8f K maxErr=%.8f", qMaxErr, kMaxErr)
	if qMaxErr > 1e-5 {
		t.Errorf("Q: RoPE backward is not inverse of forward (maxErr=%.8f)", qMaxErr)
		for i := 0; i < min(16, len(qOrig)); i++ {
			if math.Abs(float64(qOrig[i]-qAfter[i])) > 1e-6 {
				t.Logf("  Q[%d] orig=%.6f after=%.6f rotated=%.6f", i, qOrig[i], qAfter[i], qRotated[i])
			}
		}
	}
	if kMaxErr > 1e-5 {
		t.Errorf("K: RoPE backward is not inverse of forward (maxErr=%.8f)", kMaxErr)
	}
}
