package cpu_test

import (
	"math"
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/tensor"
)

func TestGELU(t *testing.T) {
	b := cpu.NewCPUBackend()

	input := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	n := len(input)

	xPtr := b.Alloc(n * 4)
	outPtr := b.Alloc(n * 4)

	b.ToDevice(xPtr, tensor.Float32ToBytes(input))
	b.GELU(xPtr, outPtr, n)

	resultBytes := make([]byte, n*4)
	b.ToHost(resultBytes, outPtr)
	result := tensor.BytesToFloat32(resultBytes)

	// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
	// or the tanh approximation used in the code.

	// GELU(0) should be 0
	if math.Abs(float64(result[2])) > 1e-6 {
		t.Errorf("GELU(0) mismatch: got %f, want 0", result[2])
	}

	// GELU(1) ≈ 0.8413
	// GELU(2) ≈ 1.9546
	if result[4] < 1.9 || result[4] > 2.0 {
		t.Errorf("GELU(2) mismatch: got %f, want ~1.9546", result[4])
	}
}
