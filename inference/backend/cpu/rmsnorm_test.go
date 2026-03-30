package cpu_test

import (
	"testing"
	"vexel/inference/backend/cpu"
)

func TestRMSNorm(t *testing.T) {
	// Input: [2, 4]
	input := []float32{
		1, 2, 3, 4,
		2, 2, 2, 2,
	}
	// Weight: [4]
	weight := []float32{1, 1, 1, 1}

	// Output container
	out := make([]float32, 8)

	b := cpu.NewBackend()
	ops, ok := b.(interface {
		RMSNorm(x, weight, out []float32, rows, cols int, eps float32)
	})
	if !ok {
		t.Fatal("Backend does not expose RMSNorm")
	}

	ops.RMSNorm(input, weight, out, 2, 4, 1e-5)

	// Verification
	// Row 0: 1, 2, 3, 4
	// Squares: 1, 4, 9, 16 -> Sum = 30
	// Mean = 30 / 4 = 7.5
	// RMS = sqrt(7.5 + 1e-5) ≈ 2.7386
	// Out[0] = 1 / 2.7386 ≈ 0.365
	// Out[3] = 4 / 2.7386 ≈ 1.460

	// Just check first element roughly
	expected0 := float32(1.0 / 2.73861278)
	if abs(out[0]-expected0) > 1e-4 {
		t.Errorf("Expected %f, got %f", expected0, out[0])
	}
}

func abs(a float32) float32 {
	if a < 0 {
		return -a
	}
	return a
}
