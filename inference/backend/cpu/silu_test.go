package cpu_test

import (
	"math"
	"testing"
	"vexel/inference/backend/cpu"
)

func TestSiLU(t *testing.T) {
	// Input: [-1, 0, 1]
	input := []float32{-1, 0, 1}
	out := make([]float32, 3)

	// Expected: x * sigmoid(x)
	// -1 * (1 / (1 + exp(1))) = -1 * (1 / 3.718) = -0.2689
	// 0 * 0.5 = 0
	// 1 * (1 / (1 + exp(-1))) = 1 * (1 / 1.367) = 0.7310

	b := cpu.NewBackend()
	ops, ok := b.(interface {
		SiLU(x, out []float32, n int)
	})
	if !ok {
		t.Fatal("Backend does not expose SiLU")
	}

	ops.SiLU(input, out, 3)

	expected := []float32{
		float32(-1.0 / (1.0 + math.Exp(1.0))),
		0.0,
		float32(1.0 / (1.0 + math.Exp(-1.0))),
	}

	for i, v := range out {
		if abs(v-expected[i]) > 1e-4 {
			t.Errorf("Index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}
