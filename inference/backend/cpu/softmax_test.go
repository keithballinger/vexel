package cpu_test

import (
	"testing"
	"vexel/inference/backend/cpu"
)

func TestSoftmax(t *testing.T) {
	// Input: [1.0, 2.0, 3.0]
	// Exp: [2.718, 7.389, 20.085]
	// Sum: 30.192
	// Softmax: [0.090, 0.244, 0.665]
	
	input := []float32{1.0, 2.0, 3.0}
	out := make([]float32, 3)
	
	b := cpu.NewBackend()
	ops, ok := b.(interface {
		Softmax(x, out []float32, rows, cols int)
	})
	if !ok {
		t.Fatal("Backend does not expose Softmax")
	}
	
	ops.Softmax(input, out, 1, 3)
	
	expected := []float32{0.09003, 0.24473, 0.66524}
	
	for i, v := range out {
		if abs(v-expected[i]) > 1e-4 {
			t.Errorf("Index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}
