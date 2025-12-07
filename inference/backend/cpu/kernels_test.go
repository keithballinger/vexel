package cpu_test

import (
	"testing"
	"vexel/inference/backend/cpu"
)

func TestMatmul(t *testing.T) {
	// A [2x3]
	aData := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	// B [3x2]
	bData := []float32{
		7, 8,
		9, 1,
		2, 3,
	}
	
	// Expected C [2x2]
	// Row 0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
	//        1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
	// Row 1: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
	//        4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
	expected := []float32{31, 19, 85, 55}

	// Create tensors
	// Ideally we use a helper to create tensors with data.
	// Since Tensor struct is complex (DevicePtr, Shape, etc.), we need to mock it or use a helper.
	// For this test, we might test the *kernel function* directly if exposed, 
	// or the Backend method if it exposes `Matmul`.
	
	// The CPU backend likely has a method `Matmul(a, b, out Tensor)`.
	// Let's assume we can call an internal function or the backend exposes it.
	
	// Since `backend/cpu` is the package under test, we can access internals if we are in `cpu` package,
	// but we are in `cpu_test`.
	
	// Let's verify the `cpuBackend` has a `Matmul` method (or similar execution capability).
	b := cpu.NewBackend()
	
	ops, ok := b.(interface {
		Matmul(a, b, out []float32, m, n, k int)
	})
	
	if !ok {
		t.Fatal("Backend does not expose Matmul for testing")
	}
	
	cData := make([]float32, 4)
	ops.Matmul(aData, bData, cData, 2, 2, 3)
	
	for i, v := range cData {
		if v != expected[i] {
			t.Errorf("Index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}
