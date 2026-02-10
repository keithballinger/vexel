package cpu_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/tensor"
)

func TestAddBias(t *testing.T) {
	b := cpu.NewCPUBackend()
	
	rows, cols := 2, 3
	input := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	bias := []float32{0.1, 0.2, 0.3}

	xPtr := b.Alloc(len(input) * 4)
	bPtr := b.Alloc(len(bias) * 4)
	outPtr := b.Alloc(len(input) * 4)
	
	b.ToDevice(xPtr, tensor.Float32ToBytes(input))
	b.ToDevice(bPtr, tensor.Float32ToBytes(bias))

	b.AddBias(xPtr, bPtr, outPtr, rows, cols)
	
	resultBytes := make([]byte, len(input)*4)
	b.ToHost(resultBytes, outPtr)
	result := tensor.BytesToFloat32(resultBytes)

	expected := []float32{
		1.1, 2.2, 3.3,
		4.1, 5.2, 6.3,
	}

	for i := range result {
		if abs(result[i]-expected[i]) > 1e-5 {
			t.Errorf("AddBias mismatch at %d: got %f, want %f", i, result[i], expected[i])
		}
	}
}
