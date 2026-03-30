package cpu_test

import (
	"math"
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/tensor"
)

func TestLayerNorm(t *testing.T) {
	b := cpu.NewCPUBackend()

	rows, cols := 2, 4
	eps := float32(1e-5)

	input := []float32{
		1, 2, 3, 4,
		-1, 0, 1, 2,
	}
	weight := []float32{1, 1, 1, 1}
	bias := []float32{0.1, 0.2, 0.3, 0.4}

	xPtr := b.Alloc(len(input) * 4)
	wPtr := b.Alloc(len(weight) * 4)
	bPtr := b.Alloc(len(bias) * 4)
	outPtr := b.Alloc(len(input) * 4)

	b.ToDevice(xPtr, tensor.Float32ToBytes(input))
	b.ToDevice(wPtr, tensor.Float32ToBytes(weight))
	b.ToDevice(bPtr, tensor.Float32ToBytes(bias))

	b.LayerNorm(xPtr, wPtr, bPtr, outPtr, rows, cols, eps)

	resultBytes := make([]byte, len(input)*4)
	b.ToHost(resultBytes, outPtr)
	result := tensor.BytesToFloat32(resultBytes)

	// Reference check for first row
	// Mean = (1+2+3+4)/4 = 2.5
	// Var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2)/4 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
	// Out[0] = (1 - 2.5) / sqrt(1.25 + 1e-5) * weight[0] + bias[0]
	// Out[0] = -1.5 / 1.11803 * 1.0 + 0.1 = -1.34164 + 0.1 = -1.24164

	expected0 := float32((1.0-2.5)/math.Sqrt(1.25+1e-5)*1.0 + 0.1)
	if math.Abs(float64(result[0]-expected0)) > 1e-5 {
		t.Errorf("LayerNorm row 0 mismatch: got %f, want %f", result[0], expected0)
	}

	// Check row 1
	// Mean = (-1+0+1+2)/4 = 0.5
	// Var = ((-1-0.5)^2 + (0-0.5)^2 + (1-0.5)^2 + (2-0.5)^2)/4 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
	// Out[4] = (-1 - 0.5) / sqrt(1.25 + 1e-5) * weight[0] + bias[0]
	// Out[4] = -1.5 / 1.11803 * 1.0 + 0.1 = -1.24164
	expected4 := float32((-1.0-0.5)/math.Sqrt(1.25+1e-5)*1.0 + 0.1)
	if math.Abs(float64(result[4]-expected4)) > 1e-5 {
		t.Errorf("LayerNorm row 1 mismatch: got %f, want %f", result[4], expected4)
	}
}
