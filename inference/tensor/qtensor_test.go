package tensor_test

import (
	"testing"
	"vexel/inference/tensor"
)

func TestQuantizedTensor(t *testing.T) {
	// Setup base tensor
	shape := tensor.NewShape(10, 10)
	dtype := tensor.Uint8 // Typically quantized data is uint8 blocks
	ptr := tensor.NewDevicePtr(tensor.CPU, 0x1234)
	base := tensor.NewTensor(shape, dtype, ptr)

	// Create quantized tensor
	profile := tensor.Q8_0
	qt := tensor.NewQuantizedTensor(base, profile)

	// Verify delegation and specific fields
	if !qt.Tensor().Shape().Equal(shape) {
		t.Error("QuantizedTensor should delegate Shape() to base tensor")
	}

	if qt.Profile() != profile {
		t.Errorf("QuantizedTensor.Profile() = %v, want %v", qt.Profile(), profile)
	}
}
