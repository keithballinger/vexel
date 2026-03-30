package tensor_test

import (
	"testing"
	"vexel/inference/tensor"
)

func TestTensor(t *testing.T) {
	// Create components
	shape := tensor.NewShape(2, 3)
	dtype := tensor.Float32
	// For testing, we use a dummy pointer address
	ptrAddr := uintptr(0xDEADBEEF)
	devicePtr := tensor.NewDevicePtr(tensor.CPU, ptrAddr)

	// Test creating a new Tensor
	// Ideally, we'd have a factory method or a direct struct initializer
	// Let's assume a NewTensor function for now
	T := tensor.NewTensor(shape, dtype, devicePtr)

	// Verify fields
	if !T.Shape().Equal(shape) {
		t.Error("Tensor.Shape() does not match input shape")
	}

	if T.DType() != dtype {
		t.Errorf("Tensor.DType() = %v, want %v", T.DType(), dtype)
	}

	if T.DevicePtr().Addr() != ptrAddr {
		t.Errorf("Tensor.DevicePtr().Addr() = %v, want %v", T.DevicePtr().Addr(), ptrAddr)
	}

	// Verify Location convenience method
	if T.Location() != tensor.CPU {
		t.Errorf("Tensor.Location() = %v, want %v", T.Location(), tensor.CPU)
	}

	// Verify NumElements convenience method
	if T.NumElements() != 6 {
		t.Errorf("Tensor.NumElements() = %v, want 6", T.NumElements())
	}
}
