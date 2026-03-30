package tensor_test

import (
	"testing"
	"vexel/inference/tensor"
)

func TestDevicePtr(t *testing.T) {
	// Test creating a DevicePtr
	// We simulate a pointer address using a uintptr
	addr := uintptr(0x12345678)
	loc := tensor.CUDA

	ptr := tensor.NewDevicePtr(loc, addr)

	if ptr.Addr() != addr {
		t.Errorf("DevicePtr.Addr() = %v, want %v", ptr.Addr(), addr)
	}

	if ptr.Location() != loc {
		t.Errorf("DevicePtr.Location() = %v, want %v", ptr.Location(), loc)
	}

	if ptr.IsNil() {
		t.Error("DevicePtr should not be nil")
	}

	// Test nil pointer
	nilPtr := tensor.NewDevicePtr(tensor.CPU, 0)
	if !nilPtr.IsNil() {
		t.Error("DevicePtr with addr 0 should be nil")
	}
}

func TestDevice(t *testing.T) {
	// Test the Device struct which might wrap backend-specific device info
	// For now, it just holds the Location and an ordinal index
	dev := tensor.NewDevice(tensor.CUDA, 0)

	if dev.Location != tensor.CUDA {
		t.Errorf("Device.Location = %v, want %v", dev.Location, tensor.CUDA)
	}
	if dev.Index != 0 {
		t.Errorf("Device.Index = %v, want 0", dev.Index)
	}
}
