//go:build metal

package metal_test

import (
	"testing"
	"vexel/inference/backend/metal"
	"vexel/inference/tensor"
)

func TestMemoryTransfer(t *testing.T) {
	b, _ := metal.NewBackend(0)

	// Verify Backend implements memory transfer methods
	memOps, ok := b.(interface {
		HostToDevice(dst tensor.DevicePtr, src []byte) error
		DeviceToHost(dst []byte, src tensor.DevicePtr) error
	})

	if !ok {
		t.Fatal("Backend does not implement HostToDevice/DeviceToHost")
	}

	// Mock DevicePtr
	devPtr := tensor.NewDevicePtr(tensor.Metal, 0x1000)
	data := []byte{1, 2, 3, 4}

	// Test HostToDevice
	err := memOps.HostToDevice(devPtr, data)
	if err != nil {
		t.Errorf("HostToDevice failed: %v", err)
	}

	// Test DeviceToHost
	readBack := make([]byte, len(data))
	err = memOps.DeviceToHost(readBack, devPtr)
	if err != nil {
		t.Errorf("DeviceToHost failed: %v", err)
	}
}
