//go:build metal

package metal

import (
	"vexel/inference/backend/cpu"
	"vexel/inference/tensor"
)

// metalBackend implements the Backend interface for Apple Metal.
type metalBackend struct {
	deviceID int
}

// NewBackend creates a new Metal backend for the specified device index.
func NewBackend(deviceID int) (cpu.Backend, error) {
	// TODO: Initialize Metal device (MTLCreateSystemDefaultDevice)
	// For now, we return the struct to satisfy the interface.
	return &metalBackend{
		deviceID: deviceID,
	}, nil
}

// CreateStream creates a new Metal command queue (analogous to a stream).
func (b *metalBackend) CreateStream() (interface{}, error) {
	// TODO: Create MTLCommandQueue
	// Return a dummy placeholder
	return struct{}{}, nil
}

// Device returns the Metal device description.
func (b *metalBackend) Device() tensor.Device {
	return tensor.NewDevice(tensor.Metal, b.deviceID)
}
