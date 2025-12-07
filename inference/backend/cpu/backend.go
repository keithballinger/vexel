package cpu

import "vexel/inference/tensor"

// cpuBackend implements the Backend interface for CPU execution.
type cpuBackend struct{}

// NewBackend creates a new CPU backend.
func NewBackend() Backend {
	return &cpuBackend{}
}

// CreateStream returns a dummy stream for CPU (synchronous execution).
func (b *cpuBackend) CreateStream() (interface{}, error) {
	// For CPU, we don't need a real stream object, but we return a
	// placeholder struct so it's not nil, to be safe.
	return struct{}{}, nil
}

// Device returns the CPU device description.
func (b *cpuBackend) Device() tensor.Device {
	return tensor.NewDevice(tensor.CPU, 0)
}
