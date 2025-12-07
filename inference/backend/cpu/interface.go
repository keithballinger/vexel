package cpu

import "vexel/inference/tensor"

// Backend represents a compute backend (CPU, CUDA, Metal).
// It manages execution streams and device-specific resources.
type Backend interface {
	// CreateStream creates a new execution stream (or command queue).
	// Returns an opaque handle (interface{}).
	CreateStream() (interface{}, error)

	// Device returns the device associated with this backend.
	Device() tensor.Device
}
