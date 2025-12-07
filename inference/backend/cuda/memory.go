//go:build cuda

package cuda

import (
	"fmt"
	"vexel/inference/tensor"
)

// HostToDevice copies data from host memory to device memory.
func (b *cudaBackend) HostToDevice(dst tensor.DevicePtr, src []byte) error {
	if dst.Location() != tensor.CUDA {
		return fmt.Errorf("destination pointer is not on CUDA device")
	}
	if len(src) == 0 {
		return nil
	}

	// TODO: Implement actual cudaMemcpy(HostToDevice)
	// For now, we mock success.
	return nil
}

// DeviceToHost copies data from device memory to host memory.
func (b *cudaBackend) DeviceToHost(dst []byte, src tensor.DevicePtr) error {
	if src.Location() != tensor.CUDA {
		return fmt.Errorf("source pointer is not on CUDA device")
	}
	if len(dst) == 0 {
		return nil
	}

	// TODO: Implement actual cudaMemcpy(DeviceToHost)
	// For now, we mock success.
	return nil
}
