//go:build metal

package metal

import (
	"fmt"
	"vexel/inference/tensor"
)

// HostToDevice copies data from host memory to device memory.
func (b *metalBackend) HostToDevice(dst tensor.DevicePtr, src []byte) error {
	if dst.Location() != tensor.Metal {
		return fmt.Errorf("destination pointer is not on Metal device")
	}
	if len(src) == 0 {
		return nil
	}

	// TODO: Implement actual memcpy (via MTLBuffer.contents())
	// Metal uses unified memory (mostly), so this might be a direct copy or a managed buffer copy.
	return nil
}

// DeviceToHost copies data from device memory to host memory.
func (b *metalBackend) DeviceToHost(dst []byte, src tensor.DevicePtr) error {
	if src.Location() != tensor.Metal {
		return fmt.Errorf("source pointer is not on Metal device")
	}
	if len(dst) == 0 {
		return nil
	}

	// TODO: Implement actual memcpy
	return nil
}
