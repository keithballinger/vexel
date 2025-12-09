//go:build metal && darwin && cgo

package metal

/*
#include "metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"unsafe"

	"vexel/inference/tensor"
)

// HostToDevice copies data from host memory to device memory.
func (b *Backend) HostToDevice(dst tensor.DevicePtr, src []byte) error {
	if dst.Location() != tensor.Metal {
		return fmt.Errorf("destination pointer is not on Metal device")
	}
	if len(src) == 0 {
		return nil
	}

	// Metal uses shared memory, so we can copy directly
	C.metal_copy_to_buffer(unsafe.Pointer(dst.Addr()), unsafe.Pointer(&src[0]), C.size_t(len(src)))
	return nil
}

// DeviceToHost copies data from device memory to host memory.
func (b *Backend) DeviceToHost(dst []byte, src tensor.DevicePtr) error {
	if src.Location() != tensor.Metal {
		return fmt.Errorf("source pointer is not on Metal device")
	}
	if len(dst) == 0 {
		return nil
	}

	C.metal_copy_from_buffer(unsafe.Pointer(&dst[0]), unsafe.Pointer(src.Addr()), C.size_t(len(dst)))
	return nil
}
