package tensor

import (
	"unsafe"
)

// ToFloat32Slice returns a slice view of the tensor's memory.
// WARNING: This is unsafe and assumes the tensor is on CPU and stores float32.
// It also assumes the memory is kept alive elsewhere (Arena).
func ToFloat32Slice(t Tensor) []float32 {
	if t.DevicePtr().IsNil() {
		return nil
	}
	
	// Create slice header
	count := t.NumElements()
	var sl = struct {
		addr uintptr
		len  int
		cap  int
	}{t.DevicePtr().Addr(), count, count}

	return *(*[]float32)(unsafe.Pointer(&sl))
}
