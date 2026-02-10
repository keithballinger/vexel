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

// Float32ToBytes converts a float32 slice to a byte slice using unsafe pointers.
func Float32ToBytes(f []float32) []byte {
	if len(f) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}

// BytesToFloat32 converts a byte slice to a float32 slice using unsafe pointers.
func BytesToFloat32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}
