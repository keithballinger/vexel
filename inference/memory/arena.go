package memory

import (
	"fmt"
	"unsafe"
	"vexel/inference/tensor"
)

// AllocFunc is a function that allocates memory on a device.
type AllocFunc func(bytes int) tensor.DevicePtr

// Arena is a linear memory allocator.
// For CPU: uses a single contiguous buffer with offset-based sub-allocation.
// For GPU: allocates individual buffers per request (like candle's approach).
type Arena struct {
	loc        tensor.Location
	kind       ArenaKind
	basePtr    uintptr
	size       int
	offset     int
	// For CPU simulation, we keep the slice alive
	cpuBuffer  []byte
	// For GPU, we store the DevicePtr and allocator function
	devicePtr  tensor.DevicePtr
	allocFunc  AllocFunc
}

// NewArena creates a new memory arena for CPU.
func NewArena(loc tensor.Location, size int, kind ArenaKind) *Arena {
	var basePtr uintptr
	var cpuBuffer []byte

	if loc == tensor.CPU {
		cpuBuffer = make([]byte, size)
		basePtr = uintptr(unsafe.Pointer(&cpuBuffer[0]))
	} else {
		// For GPU without backend, use placeholder (will crash if used)
		basePtr = 0xBADF00D
	}

	return &Arena{
		loc:       loc,
		kind:      kind,
		basePtr:   basePtr,
		size:      size,
		offset:    0,
		cpuBuffer: cpuBuffer,
	}
}

// NewArenaWithBackend creates a new memory arena using the backend for allocation.
// For GPU: stores the allocator function and allocates individual buffers per request.
// This matches candle's approach of not doing sub-allocation for Metal buffers.
func NewArenaWithBackend(loc tensor.Location, size int, kind ArenaKind, alloc AllocFunc) *Arena {
	if loc == tensor.CPU || alloc == nil {
		return NewArena(loc, size, kind)
	}

	// For GPU, we don't pre-allocate a single big buffer.
	// Instead, we store the allocFunc and allocate per-request in Alloc().
	return &Arena{
		loc:       loc,
		kind:      kind,
		basePtr:   0, // Not used for GPU per-allocation mode
		size:      size,
		offset:    0,
		allocFunc: alloc,
	}
}

// Alloc allocates memory from the arena.
// For CPU: sub-allocates from the pre-allocated buffer.
// For GPU: allocates a fresh buffer per request (no sub-allocation).
func (a *Arena) Alloc(size int) (tensor.DevicePtr, error) {
	// Track offset for OOM checking (even for GPU per-allocation mode)
	if a.offset+size > a.size {
		return tensor.DevicePtr{}, fmt.Errorf("OOM: requested %d bytes, only %d remaining", size, a.size-a.offset)
	}
	a.offset += size

	if a.loc == tensor.CPU {
		// For CPU, use raw pointer arithmetic on the pre-allocated buffer
		addr := a.basePtr + uintptr(a.offset-size)
		return tensor.NewDevicePtr(a.loc, addr), nil
	}

	// For GPU, allocate a fresh buffer per request
	// This avoids the complexity of buffer+offset parameters in Metal kernels
	if a.allocFunc != nil {
		return a.allocFunc(size), nil
	}

	return tensor.DevicePtr{}, fmt.Errorf("GPU arena has no allocator function")
}

// Reset resets the arena offset to zero.
func (a *Arena) Reset() {
	a.offset = 0
}

// UsedBytes returns the number of bytes currently allocated.
func (a *Arena) UsedBytes() int {
	return a.offset
}

// TotalBytes returns the total capacity of the arena.
func (a *Arena) TotalBytes() int {
	return a.size
}
