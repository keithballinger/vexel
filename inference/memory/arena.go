package memory

import (
	"fmt"
	"unsafe"
	"vexel/inference/tensor"
)

// Arena is a linear memory allocator.
type Arena struct {
	loc        tensor.Location
	kind       ArenaKind
	basePtr    uintptr
	size       int
	offset     int
	// For CPU simulation, we keep the slice alive
	cpuBuffer  []byte
}

// NewArena creates a new memory arena.
func NewArena(loc tensor.Location, size int, kind ArenaKind) *Arena {
	var basePtr uintptr
	var cpuBuffer []byte

	if loc == tensor.CPU {
		cpuBuffer = make([]byte, size)
		basePtr = uintptr(unsafe.Pointer(&cpuBuffer[0]))
	} else {
		// TODO: Implement GPU allocation
		// For now, we panic or use a mock address
		basePtr = 0xBADF00D // Placeholder
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

// Alloc allocates memory from the arena.
func (a *Arena) Alloc(size int) (tensor.DevicePtr, error) {
	if a.offset+size > a.size {
		return tensor.DevicePtr{}, fmt.Errorf("OOM: requested %d bytes, only %d remaining", size, a.size-a.offset)
	}

	addr := a.basePtr + uintptr(a.offset)
	a.offset += size

	return tensor.NewDevicePtr(a.loc, addr), nil
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
