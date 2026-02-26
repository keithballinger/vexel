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

// metalAlignment is the optimal buffer offset alignment for Metal GPU access.
// Metal requires buffer offsets to be aligned to 256 bytes for optimal performance.
const metalAlignment = 256

// ScratchAllocator implements a bump allocator backed by a single pre-allocated
// MTLBuffer. Instead of creating new MTLBuffer objects for each temporary tensor,
// it sub-allocates regions from one large buffer using offsets.
//
// This eliminates per-allocation overhead from [device newBufferWithLength:] and
// reduces the total number of MTLBuffer objects tracked by the Metal runtime.
//
// Usage:
//
//	sa, _ := NewScratchAllocator(backend, scratchBytes)
//	ptr := sa.Alloc(4096)  // returns DevicePtr with base buffer + offset
//	// ... use ptr in kernel dispatch ...
//	sa.Reset()              // reset for next forward pass
type ScratchAllocator struct {
	baseBuf  unsafe.Pointer // The single pre-allocated MTLBuffer
	capacity int            // Total buffer size in bytes
	offset   int            // Current bump pointer offset
}

// NewScratchAllocator creates a scratch allocator backed by a single MTLBuffer
// of the given capacity. The buffer uses MTLResourceStorageModeShared (unified memory).
func NewScratchAllocator(b *Backend, capacity int) (*ScratchAllocator, error) {
	if capacity <= 0 {
		return nil, fmt.Errorf("scratch capacity must be positive, got %d", capacity)
	}

	buf := C.metal_alloc_buffer(b.device, C.size_t(capacity))
	if buf == nil {
		return nil, fmt.Errorf("failed to allocate %d byte Metal scratch buffer", capacity)
	}

	return &ScratchAllocator{
		baseBuf:  buf,
		capacity: capacity,
		offset:   0,
	}, nil
}

// Alloc sub-allocates a region of the given size from the scratch buffer.
// Returns a DevicePtr with the base MTLBuffer handle and an offset.
// The offset is aligned to metalAlignment (256 bytes) for optimal GPU access.
// Returns a nil DevicePtr if the scratch buffer is exhausted.
func (sa *ScratchAllocator) Alloc(bytes int) tensor.DevicePtr {
	// Align the current offset up to metalAlignment
	aligned := (sa.offset + metalAlignment - 1) & ^(metalAlignment - 1)

	if aligned+bytes > sa.capacity {
		return tensor.DevicePtr{}
	}

	sa.offset = aligned + bytes

	return tensor.NewDevicePtrWithOffset(
		tensor.Metal,
		uintptr(sa.baseBuf),
		aligned,
	)
}

// Reset resets the bump pointer to 0, making the entire scratch buffer
// available for reuse. Call this at the start of each forward pass.
func (sa *ScratchAllocator) Reset() {
	sa.offset = 0
}

// Capacity returns the total scratch buffer size in bytes.
func (sa *ScratchAllocator) Capacity() int {
	return sa.capacity
}

// Used returns the number of bytes currently allocated (including alignment padding).
func (sa *ScratchAllocator) Used() int {
	return sa.offset
}

// BaseBuffer returns the underlying MTLBuffer pointer.
// Used by kernel dispatch to bind the shared buffer with per-tensor offsets.
func (sa *ScratchAllocator) BaseBuffer() unsafe.Pointer {
	return sa.baseBuf
}

// WriteAt writes data to the scratch buffer at the given sub-allocated region.
// This works because Metal shared buffers map to CPU-accessible unified memory.
func (sa *ScratchAllocator) WriteAt(ptr tensor.DevicePtr, data []byte) {
	if ptr.IsNil() || len(data) == 0 {
		return
	}
	// Get the base buffer's CPU-accessible contents pointer and add the offset
	contents := C.metal_buffer_contents(sa.baseBuf)
	if contents == nil {
		return
	}
	dst := unsafe.Add(contents, ptr.Offset())
	copy((*[1 << 30]byte)(dst)[:len(data)], data)
}

// ReadAt reads data from the scratch buffer at the given sub-allocated region.
func (sa *ScratchAllocator) ReadAt(ptr tensor.DevicePtr, size int) []byte {
	if ptr.IsNil() || size <= 0 {
		return nil
	}
	contents := C.metal_buffer_contents(sa.baseBuf)
	if contents == nil {
		return nil
	}
	src := unsafe.Add(contents, ptr.Offset())
	result := make([]byte, size)
	copy(result, (*[1 << 30]byte)(src)[:size])
	return result
}

// Release frees the underlying MTLBuffer.
func (sa *ScratchAllocator) Release() {
	if sa.baseBuf != nil {
		C.metal_release(sa.baseBuf)
		sa.baseBuf = nil
	}
}
