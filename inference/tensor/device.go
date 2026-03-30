package tensor

// DevicePtr represents a pointer to memory on a specific device.
// It is an abstraction to handle memory addresses across CPU and accelerators (CUDA, Metal).
//
// For CPU: addr is a raw memory pointer, offset is always 0 (pointer arithmetic is done on addr)
// For GPU: addr is the buffer handle (e.g., MTLBuffer*), offset is the position within the buffer
type DevicePtr struct {
	addr   uintptr // For CPU: raw pointer. For GPU: buffer handle
	offset int     // Offset within the buffer (used for GPU sub-allocations)
	loc    Location
}

// NewDevicePtr creates a new DevicePtr with offset 0.
func NewDevicePtr(loc Location, addr uintptr) DevicePtr {
	return DevicePtr{
		addr:   addr,
		offset: 0,
		loc:    loc,
	}
}

// NewDevicePtrWithOffset creates a new DevicePtr with a specific offset.
// For GPU backends, this allows sub-allocations within a buffer.
func NewDevicePtrWithOffset(loc Location, addr uintptr, offset int) DevicePtr {
	return DevicePtr{
		addr:   addr,
		offset: offset,
		loc:    loc,
	}
}

// Addr returns the base memory address/buffer handle.
// For CPU, this includes any offset (raw pointer arithmetic).
// For GPU, this returns the buffer handle without offset.
func (p DevicePtr) Addr() uintptr {
	if p.loc == CPU {
		return p.addr + uintptr(p.offset)
	}
	return p.addr
}

// BaseAddr returns the base buffer address without any offset.
func (p DevicePtr) BaseAddr() uintptr {
	return p.addr
}

// Offset returns the offset within the buffer in bytes.
func (p DevicePtr) Offset() int {
	return p.offset
}

// Location returns the location of the memory.
func (p DevicePtr) Location() Location {
	return p.loc
}

// IsNil returns true if the pointer address is 0 (null).
func (p DevicePtr) IsNil() bool {
	return p.addr == 0
}

// DevicePtrOffset returns a new DevicePtr offset by the given number of bytes.
// For CPU: adds to the raw pointer
// For GPU: adds to the offset field (keeping the same buffer handle)
func DevicePtrOffset(base DevicePtr, offsetBytes uintptr) DevicePtr {
	if base.loc == CPU {
		return DevicePtr{
			addr:   base.addr + offsetBytes,
			offset: 0,
			loc:    base.loc,
		}
	}
	// For GPU, keep the same buffer handle and add to offset
	return DevicePtr{
		addr:   base.addr,
		offset: base.offset + int(offsetBytes),
		loc:    base.loc,
	}
}

// Device represents a physical compute device.
type Device struct {
	Location Location
	Index    int // Device index (e.g., 0 for cuda:0, 1 for cuda:1)
}

// NewDevice creates a new Device.
func NewDevice(loc Location, index int) Device {
	return Device{
		Location: loc,
		Index:    index,
	}
}
