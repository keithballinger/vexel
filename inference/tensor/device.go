package tensor

// DevicePtr represents a pointer to memory on a specific device.
// It is an abstraction to handle memory addresses across CPU and accelerators (CUDA, Metal).
type DevicePtr struct {
	addr uintptr
	loc  Location
}

// NewDevicePtr creates a new DevicePtr.
func NewDevicePtr(loc Location, addr uintptr) DevicePtr {
	return DevicePtr{
		addr: addr,
		loc:  loc,
	}
}

// Addr returns the raw memory address.
func (p DevicePtr) Addr() uintptr {
	return p.addr
}

// Location returns the location of the memory.
func (p DevicePtr) Location() Location {
	return p.loc
}

// IsNil returns true if the pointer address is 0 (null).
func (p DevicePtr) IsNil() bool {
	return p.addr == 0
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
