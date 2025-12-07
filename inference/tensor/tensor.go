package tensor

// Tensor represents a multi-dimensional array of elements.
// It holds metadata about the shape and data type, and a pointer to the underlying memory.
type Tensor struct {
	shape     Shape
	dtype     DType
	devicePtr DevicePtr
}

// NewTensor creates a new Tensor.
func NewTensor(shape Shape, dtype DType, ptr DevicePtr) Tensor {
	return Tensor{
		shape:     shape,
		dtype:     dtype,
		devicePtr: ptr,
	}
}

// Shape returns the tensor's shape.
func (t Tensor) Shape() Shape {
	return t.shape
}

// DType returns the tensor's data type.
func (t Tensor) DType() DType {
	return t.dtype
}

// DevicePtr returns the pointer to the underlying memory.
func (t Tensor) DevicePtr() DevicePtr {
	return t.devicePtr
}

// Location returns the location of the tensor's memory.
func (t Tensor) Location() Location {
	return t.devicePtr.Location()
}

// NumElements returns the total number of elements in the tensor.
func (t Tensor) NumElements() int {
	return t.shape.NumElements()
}
