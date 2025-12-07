package tensor

// Shape represents the dimensions of a tensor.
type Shape struct {
	dims []int
}

// NewShape creates a new Shape with the given dimensions.
func NewShape(dims ...int) Shape {
	// Copy dimensions to ensure immutability from the caller's side
	d := make([]int, len(dims))
	copy(d, dims)
	return Shape{dims: d}
}

// NumElements returns the total number of elements in the shape.
func (s Shape) NumElements() int {
	if len(s.dims) == 0 {
		return 1 // Scalar
	}
	n := 1
	for _, d := range s.dims {
		n *= d
	}
	return n
}

// Rank returns the number of dimensions.
func (s Shape) Rank() int {
	return len(s.dims)
}

// StridesRowMajor calculates the strides for a row-major (C-style) memory layout.
func (s Shape) StridesRowMajor() []int {
	rank := len(s.dims)
	if rank == 0 {
		return []int{}
	}

	strides := make([]int, rank)
	stride := 1
	for i := rank - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= s.dims[i]
	}
	return strides
}

// Equal returns true if the other shape has the same dimensions.
func (s Shape) Equal(other Shape) bool {
	if len(s.dims) != len(other.dims) {
		return false
	}
	for i, d := range s.dims {
		if d != other.dims[i] {
			return false
		}
	}
	return true
}

// Dims returns a copy of the dimensions.
func (s Shape) Dims() []int {
	d := make([]int, len(s.dims))
	copy(d, s.dims)
	return d
}
