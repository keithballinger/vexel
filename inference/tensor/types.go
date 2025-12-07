package tensor

// DType represents the data type of a tensor's elements.
type DType int

const (
	Float32 DType = iota
	Float16
	BFloat16
	Int8
	Uint8
)

// SizeBytes returns the size of the data type in bytes.
func (d DType) SizeBytes() int {
	switch d {
	case Float32:
		return 4
	case Float16, BFloat16:
		return 2
	case Int8, Uint8:
		return 1
	default:
		return 0
	}
}

// BitSize returns the size of the data type in bits.
func (d DType) BitSize() int {
	return d.SizeBytes() * 8
}

// Location represents the physical device where a tensor resides.
type Location int

const (
	CPU Location = iota
	CUDA
	Metal
)

func (l Location) String() string {
	switch l {
	case CPU:
		return "CPU"
	case CUDA:
		return "CUDA"
	case Metal:
		return "Metal"
	default:
		return "Unknown"
	}
}
