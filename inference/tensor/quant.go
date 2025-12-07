package tensor

// QuantProfile represents the quantization scheme used for a tensor.
type QuantProfile int

const (
	// QuantNone represents no quantization (full precision).
	QuantNone QuantProfile = iota
	// Q8_0 represents 8-bit block quantization.
	Q8_0
	// Q4_0 represents 4-bit block quantization.
	Q4_0
)

func (q QuantProfile) String() string {
	switch q {
	case QuantNone:
		return "None"
	case Q8_0:
		return "Q8_0"
	case Q4_0:
		return "Q4_0"
	default:
		return "Unknown"
	}
}
