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
	// Q6_K represents 6-bit k-quant quantization (used for lm_head).
	Q6_K
	// Q4_K represents 4-bit k-quant quantization (better alignment than Q4_0).
	Q4_K
	// Q5_K represents 5-bit k-quant quantization.
	Q5_K
	// BF16 represents bfloat16 format (not quantized, but stored in reduced precision).
	BF16
)

func (q QuantProfile) String() string {
	switch q {
	case QuantNone:
		return "None"
	case Q8_0:
		return "Q8_0"
	case Q4_0:
		return "Q4_0"
	case Q6_K:
		return "Q6_K"
	case Q4_K:
		return "Q4_K"
	case Q5_K:
		return "Q5_K"
	case BF16:
		return "BF16"
	default:
		return "Unknown"
	}
}
