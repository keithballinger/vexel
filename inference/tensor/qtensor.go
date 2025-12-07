package tensor

// QuantizedTensor wraps a Tensor with quantization metadata.
// The underlying Tensor holds the raw block data (scales + weights).
type QuantizedTensor struct {
	base    Tensor
	profile QuantProfile
}

// NewQuantizedTensor creates a new QuantizedTensor.
func NewQuantizedTensor(base Tensor, profile QuantProfile) QuantizedTensor {
	return QuantizedTensor{
		base:    base,
		profile: profile,
	}
}

// Tensor returns the underlying raw tensor.
func (qt QuantizedTensor) Tensor() Tensor {
	return qt.base
}

// Profile returns the quantization profile.
func (qt QuantizedTensor) Profile() QuantProfile {
	return qt.profile
}
