package bf16

import "unsafe"

// ConvertToFP32 converts a byte slice of BF16 values to a slice of float32.
func ConvertToFP32(data []byte) []float32 {
	count := len(data) / 2
	out := make([]float32, count)

	for i := 0; i < count; i++ {
		b0 := data[i*2]
		b1 := data[i*2+1]

		val := uint32(uint16(b1)<<8|uint16(b0)) << 16
		out[i] = *(*float32)(unsafeAddr(&val))
	}
	return out
}

func unsafeAddr(p *uint32) unsafe.Pointer {
	return unsafe.Pointer(p)
}
