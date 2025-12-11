package gguf

import (
	"encoding/binary"
	"math"
)

// DequantizeQ4_0 converts Q4_0 quantized data to float32.
// Q4_0 format: 32 4-bit weights per block with one f16 scale.
// Block layout: [scale:f16][16 bytes of packed 4-bit weights]
// Total: 18 bytes per 32 elements.
//
// Nibble ordering follows llama.cpp convention:
// - Low nibbles (bits 0-3) go to positions 0..15
// - High nibbles (bits 4-7) go to positions 16..31
func DequantizeQ4_0(data []byte, numElements int) []float32 {
	const blockSize = 32
	const bytesPerBlock = 18 // 2 (scale) + 16 (32 nibbles)

	numBlocks := (numElements + blockSize - 1) / blockSize
	result := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock
		if blockOffset+bytesPerBlock > len(data) {
			break
		}

		// Read f16 scale and convert to f32
		scaleU16 := binary.LittleEndian.Uint16(data[blockOffset:])
		scale := float16ToFloat32(scaleU16)

		// Unpack 32 4-bit values from 16 bytes
		// Low nibbles go to first half (0..15), high nibbles to second half (16..31)
		for i := 0; i < 16; i++ {
			byteVal := data[blockOffset+2+i]

			// Low nibble -> position i (first half of block)
			idx := b*blockSize + i
			if idx < numElements {
				q := int(byteVal & 0x0F)
				// Q4_0 uses signed representation: subtract 8 to center around 0
				result[idx] = scale * float32(q-8)
			}

			// High nibble -> position i + 16 (second half of block)
			idx = b*blockSize + i + 16
			if idx < numElements {
				q := int((byteVal >> 4) & 0x0F)
				result[idx] = scale * float32(q-8)
			}
		}
	}

	return result
}

// DequantizeQ8_0 converts Q8_0 quantized data to float32.
// Q8_0 format: 32 8-bit weights per block with one f16 scale.
// Block layout: [scale:f16][32 int8 weights]
// Total: 34 bytes per 32 elements.
func DequantizeQ8_0(data []byte, numElements int) []float32 {
	const blockSize = 32
	const bytesPerBlock = 34 // 2 (scale) + 32 (int8s)

	numBlocks := (numElements + blockSize - 1) / blockSize
	result := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock
		if blockOffset+bytesPerBlock > len(data) {
			break
		}

		// Read f16 scale and convert to f32
		scaleU16 := binary.LittleEndian.Uint16(data[blockOffset:])
		scale := float16ToFloat32(scaleU16)

		// Read 32 int8 values
		for i := 0; i < 32; i++ {
			idx := b*blockSize + i
			if idx < numElements {
				q := int8(data[blockOffset+2+i])
				result[idx] = scale * float32(q)
			}
		}
	}

	return result
}

// DequantizeF16 converts F16 data to float32.
func DequantizeF16(data []byte, numElements int) []float32 {
	result := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		offset := i * 2
		if offset+2 > len(data) {
			break
		}
		u16 := binary.LittleEndian.Uint16(data[offset:])
		result[i] = float16ToFloat32(u16)
	}
	return result
}

// DequantizeBF16 converts BF16 data to float32.
// BF16 is just the upper 16 bits of float32.
func DequantizeBF16(data []byte, numElements int) []float32 {
	result := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		offset := i * 2
		if offset+2 > len(data) {
			break
		}
		// BF16 is upper 16 bits of f32, so shift left by 16
		u16 := binary.LittleEndian.Uint16(data[offset:])
		u32 := uint32(u16) << 16
		result[i] = math.Float32frombits(u32)
	}
	return result
}

// float16ToFloat32 converts an IEEE 754 half-precision float to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 1)
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	var f32 uint32

	if exp == 0 {
		if mant == 0 {
			// Zero
			f32 = sign << 31
		} else {
			// Denormalized number
			exp = 127 - 14
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF
			f32 = (sign << 31) | ((exp) << 23) | (mant << 13)
		}
	} else if exp == 0x1F {
		// Inf or NaN
		f32 = (sign << 31) | (0xFF << 23) | (mant << 13)
	} else {
		// Normalized number
		f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}

	return math.Float32frombits(f32)
}

// DequantizeQ6_K converts Q6_K quantized data to float32.
// Q6_K format: 256 elements per block with complex structure.
// Block layout (210 bytes):
//   - ql[128]: lower 4 bits of 6-bit quants
//   - qh[64]: upper 2 bits of 6-bit quants
//   - scales[16]: 8-bit scales for 16-element super-blocks
//   - d: f16 global scale
//
// Implementation follows llama.cpp's dequantize_row_q6_K
func DequantizeQ6_K(data []byte, numElements int) []float32 {
	const blockSize = 256
	const bytesPerBlock = 210

	numBlocks := (numElements + blockSize - 1) / blockSize
	result := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock
		if blockOffset+bytesPerBlock > len(data) {
			break
		}

		// Parse block structure
		ql := data[blockOffset : blockOffset+128]
		qh := data[blockOffset+128 : blockOffset+192]
		scales := data[blockOffset+192 : blockOffset+208]
		dU16 := binary.LittleEndian.Uint16(data[blockOffset+208:])
		d := float16ToFloat32(dU16)

		// Dequantize following llama.cpp's layout
		// Process in two 128-element chunks
		for n := 0; n < 2; n++ {
			qlOff := n * 64
			qhOff := n * 32
			scOff := n * 8
			yOff := b*blockSize + n*128

			for l := 0; l < 32; l++ {
				is := l / 16

				// Extract 4 6-bit values from ql and qh
				q1 := int8((ql[qlOff+l]&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32
				q2 := int8((ql[qlOff+l+32]&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32
				q3 := int8((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32

				sc0 := float32(int8(scales[scOff+is+0]))
				sc2 := float32(int8(scales[scOff+is+2]))
				sc4 := float32(int8(scales[scOff+is+4]))
				sc6 := float32(int8(scales[scOff+is+6]))

				if yOff+l < numElements {
					result[yOff+l] = d * sc0 * float32(q1)
				}
				if yOff+l+32 < numElements {
					result[yOff+l+32] = d * sc2 * float32(q2)
				}
				if yOff+l+64 < numElements {
					result[yOff+l+64] = d * sc4 * float32(q3)
				}
				if yOff+l+96 < numElements {
					result[yOff+l+96] = d * sc6 * float32(q4)
				}
			}
		}
	}

	return result
}

// DequantizeQ4_K converts Q4_K quantized data to float32.
// Q4_K format: 256 elements per block (8 sub-blocks of 32 elements).
// Block layout (144 bytes):
//   - d (2 bytes): fp16 super-block scale for quantized scales
//   - dmin (2 bytes): fp16 super-block scale for quantized mins
//   - scales[12] (12 bytes): 6-bit quantized scales/mins packed
//   - qs[128] (128 bytes): 4-bit quantized values
//
// Implementation follows llama.cpp's dequantize_row_q4_K exactly
func DequantizeQ4_K(data []byte, numElements int) []float32 {
	const blockSize = 256
	const bytesPerBlock = 144

	numBlocks := (numElements + blockSize - 1) / blockSize
	result := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock
		if blockOffset+bytesPerBlock > len(data) {
			break
		}

		// Parse block header
		dU16 := binary.LittleEndian.Uint16(data[blockOffset:])
		dminU16 := binary.LittleEndian.Uint16(data[blockOffset+2:])
		d := float16ToFloat32(dU16)
		dmin := float16ToFloat32(dminU16)

		// Scales are in bytes 4-15 (12 bytes total)
		scalesData := data[blockOffset+4 : blockOffset+16]
		qs := data[blockOffset+16 : blockOffset+144]

		// Process 4 groups of 64 elements (using is=0,2,4,6 for scale indices)
		qOffset := 0
		is := 0
		for j := 0; j < blockSize; j += 64 {
			// Get scale and min for first 32 elements in this group
			sc1, m1 := getScaleMinK4(is+0, scalesData)
			d1 := d * float32(sc1)
			dm1 := dmin * float32(m1)

			// Get scale and min for second 32 elements in this group
			sc2, m2 := getScaleMinK4(is+1, scalesData)
			d2 := d * float32(sc2)
			dm2 := dmin * float32(m2)

			// Process 32 bytes containing 64 elements
			for l := 0; l < 32; l++ {
				idx1 := b*blockSize + j + l
				idx2 := b*blockSize + j + l + 32
				if idx1 < numElements {
					result[idx1] = d1*float32(qs[qOffset+l]&0x0F) - dm1
				}
				if idx2 < numElements {
					result[idx2] = d2*float32(qs[qOffset+l]>>4) - dm2
				}
			}
			qOffset += 32
			is += 2
		}
	}

	return result
}

// getScaleMinK4 extracts a 6-bit scale and min from the packed scales array.
// This matches llama.cpp's get_scale_min_k4 function exactly.
func getScaleMinK4(j int, q []byte) (uint8, uint8) {
	if j < 4 {
		// First 4 sub-blocks: both scale and min from lower 6 bits
		return q[j] & 63, q[j+4] & 63
	}
	// Last 4 sub-blocks: scale and min use bits from bytes 8-11 combined with upper bits
	scale := (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
	min := (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	return scale, min
}

// Dequantize converts quantized tensor data to float32 based on tensor type.
func Dequantize(data []byte, tensorType TensorType, numElements int) []float32 {
	switch tensorType {
	case TensorTypeF32:
		// Already float32, just reinterpret
		result := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			offset := i * 4
			if offset+4 > len(data) {
				break
			}
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
		}
		return result
	case TensorTypeF16:
		return DequantizeF16(data, numElements)
	case TensorTypeBF16:
		return DequantizeBF16(data, numElements)
	case TensorTypeQ4_0:
		return DequantizeQ4_0(data, numElements)
	case TensorTypeQ8_0:
		return DequantizeQ8_0(data, numElements)
	case TensorTypeQ6_K:
		return DequantizeQ6_K(data, numElements)
	case TensorTypeQ4_K:
		return DequantizeQ4_K(data, numElements)
	default:
		// Unsupported type - return zeros
		return make([]float32, numElements)
	}
}
