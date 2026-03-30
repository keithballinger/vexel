package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		name string
		h    uint16
		want float32
	}{
		{"zero", 0x0000, 0.0},
		{"one", 0x3C00, 1.0},
		{"two", 0x4000, 2.0},
		{"half", 0x3800, 0.5},
		{"neg_one", 0xBC00, -1.0},
		{"infinity", 0x7C00, float32(math.Inf(1))},
		{"neg_infinity", 0xFC00, float32(math.Inf(-1))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := float16ToFloat32(tt.h)
			if math.IsInf(float64(tt.want), 0) {
				if !math.IsInf(float64(got), 0) || math.Signbit(float64(got)) != math.Signbit(float64(tt.want)) {
					t.Errorf("float16ToFloat32(0x%04X) = %v, want %v", tt.h, got, tt.want)
				}
			} else if got != tt.want {
				t.Errorf("float16ToFloat32(0x%04X) = %v, want %v", tt.h, got, tt.want)
			}
		})
	}
}

func TestDequantizeQ8_0(t *testing.T) {
	// Create a Q8_0 block with known values
	// Block: scale=1.0 (f16: 0x3C00), then 32 int8 values
	data := make([]byte, 34)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0

	// Set int8 values: 0, 1, 2, ..., 31
	for i := 0; i < 32; i++ {
		data[2+i] = byte(i)
	}

	result := DequantizeQ8_0(data, 32)

	if len(result) != 32 {
		t.Fatalf("DequantizeQ8_0 returned %d elements, want 32", len(result))
	}

	for i := 0; i < 32; i++ {
		want := float32(i) // scale=1.0 * q=i
		if result[i] != want {
			t.Errorf("result[%d] = %v, want %v", i, result[i], want)
		}
	}
}

func TestDequantizeQ8_0WithScale(t *testing.T) {
	// Test with scale = 0.5 (f16: 0x3800)
	data := make([]byte, 34)
	binary.LittleEndian.PutUint16(data[0:], 0x3800) // scale = 0.5

	// Set int8 values: all 2
	for i := 0; i < 32; i++ {
		data[2+i] = 2
	}

	result := DequantizeQ8_0(data, 32)

	for i := 0; i < 32; i++ {
		want := float32(1.0) // scale=0.5 * q=2
		if result[i] != want {
			t.Errorf("result[%d] = %v, want %v", i, result[i], want)
		}
	}
}

func TestDequantizeQ4_0(t *testing.T) {
	// Create a Q4_0 block with known values
	// Block: scale=1.0 (f16: 0x3C00), then 16 bytes (32 nibbles)
	data := make([]byte, 18)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0

	// llama.cpp Q4_0 layout:
	// - Low nibbles of bytes 0-15 go to positions 0-15
	// - High nibbles of bytes 0-15 go to positions 16-31
	// Set byte 0: low=8 (0 centered), high=9 (1 centered)
	data[2] = 0x98 // high=9, low=8
	// Set byte 1: low=10 (2 centered), high=11 (3 centered)
	data[3] = 0xBA // high=11, low=10

	result := DequantizeQ4_0(data, 32)

	if len(result) != 32 {
		t.Fatalf("DequantizeQ4_0 returned %d elements, want 32", len(result))
	}

	// Position 0: low nibble of byte 0 = 8-8 = 0
	if result[0] != 0.0 {
		t.Errorf("result[0] = %v, want 0", result[0])
	}
	// Position 1: low nibble of byte 1 = 10-8 = 2
	if result[1] != 2.0 {
		t.Errorf("result[1] = %v, want 2", result[1])
	}
	// Position 16: high nibble of byte 0 = 9-8 = 1
	if result[16] != 1.0 {
		t.Errorf("result[16] = %v, want 1", result[16])
	}
	// Position 17: high nibble of byte 1 = 11-8 = 3
	if result[17] != 3.0 {
		t.Errorf("result[17] = %v, want 3", result[17])
	}
}

func TestDequantizeF16(t *testing.T) {
	// Create F16 data: [1.0, 2.0, 0.5]
	data := make([]byte, 6)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // 1.0
	binary.LittleEndian.PutUint16(data[2:], 0x4000) // 2.0
	binary.LittleEndian.PutUint16(data[4:], 0x3800) // 0.5

	result := DequantizeF16(data, 3)

	if len(result) != 3 {
		t.Fatalf("DequantizeF16 returned %d elements, want 3", len(result))
	}

	want := []float32{1.0, 2.0, 0.5}
	for i, w := range want {
		if result[i] != w {
			t.Errorf("result[%d] = %v, want %v", i, result[i], w)
		}
	}
}

func TestDequantizeBF16(t *testing.T) {
	// Create BF16 data: bf16 is just upper 16 bits of f32
	// 1.0 in f32 = 0x3F800000, so bf16 = 0x3F80
	data := make([]byte, 4)
	binary.LittleEndian.PutUint16(data[0:], 0x3F80) // 1.0
	binary.LittleEndian.PutUint16(data[2:], 0x4000) // 2.0

	result := DequantizeBF16(data, 2)

	if len(result) != 2 {
		t.Fatalf("DequantizeBF16 returned %d elements, want 2", len(result))
	}

	if result[0] != 1.0 {
		t.Errorf("result[0] = %v, want 1.0", result[0])
	}
	if result[1] != 2.0 {
		t.Errorf("result[1] = %v, want 2.0", result[1])
	}
}

func TestDequantizeF32(t *testing.T) {
	// Create F32 data
	data := make([]byte, 12)
	binary.LittleEndian.PutUint32(data[0:], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(data[4:], math.Float32bits(2.5))
	binary.LittleEndian.PutUint32(data[8:], math.Float32bits(-3.0))

	result := Dequantize(data, TensorTypeF32, 3)

	if len(result) != 3 {
		t.Fatalf("Dequantize returned %d elements, want 3", len(result))
	}

	want := []float32{1.0, 2.5, -3.0}
	for i, w := range want {
		if result[i] != w {
			t.Errorf("result[%d] = %v, want %v", i, result[i], w)
		}
	}
}

func TestDequantizeQ5_0(t *testing.T) {
	// Q5_0 block: 22 bytes = 2 (d) + 4 (qh) + 16 (qs)
	// Test 1: scale=1.0, all q5=16 → dequant = (16-16)*1 = 0
	t.Run("all_zero", func(t *testing.T) {
		data := make([]byte, 22)
		binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0
		// qh: all high bits set → all 5th bits are 1
		binary.LittleEndian.PutUint32(data[2:], 0xFFFFFFFF)
		// qs: all low nibbles 0 → q5 = 0|16 = 16 for all

		result := DequantizeQ5_0(data, 32)
		if len(result) != 32 {
			t.Fatalf("got %d elements, want 32", len(result))
		}
		for i, v := range result {
			if v != 0.0 {
				t.Errorf("result[%d] = %v, want 0.0", i, v)
			}
		}
	})

	// Test 2: scale=1.0, first half q5=8 (dequant=-8), second half q5=24 (dequant=8)
	t.Run("split_halves", func(t *testing.T) {
		data := make([]byte, 22)
		binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0
		// qh: bits 0-15 = 0 (first half high bit 0), bits 16-31 = 1 (second half high bit 1)
		binary.LittleEndian.PutUint32(data[2:], 0xFFFF0000)
		// qs: all bytes = 0x88 → low nibble=8, high nibble=8
		for i := 0; i < 16; i++ {
			data[6+i] = 0x88
		}

		result := DequantizeQ5_0(data, 32)
		// First half (0-15): q=8|0=8, dequant=(8-16)*1=-8
		for i := 0; i < 16; i++ {
			if result[i] != -8.0 {
				t.Errorf("result[%d] = %v, want -8.0", i, result[i])
			}
		}
		// Second half (16-31): q=8|16=24, dequant=(24-16)*1=8
		for i := 16; i < 32; i++ {
			if result[i] != 8.0 {
				t.Errorf("result[%d] = %v, want 8.0", i, result[i])
			}
		}
	})

	// Test 3: scale=0.5, varying values
	t.Run("with_scale", func(t *testing.T) {
		data := make([]byte, 22)
		binary.LittleEndian.PutUint16(data[0:], 0x3800) // scale = 0.5
		// qh: all 0 → no high bits
		binary.LittleEndian.PutUint32(data[2:], 0x00000000)
		// qs[0]: low=15, high=0 → q5=15 (first half), q5=0 (second half)
		data[6] = 0x0F // high nibble=0, low nibble=15

		result := DequantizeQ5_0(data, 32)
		// Position 0: q=15, dequant=(15-16)*0.5 = -0.5
		if result[0] != -0.5 {
			t.Errorf("result[0] = %v, want -0.5", result[0])
		}
		// Position 16: q=0, dequant=(0-16)*0.5 = -8
		if result[16] != -8.0 {
			t.Errorf("result[16] = %v, want -8.0", result[16])
		}
	})
}

func TestDequantizeQ5_1(t *testing.T) {
	// Q5_1 block: 24 bytes = 2 (d) + 2 (m) + 4 (qh) + 16 (qs)
	// Test 1: scale=1.0, min=0, all q5=0 → dequant = 0*1+0 = 0
	t.Run("all_zero", func(t *testing.T) {
		data := make([]byte, 24)
		binary.LittleEndian.PutUint16(data[0:], 0x3C00) // d = 1.0
		binary.LittleEndian.PutUint16(data[2:], 0x0000) // m = 0.0
		binary.LittleEndian.PutUint32(data[4:], 0x00000000)
		// all qs = 0

		result := DequantizeQ5_1(data, 32)
		if len(result) != 32 {
			t.Fatalf("got %d elements, want 32", len(result))
		}
		for i, v := range result {
			if v != 0.0 {
				t.Errorf("result[%d] = %v, want 0.0", i, v)
			}
		}
	})

	// Test 2: scale=1.0, min=10.0, q=0 → dequant = 0*1+10 = 10
	t.Run("min_offset", func(t *testing.T) {
		data := make([]byte, 24)
		binary.LittleEndian.PutUint16(data[0:], 0x3C00) // d = 1.0
		// 10.0 in f16: sign=0, exp=10+15=25=0b11001, mant=0b0100000000 → 0x4900
		binary.LittleEndian.PutUint16(data[2:], 0x4900) // m = 10.0
		binary.LittleEndian.PutUint32(data[4:], 0x00000000)

		result := DequantizeQ5_1(data, 32)
		expected := float16ToFloat32(0x4900) // verify the float16 encoding
		for i := 0; i < 32; i++ {
			if result[i] != expected {
				t.Errorf("result[%d] = %v, want %v (10.0)", i, result[i], expected)
			}
		}
	})

	// Test 3: scale=0.5, min=1.0, first half q5=2 → dequant = 2*0.5+1 = 2
	t.Run("with_scale_and_min", func(t *testing.T) {
		data := make([]byte, 24)
		binary.LittleEndian.PutUint16(data[0:], 0x3800)     // d = 0.5
		binary.LittleEndian.PutUint16(data[2:], 0x3C00)     // m = 1.0
		binary.LittleEndian.PutUint32(data[4:], 0x00000000) // no high bits
		// qs[0]: low nibble = 2 → q5 for position 0 = 2
		data[8] = 0x02

		result := DequantizeQ5_1(data, 32)
		// Position 0: q=2, dequant = 2*0.5 + 1.0 = 2.0
		if result[0] != 2.0 {
			t.Errorf("result[0] = %v, want 2.0", result[0])
		}
	})
}

func TestDequantizeUnsupported(t *testing.T) {
	data := make([]byte, 100)
	result := Dequantize(data, TensorTypeQ4_K, 32)

	// Should return zeros for unsupported types
	if len(result) != 32 {
		t.Fatalf("Dequantize returned %d elements, want 32", len(result))
	}
	for i, v := range result {
		if v != 0 {
			t.Errorf("result[%d] = %v, want 0", i, v)
		}
	}
}

// Benchmark dequantization
func BenchmarkDequantizeQ8_0(b *testing.B) {
	// 4096x4096 matrix worth of blocks
	numElements := 4096 * 4096
	numBlocks := numElements / 32
	data := make([]byte, numBlocks*34)

	// Fill with test data
	for i := 0; i < numBlocks; i++ {
		binary.LittleEndian.PutUint16(data[i*34:], 0x3C00) // scale = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DequantizeQ8_0(data, numElements)
	}
}

func BenchmarkDequantizeQ4_0(b *testing.B) {
	numElements := 4096 * 4096
	numBlocks := numElements / 32
	data := make([]byte, numBlocks*18)

	for i := 0; i < numBlocks; i++ {
		binary.LittleEndian.PutUint16(data[i*18:], 0x3C00)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DequantizeQ4_0(data, numElements)
	}
}
