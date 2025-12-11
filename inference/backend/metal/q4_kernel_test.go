//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"math"
	"testing"
)

// Q4_0 format constants
const (
	Q4BlockSize     = 32  // Elements per Q4_0 block
	Q4BytesPerBlock = 18  // 2 (f16 scale) + 16 (32 nibbles)
)

// createQ4_0Block creates a single Q4_0 block with the given scale and values.
// values should have 32 elements, each in range 0-15 (will be stored as q-8 internally).
func createQ4_0Block(scale float32, values []int) []byte {
	if len(values) != 32 {
		panic("Q4_0 block requires exactly 32 values")
	}

	block := make([]byte, Q4BytesPerBlock)

	// Store scale as f16
	scaleU16 := float32ToFloat16(scale)
	binary.LittleEndian.PutUint16(block[0:], scaleU16)

	// Pack nibbles: low nibbles → positions 0-15, high nibbles → positions 16-31
	for i := 0; i < 16; i++ {
		lowNibble := byte(values[i] & 0x0F)
		highNibble := byte(values[i+16] & 0x0F)
		block[2+i] = lowNibble | (highNibble << 4)
	}

	return block
}

// float32ToFloat16 converts a float32 to float16 bits.
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := (bits >> 23) & 0xFF
	mant := bits & 0x7FFFFF

	if exp == 0 {
		// Zero or denormal
		return uint16(sign << 15)
	} else if exp == 0xFF {
		// Inf or NaN
		return uint16((sign << 15) | 0x7C00)
	}

	// Normal number
	newExp := int(exp) - 127 + 15
	if newExp <= 0 {
		// Underflow to zero
		return uint16(sign << 15)
	} else if newExp >= 31 {
		// Overflow to inf
		return uint16((sign << 15) | 0x7C00)
	}

	newMant := mant >> 13
	return uint16((sign << 15) | (uint32(newExp) << 10) | newMant)
}

// dequantizeQ4_0 is a reference CPU implementation for verification.
func dequantizeQ4_0(data []byte, numElements int) []float32 {
	numBlocks := (numElements + Q4BlockSize - 1) / Q4BlockSize
	result := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * Q4BytesPerBlock
		if blockOffset+Q4BytesPerBlock > len(data) {
			break
		}

		// Read f16 scale and convert to f32
		scaleU16 := binary.LittleEndian.Uint16(data[blockOffset:])
		scale := float16ToFloat32CPU(scaleU16)

		// Unpack nibbles
		for i := 0; i < 16; i++ {
			byteVal := data[blockOffset+2+i]

			// Low nibble → position i
			idx := b*Q4BlockSize + i
			if idx < numElements {
				q := int(byteVal & 0x0F)
				result[idx] = scale * float32(q-8)
			}

			// High nibble → position i + 16
			idx = b*Q4BlockSize + i + 16
			if idx < numElements {
				q := int((byteVal >> 4) & 0x0F)
				result[idx] = scale * float32(q-8)
			}
		}
	}

	return result
}

// float16ToFloat32CPU converts f16 bits to f32.
func float16ToFloat32CPU(h uint16) float32 {
	sign := (h >> 15) & 1
	exp := (h >> 10) & 0x1F
	mant := h & 0x3FF

	if exp == 0 {
		if mant == 0 {
			if sign == 1 {
				return float32(math.Copysign(0, -1))
			}
			return 0
		}
		// Denormalized
		f := float32(mant) * (1.0 / 1024.0) * (1.0 / 16384.0)
		if sign == 1 {
			return -f
		}
		return f
	} else if exp == 31 {
		if mant == 0 {
			if sign == 1 {
				return float32(math.Inf(-1))
			}
			return float32(math.Inf(1))
		}
		return float32(math.NaN())
	}

	f := (1.0 + float32(mant)/1024.0) * float32(math.Pow(2, float64(int(exp)-15)))
	if sign == 1 {
		return -f
	}
	return f
}

// cpuMatMulQ4_0 is a CPU reference implementation of Q4_0 matmul.
// Computes C = A @ B^T where A is [M,K] F32, B is [N,K] Q4_0, C is [M,N].
func cpuMatMulQ4_0(a []float32, bQ4 []byte, m, n, k int) []float32 {
	// Dequantize B
	bF32 := dequantizeQ4_0(bQ4, n*k)

	// Compute C = A @ B^T
	c := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += a[i*k+l] * bF32[j*k+l]
			}
			c[i*n+j] = sum
		}
	}
	return c
}

// createQ4_0Matrix creates a Q4_0 matrix with known values for testing.
// Returns [N,K] in Q4_0 format.
func createQ4_0Matrix(n, k int, valueFunc func(row, col int) int) []byte {
	numBlocksPerRow := (k + Q4BlockSize - 1) / Q4BlockSize
	bytesPerRow := numBlocksPerRow * Q4BytesPerBlock
	data := make([]byte, n*bytesPerRow)

	for row := 0; row < n; row++ {
		for blockIdx := 0; blockIdx < numBlocksPerRow; blockIdx++ {
			values := make([]int, 32)
			for i := 0; i < 32; i++ {
				col := blockIdx*Q4BlockSize + i
				if col < k {
					// Offset by 8 since Q4_0 stores (q-8)
					values[i] = valueFunc(row, col) + 8
				} else {
					values[i] = 8 // Zero after offset
				}
			}
			block := createQ4_0Block(1.0, values) // Scale = 1.0 for simplicity
			blockOffset := row*bytesPerRow + blockIdx*Q4BytesPerBlock
			copy(data[blockOffset:], block)
		}
	}
	return data
}

// TestQ4_0MatVec_Simple tests the M=1 (matvec) path with a simple case.
func TestQ4_0MatVec_Simple(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Simple test: K=32 (one block), N=1 (one output)
	// A = [1.0, 1.0, ..., 1.0] (32 ones)
	// B = all zeros in Q4_0 (quant value 8, which becomes 0 after -8)
	// Expected: 0.0

	M, N, K := 1, 1, 32

	// Create input A (all ones)
	a := make([]float32, K)
	for i := range a {
		a[i] = 1.0
	}

	// Create B as Q4_0 with scale=1.0 and all values = 8 (0 after offset)
	values := make([]int, 32)
	for i := range values {
		values[i] = 8 // Results in 0 after (q-8)
	}
	bQ4 := createQ4_0Block(1.0, values)

	// CPU reference
	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)
	b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	// Compare
	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > 0.01 {
			t.Errorf("Index %d: GPU=%f, CPU=%f, diff=%f", i, result[i], expected[i], diff)
		}
	}
	t.Logf("Q4_0 MatVec Simple: GPU=%v, Expected=%v", result, expected)
}

// TestQ4_0MatVec_NonZero tests matvec with non-zero values.
func TestQ4_0MatVec_NonZero(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	M, N, K := 1, 1, 32

	// A = [1.0, 1.0, ..., 1.0]
	a := make([]float32, K)
	for i := range a {
		a[i] = 1.0
	}

	// B = all ones (quant value 9, which becomes 1 after -8)
	// Expected: sum of 32 ones = 32.0
	values := make([]int, 32)
	for i := range values {
		values[i] = 9 // Results in 1 after (q-8)
	}
	bQ4 := createQ4_0Block(1.0, values)

	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)
	t.Logf("Expected result: %v (should be ~32.0)", expected)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)
	b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > 0.1 {
			t.Errorf("Index %d: GPU=%f, CPU=%f, diff=%f", i, result[i], expected[i], diff)
		}
	}
	t.Logf("Q4_0 MatVec NonZero: GPU=%v, Expected=%v", result, expected)
}

// TestQ4_0MatVec_MultipleOutputs tests matvec with multiple output rows (N>1).
func TestQ4_0MatVec_MultipleOutputs(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	M, N, K := 1, 4, 32

	// A = [1.0, 1.0, ..., 1.0]
	a := make([]float32, K)
	for i := range a {
		a[i] = 1.0
	}

	// B rows with different values
	// Row 0: all 0s → sum = 0
	// Row 1: all 1s → sum = 32
	// Row 2: all -1s → sum = -32
	// Row 3: alternating → sum = 0
	bQ4 := make([]byte, N*Q4BytesPerBlock)

	// Row 0: quant = 8 → dequant = 0
	row0Vals := make([]int, 32)
	for i := range row0Vals {
		row0Vals[i] = 8
	}
	copy(bQ4[0:], createQ4_0Block(1.0, row0Vals))

	// Row 1: quant = 9 → dequant = 1
	row1Vals := make([]int, 32)
	for i := range row1Vals {
		row1Vals[i] = 9
	}
	copy(bQ4[Q4BytesPerBlock:], createQ4_0Block(1.0, row1Vals))

	// Row 2: quant = 7 → dequant = -1
	row2Vals := make([]int, 32)
	for i := range row2Vals {
		row2Vals[i] = 7
	}
	copy(bQ4[2*Q4BytesPerBlock:], createQ4_0Block(1.0, row2Vals))

	// Row 3: alternating 7,9 → dequant alternates -1, 1
	row3Vals := make([]int, 32)
	for i := range row3Vals {
		if i%2 == 0 {
			row3Vals[i] = 7
		} else {
			row3Vals[i] = 9
		}
	}
	copy(bQ4[3*Q4BytesPerBlock:], createQ4_0Block(1.0, row3Vals))

	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)
	t.Logf("Expected: %v (should be [0, 32, -32, 0])", expected)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)
	b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > 0.1 {
			t.Errorf("Index %d: GPU=%f, CPU=%f, diff=%f", i, result[i], expected[i], diff)
		}
	}
	t.Logf("Q4_0 MatVec MultiOutput: GPU=%v", result)
}

// TestQ4_0MatVec_LargeK tests matvec with K > 32 (multiple blocks).
func TestQ4_0MatVec_LargeK(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	M, N, K := 1, 1, 128 // 4 blocks

	// A = [1.0, 1.0, ..., 1.0]
	a := make([]float32, K)
	for i := range a {
		a[i] = 1.0
	}

	// B = all ones (quant value 9, scale=1.0)
	// Expected: 128.0
	numBlocks := K / Q4BlockSize
	bQ4 := make([]byte, numBlocks*Q4BytesPerBlock)
	for blk := 0; blk < numBlocks; blk++ {
		values := make([]int, 32)
		for i := range values {
			values[i] = 9
		}
		copy(bQ4[blk*Q4BytesPerBlock:], createQ4_0Block(1.0, values))
	}

	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)
	t.Logf("Expected: %v (should be ~128.0)", expected)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)
	b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > 0.5 {
			t.Errorf("Index %d: GPU=%f, CPU=%f, diff=%f", i, result[i], expected[i], diff)
		}
	}
	t.Logf("Q4_0 MatVec LargeK: GPU=%v", result)
}

// TestQ4_0Batched_Simple tests the batched (M>1) path.
func TestQ4_0Batched_Simple(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	M, N, K := 2, 1, 32

	// A = [[1,1,...], [2,2,...]]
	a := make([]float32, M*K)
	for i := 0; i < K; i++ {
		a[i] = 1.0     // Row 0
		a[K+i] = 2.0   // Row 1
	}

	// B = all ones
	values := make([]int, 32)
	for i := range values {
		values[i] = 9
	}
	bQ4 := createQ4_0Block(1.0, values)

	// Expected: [32.0, 64.0]
	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)
	t.Logf("Expected: %v (should be [32.0, 64.0])", expected)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)
	b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > 0.5 {
			t.Errorf("Index %d: GPU=%f, CPU=%f, diff=%f", i, result[i], expected[i], diff)
		}
	}
	t.Logf("Q4_0 Batched Simple: GPU=%v", result)
}

// TestQ4_0Batched_MultiOutput tests batched path with multiple outputs.
func TestQ4_0Batched_MultiOutput(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	M, N, K := 2, 4, 32

	// A = [[1,1,...], [1,1,...]]
	a := make([]float32, M*K)
	for i := range a {
		a[i] = 1.0
	}

	// Same B as TestQ4_0MatVec_MultipleOutputs
	bQ4 := make([]byte, N*Q4BytesPerBlock)

	row0Vals := make([]int, 32)
	for i := range row0Vals {
		row0Vals[i] = 8
	}
	copy(bQ4[0:], createQ4_0Block(1.0, row0Vals))

	row1Vals := make([]int, 32)
	for i := range row1Vals {
		row1Vals[i] = 9
	}
	copy(bQ4[Q4BytesPerBlock:], createQ4_0Block(1.0, row1Vals))

	row2Vals := make([]int, 32)
	for i := range row2Vals {
		row2Vals[i] = 7
	}
	copy(bQ4[2*Q4BytesPerBlock:], createQ4_0Block(1.0, row2Vals))

	row3Vals := make([]int, 32)
	for i := range row3Vals {
		row3Vals[i] = 8
	}
	copy(bQ4[3*Q4BytesPerBlock:], createQ4_0Block(1.0, row3Vals))

	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)
	t.Logf("Expected: %v", expected)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)
	b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > 0.5 {
			t.Errorf("Index %d: GPU=%f, CPU=%f, diff=%f", i, result[i], expected[i], diff)
		}
	}
	t.Logf("Q4_0 Batched MultiOutput: GPU=%v", result)
}

// TestQ4_0_RealisticSize tests with realistic LLM dimensions.
func TestQ4_0_RealisticSize(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Typical FFN dimension for smaller models
	M, N, K := 1, 2048, 2048

	// Random-ish input
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(i%7-3) * 0.1 // Values in [-0.3, 0.3]
	}

	// Create Q4_0 matrix with pattern
	numBlocksPerRow := (K + Q4BlockSize - 1) / Q4BlockSize
	bytesPerRow := numBlocksPerRow * Q4BytesPerBlock
	bQ4 := make([]byte, N*bytesPerRow)

	for row := 0; row < N; row++ {
		for blk := 0; blk < numBlocksPerRow; blk++ {
			values := make([]int, 32)
			for i := range values {
				// Pattern: value depends on position
				values[i] = (row + blk + i) % 15
			}
			block := createQ4_0Block(0.5, values) // Scale = 0.5
			blockOffset := row*bytesPerRow + blk*Q4BytesPerBlock
			copy(bQ4[blockOffset:], block)
		}
	}

	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)
	b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	// Compare with tolerance
	var maxDiff float64
	var mismatchCount int
	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1.0 { // Allow 1.0 tolerance for accumulated errors
			mismatchCount++
			if mismatchCount <= 5 {
				t.Errorf("Index %d: GPU=%f, CPU=%f, diff=%f", i, result[i], expected[i], diff)
			}
		}
	}

	if mismatchCount > 0 {
		t.Errorf("Total mismatches: %d/%d (max diff: %f)", mismatchCount, len(expected), maxDiff)
	} else {
		t.Logf("Q4_0 Realistic [1,%d]x[%d,%d]: PASS (max diff: %f)", K, N, K, maxDiff)
	}
}

// TestQ4_0_Scale tests that different scales work correctly.
func TestQ4_0_Scale(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	M, N, K := 1, 1, 32

	a := make([]float32, K)
	for i := range a {
		a[i] = 1.0
	}

	testScales := []float32{0.5, 1.0, 2.0, 0.125}

	for _, scale := range testScales {
		values := make([]int, 32)
		for i := range values {
			values[i] = 9 // dequant = 1
		}
		bQ4 := createQ4_0Block(scale, values)

		// Expected: 32 * scale
		expectedVal := 32.0 * scale
		expected := []float32{expectedVal}

		aBuf := b.Alloc(len(a) * 4)
		bBuf := b.Alloc(len(bQ4))
		outBuf := b.Alloc(4)
		defer b.Free(aBuf)
		defer b.Free(bBuf)
		defer b.Free(outBuf)

		b.ToDevice(aBuf, float32ToBytes(a))
		b.ToDevice(bBuf, bQ4)
		b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
		b.Sync()

		resultBytes := make([]byte, 4)
		b.ToHost(resultBytes, outBuf)
		result := bytesToFloat32(resultBytes)

		diff := math.Abs(float64(result[0] - expected[0]))
		if diff > 0.5 {
			t.Errorf("Scale %f: GPU=%f, Expected=%f, diff=%f", scale, result[0], expected[0], diff)
		} else {
			t.Logf("Scale %f: GPU=%f, Expected=%f (PASS)", scale, result[0], expected[0])
		}
	}
}

// TestQ4_0_SimdgroupKernel tests the simdgroup_matrix kernel for M >= 8.
// This kernel is used for prefill operations where batch size is larger.
func TestQ4_0_SimdgroupKernel(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Test with M=16 to ensure simdgroup kernel is used (threshold is M >= 8)
	M, N, K := 16, 2048, 2048

	// Create random input (reproducible)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32((i*7+13)%101-50) * 0.01 // Values in [-0.5, 0.5]
	}

	// Create Q4_0 matrix with pattern
	numBlocksPerRow := (K + Q4BlockSize - 1) / Q4BlockSize
	bytesPerRow := numBlocksPerRow * Q4BytesPerBlock
	bQ4 := make([]byte, N*bytesPerRow)

	for row := 0; row < N; row++ {
		for blk := 0; blk < numBlocksPerRow; blk++ {
			values := make([]int, 32)
			for i := range values {
				values[i] = (row + blk + i) % 15
			}
			block := createQ4_0Block(0.5, values)
			blockOffset := row*bytesPerRow + blk*Q4BytesPerBlock
			copy(bQ4[blockOffset:], block)
		}
	}

	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)
	b.MatMulQ4_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	// Compare with tolerance
	var maxDiff float64
	var mismatchCount int
	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1.0 { // Allow 1.0 tolerance for accumulated errors
			mismatchCount++
			if mismatchCount <= 5 {
				t.Errorf("Index %d [m=%d,n=%d]: GPU=%f, CPU=%f, diff=%f",
					i, i/N, i%N, result[i], expected[i], diff)
			}
		}
	}

	if mismatchCount > 0 {
		t.Errorf("Total mismatches: %d/%d (max diff: %f)", mismatchCount, len(expected), maxDiff)
	} else {
		t.Logf("Q4_0 Simdgroup [%d,%d]x[%d,%d]: PASS (max diff: %f)", M, K, N, K, maxDiff)
	}
}
