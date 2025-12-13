//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"math"
	"testing"
)

// Q6_K format constants
const (
	Q6KBlockSize     = 256 // Elements per Q6_K block
	Q6KBytesPerBlock = 210 // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
)

// createQ6_KBlock creates a single Q6_K block with the given global scale,
// per-superblock scales, and 6-bit values.
func createQ6_KBlock(d float32, scales [16]int8, values [256]int8) []byte {
	block := make([]byte, Q6KBytesPerBlock)

	// Layout:
	// ql[128]: lower 4 bits of 6-bit quants
	// qh[64]: upper 2 bits of 6-bit quants
	// scales[16]: 8-bit signed scales
	// d[2]: f16 global scale

	// values are 6-bit signed: -32 to 31 (stored as 0-63)
	// To encode: store_val = val + 32
	// ql stores lower 4 bits, qh stores upper 2 bits

	// Process in two 128-element halves
	for n := 0; n < 2; n++ {
		qlOff := n * 64
		qhOff := 128 + n*32

		for l := 0; l < 32; l++ {
			// Get the 4 values that share this ql/qh position
			// chunk 0: l, chunk 1: l+32, chunk 2: l+64, chunk 3: l+96
			v1 := int(values[n*128+l]) + 32       // chunk 0
			v2 := int(values[n*128+l+32]) + 32    // chunk 1
			v3 := int(values[n*128+l+64]) + 32    // chunk 2
			v4 := int(values[n*128+l+96]) + 32    // chunk 3

			// ql[l] = (v1 & 0xF) | ((v3 & 0xF) << 4)
			block[qlOff+l] = byte(v1&0xF) | byte((v3&0xF)<<4)
			// ql[l+32] = (v2 & 0xF) | ((v4 & 0xF) << 4)
			block[qlOff+l+32] = byte(v2&0xF) | byte((v4&0xF)<<4)

			// qh[l] encodes upper 2 bits of all 4 values
			qh := byte(0)
			qh |= byte((v1 >> 4) & 3)        // bits 0-1: v1 high
			qh |= byte((v2>>4)&3) << 2       // bits 2-3: v2 high
			qh |= byte((v3>>4)&3) << 4       // bits 4-5: v3 high
			qh |= byte((v4>>4)&3) << 6       // bits 6-7: v4 high
			block[qhOff+l] = qh
		}
	}

	// Store scales
	for i := 0; i < 16; i++ {
		block[192+i] = byte(scales[i])
	}

	// Store d as f16
	dU16 := float32ToFloat16(d)
	binary.LittleEndian.PutUint16(block[208:], dU16)

	return block
}

// dequantizeQ6_KRef is a reference CPU implementation for verification.
func dequantizeQ6_KRef(data []byte, numElements int) []float32 {
	numBlocks := (numElements + Q6KBlockSize - 1) / Q6KBlockSize
	result := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * Q6KBytesPerBlock
		if blockOffset+Q6KBytesPerBlock > len(data) {
			break
		}

		ql := data[blockOffset : blockOffset+128]
		qh := data[blockOffset+128 : blockOffset+192]
		scales := data[blockOffset+192 : blockOffset+208]
		dU16 := binary.LittleEndian.Uint16(data[blockOffset+208:])
		d := float16ToFloat32CPU(dU16)

		// Process in two 128-element halves
		for n := 0; n < 2; n++ {
			qlOff := n * 64
			qhOff := n * 32
			scOff := n * 8
			yOff := b*Q6KBlockSize + n*128

			for l := 0; l < 32; l++ {
				is := l / 16

				// Extract 4 6-bit values
				q1 := int(int8((ql[qlOff+l]&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32)
				q2 := int(int8((ql[qlOff+l+32]&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32)
				q3 := int(int8((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32)
				q4 := int(int8((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32)

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

// cpuMatVecQ6_K computes matvec using CPU dequantization.
func cpuMatVecQ6_K(a []float32, b []byte, m, n, k int) []float32 {
	// a: [m, k] = [1, k]
	// b: [n, k] in Q6_K format
	// out: [m, n] = [1, n]
	numBlocksPerRow := (k + Q6KBlockSize - 1) / Q6KBlockSize
	bytesPerRow := numBlocksPerRow * Q6KBytesPerBlock

	out := make([]float32, m*n)
	for row := 0; row < n; row++ {
		rowData := b[row*bytesPerRow : (row+1)*bytesPerRow]
		bF32 := dequantizeQ6_KRef(rowData, k)
		for col := 0; col < m; col++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += a[col*k+l] * bF32[l]
			}
			out[col*n+row] = sum
		}
	}
	return out
}

// TestQ6_KDequantization verifies the CPU dequantization reference.
func TestQ6_KDequantization(t *testing.T) {
	// Create a simple test block with known values
	var values [256]int8
	var scales [16]int8

	// Set all values to a simple pattern
	for i := 0; i < 256; i++ {
		values[i] = int8((i % 32) - 16) // -16 to 15
	}
	for i := 0; i < 16; i++ {
		scales[i] = int8(i + 1) // 1 to 16
	}

	d := float32(0.5)
	block := createQ6_KBlock(d, scales, values)

	// Dequantize
	result := dequantizeQ6_KRef(block, 256)

	// Verify first few values manually
	// values[0] = -16, scales[0] = 1, d = 0.5
	// expected = 0.5 * 1 * (-16) = -8.0
	expected0 := d * float32(scales[0]) * float32(values[0])
	if math.Abs(float64(result[0]-expected0)) > 0.001 {
		t.Errorf("result[0] = %f, expected %f", result[0], expected0)
	}

	t.Logf("Q6_K dequantization test passed")
}

// TestQ6_KMatVec_Simple tests the GPU Q6_K matvec with simple data.
func TestQ6_KMatVec_Simple(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Test dimensions: A[1, K] @ B[N, K]^T = C[1, N]
	K := 256 // One Q6_K block per row
	N := 8   // Number of output elements
	M := 1

	// Create A vector with all 1s
	a := make([]float32, K)
	for i := range a {
		a[i] = 1.0
	}

	// Create B matrix in Q6_K format
	// Each row is one Q6_K block (256 elements)
	var values [256]int8
	var scales [16]int8
	for i := 0; i < 256; i++ {
		values[i] = 1 // All 1s (dequantized = d * scale * 1)
	}
	for i := 0; i < 16; i++ {
		scales[i] = 1 // Unit scales
	}
	d := float32(1.0)

	// Create N identical Q6_K blocks (one per output)
	bQ6K := make([]byte, N*Q6KBytesPerBlock)
	block := createQ6_KBlock(d, scales, values)
	for i := 0; i < N; i++ {
		copy(bQ6K[i*Q6KBytesPerBlock:], block)
	}

	// CPU reference
	expected := cpuMatVecQ6_K(a, bQ6K, M, N, K)
	t.Logf("CPU expected: %v", expected)

	// Allocate GPU buffers
	aBuf := b.Alloc(K * 4)
	bBuf := b.Alloc(len(bQ6K))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	// Copy data to GPU
	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ6K)

	// Run GPU kernel
	b.MatMulQ6_K(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	// Copy result back
	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	t.Logf("GPU result: %v", result)
	t.Logf("Expected: %v", expected)

	// Compare
	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > 1.0 {
			t.Errorf("Index %d: GPU=%f, CPU=%f, diff=%f", i, result[i], expected[i], diff)
		}
	}
}

// TestQ6_KMatVec_VsReference compares GPU against CPU reference with larger data.
func TestQ6_KMatVec_VsReference(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Larger test: K=2048 (8 blocks), N=32
	K := 2048
	N := 32
	M := 1
	numBlocksPerRow := K / Q6KBlockSize

	// Create random-ish A vector
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(i%10) * 0.1 // 0.0, 0.1, 0.2, ..., 0.9, 0.0, ...
	}

	// Create B matrix with varying Q6_K data
	bQ6K := make([]byte, N*numBlocksPerRow*Q6KBytesPerBlock)
	for row := 0; row < N; row++ {
		for blk := 0; blk < numBlocksPerRow; blk++ {
			var values [256]int8
			var scales [16]int8
			for i := 0; i < 256; i++ {
				values[i] = int8((i + row + blk) % 32 - 16)
			}
			for i := 0; i < 16; i++ {
				scales[i] = int8((row + blk + i) % 8 + 1)
			}
			d := float32(0.01 * float32(row+1))
			block := createQ6_KBlock(d, scales, values)
			offset := (row*numBlocksPerRow + blk) * Q6KBytesPerBlock
			copy(bQ6K[offset:], block)
		}
	}

	// CPU reference
	expected := cpuMatVecQ6_K(a, bQ6K, M, N, K)

	// Allocate GPU buffers
	aBuf := b.Alloc(K * 4)
	bBuf := b.Alloc(len(bQ6K))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	// Copy data to GPU
	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ6K)

	// Run GPU kernel
	b.MatMulQ6_K(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	// Copy result back
	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	// Compare
	maxDiff := float32(0)
	for i := 0; i < N; i++ {
		diff := float32(math.Abs(float64(result[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		// Relative tolerance for numerical precision
		relDiff := diff / (float32(math.Abs(float64(expected[i]))) + 1e-6)
		if relDiff > 0.01 { // 1% tolerance
			t.Errorf("Row %d: GPU=%f, CPU=%f, diff=%f (%.2f%%)",
				i, result[i], expected[i], diff, relDiff*100)
		}
	}

	t.Logf("Q6_K matvec test: max diff = %f", maxDiff)
}
