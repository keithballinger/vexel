//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"vexel/inference/tensor"
)

func cpuRMSNorm(x, weight []float32, eps float32) []float32 {
	n := len(x)
	out := make([]float32, n)
	var sumSq float32
	for _, v := range x {
		sumSq += v * v
	}
	rms := float32(1.0 / math.Sqrt(float64(sumSq/float32(n)+eps)))
	for i := range x {
		out[i] = x[i] * rms * weight[i]
	}
	return out
}

func cpuMatVecQ4_0(a []float32, b []byte, m, n, k int) []float32 {
	// a: [k], b: [n, k] (Q4_0)
	out := make([]float32, n)
	blockSize := 32
	bytesPerBlock := 18

	for i := 0; i < n; i++ {
		var sum float32
		bRow := b[i*(k/blockSize*bytesPerBlock):]
		for blk := 0; blk < k/blockSize; blk++ {
			blockPtr := bRow[blk*bytesPerBlock:]
			scaleVal := binary.LittleEndian.Uint16(blockPtr[0:2])
			scale := q4_f16_to_f32_cpu(scaleVal)

            // Q4_0 Layout: 
            // 16 bytes of nibbles. 
            // byte[j] low nibble = w[j], high nibble = w[j+16]
			for j := 0; j < 16; j++ {
				byteVal := blockPtr[2+j]
                
                // Low nibble -> w[j]
				valLow := scale * (float32(byteVal & 0x0F) - 8.0)
				sum += a[blk*blockSize+j] * valLow

                // High nibble -> w[j+16]
				valHigh := scale * (float32(byteVal >> 4) - 8.0)
				sum += a[blk*blockSize+j+16] * valHigh
			}
		}
		out[i] = sum
	}
	return out
}

// Re-implement q4_f16_to_f32 for test (or make public in q4_kernel_test.go?)
// Copying simple version for self-containment
func q4_f16_to_f32_cpu(h uint16) float32 {
	sign := (h >> 15) & 0x1
	exp := (h >> 10) & 0x1f
	mant := h & 0x3ff
	if exp == 0 {
		return 0 // Subnormal not handled for simplicity in scale
	}
	return float32(math.Pow(2, float64(exp)-15)) * (1.0 + float32(mant)/1024.0) * float32(1-2*int(sign))
}

func TestFusedRMSNormMatMulQ4_0(t *testing.T) {
	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	// Dimensions
	m, n, k := 1, 128, 256 // Small for test
	eps := float32(1e-5)

	// Random Input
	x := make([]float32, k)
	for i := range x {
		x[i] = (rand.Float32() - 0.5) * 2
	}

	// RMSNorm Weights
	normWeight := make([]float32, k)
	for i := range normWeight {
		normWeight[i] = rand.Float32() + 0.5
	}

	// Q4_0 Weights (randomly generated)
	// We need to quantize them properly or just generate random Q4 blocks
	// Generating random Q4 blocks is easier and sufficient for MatMul check
	numBlocks := k / 32
	q4Bytes := n * numBlocks * 18
	weightsQ4 := make([]byte, q4Bytes)
	rand.Read(weightsQ4)
	// Ensure scales are valid (positive, non-zero/subnormal for simplicity)
	for i := 0; i < len(weightsQ4); i += 18 {
		// Set scale to 1.0 (0x3C00) to keep math simple, or random
		binary.LittleEndian.PutUint16(weightsQ4[i:], 0x3C00) 
	}

	// Reference Computation
	// 1. RMSNorm
	xNorm := cpuRMSNorm(x, normWeight, eps)
	// 2. MatVec
	expected := cpuMatVecQ4_0(xNorm, weightsQ4, m, n, k)

	// GPU Buffers
	xBuf := backend.Alloc(k * 4)
	wNormBuf := backend.Alloc(k * 4)
	wMatBuf := backend.Alloc(q4Bytes)
	outBuf := backend.Alloc(n * 4)
	defer backend.Free(xBuf)
	defer backend.Free(wNormBuf)
	defer backend.Free(wMatBuf)
	defer backend.Free(outBuf)

	backend.ToDevice(xBuf, float32ToBytes(x))
	backend.ToDevice(wNormBuf, float32ToBytes(normWeight))
	backend.ToDevice(wMatBuf, weightsQ4)

	// Run Fused Kernel
	// Need to add this method to Backend first, but for now we fail compilation if not present
	// This confirms test-first methodology :)
	// We assume interface: MatMulQ4_0_FusedRMSNorm(x, normWeight, wMat, out, m, n, k, eps)
	if fused, ok := interface{}(backend).(interface {
		MatMulQ4_0_FusedRMSNorm(x, normWeight, wMat, out tensor.DevicePtr, m, n, k int, eps float32)
	}); ok {
		fused.MatMulQ4_0_FusedRMSNorm(xBuf, wNormBuf, wMatBuf, outBuf, m, n, k, eps)
	} else {
		t.Fatalf("Backend does not implement MatMulQ4_0_FusedRMSNorm")
	}
	
	backend.Sync()

	// Verify
	outBytes := make([]byte, n*4)
	backend.ToHost(outBytes, outBuf)
	out := bytesToFloat32(outBytes)

	maxDiff := float32(0)
	for i := range expected {
		diff := float32(math.Abs(float64(out[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	t.Logf("Max Diff: %f", maxDiff)
	if maxDiff > 1e-2 {
		t.Fatalf("Mismatch too large")
	}
}
