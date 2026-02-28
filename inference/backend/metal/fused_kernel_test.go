//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"fmt"
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

// TestFusedRMSNormQKV_F16 tests the fused RMSNorm + Q/K/V projection kernel (3→1 dispatch).
// Verifies that a single fused dispatch produces the same FP16 outputs as 3 separate
// FusedRMSNormF16 dispatches at all LLaMA 2 7B dimensions.
func TestFusedRMSNormQKV_F16(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	if b.matvecQ4FusedRMSNormQKVF16Pipeline == nil {
		t.Skip("Fused QKV F16 pipeline not available")
	}
	if b.matvecQ4FusedRMSNormF16Pipeline == nil {
		t.Skip("FusedRMSNormF16 pipeline not available (needed for reference)")
	}

	configs := []struct {
		name  string
		qDim  int // Q output dimension
		kvDim int // K, V output dimension
		k     int // hidden dimension
	}{
		// Small for debugging
		{"small_128_128_256", 128, 128, 256},
		// LLaMA 2 7B: 32 Q heads, 32 KV heads (MHA), headDim=128
		{"llama2_7b", 4096, 4096, 4096},
		// LLaMA 2 7B GQA variant (e.g. 70B has 8 KV heads)
		{"llama2_gqa", 4096, 1024, 4096},
	}

	eps := float32(1e-5)
	rng := rand.New(rand.NewSource(42))

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			k := cfg.k
			qDim := cfg.qDim
			kvDim := cfg.kvDim

			// Random input x [1, K]
			x := make([]float32, k)
			for i := range x {
				x[i] = (rng.Float32() - 0.5) * 2
			}

			// RMSNorm weights [K]
			normWeight := make([]float32, k)
			for i := range normWeight {
				normWeight[i] = rng.Float32() + 0.5
			}

			// Q4_0 weight matrices
			numBlocksPerRow := k / 32
			bytesPerRow := numBlocksPerRow * 18

			wqBytes := make([]byte, qDim*bytesPerRow)
			wkBytes := make([]byte, kvDim*bytesPerRow)
			wvBytes := make([]byte, kvDim*bytesPerRow)
			rng.Read(wqBytes)
			rng.Read(wkBytes)
			rng.Read(wvBytes)

			// Ensure valid scales (1.0 = 0x3C00)
			for _, w := range [][]byte{wqBytes, wkBytes, wvBytes} {
				for i := 0; i < len(w); i += 18 {
					binary.LittleEndian.PutUint16(w[i:], 0x3C00)
				}
			}

			// Allocate GPU buffers
			xBuf := b.Alloc(k * 4)
			normBuf := b.Alloc(k * 4)
			wqBuf := b.Alloc(len(wqBytes))
			wkBuf := b.Alloc(len(wkBytes))
			wvBuf := b.Alloc(len(wvBytes))
			// FP16 output buffers (2 bytes per element)
			outQFused := b.Alloc(qDim * 2)
			outKFused := b.Alloc(kvDim * 2)
			outVFused := b.Alloc(kvDim * 2)
			outQRef := b.Alloc(qDim * 2)
			outKRef := b.Alloc(kvDim * 2)
			outVRef := b.Alloc(kvDim * 2)
			defer b.Free(xBuf)
			defer b.Free(normBuf)
			defer b.Free(wqBuf)
			defer b.Free(wkBuf)
			defer b.Free(wvBuf)
			defer b.Free(outQFused)
			defer b.Free(outKFused)
			defer b.Free(outVFused)
			defer b.Free(outQRef)
			defer b.Free(outKRef)
			defer b.Free(outVRef)

			b.ToDevice(xBuf, float32ToBytes(x))
			b.ToDevice(normBuf, float32ToBytes(normWeight))
			b.ToDevice(wqBuf, wqBytes)
			b.ToDevice(wkBuf, wkBytes)
			b.ToDevice(wvBuf, wvBytes)

			// Run fused QKV kernel (single dispatch)
			b.MatMulQ4_0_FusedRMSNormQKV_F16(xBuf, normBuf, wqBuf, wkBuf, wvBuf,
				outQFused, outKFused, outVFused, qDim, kvDim, k, eps)
			b.Sync()

			// Run 3 separate FusedRMSNormF16 as reference
			b.MatMulQ4_0_FusedRMSNormF16(xBuf, normBuf, wqBuf, outQRef, 1, qDim, k, eps)
			b.MatMulQ4_0_FusedRMSNormF16(xBuf, normBuf, wkBuf, outKRef, 1, kvDim, k, eps)
			b.MatMulQ4_0_FusedRMSNormF16(xBuf, normBuf, wvBuf, outVRef, 1, kvDim, k, eps)
			b.Sync()

			// Compare fused vs reference for each projection
			checkF16Match := func(name string, fusedBuf, refBuf tensor.DevicePtr, dim int) {
				fusedBytes := make([]byte, dim*2)
				refBytes := make([]byte, dim*2)
				b.ToHost(fusedBytes, fusedBuf)
				b.ToHost(refBytes, refBuf)

				maxDiff := 0.0
				maxDiffIdx := 0
				for i := 0; i < dim; i++ {
					fusedVal := float16ToFloat32CPU(binary.LittleEndian.Uint16(fusedBytes[i*2 : i*2+2]))
					refVal := float16ToFloat32CPU(binary.LittleEndian.Uint16(refBytes[i*2 : i*2+2]))
					diff := math.Abs(float64(fusedVal - refVal))
					if diff > maxDiff {
						maxDiff = diff
						maxDiffIdx = i
					}
				}

				t.Logf("  %s: max_diff=%.6f at idx=%d (dim=%d)", name, maxDiff, maxDiffIdx, dim)
				if maxDiff > 1e-3 {
					fusedVal := float16ToFloat32CPU(binary.LittleEndian.Uint16(fusedBytes[maxDiffIdx*2 : maxDiffIdx*2+2]))
					refVal := float16ToFloat32CPU(binary.LittleEndian.Uint16(refBytes[maxDiffIdx*2 : maxDiffIdx*2+2]))
					t.Fatalf("  %s: fused vs ref mismatch: diff=%.6f at idx=%d (fused=%.6f, ref=%.6f)",
						name, maxDiff, maxDiffIdx, fusedVal, refVal)
				}
			}

			checkF16Match("Q", outQFused, outQRef, qDim)
			checkF16Match("K", outKFused, outKRef, kvDim)
			checkF16Match("V", outVFused, outVRef, kvDim)
		})
	}
}

// TestFusedRMSNormQKV_F16_vsCPU tests the fused QKV kernel against CPU reference.
func TestFusedRMSNormQKV_F16_vsCPU(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	if b.matvecQ4FusedRMSNormQKVF16Pipeline == nil {
		t.Skip("Fused QKV F16 pipeline not available")
	}

	// LLaMA 2 7B dimensions
	k := 4096
	qDim := 4096
	kvDim := 4096
	eps := float32(1e-5)
	rng := rand.New(rand.NewSource(99))

	// Random input
	x := make([]float32, k)
	for i := range x {
		x[i] = (rng.Float32() - 0.5) * 2
	}

	normWeight := make([]float32, k)
	for i := range normWeight {
		normWeight[i] = rng.Float32() + 0.5
	}

	numBlocksPerRow := k / 32
	bytesPerRow := numBlocksPerRow * 18

	wqBytes := make([]byte, qDim*bytesPerRow)
	wkBytes := make([]byte, kvDim*bytesPerRow)
	wvBytes := make([]byte, kvDim*bytesPerRow)
	rng.Read(wqBytes)
	rng.Read(wkBytes)
	rng.Read(wvBytes)
	for _, w := range [][]byte{wqBytes, wkBytes, wvBytes} {
		for i := 0; i < len(w); i += 18 {
			binary.LittleEndian.PutUint16(w[i:], 0x3C00)
		}
	}

	// CPU reference: RMSNorm then MatVec
	xNorm := cpuRMSNorm(x, normWeight, eps)
	expectedQ := cpuMatVecQ4_0(xNorm, wqBytes, 1, qDim, k)
	expectedK := cpuMatVecQ4_0(xNorm, wkBytes, 1, kvDim, k)
	expectedV := cpuMatVecQ4_0(xNorm, wvBytes, 1, kvDim, k)

	// GPU
	xBuf := b.Alloc(k * 4)
	normBuf := b.Alloc(k * 4)
	wqBuf := b.Alloc(len(wqBytes))
	wkBuf := b.Alloc(len(wkBytes))
	wvBuf := b.Alloc(len(wvBytes))
	outQBuf := b.Alloc(qDim * 2)
	outKBuf := b.Alloc(kvDim * 2)
	outVBuf := b.Alloc(kvDim * 2)
	defer b.Free(xBuf)
	defer b.Free(normBuf)
	defer b.Free(wqBuf)
	defer b.Free(wkBuf)
	defer b.Free(wvBuf)
	defer b.Free(outQBuf)
	defer b.Free(outKBuf)
	defer b.Free(outVBuf)

	b.ToDevice(xBuf, float32ToBytes(x))
	b.ToDevice(normBuf, float32ToBytes(normWeight))
	b.ToDevice(wqBuf, wqBytes)
	b.ToDevice(wkBuf, wkBytes)
	b.ToDevice(wvBuf, wvBytes)

	b.MatMulQ4_0_FusedRMSNormQKV_F16(xBuf, normBuf, wqBuf, wkBuf, wvBuf,
		outQBuf, outKBuf, outVBuf, qDim, kvDim, k, eps)
	b.Sync()

	checkVsCPU := func(name string, buf tensor.DevicePtr, expected []float32, dim int) {
		bytes := make([]byte, dim*2)
		b.ToHost(bytes, buf)

		maxDiff := 0.0
		maxDiffIdx := 0
		for i := 0; i < dim; i++ {
			gpuVal := float64(float16ToFloat32CPU(binary.LittleEndian.Uint16(bytes[i*2 : i*2+2])))
			cpuVal := float64(expected[i])
			diff := math.Abs(gpuVal - cpuVal)
			if diff > maxDiff {
				maxDiff = diff
				maxDiffIdx = i
			}
		}

		// FP16 output quantization: at value ~1000, step size is 1.0, so absolute error can be 0.5+.
		// Use relative tolerance: allow 0.1% relative error or 0.5 absolute (FP16 quantization).
		tol := math.Max(0.5, 0.001*math.Abs(float64(expected[maxDiffIdx])))
		t.Logf("  %s vs CPU: max_diff=%.6f at idx=%d (tol=%.3f)", name, maxDiff, maxDiffIdx, tol)
		if maxDiff > tol {
			gpuVal := float16ToFloat32CPU(binary.LittleEndian.Uint16(bytes[maxDiffIdx*2 : maxDiffIdx*2+2]))
			t.Fatalf("  %s: GPU vs CPU mismatch: diff=%.6f at idx=%d (gpu=%.6f, cpu=%.6f, tol=%.3f)",
				name, maxDiff, maxDiffIdx, gpuVal, expected[maxDiffIdx], tol)
		}
	}

	checkVsCPU("Q", outQBuf, expectedQ, qDim)
	checkVsCPU("K", outKBuf, expectedK, kvDim)
	checkVsCPU("V", outVBuf, expectedV, kvDim)
	_ = fmt.Sprintf("") // use fmt
}
