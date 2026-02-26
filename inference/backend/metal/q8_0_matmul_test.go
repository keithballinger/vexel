//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// Q8_0 format constants
const (
	Q8_0BlockSize     = 32 // Elements per Q8_0 block
	Q8_0BytesPerBlock = 34 // 2 (f16 scale) + 32 (int8 values)
)

// createQ8_0Block creates a single Q8_0 block with the given scale and int8 values.
// values should have 32 elements, each in range -128..127.
func createQ8_0Block(scale float32, values []int8) []byte {
	if len(values) != 32 {
		panic("Q8_0 block requires exactly 32 values")
	}
	block := make([]byte, Q8_0BytesPerBlock)
	// Store scale as f16
	scaleU16 := float32ToFloat16(scale)
	binary.LittleEndian.PutUint16(block[0:], scaleU16)
	// Store int8 values
	for i := 0; i < 32; i++ {
		block[2+i] = byte(values[i])
	}
	return block
}

// dequantizeQ8_0Ref dequantizes a Q8_0 block for CPU reference.
func dequantizeQ8_0Ref(data []byte, numElements int) []float32 {
	numBlocks := (numElements + Q8_0BlockSize - 1) / Q8_0BlockSize
	result := make([]float32, numElements)
	for b := 0; b < numBlocks; b++ {
		blockOffset := b * Q8_0BytesPerBlock
		if blockOffset+Q8_0BytesPerBlock > len(data) {
			break
		}
		// Read f16 scale
		scaleU16 := binary.LittleEndian.Uint16(data[blockOffset:])
		scale := float16ToFloat32CPU(scaleU16)
		// Read int8 values and dequantize
		for i := 0; i < 32; i++ {
			idx := b*Q8_0BlockSize + i
			if idx >= numElements {
				break
			}
			val := int8(data[blockOffset+2+i])
			result[idx] = scale * float32(val)
		}
	}
	return result
}

// cpuMatMulQ8_0 computes C = A @ B^T on CPU where B is Q8_0 encoded.
// A: [m, k] float32, B: [n, k] Q8_0, C: [m, n] float32.
func cpuMatMulQ8_0(a []float32, bQ8 []byte, m, n, k int) []float32 {
	numBlocksPerRow := (k + Q8_0BlockSize - 1) / Q8_0BlockSize
	bytesPerRow := numBlocksPerRow * Q8_0BytesPerBlock
	result := make([]float32, m*n)
	for row := 0; row < m; row++ {
		for col := 0; col < n; col++ {
			var sum float64
			rowData := bQ8[col*bytesPerRow : col*bytesPerRow+bytesPerRow]
			bDequant := dequantizeQ8_0Ref(rowData, k)
			for i := 0; i < k; i++ {
				sum += float64(a[row*k+i]) * float64(bDequant[i])
			}
			result[row*n+col] = float32(sum)
		}
	}
	return result
}

// TestQ8_0MatVecBasic tests basic Q8_0 matvec (M=1) correctness.
func TestQ8_0MatVecBasic(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		N = 4  // Output dimension (number of B rows)
		K = 32 // Inner dimension (must be multiple of 32 for Q8_0)
	)

	// Create A vector: [1, K]
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(i+1) * 0.01
	}

	// Create B matrix: [N, K] in Q8_0, one block per row
	bQ8 := make([]byte, N*Q8_0BytesPerBlock)

	// Row 0: scale=0.1, all values=1 → each element=0.1
	vals0 := make([]int8, 32)
	for i := range vals0 {
		vals0[i] = 1
	}
	copy(bQ8[0:], createQ8_0Block(0.1, vals0))

	// Row 1: scale=0.5, all values=2 → each element=1.0
	vals1 := make([]int8, 32)
	for i := range vals1 {
		vals1[i] = 2
	}
	copy(bQ8[Q8_0BytesPerBlock:], createQ8_0Block(0.5, vals1))

	// Row 2: scale=1.0, values alternate +1/-1
	vals2 := make([]int8, 32)
	for i := range vals2 {
		if i%2 == 0 {
			vals2[i] = 1
		} else {
			vals2[i] = -1
		}
	}
	copy(bQ8[2*Q8_0BytesPerBlock:], createQ8_0Block(1.0, vals2))

	// Row 3: scale=0.0, all zeros
	vals3 := make([]int8, 32)
	copy(bQ8[3*Q8_0BytesPerBlock:], createQ8_0Block(0.0, vals3))

	// CPU reference
	expected := cpuMatMulQ8_0(a, bQ8, 1, N, K)

	// GPU
	aBuf := b.Alloc(K * 4)
	bBuf := b.Alloc(len(bQ8))
	outBuf := b.Alloc(N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ8)
	b.MatMulQ8_0(aBuf, bBuf, outBuf, 1, N, K)
	b.Sync()

	resultBytes := make([]byte, N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	tol := 0.01
	for i := 0; i < N; i++ {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > tol {
			t.Errorf("output[%d]: GPU=%.6f CPU=%.6f diff=%.6f", i, result[i], expected[i], diff)
		}
	}
	t.Logf("Q8_0 MatVec PASS: max output values: GPU[0]=%.4f CPU[0]=%.4f", result[0], expected[0])
}

// TestQ8_0BatchedBasic tests Q8_0 batched matmul (M>1) correctness.
func TestQ8_0BatchedBasic(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		M = 4
		N = 8
		K = 64 // 2 blocks per row
	)

	rng := rand.New(rand.NewSource(42))

	// Create random A matrix [M, K]
	a := make([]float32, M*K)
	for i := range a {
		a[i] = (rng.Float32() - 0.5) * 0.2
	}

	// Create random Q8_0 B matrix [N, K]
	numBlocksPerRow := (K + Q8_0BlockSize - 1) / Q8_0BlockSize
	bytesPerRow := numBlocksPerRow * Q8_0BytesPerBlock
	bQ8 := make([]byte, N*bytesPerRow)

	for row := 0; row < N; row++ {
		for blk := 0; blk < numBlocksPerRow; blk++ {
			scale := (rng.Float32() - 0.5) * 0.1
			vals := make([]int8, 32)
			for i := range vals {
				vals[i] = int8(rng.Intn(256) - 128)
			}
			block := createQ8_0Block(scale, vals)
			blockOffset := row*bytesPerRow + blk*Q8_0BytesPerBlock
			copy(bQ8[blockOffset:], block)
		}
	}

	// CPU reference
	expected := cpuMatMulQ8_0(a, bQ8, M, N, K)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ8))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ8)
	b.MatMulQ8_0(aBuf, bBuf, outBuf, M, N, K)
	b.Sync()

	resultBytes := make([]byte, M*N*4)
	b.ToHost(resultBytes, outBuf)
	result := bytesToFloat32(resultBytes)

	var maxDiff float64
	tol := 0.01
	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tol {
			row := i / N
			col := i % N
			t.Errorf("[m=%d,n=%d] GPU=%.6f CPU=%.6f diff=%.6f", row, col, result[i], expected[i], diff)
		}
	}
	t.Logf("Q8_0 Batched PASS [%dx%d]x[%dx%d] max_diff=%.6f", M, K, N, K, maxDiff)
}

// TestQ8_0BatchedPrefillCorrectness tests Q8_0 matmul at realistic prefill batch sizes
// with production LLaMA dimensions.
//
// Track 4: Quantization Expansion, Phase 2 Task 2.
func TestQ8_0BatchedPrefillCorrectness(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	configs := []struct {
		name string
		m    int // batch/sequence length
		n    int // output dimension
		k    int // input dimension
	}{
		// Core prefill sizes
		{"prefill_32_4096x4096", 32, 4096, 4096},
		{"prefill_64_4096x4096", 64, 4096, 4096},
		{"prefill_128_4096x4096", 128, 4096, 4096},
		// GQA projections (K_proj, V_proj have fewer output dims)
		{"prefill_64_1024x4096", 64, 1024, 4096},
		// FFN intermediate
		{"prefill_64_11008x4096", 64, 11008, 4096},
		// Small batch (exercises batched kernel for M=2-7)
		{"small_batch_4_4096x4096", 4, 4096, 4096},
		// Edge: N not divisible by 16 (NR2 outputs per TG)
		{"edge_N_odd_64_2000x4096", 64, 2000, 4096},
	}

	rng := rand.New(rand.NewSource(42))

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			// Create random A matrix [M, K]
			a := make([]float32, cfg.m*cfg.k)
			for i := range a {
				a[i] = (rng.Float32() - 0.5) * 0.2
			}

			// Create random Q8_0 B matrix [N, K]
			numBlocksPerRow := (cfg.k + Q8_0BlockSize - 1) / Q8_0BlockSize
			bytesPerRow := numBlocksPerRow * Q8_0BytesPerBlock
			bQ8 := make([]byte, cfg.n*bytesPerRow)

			for row := 0; row < cfg.n; row++ {
				for blk := 0; blk < numBlocksPerRow; blk++ {
					scale := (rng.Float32() - 0.5) * 0.1
					vals := make([]int8, 32)
					for i := range vals {
						vals[i] = int8(rng.Intn(256) - 128)
					}
					block := createQ8_0Block(scale, vals)
					blockOffset := row*bytesPerRow + blk*Q8_0BytesPerBlock
					copy(bQ8[blockOffset:], block)
				}
			}

			// CPU reference
			expected := cpuMatMulQ8_0(a, bQ8, cfg.m, cfg.n, cfg.k)

			// GPU
			aBuf := b.Alloc(len(a) * 4)
			bBuf := b.Alloc(len(bQ8))
			outBuf := b.Alloc(cfg.m * cfg.n * 4)
			defer b.Free(aBuf)
			defer b.Free(bBuf)
			defer b.Free(outBuf)

			b.ToDevice(aBuf, float32ToBytes(a))
			b.ToDevice(bBuf, bQ8)
			b.MatMulQ8_0(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
			b.Sync()

			resultBytes := make([]byte, cfg.m*cfg.n*4)
			b.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			// Compare with tolerance (Q8_0 has less quantization error than Q4)
			var maxDiff float64
			var mismatchCount int
			tol := 0.5 // Q8_0 should be much more precise than Q4
			for i := range expected {
				diff := math.Abs(float64(result[i] - expected[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > tol {
					mismatchCount++
					if mismatchCount <= 3 {
						row := i / cfg.n
						col := i % cfg.n
						t.Errorf("[m=%d,n=%d] GPU=%.6f CPU=%.6f diff=%.6f",
							row, col, result[i], expected[i], diff)
					}
				}
			}

			if mismatchCount > 0 {
				t.Errorf("Mismatches: %d/%d (max diff: %.6f, tol: %.1f)",
					mismatchCount, len(expected), maxDiff, tol)
			} else {
				t.Logf("PASS [%dx%d]x[%dx%d] max_diff=%.6f",
					cfg.m, cfg.k, cfg.n, cfg.k, maxDiff)
			}
		})
	}
}

// BenchmarkQ8_0PrefillMatMul benchmarks Q8_0 matmul throughput at different batch sizes.
//
// Track 4: Quantization Expansion, Phase 2 Task 2.
func BenchmarkQ8_0PrefillMatMul(b *testing.B) {
	be, err := NewBackend(0)
	if err != nil {
		b.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	configs := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{"M1_decode", 1, 4096, 4096},
		{"M4_small_batch", 4, 4096, 4096},
		{"M8_batch", 8, 4096, 4096},
		{"M32_prefill", 32, 4096, 4096},
		{"M64_prefill", 64, 4096, 4096},
		{"M128_prefill", 128, 4096, 4096},
		// FFN dimensions (wider N)
		{"M64_ffn", 64, 11008, 4096},
	}

	for _, cfg := range configs {
		// Allocate buffers
		aSize := cfg.m * cfg.k
		numBlocksPerRow := (cfg.k + Q8_0BlockSize - 1) / Q8_0BlockSize
		bSize := cfg.n * numBlocksPerRow * Q8_0BytesPerBlock
		outSize := cfg.m * cfg.n

		aBuf := be.Alloc(aSize * 4)
		bBuf := be.Alloc(bSize)
		outBuf := be.Alloc(outSize * 4)

		// Fill with random data
		aData := make([]float32, aSize)
		rng := rand.New(rand.NewSource(42))
		for i := range aData {
			aData[i] = (rng.Float32() - 0.5) * 0.2
		}
		be.ToDevice(aBuf, float32ToBytes(aData))
		be.Zero(bBuf, bSize)
		be.Sync()

		b.Run(cfg.name, func(b *testing.B) {
			// Warmup
			for i := 0; i < 5; i++ {
				be.MatMulQ8_0(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
				be.Sync()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				be.MatMulQ8_0(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
				be.Sync()
			}
		})

		be.Free(aBuf)
		be.Free(bBuf)
		be.Free(outBuf)
	}
}

// TestQ8_0PrefillThroughputReport generates a human-readable throughput report
// comparing Q8_0 matmul performance across batch sizes.
func TestQ8_0PrefillThroughputReport(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	const (
		N     = 4096
		K     = 4096
		iters = 50
	)

	batchSizes := []int{1, 4, 8, 16, 32, 64, 128}

	// Create Q8_0 weight matrix
	numBlocksPerRow := (K + Q8_0BlockSize - 1) / Q8_0BlockSize
	bSize := N * numBlocksPerRow * Q8_0BytesPerBlock
	bBuf := be.Alloc(bSize)
	be.Zero(bBuf, bSize)
	be.Sync()

	fmt.Println("\n[Q8_0 BATCHED MATMUL THROUGHPUT]")
	fmt.Printf("Weight matrix: [%d, %d] Q8_0\n", N, K)
	fmt.Printf("Benchmark: %d iterations per measurement\n\n", iters)
	fmt.Printf("%-8s %12s %12s %12s %12s\n", "M", "Time/iter", "GFLOPS", "Throughput", "Kernel")
	fmt.Printf("%-8s %12s %12s %12s %12s\n", "---", "--------", "------", "----------", "------")

	for _, m := range batchSizes {
		aSize := m * K
		outSize := m * N

		aBuf := be.Alloc(aSize * 4)
		outBuf := be.Alloc(outSize * 4)

		aData := make([]float32, aSize)
		rng := rand.New(rand.NewSource(42))
		for i := range aData {
			aData[i] = (rng.Float32() - 0.5) * 0.2
		}
		be.ToDevice(aBuf, float32ToBytes(aData))
		be.Sync()

		// Warmup
		for i := 0; i < 10; i++ {
			be.MatMulQ8_0(aBuf, bBuf, outBuf, m, N, K)
			be.Sync()
		}

		start := time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulQ8_0(aBuf, bBuf, outBuf, m, N, K)
			be.Sync()
		}
		elapsed := time.Since(start) / time.Duration(iters)

		// Calculate GFLOPS: 2*M*N*K FLOPs (multiply-add)
		flops := float64(2) * float64(m) * float64(N) * float64(K)
		gflops := flops / float64(elapsed.Nanoseconds())

		// Determine which kernel path is used
		kernel := "nr2_batched"
		if m == 1 {
			kernel = "nr2_matvec"
		}

		fmt.Printf("%-8d %12v %10.1f %10.1f GB/s %12s\n",
			m, elapsed, gflops,
			// Memory throughput (read A + read B + write C)
			float64(aSize*4+bSize+outSize*4)/float64(elapsed.Nanoseconds()),
			kernel)

		be.Free(aBuf)
		be.Free(outBuf)
	}
	fmt.Println()

	be.Free(bBuf)
}
