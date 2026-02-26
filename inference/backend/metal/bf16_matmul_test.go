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

// float32ToBF16 converts a float32 to BF16 bits (upper 16 bits of float32).
func float32ToBF16(f float32) uint16 {
	bits := math.Float32bits(f)
	return uint16(bits >> 16)
}

// bf16ToFloat32 converts BF16 bits to float32.
func bf16ToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// createBF16Matrix creates a BF16-encoded matrix [rows, cols] from float32 values.
func createBF16Matrix(data []float32, rows, cols int) []byte {
	result := make([]byte, rows*cols*2)
	for i, v := range data {
		bf16 := float32ToBF16(v)
		binary.LittleEndian.PutUint16(result[i*2:], bf16)
	}
	return result
}

// cpuMatMulBF16 computes C = A @ B^T on CPU where B is BF16 encoded.
// A: [m, k] float32, B: [n, k] BF16 (2 bytes per element), C: [m, n] float32.
func cpuMatMulBF16(a []float32, bBF16 []byte, m, n, k int) []float32 {
	result := make([]float32, m*n)
	for row := 0; row < m; row++ {
		for col := 0; col < n; col++ {
			var sum float64
			for i := 0; i < k; i++ {
				bIdx := (col*k + i) * 2
				bVal := bf16ToFloat32(binary.LittleEndian.Uint16(bBF16[bIdx:]))
				sum += float64(a[row*k+i]) * float64(bVal)
			}
			result[row*n+col] = float32(sum)
		}
	}
	return result
}

// TestBF16MatVecBasic tests basic BF16 matvec (M=1) correctness.
func TestBF16MatVecBasic(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		N = 4  // Output dimension
		K = 32 // Inner dimension
	)

	// Create A vector: [1, K]
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(i+1) * 0.01
	}

	// Create B matrix: [N, K] in BF16
	bData := make([]float32, N*K)
	// Row 0: all 1.0
	for i := 0; i < K; i++ {
		bData[0*K+i] = 1.0
	}
	// Row 1: all 0.5
	for i := 0; i < K; i++ {
		bData[1*K+i] = 0.5
	}
	// Row 2: alternating +1/-1
	for i := 0; i < K; i++ {
		if i%2 == 0 {
			bData[2*K+i] = 1.0
		} else {
			bData[2*K+i] = -1.0
		}
	}
	// Row 3: all zeros
	for i := 0; i < K; i++ {
		bData[3*K+i] = 0.0
	}

	bBF16 := createBF16Matrix(bData, N, K)

	// CPU reference
	expected := cpuMatMulBF16(a, bBF16, 1, N, K)

	// GPU
	aBuf := b.Alloc(K * 4)
	bBuf := b.Alloc(len(bBF16))
	outBuf := b.Alloc(N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bBF16)
	b.MatMulBF16(aBuf, bBuf, outBuf, 1, N, K)
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
	t.Logf("BF16 MatVec PASS: GPU[0]=%.4f CPU[0]=%.4f, GPU[1]=%.4f CPU[1]=%.4f",
		result[0], expected[0], result[1], expected[1])
}

// TestBF16BatchedBasic tests BF16 batched matmul (M>1) correctness.
func TestBF16BatchedBasic(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		M = 4
		N = 8
		K = 64
	)

	rng := rand.New(rand.NewSource(42))

	// Create random A matrix [M, K]
	a := make([]float32, M*K)
	for i := range a {
		a[i] = (rng.Float32() - 0.5) * 0.2
	}

	// Create random BF16 B matrix [N, K]
	bData := make([]float32, N*K)
	for i := range bData {
		bData[i] = (rng.Float32() - 0.5) * 0.2
	}
	bBF16 := createBF16Matrix(bData, N, K)

	// CPU reference
	expected := cpuMatMulBF16(a, bBF16, M, N, K)

	// GPU
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bBF16))
	outBuf := b.Alloc(M * N * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bBF16)
	b.MatMulBF16(aBuf, bBuf, outBuf, M, N, K)
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
	t.Logf("BF16 Batched PASS [%dx%d]x[%dx%d] max_diff=%.6f", M, K, N, K, maxDiff)
}

// TestBF16BatchedPrefillCorrectness tests BF16 matmul at realistic prefill batch sizes.
//
// Track 4: Quantization Expansion, Phase 2 Task 3.
func TestBF16BatchedPrefillCorrectness(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	configs := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{"prefill_32_4096x4096", 32, 4096, 4096},
		{"prefill_64_4096x4096", 64, 4096, 4096},
		{"prefill_128_4096x4096", 128, 4096, 4096},
		{"prefill_64_1024x4096", 64, 1024, 4096},
		{"prefill_64_11008x4096", 64, 11008, 4096},
		{"small_batch_4_4096x4096", 4, 4096, 4096},
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

			// Create random BF16 B matrix [N, K]
			bData := make([]float32, cfg.n*cfg.k)
			for i := range bData {
				bData[i] = (rng.Float32() - 0.5) * 0.2
			}
			bBF16 := createBF16Matrix(bData, cfg.n, cfg.k)

			// CPU reference
			expected := cpuMatMulBF16(a, bBF16, cfg.m, cfg.n, cfg.k)

			// GPU
			aBuf := b.Alloc(len(a) * 4)
			bBuf := b.Alloc(len(bBF16))
			outBuf := b.Alloc(cfg.m * cfg.n * 4)
			defer b.Free(aBuf)
			defer b.Free(bBuf)
			defer b.Free(outBuf)

			b.ToDevice(aBuf, float32ToBytes(a))
			b.ToDevice(bBuf, bBF16)
			b.MatMulBF16(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
			b.Sync()

			resultBytes := make([]byte, cfg.m*cfg.n*4)
			b.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			// BF16 has ~7 bits of mantissa precision (vs 23 for F32)
			// Accumulated error over K=4096 can be noticeable
			var maxDiff float64
			var mismatchCount int
			tol := 0.5
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

// BenchmarkBF16PrefillMatMul benchmarks BF16 matmul throughput at different batch sizes.
func BenchmarkBF16PrefillMatMul(b *testing.B) {
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
		{"M64_ffn", 64, 11008, 4096},
	}

	for _, cfg := range configs {
		aSize := cfg.m * cfg.k
		bSize := cfg.n * cfg.k * 2 // BF16: 2 bytes per element
		outSize := cfg.m * cfg.n

		aBuf := be.Alloc(aSize * 4)
		bBuf := be.Alloc(bSize)
		outBuf := be.Alloc(outSize * 4)

		aData := make([]float32, aSize)
		rng := rand.New(rand.NewSource(42))
		for i := range aData {
			aData[i] = (rng.Float32() - 0.5) * 0.2
		}
		be.ToDevice(aBuf, float32ToBytes(aData))
		be.Zero(bBuf, bSize)
		be.Sync()

		b.Run(cfg.name, func(b *testing.B) {
			for i := 0; i < 5; i++ {
				be.MatMulBF16(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
				be.Sync()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				be.MatMulBF16(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
				be.Sync()
			}
		})

		be.Free(aBuf)
		be.Free(bBuf)
		be.Free(outBuf)
	}
}

// TestBF16PrefillThroughputReport generates a human-readable throughput report.
func TestBF16PrefillThroughputReport(t *testing.T) {
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

	bSize := N * K * 2 // BF16: 2 bytes per element
	bBuf := be.Alloc(bSize)
	be.Zero(bBuf, bSize)
	be.Sync()

	fmt.Println("\n[BF16 BATCHED MATMUL THROUGHPUT]")
	fmt.Printf("Weight matrix: [%d, %d] BF16\n", N, K)
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

		for i := 0; i < 10; i++ {
			be.MatMulBF16(aBuf, bBuf, outBuf, m, N, K)
			be.Sync()
		}

		start := time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulBF16(aBuf, bBuf, outBuf, m, N, K)
			be.Sync()
		}
		elapsed := time.Since(start) / time.Duration(iters)

		flops := float64(2) * float64(m) * float64(N) * float64(K)
		gflops := flops / float64(elapsed.Nanoseconds())

		kernel := "nr2_batched"
		if m == 1 {
			kernel = "nr2_matvec"
		}

		fmt.Printf("%-8d %12v %10.1f %10.1f GB/s %12s\n",
			m, elapsed, gflops,
			float64(aSize*4+bSize+outSize*4)/float64(elapsed.Nanoseconds()),
			kernel)

		be.Free(aBuf)
		be.Free(outBuf)
	}
	fmt.Println()

	be.Free(bBuf)
}
