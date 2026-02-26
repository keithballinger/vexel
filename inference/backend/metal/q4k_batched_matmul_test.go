//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestQ4_KBatchedPrefillCorrectness tests Q4_K matmul at realistic prefill batch sizes
// with production LLaMA dimensions. These are the target sizes for kernel optimization.
//
// Track 4: Quantization Expansion, Phase 1 Task 2.
func TestQ4_KBatchedPrefillCorrectness(t *testing.T) {
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

			// Create random Q4_K B matrix [N, K]
			numBlocksPerRow := (cfg.k + Q4KBlockSize - 1) / Q4KBlockSize
			bytesPerRow := numBlocksPerRow * Q4KBytesPerBlock
			bQ4K := make([]byte, cfg.n*bytesPerRow)

			for row := 0; row < cfg.n; row++ {
				for blk := 0; blk < numBlocksPerRow; blk++ {
					d := (rng.Float32() - 0.5) * 0.1
					dmin := rng.Float32() * 0.05
					var scales [8]uint8
					var mins [8]uint8
					var values [256]uint8
					for i := range scales {
						scales[i] = uint8(rng.Intn(64))
						mins[i] = uint8(rng.Intn(64))
					}
					for i := range values {
						values[i] = uint8(rng.Intn(16))
					}
					block := createQ4_KBlock(d, dmin, scales, mins, values)
					blockOffset := row*bytesPerRow + blk*Q4KBytesPerBlock
					copy(bQ4K[blockOffset:], block)
				}
			}

			// CPU reference
			expected := cpuMatVecQ4_K(a, bQ4K, cfg.m, cfg.n, cfg.k)

			// GPU
			aBuf := b.Alloc(len(a) * 4)
			bBuf := b.Alloc(len(bQ4K))
			outBuf := b.Alloc(cfg.m * cfg.n * 4)
			defer b.Free(aBuf)
			defer b.Free(bBuf)
			defer b.Free(outBuf)

			b.ToDevice(aBuf, float32ToBytes(a))
			b.ToDevice(bBuf, bQ4K)
			b.MatMulQ4_K(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
			b.Sync()

			resultBytes := make([]byte, cfg.m*cfg.n*4)
			b.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			// Compare with tolerance (Q4_K has more quantization error than Q4_0)
			var maxDiff float64
			var mismatchCount int
			tol := 1.0 // Tolerance for accumulated quantization error over K=4096
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

// BenchmarkQ4_KPrefillMatMul benchmarks Q4_K matmul throughput at different batch sizes.
// This establishes the performance baseline for the NR2 batched kernel.
//
// Track 4: Quantization Expansion, Phase 1 Task 2.
func BenchmarkQ4_KPrefillMatMul(b *testing.B) {
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
		numBlocksPerRow := (cfg.k + Q4KBlockSize - 1) / Q4KBlockSize
		bSize := cfg.n * numBlocksPerRow * Q4KBytesPerBlock
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
				be.MatMulQ4_K(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
				be.Sync()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				be.MatMulQ4_K(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
				be.Sync()
			}
		})

		be.Free(aBuf)
		be.Free(bBuf)
		be.Free(outBuf)
	}
}

// TestQ4_KPrefillThroughputReport generates a human-readable throughput report
// comparing Q4_K matmul performance across batch sizes.
func TestQ4_KPrefillThroughputReport(t *testing.T) {
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

	// Create Q4_K weight matrix
	numBlocksPerRow := (K + Q4KBlockSize - 1) / Q4KBlockSize
	bSize := N * numBlocksPerRow * Q4KBytesPerBlock
	bBuf := be.Alloc(bSize)
	be.Zero(bBuf, bSize)
	be.Sync()

	fmt.Println("\n[Q4_K BATCHED MATMUL THROUGHPUT]")
	fmt.Printf("Weight matrix: [%d, %d] Q4_K\n", N, K)
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
			be.MatMulQ4_K(aBuf, bBuf, outBuf, m, N, K)
			be.Sync()
		}

		start := time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulQ4_K(aBuf, bBuf, outBuf, m, N, K)
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
