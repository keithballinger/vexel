//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestQ4_0BatchedPrefillCorrectness tests Q4_0 matmul at realistic prefill batch sizes
// with production LLaMA dimensions. These are the target sizes for kernel optimization.
//
// Track 4: Quantization Expansion, Phase 1 Task 1.
func TestQ4_0BatchedPrefillCorrectness(t *testing.T) {
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
		// Core prefill sizes from plan
		{"prefill_32_4096x4096", 32, 4096, 4096},
		{"prefill_64_4096x4096", 64, 4096, 4096},
		{"prefill_128_4096x4096", 128, 4096, 4096},
		// GQA projections (K_proj, V_proj have fewer output dims)
		{"prefill_64_1024x4096", 64, 1024, 4096},
		// FFN intermediate
		{"prefill_64_11008x4096", 64, 11008, 4096},
		// Small batch (exercises batched kernel path M=2-7)
		{"small_batch_4_4096x4096", 4, 4096, 4096},
	}

	rng := rand.New(rand.NewSource(42))

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			// Create random A matrix [M, K]
			a := make([]float32, cfg.m*cfg.k)
			for i := range a {
				a[i] = (rng.Float32() - 0.5) * 0.2
			}

			// Create random Q4_0 B matrix [N, K]
			numBlocksPerRow := (cfg.k + Q4BlockSize - 1) / Q4BlockSize
			bytesPerRow := numBlocksPerRow * Q4BytesPerBlock
			bQ4 := make([]byte, cfg.n*bytesPerRow)

			for row := 0; row < cfg.n; row++ {
				for blk := 0; blk < numBlocksPerRow; blk++ {
					values := make([]int, 32)
					for i := range values {
						values[i] = rng.Intn(16) // 0-15 range
					}
					scale := (rng.Float32() - 0.5) * 0.1
					block := createQ4_0Block(scale, values)
					blockOffset := row*bytesPerRow + blk*Q4BytesPerBlock
					copy(bQ4[blockOffset:], block)
				}
			}

			// CPU reference
			expected := cpuMatMulQ4_0(a, bQ4, cfg.m, cfg.n, cfg.k)

			// GPU
			aBuf := b.Alloc(len(a) * 4)
			bBuf := b.Alloc(len(bQ4))
			outBuf := b.Alloc(cfg.m * cfg.n * 4)
			defer b.Free(aBuf)
			defer b.Free(bBuf)
			defer b.Free(outBuf)

			b.ToDevice(aBuf, float32ToBytes(a))
			b.ToDevice(bBuf, bQ4)
			b.MatMulQ4_0(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
			b.Sync()

			resultBytes := make([]byte, cfg.m*cfg.n*4)
			b.ToHost(resultBytes, outBuf)
			result := bytesToFloat32(resultBytes)

			// Compare with tolerance (Q4_0 has quantization error that accumulates)
			var maxDiff float64
			var mismatchCount int
			tol := 1.0 // Tolerance for accumulated quantization error
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
				t.Logf("PASS [%dx%d]×[%dx%d] max_diff=%.6f",
					cfg.m, cfg.k, cfg.n, cfg.k, maxDiff)
			}
		})
	}
}

// BenchmarkQ4_0PrefillMatMul benchmarks Q4_0 matmul throughput at different batch sizes.
// This establishes the performance baseline for kernel optimization.
//
// Track 4: Quantization Expansion, Phase 1 Task 1.
func BenchmarkQ4_0PrefillMatMul(b *testing.B) {
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
		{"M8_simdgroup_entry", 8, 4096, 4096},
		{"M32_prefill", 32, 4096, 4096},
		{"M64_prefill", 64, 4096, 4096},
		{"M128_prefill", 128, 4096, 4096},
		// FFN dimensions (wider N)
		{"M64_ffn", 64, 11008, 4096},
	}

	for _, cfg := range configs {
		// Allocate buffers
		aSize := cfg.m * cfg.k
		numBlocksPerRow := (cfg.k + Q4BlockSize - 1) / Q4BlockSize
		bSize := cfg.n * numBlocksPerRow * Q4BytesPerBlock
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
				be.MatMulQ4_0(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
				be.Sync()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				be.MatMulQ4_0(aBuf, bBuf, outBuf, cfg.m, cfg.n, cfg.k)
				be.Sync()
			}
		})

		be.Free(aBuf)
		be.Free(bBuf)
		be.Free(outBuf)
	}
}

// TestQ4_0PrefillThroughputReport generates a human-readable throughput report
// comparing Q4_0 matmul performance across batch sizes.
func TestQ4_0PrefillThroughputReport(t *testing.T) {
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

	// Create Q4_0 weight matrix
	numBlocksPerRow := (K + Q4BlockSize - 1) / Q4BlockSize
	bSize := N * numBlocksPerRow * Q4BytesPerBlock
	bBuf := be.Alloc(bSize)
	be.Zero(bBuf, bSize)
	be.Sync()

	fmt.Println("\n[Q4_0 BATCHED MATMUL THROUGHPUT]")
	fmt.Printf("Weight matrix: [%d, %d] Q4_0\n", N, K)
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
			be.MatMulQ4_0(aBuf, bBuf, outBuf, m, N, K)
			be.Sync()
		}

		start := time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulQ4_0(aBuf, bBuf, outBuf, m, N, K)
			be.Sync()
		}
		elapsed := time.Since(start) / time.Duration(iters)

		// Calculate GFLOPS: 2*M*N*K FLOPs (multiply-add)
		flops := float64(2) * float64(m) * float64(N) * float64(K)
		gflops := flops / float64(elapsed.Nanoseconds())

		// Determine which kernel path is used
		kernel := "unknown"
		if m == 1 {
			kernel = "multi_output"
		} else if m >= 8 {
			kernel = "simdgroup"
		} else {
			kernel = "batched"
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

// TestQ4_0PerDimensionGFLOPS measures GEMM throughput at every matmul dimension
// that occurs during a LLaMA 2 7B Q4_0 prefill forward pass. This identifies which
// dimensions are bottlenecks and establishes baseline GFLOPS for kernel optimization.
//
// Track: close_prefill_gap, Phase 0 Task 0.1.
func TestQ4_0PerDimensionGFLOPS(t *testing.T) {
	be, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer be.Close()

	const (
		M     = 128 // prefill sequence length
		iters = 50
	)

	// All matmul dimensions in one LLaMA 2 7B layer + lm_head
	// Layout: C[M,N] = A[M,K] × B[N,K]^T  (weights are [N,K] row-major)
	dims := []struct {
		name  string
		n     int
		k     int
		count int // how many times per layer (for total time estimate)
	}{
		{"Wq_4096x4096", 4096, 4096, 1},
		{"Wk_4096x4096", 4096, 4096, 1},
		{"Wv_4096x4096", 4096, 4096, 1},
		{"Wo_4096x4096", 4096, 4096, 1},
		{"W1_gate_11008x4096", 11008, 4096, 1},
		{"W3_up_11008x4096", 11008, 4096, 1},
		{"W2_down_4096x11008", 4096, 11008, 1},
		{"lm_head_32000x4096", 32000, 4096, 0}, // once total, not per layer
	}

	type result struct {
		name      string
		n, k      int
		count     int
		elapsed   time.Duration
		gflops    float64
		gbps      float64
	}

	results := make([]result, 0, len(dims))

	for _, d := range dims {
		// Create activation buffer [M, K]
		aSize := M * d.k
		aBuf := be.Alloc(aSize * 4)
		aData := make([]float32, aSize)
		rng := rand.New(rand.NewSource(42))
		for i := range aData {
			aData[i] = (rng.Float32() - 0.5) * 0.2
		}
		be.ToDevice(aBuf, float32ToBytes(aData))

		// Create Q4_0 weight buffer [N, K]
		numBlocksPerRow := (d.k + Q4BlockSize - 1) / Q4BlockSize
		bSize := d.n * numBlocksPerRow * Q4BytesPerBlock
		bBuf := be.Alloc(bSize)
		be.Zero(bBuf, bSize)

		// Output buffer [M, N]
		outSize := M * d.n
		outBuf := be.Alloc(outSize * 4)

		be.Sync()

		// Warmup
		for i := 0; i < 10; i++ {
			be.MatMulQ4_0(aBuf, bBuf, outBuf, M, d.n, d.k)
			be.Sync()
		}

		// Benchmark
		start := time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulQ4_0(aBuf, bBuf, outBuf, M, d.n, d.k)
			be.Sync()
		}
		elapsed := time.Since(start) / time.Duration(iters)

		flops := float64(2) * float64(M) * float64(d.n) * float64(d.k)
		gflops := flops / float64(elapsed.Nanoseconds())
		totalBytes := float64(aSize*4 + bSize + outSize*4)
		gbps := totalBytes / float64(elapsed.Nanoseconds())

		results = append(results, result{d.name, d.n, d.k, d.count, elapsed, gflops, gbps})

		be.Free(aBuf)
		be.Free(bBuf)
		be.Free(outBuf)
	}

	// Report
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Q4_0 GEMM THROUGHPUT — Per Dimension (M=128, LLaMA 2 7B)                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("\n%-25s %8s %8s %10s %10s %10s\n",
		"Dimension", "N", "K", "Time", "GFLOPS", "GB/s")
	fmt.Printf("%-25s %8s %8s %10s %10s %10s\n",
		"─────────", "─────", "─────", "────────", "──────", "────")

	var totalPerLayer time.Duration
	for _, r := range results {
		fmt.Printf("%-25s %8d %8d %10v %8.0f %8.1f\n",
			r.name, r.n, r.k, r.elapsed, r.gflops, r.gbps)
		if r.count > 0 {
			totalPerLayer += r.elapsed * time.Duration(r.count)
		}
	}

	// Per-layer total (7 matmuls: 4 attention + 3 MLP)
	totalModel := totalPerLayer * 32
	// Add lm_head
	for _, r := range results {
		if r.count == 0 {
			totalModel += r.elapsed
		}
	}
	prefillTokPerSec := float64(M) / totalModel.Seconds()

	fmt.Printf("\n── Per-Layer Matmul Total ──────────────────────────────────────────\n")
	fmt.Printf("  7 matmuls × 1 layer:  %v\n", totalPerLayer)
	fmt.Printf("  32 layers + lm_head:  %v\n", totalModel)
	fmt.Printf("  Estimated prefill:    %.0f tok/s (matmul only, no overhead)\n", prefillTokPerSec)
	fmt.Printf("  Measured prefill:     377 tok/s (with overhead)\n")
	fmt.Printf("  llama.cpp target:     803 tok/s\n")
	fmt.Println()
}
