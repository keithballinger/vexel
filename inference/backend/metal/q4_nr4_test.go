//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// runQ4NR4Kernel directly exercises the NR4 matvec kernel.
func runQ4NR4Kernel(t *testing.T, b *Backend, a []float32, bQ4 []byte, n, k int) []float32 {
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(n * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)

	b.MatVecQ4_0NR4(aBuf, bBuf, outBuf, n, k)
	b.Sync()

	resultBytes := make([]byte, n*4)
	b.ToHost(resultBytes, outBuf)
	return bytesToFloat32(resultBytes)
}

// TestQ4_0NR4_Correctness tests the NR4 kernel against CPU reference across
// all LLaMA 2 7B layer sizes and edge cases.
func TestQ4_0NR4_Correctness(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	if b.matvecQ4NR4Pipeline == nil {
		t.Skip("NR4 pipeline not available")
	}

	tests := []struct {
		name string
		n    int
		k    int
		tol  float64
	}{
		// Small edge cases
		{"N=1_K=32", 1, 32, 1e-3},
		{"N=4_K=32", 4, 32, 1e-3},
		{"N=7_K=32", 7, 32, 1e-3},   // N not divisible by 32
		{"N=8_K=64", 8, 64, 1e-3},   // Exactly 1 multi_output TG
		{"N=31_K=32", 31, 32, 1e-3}, // One less than NR4 TG
		{"N=32_K=32", 32, 32, 1e-3}, // Exactly 1 NR4 TG
		{"N=33_K=64", 33, 64, 1e-3}, // Partial second TG

		// LLaMA 2 7B dimensions
		{"N=4096_K=4096", 4096, 4096, 1e-2},   // Attention qkv, output
		{"N=11008_K=4096", 11008, 4096, 1e-2},  // MLP up/gate
		{"N=4096_K=11008", 4096, 11008, 2e-2},  // MLP down (large K → more accumulation error)
		{"N=32000_K=4096", 32000, 4096, 1e-2},  // LM head (vocab)
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n, k := tc.n, tc.k

			// Activation pattern with mixed signs
			a := make([]float32, k)
			for i := range a {
				sign := float32(1.0)
				if i%3 == 1 {
					sign = -1.0
				}
				a[i] = sign * float32((i%11)+1) * 0.1
			}

			// Weight matrix with row-dependent pattern
			bQ4 := createQ4_0Matrix(n, k, func(row, col int) int {
				return ((row*3 + col*7) % 7) - 3 // Range [-3..3]
			})

			expected := cpuMatMulQ4_0(a, bQ4, 1, n, k)
			result := runQ4NR4Kernel(t, b, a, bQ4, n, k)

			maxDiff := 0.0
			maxDiffIdx := 0
			for i := 0; i < n; i++ {
				if math.IsNaN(float64(result[i])) || math.IsInf(float64(result[i]), 0) {
					t.Fatalf("NaN/Inf at output %d", i)
				}
				diff := math.Abs(float64(result[i] - expected[i]))
				if diff > maxDiff {
					maxDiff = diff
					maxDiffIdx = i
				}
			}

			t.Logf("N=%d K=%d: max_diff=%.6f at idx=%d (tol=%.4f)", n, k, maxDiff, maxDiffIdx, tc.tol)
			if maxDiff > tc.tol {
				t.Fatalf("NR4 mismatch: diff=%.6f > tol=%.4f at idx=%d (expected=%.6f, got=%.6f)",
					maxDiff, tc.tol, maxDiffIdx, expected[maxDiffIdx], result[maxDiffIdx])
			}
		})
	}
}

// TestQ4_0NR4_vs_MultiOutput compares NR4 output against multi_output to
// verify both kernels produce identical results.
func TestQ4_0NR4_vs_MultiOutput(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	if b.matvecQ4NR4Pipeline == nil {
		t.Skip("NR4 pipeline not available")
	}
	if b.matvecQ4MultiOutputPipeline == nil {
		t.Skip("Multi-output pipeline not available")
	}

	// Test with all LLaMA 2 7B layer sizes
	sizes := []struct {
		n, k int
	}{
		{4096, 4096},
		{11008, 4096},
		{4096, 11008},
		{32000, 4096},
	}

	for _, sz := range sizes {
		t.Run(fmt.Sprintf("N=%d_K=%d", sz.n, sz.k), func(t *testing.T) {
			n, k := sz.n, sz.k

			a := make([]float32, k)
			for i := range a {
				a[i] = float32(i%17-8) * 0.05
			}

			bQ4 := createQ4_0Matrix(n, k, func(row, col int) int {
				return ((row + col*5) % 7) - 3
			})

			// Run both kernels
			resultMultiOutput := runQ4MultiOutputKernel(t, b, a, bQ4, n, k)
			resultNR4 := runQ4NR4Kernel(t, b, a, bQ4, n, k)

			// Compare outputs
			maxDiff := 0.0
			maxDiffIdx := 0
			for i := 0; i < n; i++ {
				diff := math.Abs(float64(resultNR4[i] - resultMultiOutput[i]))
				if diff > maxDiff {
					maxDiff = diff
					maxDiffIdx = i
				}
			}

			t.Logf("N=%d K=%d: NR4 vs multi_output max_diff=%.6f at idx=%d", n, k, maxDiff, maxDiffIdx)
			// Both should produce identical results (same float operations)
			if maxDiff > 1e-4 {
				t.Fatalf("NR4 vs multi_output divergence: diff=%.6f at idx=%d (nr4=%.6f, multi=%.6f)",
					maxDiff, maxDiffIdx, resultNR4[maxDiffIdx], resultMultiOutput[maxDiffIdx])
			}
		})
	}
}

// BenchmarkQ4_0_NR4_vs_MultiOutput runs an A/B comparison of the NR4 and
// multi_output kernels at each LLaMA 2 7B layer size.
func BenchmarkQ4_0_NR4_vs_MultiOutput(b *testing.B) {
	backend, err := NewBackend(0)
	if err != nil {
		b.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4NR4Pipeline == nil {
		b.Skip("NR4 pipeline not available")
	}

	sizes := []struct {
		name string
		n, k int
	}{
		{"attn_4096x4096", 4096, 4096},
		{"mlp_up_11008x4096", 11008, 4096},
		{"mlp_down_4096x11008", 4096, 11008},
		{"lm_head_32000x4096", 32000, 4096},
	}

	for _, sz := range sizes {
		// Create test data
		a := make([]float32, sz.k)
		for i := range a {
			a[i] = float32(i%17-8) * 0.05
		}
		bQ4 := createQ4_0Matrix(sz.n, sz.k, func(row, col int) int {
			return ((row + col*5) % 7) - 3
		})

		aBuf := backend.Alloc(len(a) * 4)
		bBuf := backend.Alloc(len(bQ4))
		outBuf := backend.Alloc(sz.n * 4)

		backend.ToDevice(aBuf, float32ToBytes(a))
		backend.ToDevice(bBuf, bQ4)
		backend.Sync()

		b.Run("multi_output/"+sz.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				backend.MatVecQ4_0MultiOutput(aBuf, bBuf, outBuf, sz.n, sz.k)
			}
			backend.Sync()
		})

		b.Run("nr4/"+sz.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				backend.MatVecQ4_0NR4(aBuf, bBuf, outBuf, sz.n, sz.k)
			}
			backend.Sync()
		})

		backend.Free(aBuf)
		backend.Free(bBuf)
		backend.Free(outBuf)
	}
}

// TestQ4_0_NR4_vs_MultiOutput_Throughput is a non-benchmark throughput test
// that prints a comparison table. More useful than Go benchmarks for GPU
// kernels since it can control warmup and iteration count.
func TestQ4_0_NR4_vs_MultiOutput_Throughput(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping throughput test in short mode")
	}

	backend, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer backend.Close()

	if backend.matvecQ4NR4Pipeline == nil {
		t.Skip("NR4 pipeline not available")
	}

	sizes := []struct {
		name string
		n, k int
	}{
		{"attn_4096x4096", 4096, 4096},
		{"mlp_up_11008x4096", 11008, 4096},
		{"mlp_down_4096x11008", 4096, 11008},
		{"lm_head_32000x4096", 32000, 4096},
	}

	t.Logf("%-25s %12s %12s %8s", "Layer", "MultiOutput", "NR4", "Speedup")
	t.Logf("%-25s %12s %12s %8s", "-----", "-----------", "---", "-------")

	for _, sz := range sizes {
		a := make([]float32, sz.k)
		for i := range a {
			a[i] = float32(i%17-8) * 0.05
		}
		bQ4 := createQ4_0Matrix(sz.n, sz.k, func(row, col int) int {
			return ((row + col*5) % 7) - 3
		})

		aBuf := backend.Alloc(len(a) * 4)
		bBuf := backend.Alloc(len(bQ4))
		outBuf := backend.Alloc(sz.n * 4)

		backend.ToDevice(aBuf, float32ToBytes(a))
		backend.ToDevice(bBuf, bQ4)
		backend.Sync()

		warmup := 50
		iters := 200

		// Warmup
		for i := 0; i < warmup; i++ {
			backend.MatVecQ4_0MultiOutput(aBuf, bBuf, outBuf, sz.n, sz.k)
		}
		backend.Sync()

		// Benchmark multi_output
		start := time.Now()
		for i := 0; i < iters; i++ {
			backend.MatVecQ4_0MultiOutput(aBuf, bBuf, outBuf, sz.n, sz.k)
		}
		backend.Sync()
		multiDur := time.Since(start)
		multiUs := float64(multiDur.Microseconds()) / float64(iters)

		// Warmup NR4
		for i := 0; i < warmup; i++ {
			backend.MatVecQ4_0NR4(aBuf, bBuf, outBuf, sz.n, sz.k)
		}
		backend.Sync()

		// Benchmark NR4
		start = time.Now()
		for i := 0; i < iters; i++ {
			backend.MatVecQ4_0NR4(aBuf, bBuf, outBuf, sz.n, sz.k)
		}
		backend.Sync()
		nr4Dur := time.Since(start)
		nr4Us := float64(nr4Dur.Microseconds()) / float64(iters)

		speedup := multiUs / nr4Us

		t.Logf("%-25s %10.1f µs %10.1f µs %7.2fx", sz.name, multiUs, nr4Us, speedup)

		backend.Free(aBuf)
		backend.Free(bBuf)
		backend.Free(outBuf)
	}
}
