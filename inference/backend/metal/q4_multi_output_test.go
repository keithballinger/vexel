//go:build metal && darwin && cgo

package metal

import (
	"math"
	"testing"
)

// runQ4MultiOutputKernel exercises the multi-output matvec path (NR2)
// by invoking the explicit multi-output kernel via backend helper.
func runQ4MultiOutputKernel(t *testing.T, b *Backend, a []float32, bQ4 []byte, n, k int) []float32 {
	aBuf := b.Alloc(len(a) * 4)
	bBuf := b.Alloc(len(bQ4))
	outBuf := b.Alloc(n * 4)
	defer b.Free(aBuf)
	defer b.Free(bBuf)
	defer b.Free(outBuf)

	b.ToDevice(aBuf, float32ToBytes(a))
	b.ToDevice(bBuf, bQ4)

	b.MatVecQ4_0MultiOutput(aBuf, bBuf, outBuf, n, k)
	b.Sync()

	resultBytes := make([]byte, n*4)
	b.ToHost(resultBytes, outBuf)
	return bytesToFloat32(resultBytes)
}

// TestQ4_0MultiOutputKernel_PartialTiles ensures correctness when N is not a multiple
// of 8 outputs-per-threadgroup and K spans multiple blocks with a partial tail.
func TestQ4_0MultiOutputKernel_PartialTiles(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// M = 1 (matvec), N = 9 (forces 2 threadgroups), K = 48 (1 full block + partial)
	const M, N, K = 1, 9, 48

	// Activation pattern with mixed signs to stress reduction accuracy.
	a := make([]float32, K)
	for i := range a {
		// Alternating +/- values with increasing magnitude.
		sign := float32(1.0)
		if i%2 == 1 {
			sign = -1.0
		}
		a[i] = sign * float32((i%7)+1) * 0.25
	}

	// Build Q4_0 weight matrix with row-dependent pattern.
	bQ4 := createQ4_0Matrix(N, K, func(row, col int) int {
		// Keep values in a small, non-wrapping range for deterministic checks.
		return ((row + col) % 7) - 3 // Range [-3..3] → dequant [-3..3]
	})

	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)
	result := runQ4MultiOutputKernel(t, b, a, bQ4, N, K)

	var maxDiff float64
	var mismatches []int
	for i := 0; i < N; i++ {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-3 {
			mismatches = append(mismatches, i)
		}
	}
	if len(mismatches) > 0 {
		t.Fatalf("mismatches=%v maxDiff=%f gpu=%v cpu=%v", mismatches, maxDiff, result, expected)
	}
	t.Logf("max diff: %f", maxDiff)
}

// TestQ4_0MultiOutputKernel_LargeN validates correctness across many threadgroups.
func TestQ4_0MultiOutputKernel_LargeN(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// M = 1, N = 33 (5 threadgroups), K = 96 (3 full blocks)
	const M, N, K = 1, 33, 96

	// Activation pattern with fractional values to surface accumulation error.
	a := make([]float32, K)
	for i := range a {
		a[i] = float32((i%11)-5) * 0.125
	}

	bQ4 := createQ4_0Matrix(N, K, func(row, col int) int {
		// Vary by row and block position in a safe 4-bit range.
		return ((row*3)+col)%7 - 3 // Range [-3..3]
	})

	expected := cpuMatMulQ4_0(a, bQ4, M, N, K)
	result := runQ4MultiOutputKernel(t, b, a, bQ4, N, K)

	var maxDiff float64
	for i := 0; i < N; i++ {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-3 {
			t.Fatalf("output %d mismatch: gpu=%f cpu=%f diff=%f", i, result[i], expected[i], diff)
		}
	}
	t.Logf("max diff: %f", maxDiff)
}
