//go:build metal && darwin && cgo

package metal

import (
	"math"
	"math/rand"
	"testing"
)

func TestMatMulQ4K_NR2_Parity(t *testing.T) {
	// Initialize backend
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Dimensions
	N := 256
	K := 256 // One block
	M := 1   // Matvec

	// Allocate A (F32) [1, K]
	sizeA := M * K * 4
	ptrA := b.Alloc(sizeA)
	defer b.Free(ptrA)

	// Allocate B (Q4_K) [N, K]
	sizeB := N * 144
	ptrB := b.Alloc(sizeB)
	defer b.Free(ptrB)

	// Allocate C (F32) [1, N]
	sizeC := M * N * 4
	ptrC1 := b.Alloc(sizeC)
	ptrC2 := b.Alloc(sizeC)
	defer b.Free(ptrC1)
	defer b.Free(ptrC2)

	// Initialize A with random data
	floatsA := make([]float32, K)
	for i := range floatsA {
		floatsA[i] = rand.Float32()
	}
	b.ToDevice(ptrA, float32ToBytes(floatsA))

	// Initialize B with random bytes
	bytesB := make([]byte, sizeB)
	rand.Read(bytesB)
	b.ToDevice(ptrB, bytesB)

	// Run with NR2 disabled (Reference)
	pipelineNR2 := b.matvecQ4KNR2Pipeline
	b.matvecQ4KNR2Pipeline = nil // Disable NR2
	
	b.MatMulQ4_K(ptrA, ptrB, ptrC1, M, N, K)
	b.Sync()

	// Read Result 1
	res1Bytes := make([]byte, sizeC)
	b.ToHost(res1Bytes, ptrC1)
	res1 := bytesToFloat32(res1Bytes)

	// Run with NR2 enabled (Target)
	b.matvecQ4KNR2Pipeline = pipelineNR2 // Restore
	b.MatMulQ4_K(ptrA, ptrB, ptrC2, M, N, K)
	b.Sync()

	// Read Result 2
	res2Bytes := make([]byte, sizeC)
	b.ToHost(res2Bytes, ptrC2)
	res2 := bytesToFloat32(res2Bytes)

	// Compare
	maxDiff := float32(0)
	for i := 0; i < N; i++ {
		diff := float32(math.Abs(float64(res1[i] - res2[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-3 {
			t.Errorf("Mismatch at %d: Ref=%f, NR2=%f, Diff=%f", i, res1[i], res2[i], diff)
			if i > 5 {
				t.FailNow()
			}
		}
	}
	t.Logf("Max difference: %f", maxDiff)
}

func BenchmarkMatMulQ4K_Reference(b_bench *testing.B) {
	b, err := NewBackend(0)
	if err != nil {
		b_bench.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Dimensions (Phi-2 size: N=10240, K=2560)
	N := 10240
	K := 2560
	M := 1

	sizeA := M * K * 4
	ptrA := b.Alloc(sizeA)
	defer b.Free(ptrA)

	// B size: N * (K/256 * 144)
	numBlocks := K / 256
	sizeB := N * numBlocks * 144
	ptrB := b.Alloc(sizeB)
	defer b.Free(ptrB)

	sizeC := M * N * 4
	ptrC := b.Alloc(sizeC)
	defer b.Free(ptrC)

	// Initialize data (optional for bench, but safer)
	b.ToDevice(ptrA, make([]byte, sizeA)) // Zeros
	
	// Disable NR2
	pipelineNR2 := b.matvecQ4KNR2Pipeline
	b.matvecQ4KNR2Pipeline = nil
	defer func() { b.matvecQ4KNR2Pipeline = pipelineNR2 }()

	b_bench.ResetTimer()
	for i := 0; i < b_bench.N; i++ {
		b.MatMulQ4_K(ptrA, ptrB, ptrC, M, N, K)
		b.Sync()
	}
}

func BenchmarkMatMulQ4K_NR2(b_bench *testing.B) {
	b, err := NewBackend(0)
	if err != nil {
		b_bench.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	N := 10240
	K := 2560
	M := 1

	sizeA := M * K * 4
	ptrA := b.Alloc(sizeA)
	defer b.Free(ptrA)

	numBlocks := K / 256
	sizeB := N * numBlocks * 144
	ptrB := b.Alloc(sizeB)
	defer b.Free(ptrB)

	sizeC := M * N * 4
	ptrC := b.Alloc(sizeC)
	defer b.Free(ptrC)

	b.ToDevice(ptrA, make([]byte, sizeA))

	// Ensure NR2 is enabled
	if b.matvecQ4KNR2Pipeline == nil {
		b_bench.Fatal("NR2 pipeline not available")
	}

	b_bench.ResetTimer()
	for i := 0; i < b_bench.N; i++ {
		b.MatMulQ4_K(ptrA, ptrB, ptrC, M, N, K)
		b.Sync()
	}
}
