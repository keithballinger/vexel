//go:build metal && darwin && cgo

package metal

import (
	"testing"
)

func BenchmarkMLP_NonFused(b_bench *testing.B) {
	b, err := NewBackend(0)
	if err != nil {
		b_bench.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	// Dimensions (Phi-2: Intermediate=10240, Hidden=2560)
	N := 10240
	K := 2560
	
	// Buffers
	xBuf := b.Alloc(K * 4)
	w1Buf := b.Alloc((K/32) * 18 * N)
	w3Buf := b.Alloc((K/32) * 18 * N)
	gateBuf := b.Alloc(N * 4)
	upBuf := b.Alloc(N * 4)
	outBuf := b.Alloc(N * 4)
	
	defer b.Free(xBuf)
	defer b.Free(w1Buf)
	defer b.Free(w3Buf)
	defer b.Free(gateBuf)
	defer b.Free(upBuf)
	defer b.Free(outBuf)
	
	// Fill with something
	b.Zero(xBuf, K)
	
	b_bench.ResetTimer()
	for i := 0; i < b_bench.N; i++ {
		// Non-fused: 3 kernel launches
		b.MatVecQ4_0MultiOutput(xBuf, w1Buf, gateBuf, N, K)
		b.MatVecQ4_0MultiOutput(xBuf, w3Buf, upBuf, N, K)
		b.SiLUMul(gateBuf, upBuf, outBuf, N)
		b.Sync()
	}
}

func BenchmarkMLP_Fused(b_bench *testing.B) {
	b, err := NewBackend(0)
	if err != nil {
		b_bench.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	N := 10240
	K := 2560
	
	xBuf := b.Alloc(K * 4)
	w1Buf := b.Alloc((K/32) * 18 * N)
	w3Buf := b.Alloc((K/32) * 18 * N)
	outBuf := b.Alloc(N * 4)
	
	defer b.Free(xBuf)
	defer b.Free(w1Buf)
	defer b.Free(w3Buf)
	defer b.Free(outBuf)
	
	b.Zero(xBuf, K)
	
	b_bench.ResetTimer()
	for i := 0; i < b_bench.N; i++ {
		// Fused: 1 kernel launch
		b.MatMulQ4_0_FusedMLP(xBuf, w1Buf, w3Buf, outBuf, 1, N, K)
		b.Sync()
	}
}
