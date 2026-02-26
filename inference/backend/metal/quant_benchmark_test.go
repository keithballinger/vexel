//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// TestQuantFormatComparisonReport generates a comprehensive comparison of all
// supported quantization formats at standard LLaMA dimensions.
//
// Track 4: Quantization Expansion, Phase 3 Task 2.
func TestQuantFormatComparisonReport(t *testing.T) {
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

	batchSizes := []int{1, 32, 128}

	rng := rand.New(rand.NewSource(42))

	// Pre-allocate all weight buffers
	numBlocksQ4 := (K + Q4BlockSize - 1) / Q4BlockSize
	bSizeQ4 := N * numBlocksQ4 * Q4BytesPerBlock
	bBufQ4 := be.Alloc(bSizeQ4)
	be.Zero(bBufQ4, bSizeQ4)

	numBlocksQ4K := (K + Q4KBlockSize - 1) / Q4KBlockSize
	bSizeQ4K := N * numBlocksQ4K * Q4KBytesPerBlock
	bBufQ4K := be.Alloc(bSizeQ4K)
	be.Zero(bBufQ4K, bSizeQ4K)

	numBlocksQ8 := (K + Q8_0BlockSize - 1) / Q8_0BlockSize
	bSizeQ8 := N * numBlocksQ8 * Q8_0BytesPerBlock
	bBufQ8 := be.Alloc(bSizeQ8)
	be.Zero(bBufQ8, bSizeQ8)

	bSizeBF16 := N * K * 2
	bBufBF16 := be.Alloc(bSizeBF16)
	be.Zero(bBufBF16, bSizeBF16)

	bSizeF32 := N * K * 4
	bBufF32 := be.Alloc(bSizeF32)
	be.Zero(bBufF32, bSizeF32)

	be.Sync()

	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║        QUANTIZATION FORMAT COMPARISON — LLaMA 4096×4096                ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	for _, m := range batchSizes {
		aSize := m * K
		outSize := m * N

		aBuf := be.Alloc(aSize * 4)
		outBuf := be.Alloc(outSize * 4)

		aData := make([]float32, aSize)
		for i := range aData {
			aData[i] = (rng.Float32() - 0.5) * 0.2
		}
		be.ToDevice(aBuf, float32ToBytes(aData))
		be.Sync()

		label := "prefill"
		if m == 1 {
			label = "decode"
		}
		fmt.Printf("\n┌─── M=%d (%s) ──────────────────────────────────────────────────────┐\n", m, label)
		fmt.Printf("│ %-6s │ %6s │ %12s │ %10s │ %10s │ %10s │\n",
			"Format", "BPW", "Time/iter", "GFLOPS", "GB/s", "WeightMB")
		fmt.Printf("│ %-6s │ %6s │ %12s │ %10s │ %10s │ %10s │\n",
			"------", "---", "--------", "------", "----", "--------")

		flops := float64(2) * float64(m) * float64(N) * float64(K)

		// Q4_0
		for i := 0; i < 10; i++ {
			be.MatMulQ4_0(aBuf, bBufQ4, outBuf, m, N, K)
			be.Sync()
		}
		start := time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulQ4_0(aBuf, bBufQ4, outBuf, m, N, K)
			be.Sync()
		}
		elapsed := time.Since(start) / time.Duration(iters)
		fmt.Printf("│ %-6s │ %6.1f │ %12v │ %10.1f │ %8.1f │ %8.1f │\n",
			"Q4_0", 4.5, elapsed, flops/float64(elapsed.Nanoseconds()),
			float64(aSize*4+bSizeQ4+outSize*4)/float64(elapsed.Nanoseconds()),
			float64(bSizeQ4)/1024/1024)

		// Q4_K
		for i := 0; i < 10; i++ {
			be.MatMulQ4_K(aBuf, bBufQ4K, outBuf, m, N, K)
			be.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulQ4_K(aBuf, bBufQ4K, outBuf, m, N, K)
			be.Sync()
		}
		elapsed = time.Since(start) / time.Duration(iters)
		fmt.Printf("│ %-6s │ %6.1f │ %12v │ %10.1f │ %8.1f │ %8.1f │\n",
			"Q4_K", 4.5, elapsed, flops/float64(elapsed.Nanoseconds()),
			float64(aSize*4+bSizeQ4K+outSize*4)/float64(elapsed.Nanoseconds()),
			float64(bSizeQ4K)/1024/1024)

		// Q8_0
		for i := 0; i < 10; i++ {
			be.MatMulQ8_0(aBuf, bBufQ8, outBuf, m, N, K)
			be.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulQ8_0(aBuf, bBufQ8, outBuf, m, N, K)
			be.Sync()
		}
		elapsed = time.Since(start) / time.Duration(iters)
		fmt.Printf("│ %-6s │ %6.1f │ %12v │ %10.1f │ %8.1f │ %8.1f │\n",
			"Q8_0", 8.5, elapsed, flops/float64(elapsed.Nanoseconds()),
			float64(aSize*4+bSizeQ8+outSize*4)/float64(elapsed.Nanoseconds()),
			float64(bSizeQ8)/1024/1024)

		// BF16
		for i := 0; i < 10; i++ {
			be.MatMulBF16(aBuf, bBufBF16, outBuf, m, N, K)
			be.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulBF16(aBuf, bBufBF16, outBuf, m, N, K)
			be.Sync()
		}
		elapsed = time.Since(start) / time.Duration(iters)
		fmt.Printf("│ %-6s │ %6.1f │ %12v │ %10.1f │ %8.1f │ %8.1f │\n",
			"BF16", 16.0, elapsed, flops/float64(elapsed.Nanoseconds()),
			float64(aSize*4+bSizeBF16+outSize*4)/float64(elapsed.Nanoseconds()),
			float64(bSizeBF16)/1024/1024)

		// F32
		for i := 0; i < 10; i++ {
			be.MatMulTransposed(aBuf, bBufF32, outBuf, m, N, K)
			be.Sync()
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			be.MatMulTransposed(aBuf, bBufF32, outBuf, m, N, K)
			be.Sync()
		}
		elapsed = time.Since(start) / time.Duration(iters)
		fmt.Printf("│ %-6s │ %6.1f │ %12v │ %10.1f │ %8.1f │ %8.1f │\n",
			"F32", 32.0, elapsed, flops/float64(elapsed.Nanoseconds()),
			float64(aSize*4+bSizeF32+outSize*4)/float64(elapsed.Nanoseconds()),
			float64(bSizeF32)/1024/1024)

		fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

		be.Free(aBuf)
		be.Free(outBuf)
	}

	// Model size comparison for 7B parameter model
	fmt.Println("\n┌─── MODEL SIZE ESTIMATES (7B parameters) ──────────────────────────────┐")
	fmt.Printf("│ %-6s │ %6s │ %10s │ %-40s │\n", "Format", "BPW", "Size (GB)", "Notes")
	fmt.Printf("│ %-6s │ %6s │ %10s │ %-40s │\n", "------", "---", "---------", "-----")
	params := float64(7e9)
	fmt.Printf("│ %-6s │ %6.1f │ %8.1f │ %-40s │\n", "Q4_0", 4.5, params*4.5/8/1e9, "Fastest decode, lowest quality")
	fmt.Printf("│ %-6s │ %6.1f │ %8.1f │ %-40s │\n", "Q4_K", 4.5, params*4.5/8/1e9, "Better quality than Q4_0 (k-quants)")
	fmt.Printf("│ %-6s │ %6.1f │ %8.1f │ %-40s │\n", "Q8_0", 8.5, params*8.5/8/1e9, "Near-F16 quality, 2x Q4 size")
	fmt.Printf("│ %-6s │ %6.1f │ %8.1f │ %-40s │\n", "BF16", 16.0, params*16.0/8/1e9, "Full precision, largest size")
	fmt.Printf("│ %-6s │ %6.1f │ %8.1f │ %-40s │\n", "F32", 32.0, params*32.0/8/1e9, "Reference only (not recommended)")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	be.Free(bBufQ4)
	be.Free(bBufQ4K)
	be.Free(bBufQ8)
	be.Free(bBufBF16)
	be.Free(bBufF32)
}
