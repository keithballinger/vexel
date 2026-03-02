//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"vexel/inference/tensor"
)

// TestQ4KKernelProfile measures individual kernel times at LLaMA 2 7B dimensions.
// Uses batched dispatches (32 back-to-back with barriers, single Sync) to accurately
// measure GPU-side kernel time without per-call Sync overhead.
//
// LLaMA 2 7B dimensions:
//   hidden=4096, heads=32, head_dim=128, intermediate=11008, vocab=32000
func TestQ4KKernelProfile(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		hidden       = 4096
		intermediate = 11008
		batchSize    = 32 // Simulate 32 layers
		warmup       = 10
		iters        = 50
	)

	q4kSize := func(n, k int) int {
		return n * (k / 256) * 144
	}

	// Allocate buffers
	xBuf := b.Alloc(hidden * 4)
	normBuf := b.Alloc(hidden * 4)
	outHidden := b.Alloc(hidden * 4)
	outInter := b.Alloc(intermediate * 4)
	attnF16 := b.Alloc(hidden * 2)

	wo := b.Alloc(q4kSize(hidden, hidden))
	w1 := b.Alloc(q4kSize(intermediate, hidden))
	w3 := b.Alloc(q4kSize(intermediate, hidden))
	w2 := b.Alloc(q4kSize(hidden, intermediate))

	defer b.Free(xBuf)
	defer b.Free(normBuf)
	defer b.Free(outHidden)
	defer b.Free(outInter)
	defer b.Free(attnF16)
	defer b.Free(wo)
	defer b.Free(w1)
	defer b.Free(w3)
	defer b.Free(w2)

	initRand := func(ptr tensor.DevicePtr, size int) {
		data := make([]byte, size)
		rand.Read(data)
		b.ToDevice(ptr, data)
	}
	initRand(xBuf, hidden*4)
	initRand(normBuf, hidden*4)
	initRand(attnF16, hidden*2)
	initRand(wo, q4kSize(hidden, hidden))
	initRand(w1, q4kSize(intermediate, hidden))
	initRand(w3, q4kSize(intermediate, hidden))
	initRand(w2, q4kSize(hidden, intermediate))

	type kernelResult struct {
		name     string
		avgUs    float64 // per-dispatch (not per-batch)
		weightMB float64
		bwGBs    float64
	}

	// benchKernel dispatches the kernel `batchSize` times in a single command buffer
	// (with barriers between dispatches), then Syncs once. This measures true GPU time.
	benchKernel := func(name string, weightBytes int, fn func()) kernelResult {
		// Warmup
		for i := 0; i < warmup; i++ {
			b.BeginBatch()
			for j := 0; j < batchSize; j++ {
				fn()
				if j < batchSize-1 {
					b.MemoryBarrier()
				}
			}
			b.EndBatch()
			b.Sync()
		}

		// Measure
		times := make([]float64, iters)
		for i := 0; i < iters; i++ {
			start := time.Now()
			b.BeginBatch()
			for j := 0; j < batchSize; j++ {
				fn()
				if j < batchSize-1 {
					b.MemoryBarrier()
				}
			}
			b.EndBatch()
			b.Sync()
			times[i] = float64(time.Since(start).Microseconds())
		}

		// Trimmed mean (middle 80%)
		sortFloat64s(times)
		lo := iters / 10
		hi := iters - lo
		sum := 0.0
		for i := lo; i < hi; i++ {
			sum += times[i]
		}
		avgBatch := sum / float64(hi-lo)
		avgPerDispatch := avgBatch / float64(batchSize)

		weightMB := float64(weightBytes) / (1024 * 1024)
		bwGBs := weightMB / 1024 / (avgPerDispatch / 1e6)

		return kernelResult{name, avgPerDispatch, weightMB, bwGBs}
	}

	var results []kernelResult

	// 1. FusedRMSNormQKV (Q4_K) — 3×[4096, 4096]
	t.Log("Benchmarking FusedRMSNormQKV (Q4_K, 3×[4096,4096])...")
	{
		qBuf := b.Alloc(hidden * 2)
		kBuf := b.Alloc(hidden * 2)
		vBuf := b.Alloc(hidden * 2)
		wq := b.Alloc(q4kSize(hidden, hidden))
		wk := b.Alloc(q4kSize(hidden, hidden))
		wv := b.Alloc(q4kSize(hidden, hidden))
		defer b.Free(qBuf)
		defer b.Free(kBuf)
		defer b.Free(vBuf)
		defer b.Free(wq)
		defer b.Free(wk)
		defer b.Free(wv)
		initRand(wq, q4kSize(hidden, hidden))
		initRand(wk, q4kSize(hidden, hidden))
		initRand(wv, q4kSize(hidden, hidden))

		r := benchKernel("FusedRMSNormQKV (Q4K)", 3*q4kSize(hidden, hidden),
			func() {
				b.MatMulQ4_K_FusedRMSNormQKV_F16(xBuf, normBuf, wq, wk, wv,
					qBuf, kBuf, vBuf, hidden, hidden, hidden, 1e-5)
			})
		results = append(results, r)
	}

	// 2. Wo+Add (F16 input) — [4096, 4096]
	t.Log("Benchmarking Wo+Add (Q4K F16in, [4096,4096])...")
	{
		r := benchKernel("Wo+Add (Q4K f16in)", q4kSize(hidden, hidden),
			func() {
				b.MatMulQ4_K_F16InAdd(attnF16, wo, outHidden, hidden, hidden)
			})
		results = append(results, r)
	}

	// 3. FusedRMSNormMLP (W1+W3) — 2×[11008, 4096]
	t.Log("Benchmarking FusedRMSNormMLP (Q4K, 2×[11008,4096])...")
	{
		r := benchKernel("FusedRMSNormMLP (Q4K)", 2*q4kSize(intermediate, hidden),
			func() {
				b.MatMulQ4_K_FusedRMSNormMLP(xBuf, normBuf, w1, w3, outInter,
					intermediate, hidden, 1e-5)
			})
		results = append(results, r)
	}

	// 4. W2+Add — [4096, 11008]
	t.Log("Benchmarking W2+Add (Q4K, [4096,11008])...")
	{
		r := benchKernel("W2+Add (Q4K)", q4kSize(hidden, intermediate),
			func() {
				b.MatMulQ4_K_Add(outInter, w2, outHidden, hidden, intermediate)
			})
		results = append(results, r)
	}

	// Print results
	t.Log("")
	t.Log("=== Q4_K Kernel Profile (LLaMA 2 7B dims, 32-batched) ===")
	t.Logf("%-30s %10s %10s %10s %8s", "Kernel", "Time/call", "Weight(MB)", "BW(GB/s)", "Util%")
	t.Logf("%-30s %10s %10s %10s %8s", "──────", "─────────", "─────────", "────────", "─────")

	totalUs := 0.0
	for _, r := range results {
		util := r.bwGBs / 400.0 * 100
		t.Logf("%-30s %8.1f µs %9.2f %9.1f %7.1f%%",
			r.name, r.avgUs, r.weightMB, r.bwGBs, util)
		totalUs += r.avgUs
	}

	// Also add SDPA estimate (negligible at ctx=16)
	t.Logf("%-30s %8s µs %9s %9s %7s", "SDPA F16 (ctx=16, est)", "~5", "0.27", "-", "-")

	t.Log("")
	t.Logf("Sum of kernel times (1 layer): %.1f µs (excl SDPA)", totalUs)
	t.Logf("Projected 32-layer total:      %.2f ms", totalUs*32/1000)
	t.Logf("Actual token time (71.2 tok/s): %.2f ms", 1000.0/71.2)
	overhead := 1000.0/71.2 - totalUs*32/1000
	t.Logf("Non-kernel overhead:           %.2f ms (%.1f%%)", overhead, overhead/(1000.0/71.2)*100)
	totalWeightMB := 0.0
	for _, r := range results {
		totalWeightMB += r.weightMB
	}
	t.Logf("Total weight per layer:        %.2f MB", totalWeightMB)
	t.Logf("Overall effective BW:          %.1f GB/s", totalWeightMB/1024/(totalUs/1e6))
}

// sortFloat64s sorts a float64 slice in place (insertion sort for small arrays).
func sortFloat64s(s []float64) {
	for i := 1; i < len(s); i++ {
		key := s[i]
		j := i - 1
		for j >= 0 && s[j] > key {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = key
	}
}

// TestQ4KvsQ4_0KernelComparison benchmarks Q4_0 vs Q4_K FusedRMSNormMLP at
// LLaMA 2 7B dimensions using batched execution for accurate GPU timing.
func TestQ4KvsQ4_0KernelComparison(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		N         = 11008
		K         = 4096
		batchSize = 32
		warmup    = 10
		iters     = 50
	)

	xBuf := b.Alloc(K * 4)
	outBuf := b.Alloc(N * 4)
	normBuf := b.Alloc(K * 4)

	q4_0_size := N * (K / 32) * 18
	wQ4_0_1 := b.Alloc(q4_0_size)
	wQ4_0_3 := b.Alloc(q4_0_size)

	q4k_size := N * (K / 256) * 144
	wQ4K_1 := b.Alloc(q4k_size)
	wQ4K_3 := b.Alloc(q4k_size)

	defer b.Free(xBuf)
	defer b.Free(outBuf)
	defer b.Free(normBuf)
	defer b.Free(wQ4_0_1)
	defer b.Free(wQ4_0_3)
	defer b.Free(wQ4K_1)
	defer b.Free(wQ4K_3)

	initRand := func(ptr tensor.DevicePtr, size int) {
		data := make([]byte, size)
		rand.Read(data)
		b.ToDevice(ptr, data)
	}
	initRand(xBuf, K*4)
	initRand(normBuf, K*4)
	initRand(wQ4_0_1, q4_0_size)
	initRand(wQ4_0_3, q4_0_size)
	initRand(wQ4K_1, q4k_size)
	initRand(wQ4K_3, q4k_size)

	measure := func(name string, weightBytes int, fn func()) float64 {
		for i := 0; i < warmup; i++ {
			b.BeginBatch()
			for j := 0; j < batchSize; j++ {
				fn()
				if j < batchSize-1 {
					b.MemoryBarrier()
				}
			}
			b.EndBatch()
			b.Sync()
		}
		times := make([]float64, iters)
		for i := 0; i < iters; i++ {
			start := time.Now()
			b.BeginBatch()
			for j := 0; j < batchSize; j++ {
				fn()
				if j < batchSize-1 {
					b.MemoryBarrier()
				}
			}
			b.EndBatch()
			b.Sync()
			times[i] = float64(time.Since(start).Microseconds())
		}
		sortFloat64s(times)
		lo := iters / 10
		hi := iters - lo
		sum := 0.0
		for i := lo; i < hi; i++ {
			sum += times[i]
		}
		avgBatch := sum / float64(hi-lo)
		avgPerCall := avgBatch / float64(batchSize)
		wMB := float64(weightBytes) / (1024 * 1024)
		bw := wMB / 1024 / (avgPerCall / 1e6)
		t.Logf("%-40s %8.1f µs/call  %6.1f GB/s (%4.1f%%)", name, avgPerCall, bw, bw/400*100)
		return avgPerCall
	}

	t.Log("=== FusedRMSNormMLP: Q4_0 vs Q4_K (32-batched, LLaMA 2 7B dims) ===")
	t.Logf("Dimensions: N=%d, K=%d", N, K)
	t.Logf("Q4_0 weight: %.2f MB each (2× for W1+W3)", float64(q4_0_size)/(1024*1024))
	t.Logf("Q4_K weight: %.2f MB each (2× for W1+W3)", float64(q4k_size)/(1024*1024))
	t.Log("")

	t0 := measure("Q4_0 FusedRMSNormMLP (W1+W3)", 2*q4_0_size, func() {
		b.MatMulQ4_0_FusedRMSNormMLP(xBuf, normBuf, wQ4_0_1, wQ4_0_3, outBuf,
			N, K, 1e-5)
	})

	tK := measure("Q4_K FusedRMSNormMLP (W1+W3)", 2*q4k_size, func() {
		b.MatMulQ4_K_FusedRMSNormMLP(xBuf, normBuf, wQ4K_1, wQ4K_3, outBuf,
			N, K, 1e-5)
	})

	t.Log("")
	t.Logf("Q4_K vs Q4_0: %.1f%% slower (%.1f µs overhead per call)", (tK/t0-1)*100, tK-t0)
	t.Logf("Projected impact: %.1f µs/layer × 32 = %.2f ms/token", tK-t0, (tK-t0)*32/1000)

	// Standalone matvec comparison
	t.Log("")
	t.Log("=== Standalone matvec: Q4_0 vs Q4_K [11008,4096] ===")
	measure("Q4_0 multi_output", 2*q4_0_size, func() {
		b.MatVecQ4_0MultiOutput(xBuf, wQ4_0_1, outBuf, N, K)
	})
	measure("Q4_K (auto-routes NR4 for N>8192)", 2*q4k_size, func() {
		b.MatMulQ4_K(xBuf, wQ4K_1, outBuf, 1, N, K)
	})

	fmt.Println()
}

// TestQ4KW2AddNR4vsSingle benchmarks the NR4 add variant vs single-output add
// at W2 dimensions [4096, 11008] to determine if NR4 routing is worthwhile.
func TestQ4KW2AddNR4vsSingle(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		batchSize = 32
		warmup    = 10
		iters     = 50
	)

	// W2 dimensions for LLaMA 2 7B: [4096, 11008]
	dims := [][2]int{
		{4096, 11008},  // W2 (down projection)
		{4096, 4096},   // Wo (attention output)
	}

	for _, dim := range dims {
		N, K := dim[0], dim[1]
		q4k_size := N * (K / 256) * 144

		xBuf := b.Alloc(K * 4)        // FP32 input
		wBuf := b.Alloc(q4k_size)     // Q4_K weights
		outBuf := b.Alloc(N * 4)      // FP32 output

		data := make([]byte, K*4)
		rand.Read(data)
		b.ToDevice(xBuf, data)
		data = make([]byte, q4k_size)
		rand.Read(data)
		b.ToDevice(wBuf, data)

		measure := func(name string, fn func()) float64 {
			for i := 0; i < warmup; i++ {
				b.BeginBatch()
				for j := 0; j < batchSize; j++ {
					fn()
					if j < batchSize-1 {
						b.MemoryBarrier()
					}
				}
				b.EndBatch()
				b.Sync()
			}
			times := make([]float64, iters)
			for i := 0; i < iters; i++ {
				start := time.Now()
				b.BeginBatch()
				for j := 0; j < batchSize; j++ {
					fn()
					if j < batchSize-1 {
						b.MemoryBarrier()
					}
				}
				b.EndBatch()
				b.Sync()
				times[i] = float64(time.Since(start).Microseconds())
			}
			sortFloat64s(times)
			lo := iters / 10
			hi := iters - lo
			sum := 0.0
			for i := lo; i < hi; i++ {
				sum += times[i]
			}
			avg := sum / float64(hi-lo) / float64(batchSize)
			wMB := float64(q4k_size) / (1024 * 1024)
			bw := wMB / 1024 / (avg / 1e6)
			t.Logf("  %-35s %8.1f µs/call  %6.1f GB/s (%4.1f%%)", name, avg, bw, bw/400*100)
			return avg
		}

		t.Logf("=== Add kernel: [%d, %d] (%.2f MB Q4_K) ===", N, K, float64(q4k_size)/(1024*1024))
		tSingle := measure("single-output (8/TG)", func() {
			b.MatMulQ4_K_Add(xBuf, wBuf, outBuf, N, K)
		})
		tShared := measure("shared-mem multi-output (32/TG)", func() {
			b.MatMulQ4_K_AddShared(xBuf, wBuf, outBuf, N, K)
		})
		tNR4 := measure("NR4 (32/TG)", func() {
			b.MatMulQ4_K_NR4_Add(xBuf, wBuf, outBuf, N, K)
		})

		best := tSingle
		bestName := "single"
		if tShared < best {
			best = tShared
			bestName = "shared"
		}
		if tNR4 < best {
			best = tNR4
			bestName = "NR4"
		}

		t.Logf("  → Best: %s (%.1f µs/call)", bestName, best)
		if best < tSingle {
			t.Logf("  → Saves %.1f µs/call vs single (%.1f%% faster), %.2f ms/token for 32 layers",
				tSingle-best, (1-best/tSingle)*100, (tSingle-best)*32/1000)
		}
		t.Log("")

		b.Free(xBuf)
		b.Free(wBuf)
		b.Free(outBuf)
	}
}

// TestQ4KW2AddF16vsF32Input benchmarks W2+Add with FP32 vs FP16 activation input
// at W2 dimensions [4096, 11008]. The hypothesis: FP16 input halves the activation
// footprint (44KB → 22KB), reducing L1 cache pressure and improving BW utilization.
func TestQ4KW2AddF16vsF32Input(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		batchSize = 32
		warmup    = 10
		iters     = 50
	)

	// W2 dimensions for LLaMA 2 7B: [4096, 11008]
	N, K := 4096, 11008
	q4k_size := N * (K / 256) * 144

	xF32 := b.Alloc(K * 4)       // FP32 input (44KB)
	xF16 := b.Alloc(K * 2)       // FP16 input (22KB)
	wBuf := b.Alloc(q4k_size)    // Q4_K weights
	outBuf := b.Alloc(N * 4)     // FP32 output

	defer b.Free(xF32)
	defer b.Free(xF16)
	defer b.Free(wBuf)
	defer b.Free(outBuf)

	data := make([]byte, K*4)
	rand.Read(data)
	b.ToDevice(xF32, data)
	data = make([]byte, K*2)
	rand.Read(data)
	b.ToDevice(xF16, data)
	data = make([]byte, q4k_size)
	rand.Read(data)
	b.ToDevice(wBuf, data)

	measure := func(name string, weightBytes int, fn func()) float64 {
		for i := 0; i < warmup; i++ {
			b.BeginBatch()
			for j := 0; j < batchSize; j++ {
				fn()
				if j < batchSize-1 {
					b.MemoryBarrier()
				}
			}
			b.EndBatch()
			b.Sync()
		}
		times := make([]float64, iters)
		for i := 0; i < iters; i++ {
			start := time.Now()
			b.BeginBatch()
			for j := 0; j < batchSize; j++ {
				fn()
				if j < batchSize-1 {
					b.MemoryBarrier()
				}
			}
			b.EndBatch()
			b.Sync()
			times[i] = float64(time.Since(start).Microseconds())
		}
		sortFloat64s(times)
		lo := iters / 10
		hi := iters - lo
		sum := 0.0
		for i := lo; i < hi; i++ {
			sum += times[i]
		}
		avg := sum / float64(hi-lo) / float64(batchSize)
		wMB := float64(weightBytes) / (1024 * 1024)
		bw := wMB / 1024 / (avg / 1e6)
		t.Logf("  %-35s %8.1f µs/call  %6.1f GB/s (%4.1f%%)", name, avg, bw, bw/400*100)
		return avg
	}

	t.Logf("=== W2+Add FP32 vs FP16 input: [%d, %d] (%.2f MB Q4_K) ===", N, K, float64(q4k_size)/(1024*1024))
	t.Logf("FP32 activation: %d bytes (%.1f KB)", K*4, float64(K*4)/1024)
	t.Logf("FP16 activation: %d bytes (%.1f KB)", K*2, float64(K*2)/1024)
	t.Log("")

	tF32 := measure("W2+Add FP32 input", q4k_size, func() {
		b.MatMulQ4_K_Add(xF32, wBuf, outBuf, N, K)
	})

	tF16 := measure("W2+Add FP16 input", q4k_size, func() {
		b.MatMulQ4_K_F16InAdd(xF16, wBuf, outBuf, N, K)
	})

	t.Log("")
	if tF16 < tF32 {
		t.Logf("→ FP16 input is %.1f%% FASTER (%.1f µs saved/call)", (1-tF16/tF32)*100, tF32-tF16)
		t.Logf("→ Projected 32-layer savings: %.2f ms/token", (tF32-tF16)*32/1000)
	} else {
		t.Logf("→ FP16 input is %.1f%% slower (%.1f µs overhead/call)", (tF16/tF32-1)*100, tF16-tF32)
	}

	// Also test Wo+Add dimensions [4096, 4096] for comparison
	t.Log("")
	N2, K2 := 4096, 4096
	q4k_size2 := N2 * (K2 / 256) * 144

	xF32_2 := b.Alloc(K2 * 4)
	xF16_2 := b.Alloc(K2 * 2)
	wBuf2 := b.Alloc(q4k_size2)
	outBuf2 := b.Alloc(N2 * 4)

	defer b.Free(xF32_2)
	defer b.Free(xF16_2)
	defer b.Free(wBuf2)
	defer b.Free(outBuf2)

	data = make([]byte, q4k_size2)
	rand.Read(data)
	b.ToDevice(wBuf2, data)

	t.Logf("=== Wo+Add FP32 vs FP16 input: [%d, %d] (%.2f MB Q4_K) ===", N2, K2, float64(q4k_size2)/(1024*1024))

	tF32_2 := measure("Wo+Add FP32 input", q4k_size2, func() {
		b.MatMulQ4_K_Add(xF32_2, wBuf2, outBuf2, N2, K2)
	})

	tF16_2 := measure("Wo+Add FP16 input", q4k_size2, func() {
		b.MatMulQ4_K_F16InAdd(xF16_2, wBuf2, outBuf2, N2, K2)
	})

	t.Log("")
	if tF16_2 < tF32_2 {
		t.Logf("→ FP16 input is %.1f%% FASTER (%.1f µs saved/call)", (1-tF16_2/tF32_2)*100, tF32_2-tF16_2)
	} else {
		t.Logf("→ FP16 input is %.1f%% slower (%.1f µs overhead/call)", (tF16_2/tF32_2-1)*100, tF16_2-tF32_2)
	}
}
