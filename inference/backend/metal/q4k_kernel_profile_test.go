//go:build metal && darwin && cgo

package metal

import (
	"encoding/binary"
	"fmt"
	"math"
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

	// 1b. FusedRMSNormQKV NR2 (Q4_K) — 3×[4096, 4096]
	t.Log("Benchmarking FusedRMSNormQKV NR2 (Q4_K, 3×[4096,4096])...")
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

		r := benchKernel("FusedRMSNormQKV NR2 (Q4K)", 3*q4kSize(hidden, hidden),
			func() {
				b.MatMulQ4_K_FusedRMSNormQKV_NR2_F16(xBuf, normBuf, wq, wk, wv,
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

	// 3b. FusedRMSNormMLP F16Out — same but FP16 output (halves intermediate footprint)
	t.Log("Benchmarking FusedRMSNormMLP F16Out (Q4K, 2×[11008,4096])...")
	{
		outInterF16 := b.Alloc(intermediate * 2) // FP16 output
		defer b.Free(outInterF16)
		r := benchKernel("FusedRMSNormMLP F16Out", 2*q4kSize(intermediate, hidden),
			func() {
				b.MatMulQ4_K_FusedRMSNormMLP_F16Out(xBuf, normBuf, w1, w3, outInterF16,
					intermediate, hidden, 1e-5)
			})
		results = append(results, r)
	}

	// 3c. Split MLP via QKV: single-matrix [11008,4096] with 4 rows/SG
	// Uses QKV kernel with qDim=11008, kvDim=1 to isolate single-matrix BW
	t.Log("Benchmarking split-MLP proxy (QKV as 1×[11008,4096])...")
	{
		bigOutQ := b.Alloc(intermediate * 2) // FP16 output for Q path
		tinyOutK := b.Alloc(1 * 2)           // FP16, 1 element
		tinyOutV := b.Alloc(1 * 2)           // FP16, 1 element
		tinyWk := b.Alloc(q4kSize(1, hidden))
		tinyWv := b.Alloc(q4kSize(1, hidden))
		defer b.Free(bigOutQ)
		defer b.Free(tinyOutK)
		defer b.Free(tinyOutV)
		defer b.Free(tinyWk)
		defer b.Free(tinyWv)
		initRand(tinyWk, q4kSize(1, hidden))
		initRand(tinyWv, q4kSize(1, hidden))

		r := benchKernel("SplitMLP proxy (1×[11008])", q4kSize(intermediate, hidden),
			func() {
				b.MatMulQ4_K_FusedRMSNormQKV_F16(xBuf, normBuf, w1, tinyWk, tinyWv,
					bigOutQ, tinyOutK, tinyOutV, intermediate, 1, hidden, 1e-5)
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

	// 4b. W2+Add F16 input — same kernel but reads FP16 activations
	t.Log("Benchmarking W2+Add F16in (Q4K, [4096,11008])...")
	{
		outInterF16 := b.Alloc(intermediate * 2)
		defer b.Free(outInterF16)
		initRand(outInterF16, intermediate*2)
		r := benchKernel("W2+Add F16in (Q4K)", q4kSize(hidden, intermediate),
			func() {
				b.MatMulQ4_K_F16InAdd(outInterF16, w2, outHidden, hidden, intermediate)
			})
		results = append(results, r)
	}

	// 4c. F16in Add at [11008, 4096] — test if 1-out/SG achieves same BW at MLP dims
	t.Log("Benchmarking f16in_add (Q4K, [11008,4096])...")
	{
		inF16 := b.Alloc(hidden * 2)   // K=4096 FP16
		wMlp := b.Alloc(q4kSize(intermediate, hidden))
		outMlp := b.Alloc(intermediate * 4) // N=11008 FP32
		defer b.Free(inF16)
		defer b.Free(wMlp)
		defer b.Free(outMlp)
		initRand(inF16, hidden*2)
		initRand(wMlp, q4kSize(intermediate, hidden))
		r := benchKernel("f16in_add [11008,4096]", q4kSize(intermediate, hidden),
			func() {
				b.MatMulQ4_K_F16InAdd(inF16, wMlp, outMlp, intermediate, hidden)
			})
		results = append(results, r)
	}

	// 5. Dimension scaling: f16in at [4096,4096] (plain, no add) vs [12288,4096] (3x QKV)
	t.Log("Benchmarking f16in [4096,4096] (standalone, no add)...")
	{
		inF16 := b.Alloc(hidden * 2)
		wAttn := b.Alloc(q4kSize(hidden, hidden))
		outAttn := b.Alloc(hidden * 4)
		defer b.Free(inF16)
		defer b.Free(wAttn)
		defer b.Free(outAttn)
		initRand(inF16, hidden*2)
		initRand(wAttn, q4kSize(hidden, hidden))
		r := benchKernel("f16in [4096,4096]", q4kSize(hidden, hidden),
			func() {
				b.MatMulQ4_K_F16In(inF16, wAttn, outAttn, hidden, hidden)
			})
		results = append(results, r)
	}

	t.Log("Benchmarking f16in [12288,4096] (3x QKV combined)...")
	{
		qkvDim := 12288 // 3 × 4096
		inF16 := b.Alloc(hidden * 2)
		wQKV := b.Alloc(q4kSize(qkvDim, hidden))
		outQKV := b.Alloc(qkvDim * 4)
		defer b.Free(inF16)
		defer b.Free(wQKV)
		defer b.Free(outQKV)
		initRand(inF16, hidden*2)
		initRand(wQKV, q4kSize(qkvDim, hidden))
		r := benchKernel("f16in [12288,4096] (3xQKV)", q4kSize(qkvDim, hidden),
			func() {
				b.MatMulQ4_K_F16In(inF16, wQKV, outQKV, qkvDim, hidden)
			})
		results = append(results, r)
	}

	// 6. Output head: [32000, 4096]
	t.Log("Benchmarking output head [32000,4096]...")
	{
		vocabDim := 32000
		wHead := b.Alloc(q4kSize(vocabDim, hidden))
		outHead := b.Alloc(vocabDim * 4)
		defer b.Free(wHead)
		defer b.Free(outHead)
		initRand(wHead, q4kSize(vocabDim, hidden))
		r := benchKernel("Output head [32000,4096]", q4kSize(vocabDim, hidden),
			func() {
				b.MatMulQ4_K_F16In(attnF16, wHead, outHead, vocabDim, hidden)
			})
		results = append(results, r)
	}

	// 7. multi_output (standard dispatch) at [4096,4096] for comparison
	t.Log("Benchmarking multi_output [4096,4096] (standard)...")
	{
		outAttn := b.Alloc(hidden * 4)
		wAttn := b.Alloc(q4kSize(hidden, hidden))
		defer b.Free(outAttn)
		defer b.Free(wAttn)
		initRand(wAttn, q4kSize(hidden, hidden))
		r := benchKernel("multi_out [4096,4096]", q4kSize(hidden, hidden),
			func() {
				b.MatMulQ4_K(xBuf, wAttn, outAttn, 1, hidden, hidden)
			})
		results = append(results, r)
	}

	// Print results
	t.Log("")
	t.Log("=== Q4_K Kernel Profile (LLaMA 2 7B dims, 32-batched) ===")
	t.Logf("%-35s %10s %10s %10s %8s", "Kernel", "Time/call", "Weight(MB)", "BW(GB/s)", "Util%")
	t.Logf("%-35s %10s %10s %10s %8s", "──────", "─────────", "─────────", "────────", "─────")

	totalUs := 0.0
	for _, r := range results {
		util := r.bwGBs / 400.0 * 100
		t.Logf("%-35s %8.1f µs %9.2f %9.1f %7.1f%%",
			r.name, r.avgUs, r.weightMB, r.bwGBs, util)
		totalUs += r.avgUs
	}

	// Also add SDPA estimate (negligible at ctx=16)
	t.Logf("%-35s %8s µs %9s %9s %7s", "SDPA F16 (ctx=16, est)", "~5", "0.27", "-", "-")

	t.Log("")
	t.Logf("Sum of kernel times (1 layer): %.1f µs (excl SDPA, excl dim scaling)", totalUs)
	t.Logf("Total weight per layer:        %.2f MB", func() float64 {
		sum := 0.0
		for _, r := range results {
			sum += r.weightMB
		}
		return sum
	}())
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
		_ = measure("single-output (8/TG)", func() {
			b.MatMulQ4_K_Add(xBuf, wBuf, outBuf, N, K)
		})
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

// TestRMSNormF32ToF16Correctness verifies rmsnorm_f32_to_f16 matches rmsnorm_f32 + f32_to_f16.
func TestRMSNormF32ToF16Correctness(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const dim = 4096
	const eps = 1e-5

	// Create input data
	xData := make([]float32, dim)
	wData := make([]float32, dim)
	rng := rand.New(rand.NewSource(42))
	for i := range xData {
		xData[i] = rng.Float32()*2 - 1 // [-1, 1]
		wData[i] = rng.Float32()*0.5 + 0.5
	}

	xBuf := b.Alloc(dim * 4)
	wBuf := b.Alloc(dim * 4)
	outF32 := b.Alloc(dim * 4)
	outF16Direct := b.Alloc(dim * 2)
	defer b.Free(xBuf)
	defer b.Free(wBuf)
	defer b.Free(outF32)
	defer b.Free(outF16Direct)

	b.ToDevice(xBuf, float32ToBytes(xData))
	b.ToDevice(wBuf, float32ToBytes(wData))

	// Reference: rmsnorm_f32
	b.RMSNorm(xBuf, wBuf, outF32, 1, dim, eps)
	b.Sync()

	// Direct: rmsnorm_f32_to_f16
	b.RMSNormF32ToF16(xBuf, wBuf, outF16Direct, 1, dim, eps)
	b.Sync()

	// Read results
	refBytes := make([]byte, dim*4)
	b.ToHost(refBytes, outF32)
	refData := bytesToFloat32(refBytes)

	directBytes := make([]byte, dim*2)
	b.ToHost(directBytes, outF16Direct)

	// Compare: direct FP16 output vs reference FP32→FP16 conversion
	maxDiff := float64(0)
	for i := 0; i < dim; i++ {
		refVal := float64(refData[i])
		directVal := float64(float16ToFloat32CPU(binary.LittleEndian.Uint16(directBytes[i*2 : i*2+2])))
		diff := math.Abs(refVal - directVal)
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	t.Logf("RMSNormF32ToF16 vs RMSNormF32: max diff = %.6e", maxDiff)
	// FP16 has ~3 decimal digits of precision, so 1e-3 is reasonable
	if maxDiff > 1e-3 {
		t.Errorf("RMSNormF32ToF16 max diff %.6e exceeds threshold 1e-3", maxDiff)
	}
}

// TestSiLUMulF32ToF16Correctness verifies silu_mul_f32_to_f16 matches silu_mul_f32 + f32_to_f16.
func TestSiLUMulF32ToF16Correctness(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const n = 11008

	// Create input data
	gateData := make([]float32, n)
	upData := make([]float32, n)
	rng := rand.New(rand.NewSource(42))
	for i := range gateData {
		gateData[i] = rng.Float32()*4 - 2 // [-2, 2]
		upData[i] = rng.Float32()*4 - 2
	}

	gateBuf := b.Alloc(n * 4)
	upBuf := b.Alloc(n * 4)
	outF32 := b.Alloc(n * 4)
	outF16Direct := b.Alloc(n * 2)
	defer b.Free(gateBuf)
	defer b.Free(upBuf)
	defer b.Free(outF32)
	defer b.Free(outF16Direct)

	b.ToDevice(gateBuf, float32ToBytes(gateData))
	b.ToDevice(upBuf, float32ToBytes(upData))

	// Reference: silu_mul_f32
	b.SiLUMul(gateBuf, upBuf, outF32, n)
	b.Sync()

	// Direct: silu_mul_f32_to_f16
	b.SiLUMulF32ToF16(gateBuf, upBuf, outF16Direct, n)
	b.Sync()

	// Read results
	refBytes := make([]byte, n*4)
	b.ToHost(refBytes, outF32)
	refData := bytesToFloat32(refBytes)

	directBytes := make([]byte, n*2)
	b.ToHost(directBytes, outF16Direct)

	// Compare
	maxDiff := float64(0)
	maxRelDiff := float64(0)
	for i := 0; i < n; i++ {
		refVal := float64(refData[i])
		directVal := float64(float16ToFloat32CPU(binary.LittleEndian.Uint16(directBytes[i*2 : i*2+2])))
		diff := math.Abs(refVal - directVal)
		if diff > maxDiff {
			maxDiff = diff
		}
		if math.Abs(refVal) > 1e-6 {
			relDiff := diff / math.Abs(refVal)
			if relDiff > maxRelDiff {
				maxRelDiff = relDiff
			}
		}
	}

	t.Logf("SiLUMulF32ToF16 vs SiLUMulF32: max abs diff = %.6e, max rel diff = %.6e", maxDiff, maxRelDiff)
	if maxDiff > 5e-3 {
		t.Errorf("SiLUMulF32ToF16 max diff %.6e exceeds threshold 5e-3", maxDiff)
	}
}

// TestUnfusedMLPPipeline benchmarks the full unfused MLP pipeline vs the fused version.
// Unfused: RMSNormF32→F16 → W1(f16in) → W3(f16in) → SiLU_Mul(f32→f16) → W2+Add(f16in)
// Fused:   FusedRMSNormMLP → W2+Add
func TestUnfusedMLPPipeline(t *testing.T) {
	b, err := NewBackend(0)
	if err != nil {
		t.Skipf("Metal backend not available: %v", err)
	}
	defer b.Close()

	const (
		hidden       = 4096
		intermediate = 11008
		batchSize    = 32 // 32 layers
		warmup       = 3
		iterations   = 10
	)

	// Q4_K weight sizes
	w1Size := intermediate * (hidden / 256) * 144 // [11008, 4096]
	w3Size := w1Size
	w2Size := hidden * (intermediate / 256) * 144 // [4096, 11008]

	// Allocate buffers
	xBuf := b.Alloc(hidden * 4)                // FP32 residual
	normWeight := b.Alloc(hidden * 4)           // RMSNorm weight
	normF16Buf := b.Alloc(hidden * 2)           // FP16 normalized activations
	w1Buf := b.Alloc(w1Size)                    // W1 Q4_K weights
	w3Buf := b.Alloc(w3Size)                    // W3 Q4_K weights
	w2Buf := b.Alloc(w2Size)                    // W2 Q4_K weights
	gateBuf := b.Alloc(intermediate * 4)        // FP32 gate output
	upBuf := b.Alloc(intermediate * 4)          // FP32 up output
	interF16Buf := b.Alloc(intermediate * 2)    // FP16 SiLU_Mul output
	outBuf := b.Alloc(hidden * 4)               // FP32 output/residual
	interF32Buf := b.Alloc(intermediate * 4)    // FP32 SiLU_Mul output (fused path)

	defer b.Free(xBuf)
	defer b.Free(normWeight)
	defer b.Free(normF16Buf)
	defer b.Free(w1Buf)
	defer b.Free(w3Buf)
	defer b.Free(w2Buf)
	defer b.Free(gateBuf)
	defer b.Free(upBuf)
	defer b.Free(interF16Buf)
	defer b.Free(outBuf)
	defer b.Free(interF32Buf)

	// Initialize with random data
	data := make([]byte, w2Size)
	rand.Read(data)
	b.ToDevice(w1Buf, data[:w1Size])
	b.ToDevice(w3Buf, data[:w3Size])
	b.ToDevice(w2Buf, data)

	xData := make([]float32, hidden)
	wData := make([]float32, hidden)
	rng := rand.New(rand.NewSource(42))
	for i := range xData {
		xData[i] = rng.Float32()*2 - 1
		wData[i] = rng.Float32()*0.5 + 0.5
	}
	b.ToDevice(xBuf, float32ToBytes(xData))
	b.ToDevice(normWeight, float32ToBytes(wData))

	eps := float32(1e-5)

	// Total weight bytes per layer for BW calculation
	totalWeightBytes := float64(w1Size + w3Size + w2Size)

	measure := func(name string, fn func()) float64 {
		for i := 0; i < warmup; i++ {
			b.BeginBatch()
			for j := 0; j < batchSize; j++ {
				fn()
				b.MemoryBarrier()
			}
			b.EndBatch()
			b.Sync()
		}

		var total time.Duration
		for i := 0; i < iterations; i++ {
			b.BeginBatch()
			for j := 0; j < batchSize; j++ {
				fn()
				b.MemoryBarrier()
			}
			b.EndBatch()
			start := time.Now()
			b.Sync()
			total += time.Since(start)
		}

		avg := float64(total.Microseconds()) / float64(iterations) / float64(batchSize)
		bw := totalWeightBytes / (avg / 1e6) / 1e9
		t.Logf("  %-40s %8.1f µs/layer  %6.1f GB/s (%4.1f%%)", name, avg, bw, bw/400*100)
		return avg
	}

	t.Logf("=== Unfused vs Fused MLP Pipeline: hidden=%d, intermediate=%d ===", hidden, intermediate)
	t.Logf("Total weight per layer: %.2f MB (W1+W3+W2 Q4_K)", totalWeightBytes/(1024*1024))
	t.Log("")

	// Fused MLP pipeline (current): FusedRMSNormMLP + W2+Add(f16in)
	tFused := measure("Fused: RMSNormMLP + W2+Add(f16in)", func() {
		b.MatMulQ4_K_FusedRMSNormMLP_F16Out(xBuf, normWeight, w1Buf, w3Buf, interF16Buf, intermediate, hidden, eps)
		b.MemoryBarrier()
		b.MatMulQ4_K_F16InAdd(interF16Buf, w2Buf, outBuf, hidden, intermediate)
	})

	// Fused MLP pipeline (f32 intermediate): FusedRMSNormMLP + W2+Add(f32)
	tFusedF32 := measure("Fused: RMSNormMLP(f32) + W2+Add(f32)", func() {
		b.MatMulQ4_K_FusedRMSNormMLP(xBuf, normWeight, w1Buf, w3Buf, interF32Buf, intermediate, hidden, eps)
		b.MemoryBarrier()
		b.MatMulQ4_K_Add(interF32Buf, w2Buf, outBuf, intermediate, hidden)
	})

	// Unfused MLP pipeline: RMSNorm→F16 + W1(f16in) + W3(f16in) + SiLU_Mul→F16 + W2+Add(f16in)
	tUnfused := measure("Unfused: Norm→F16 + W1 + W3 + SiLU→F16 + W2", func() {
		b.RMSNormF32ToF16(xBuf, normWeight, normF16Buf, 1, hidden, eps)
		b.MemoryBarrier()
		b.MatMulQ4_K_F16In(normF16Buf, w1Buf, gateBuf, intermediate, hidden)
		b.MatMulQ4_K_F16In(normF16Buf, w3Buf, upBuf, intermediate, hidden)
		b.MemoryBarrier()
		b.SiLUMulF32ToF16(gateBuf, upBuf, interF16Buf, intermediate)
		b.MemoryBarrier()
		b.MatMulQ4_K_F16InAdd(interF16Buf, w2Buf, outBuf, hidden, intermediate)
	})

	// NR1 Fused MLP pipeline (f16 out): 1 output/SG for higher BW
	tNR1F16 := measure("NR1 Fused: RMSNormMLP_NR1 + W2+Add(f16in)", func() {
		b.MatMulQ4_K_FusedRMSNormMLP_NR1_F16Out(xBuf, normWeight, w1Buf, w3Buf, interF16Buf, intermediate, hidden, eps)
		b.MemoryBarrier()
		b.MatMulQ4_K_F16InAdd(interF16Buf, w2Buf, outBuf, hidden, intermediate)
	})

	// NR1 Fused MLP pipeline (f32 out): 1 output/SG for higher BW
	tNR1F32 := measure("NR1 Fused: RMSNormMLP_NR1(f32) + W2+Add(f32)", func() {
		b.MatMulQ4_K_FusedRMSNormMLP_NR1(xBuf, normWeight, w1Buf, w3Buf, interF32Buf, intermediate, hidden, eps)
		b.MemoryBarrier()
		b.MatMulQ4_K_Add(interF32Buf, w2Buf, outBuf, intermediate, hidden)
	})

	t.Log("")
	t.Logf("=== Summary ===")
	t.Logf("Fused NR4 (f16 inter):  %.1f µs/layer → %.1f ms total (32 layers)", tFused, tFused*32/1000)
	t.Logf("Fused NR4 (f32 inter):  %.1f µs/layer → %.1f ms total (32 layers)", tFusedF32, tFusedF32*32/1000)
	t.Logf("NR1 Fused (f16 inter):  %.1f µs/layer → %.1f ms total (32 layers)", tNR1F16, tNR1F16*32/1000)
	t.Logf("NR1 Fused (f32 inter):  %.1f µs/layer → %.1f ms total (32 layers)", tNR1F32, tNR1F32*32/1000)
	t.Logf("Unfused (all f16in):    %.1f µs/layer → %.1f ms total (32 layers)", tUnfused, tUnfused*32/1000)

	best := tFused
	bestName := "Fused NR4 (f16)"
	for _, c := range []struct {
		name string
		t    float64
	}{
		{"NR1 (f16)", tNR1F16},
		{"NR1 (f32)", tNR1F32},
		{"Unfused", tUnfused},
	} {
		if c.t < best {
			best = c.t
			bestName = c.name
		}
	}
	t.Log("")
	if best < tFused {
		saved := (tFused - best) * 32 / 1000
		t.Logf("→ WINNER: %s — %.1f%% faster than NR4 fused, saving %.2f ms/token", bestName, (1-best/tFused)*100, saved)
	} else {
		t.Logf("→ NR4 fused remains fastest")
	}
}
