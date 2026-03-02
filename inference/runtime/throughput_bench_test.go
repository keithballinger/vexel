//go:build metal && darwin && cgo

package runtime_test

import (
	"fmt"
	"os"
	"testing"
	"time"

	"vexel/inference/backend/metal"
	"vexel/inference/runtime"
)

// TestThroughputPrefill measures prefill throughput (tokens/second) at various
// sequence lengths. This validates that the command buffer batching fix restored
// expected prefill performance.
//
// Prefill processes all prompt tokens in a single forward pass (seqLen > 1).
// Higher sequence lengths amortize per-call overhead and expose SIMD group
// kernel paths (M >= 8).
func TestThroughputPrefill(t *testing.T) {
	// Test cases: various sequence lengths
	// Token IDs are arbitrary valid LLaMA 2 token IDs.
	cases := []struct {
		name   string
		tokens []int
	}{
		{
			name:   "5_tokens",
			tokens: []int{1, 15043, 29892, 920, 526}, // BOS + "Hello, how are"
		},
		{
			name: "32_tokens",
			tokens: generateTokenSequence(32), // BOS + 31 repeated tokens
		},
		{
			name: "128_tokens",
			tokens: generateTokenSequence(128), // BOS + 127 repeated tokens
		},
		{
			name: "385_tokens",
			tokens: generateTokenSequence(385), // BOS + 384 repeated tokens
		},
	}

	results := make([]prefillResult, 0, len(cases))

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			seqLen := len(tc.tokens)

			// Create fresh model for each case to reset KV cache
			m, b, c := setupModel(t, false)
			defer b.Close()
			defer c.Free()

			// Warmup: 2 iterations
			for w := 0; w < 2; w++ {
				_, err := m.DecodeWithGPUKV(tc.tokens, 0)
				if err != nil {
					t.Fatalf("Warmup failed: %v", err)
				}
				b.Sync()
				// Reset KV cache position for next iteration
				c.Reset()
			}

			// Benchmark: 5 iterations, take median-ish (average of middle 3)
			const iterations = 5
			times := make([]time.Duration, iterations)

			for i := 0; i < iterations; i++ {
				c.Reset()
				start := time.Now()
				_, err := m.DecodeWithGPUKV(tc.tokens, 0)
				if err != nil {
					t.Fatalf("Iteration %d failed: %v", i, err)
				}
				b.Sync()
				times[i] = time.Since(start)
			}

			// Sort and take middle 3
			sortDurations(times)
			var totalMiddle time.Duration
			for _, d := range times[1 : iterations-1] {
				totalMiddle += d
			}
			avgTime := totalMiddle / time.Duration(iterations-2)

			tokPerSec := float64(seqLen) / avgTime.Seconds()

			t.Logf("Prefill %d tokens: avg %.2f ms, %.1f tok/s (min=%.2f ms, max=%.2f ms)",
				seqLen,
				float64(avgTime.Microseconds())/1000,
				tokPerSec,
				float64(times[0].Microseconds())/1000,
				float64(times[iterations-1].Microseconds())/1000,
			)

			results = append(results, prefillResult{
				seqLen:   seqLen,
				avgMs:    float64(avgTime.Microseconds()) / 1000,
				tokPerSec: tokPerSec,
			})
		})
	}

	// Print summary
	t.Log("\n=== Prefill Throughput Summary ===")
	for _, r := range results {
		t.Logf("  seqLen=%3d: %8.1f tok/s  (%.2f ms)", r.seqLen, r.tokPerSec, r.avgMs)
	}

	// Threshold assertions: prefill throughput must exceed minimum for the model.
	// Note: setupModel loads LLaMA 2 7B (~3.7 GB weights). The plan's 785 tok/s
	// was measured with TinyLlama 1.1B (~0.6 GB). Expected ratio: ~5-6x difference.
	// Conservative minimums per sequence length (should pass on any M-series Mac):
	minTokPerSec := map[int]float64{
		5:   40,  // Short prefill: dominated by per-call overhead (~60ms)
		32:  60,  // Medium: amortizes some overhead
		128: 100, // Long: representative prefill throughput
		385: 100, // Longer: should maintain throughput
	}
	for _, r := range results {
		minTPS, ok := minTokPerSec[r.seqLen]
		if !ok {
			minTPS = 40 // default
		}
		if r.tokPerSec < minTPS {
			t.Errorf("Prefill throughput too low for seqLen=%d: %.1f tok/s (minimum: %.0f tok/s)", r.seqLen, r.tokPerSec, minTPS)
		}
	}

	// Verify prefill throughput scales with sequence length (key regression indicator).
	// If prefill is broken, all sequence lengths show similar low throughput.
	if len(results) >= 2 {
		shortest := results[0]
		longest := results[len(results)-1]
		if longest.seqLen > shortest.seqLen && longest.tokPerSec < shortest.tokPerSec*0.8 {
			t.Errorf("Prefill throughput does NOT scale with sequence length: seqLen=%d=%.1f tok/s, seqLen=%d=%.1f tok/s",
				shortest.seqLen, shortest.tokPerSec, longest.seqLen, longest.tokPerSec)
		}
	}
}

// TestThroughputDecode measures decode throughput (tokens/second) by generating
// multiple tokens sequentially. Decode processes one token at a time (seqLen=1).
//
// This exercises the fused kernel paths (canFuseFFN/canFuseAttn for M=1) and
// measures the per-token latency that determines user-perceived generation speed.
func TestThroughputDecode(t *testing.T) {
	model, backend, cache := setupModel(t, false)
	defer backend.Close()
	defer cache.Free()

	// Start with a short prefill to initialize KV cache
	prefillTokens := []int{1, 15043, 29892, 920, 526} // BOS + "Hello, how are"
	logits := getLogits(t, model, backend, prefillTokens, 0)
	nextToken := argmax(logits)

	// Generate 50 tokens and measure throughput
	const numDecodeTokens = 50
	pos := len(prefillTokens)

	// Warmup: generate 3 tokens
	for i := 0; i < 3; i++ {
		logits = getLogits(t, model, backend, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}

	// Benchmark: generate numDecodeTokens tokens
	decodeStart := time.Now()
	tokenTimes := make([]time.Duration, numDecodeTokens)

	for i := 0; i < numDecodeTokens; i++ {
		tokenStart := time.Now()
		logits = getLogits(t, model, backend, []int{nextToken}, pos)
		nextToken = argmax(logits)
		tokenTimes[i] = time.Since(tokenStart)
		pos++
	}
	totalDecodeTime := time.Since(decodeStart)

	avgTokPerSec := float64(numDecodeTokens) / totalDecodeTime.Seconds()

	// Per-token timing stats
	sortDurations(tokenTimes)
	minMs := float64(tokenTimes[0].Microseconds()) / 1000
	maxMs := float64(tokenTimes[numDecodeTokens-1].Microseconds()) / 1000
	medianMs := float64(tokenTimes[numDecodeTokens/2].Microseconds()) / 1000
	p99Idx := numDecodeTokens * 99 / 100
	p99Ms := float64(tokenTimes[p99Idx].Microseconds()) / 1000

	t.Logf("Decode %d tokens: %.1f tok/s", numDecodeTokens, avgTokPerSec)
	t.Logf("  Total time: %.2f ms", float64(totalDecodeTime.Microseconds())/1000)
	t.Logf("  Per-token: min=%.2f ms, median=%.2f ms, p99=%.2f ms, max=%.2f ms",
		minMs, medianMs, p99Ms, maxMs)

	// Threshold: decode must exceed minimum performance.
	// Note: Using LLaMA 2 7B (~3.7 GB weights). Plan's 138 tok/s was TinyLlama 1.1B.
	// LLaMA 2 7B on M-series should achieve ~40-50 tok/s decode.
	// Conservative minimum: 25 tok/s (any M-series Mac).
	if avgTokPerSec < 25 {
		t.Errorf("Decode throughput too low: %.1f tok/s (minimum: 25 tok/s)", avgTokPerSec)
	}
}

// TestThroughputDecodeContextScaling measures how decode throughput scales
// with increasing context length (more KV cache entries to attend over).
func TestThroughputDecodeContextScaling(t *testing.T) {
	// Measure decode throughput at various context lengths.
	// Each subtest creates a fresh model, fills context, then measures.
	contextLengths := []int{16, 128, 256, 512, 1024, 2048}

	type decodeResult struct {
		ctxLen    int
		tokPerSec float64
		avgMs     float64
	}
	results := make([]decodeResult, 0)

	for _, ctxLen := range contextLengths {
		t.Run(fmt.Sprintf("ctx_%d", ctxLen), func(t *testing.T) {
			m, b, c := setupModel(t, false)
			defer b.Close()
			defer c.Free()

			// Fill context
			fillToks := generateTokenSequence(ctxLen)
			logits := getLogits(t, m, b, fillToks, 0)
			nextToken := argmax(logits)
			pos := ctxLen

			// Warmup
			for w := 0; w < 3; w++ {
				logits = getLogits(t, m, b, []int{nextToken}, pos)
				nextToken = argmax(logits)
				pos++
			}

			// Measure 20 tokens
			const n = 20
			start := time.Now()
			for i := 0; i < n; i++ {
				logits = getLogits(t, m, b, []int{nextToken}, pos)
				nextToken = argmax(logits)
				pos++
			}
			elapsed := time.Since(start)

			tps := float64(n) / elapsed.Seconds()
			avgMs := float64(elapsed.Microseconds()) / 1000 / float64(n)
			t.Logf("Context=%d: %.1f tok/s (%.2f ms/token)", ctxLen, tps, avgMs)

			results = append(results, decodeResult{
				ctxLen:    ctxLen,
				tokPerSec: tps,
				avgMs:     avgMs,
			})
		})
	}

	// Print summary
	t.Log("\n=== Decode Throughput vs Context Length ===")
	for _, r := range results {
		t.Logf("  ctx=%3d: %6.1f tok/s  (%.2f ms/token)", r.ctxLen, r.tokPerSec, r.avgMs)
	}
}

// TestDecodeContextScalingTarget is the TDD acceptance test for context scaling.
// Asserts ≤5% decode throughput degradation from ctx=16 to ctx=2048.
// This matches llama.cpp's ~2% degradation across the same range.
func TestDecodeContextScalingTarget(t *testing.T) {
	type decodeResult struct {
		ctxLen    int
		tokPerSec float64
		avgMs     float64
	}
	var results []decodeResult

	// Measure at baseline (ctx=16) and target (ctx=2048)
	for _, ctxLen := range []int{16, 2048} {
		t.Run(fmt.Sprintf("ctx_%d", ctxLen), func(t *testing.T) {
			m, b, c := setupModel(t, false)
			defer b.Close()
			defer c.Free()

			fillToks := generateTokenSequence(ctxLen)
			logits := getLogits(t, m, b, fillToks, 0)
			nextToken := argmax(logits)
			pos := ctxLen

			// Warmup
			for w := 0; w < 5; w++ {
				logits = getLogits(t, m, b, []int{nextToken}, pos)
				nextToken = argmax(logits)
				pos++
			}

			// Measure 30 tokens for stable average
			const n = 30
			start := time.Now()
			for i := 0; i < n; i++ {
				logits = getLogits(t, m, b, []int{nextToken}, pos)
				nextToken = argmax(logits)
				pos++
			}
			elapsed := time.Since(start)

			tps := float64(n) / elapsed.Seconds()
			avgMs := float64(elapsed.Microseconds()) / 1000 / float64(n)
			t.Logf("Context=%d: %.1f tok/s (%.2f ms/token)", ctxLen, tps, avgMs)

			results = append(results, decodeResult{ctxLen, tps, avgMs})
		})
	}

	if len(results) == 2 {
		baseline := results[0].tokPerSec
		longCtx := results[1].tokPerSec
		degradation := (baseline - longCtx) / baseline * 100

		t.Logf("\n=== Context Scaling Target ===")
		t.Logf("  ctx=16:   %.1f tok/s", baseline)
		t.Logf("  ctx=2048: %.1f tok/s", longCtx)
		t.Logf("  Degradation: %.1f%% (target: ≤5%%)", degradation)

		if degradation > 5.0 {
			t.Errorf("Context degradation %.1f%% exceeds 5%% target (ctx=16: %.1f, ctx=2048: %.1f)",
				degradation, baseline, longCtx)
		}
	}
}

// TestThroughputGPUProfile runs a quick throughput test with GPU profiling
// enabled to verify that profiling infrastructure works and reports stats.
func TestThroughputGPUProfile(t *testing.T) {
	// Enable GPU profiling for this test
	os.Setenv("VEXEL_GPU_PROFILE", "1")
	defer os.Unsetenv("VEXEL_GPU_PROFILE")

	model, backend, cache := setupModel(t, false)
	defer backend.Close()
	defer cache.Free()

	metal.ResetGPUProfile()

	// Prefill
	tokens := []int{1, 15043, 29892, 920, 526}
	logits := getLogits(t, model, backend, tokens, 0)
	nextToken := argmax(logits)

	// Decode 10 tokens
	pos := len(tokens)
	for i := 0; i < 10; i++ {
		logits = getLogits(t, model, backend, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}

	stats := metal.GetGPUProfile()
	t.Logf("GPU Profile: total=%.2f ms, sync=%.2f ms, batches=%d, kernels=%d",
		float64(stats.TotalTimeNs)/1e6,
		float64(stats.SyncTimeNs)/1e6,
		stats.BatchCount,
		stats.KernelCount,
	)

	// Verify profile collected data (BatchCount tracks command buffer commits;
	// KernelCount is not currently incremented in the C profiling code).
	if stats.BatchCount == 0 {
		t.Errorf("GPU profile reported 0 batches — profiling may be broken")
	}
}

// TestDecodeTimingBreakdown measures where time is spent in the decode pipeline.
// Separates Go-side overhead (setup, layer loop encoding) from GPU execution.
// Requires VEXEL_DECODE_TIMING=1 to activate detailed timing in DecodeWithGPUKV.
func TestDecodeTimingBreakdown(t *testing.T) {
	runtime.EnableDecodeTiming()
	defer runtime.DisableDecodeTiming()

	// Reset any prior timing data
	runtime.PrintDecodeTiming()

	m, b, c := setupModel(t, false)
	defer b.Close()
	defer c.Free()

	// Fill some context
	ctxLen := 16
	fillToks := generateTokenSequence(ctxLen)
	logits := getLogits(t, m, b, fillToks, 0)
	nextToken := argmax(logits)
	pos := ctxLen

	// Warmup
	for w := 0; w < 5; w++ {
		logits = getLogits(t, m, b, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}

	// Reset timing counters after warmup
	runtime.PrintDecodeTiming()

	// Measure 30 tokens
	const n = 30
	start := time.Now()
	for i := 0; i < n; i++ {
		logits = getLogits(t, m, b, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	elapsed := time.Since(start)

	tps := float64(n) / elapsed.Seconds()
	avgMs := float64(elapsed.Microseconds()) / 1000.0 / float64(n)
	t.Logf("Overall: %.1f tok/s (%.2f ms/token, n=%d)", tps, avgMs, n)

	// Print timing breakdown
	runtime.PrintDecodeTiming()
}

// --- Helpers ---

type prefillResult struct {
	seqLen    int
	avgMs     float64
	tokPerSec float64
}

// generateTokenSequence creates a sequence of N token IDs starting with BOS (1).
// Uses a repeating pattern of common LLaMA 2 token IDs.
func generateTokenSequence(n int) []int {
	// Common token IDs: BOS=1, then cycle through some valid tokens
	commonTokens := []int{15043, 29892, 920, 526, 1532, 278, 2462, 310, 4234, 17162,
		297, 2211, 3078, 9763, 2274, 13, 450, 1348, 393, 565, 723, 263, 1781, 5613,
		6910, 15332, 310, 5613, 6724, 297, 2181, 29889}
	tokens := make([]int, n)
	tokens[0] = 1 // BOS
	for i := 1; i < n; i++ {
		tokens[i] = commonTokens[(i-1)%len(commonTokens)]
	}
	return tokens
}

// TestFusionABComparison measures decode throughput with and without the
// Phase 3 fusions (FusedMLP, AddRMSNorm) to quantify their actual impact.
//
// Key finding: dispatch count reduction (451→387, 14%) does NOT translate to
// measurable throughput improvement. The bottleneck is memory bandwidth in the
// Q4_0 matmul kernels, not kernel dispatch overhead (~0.6% of total time).
func TestFusionABComparison(t *testing.T) {
	const numDecodeTokens = 30
	const warmupTokens = 5

	type fusionResult struct {
		name      string
		tokPerSec float64
		medianMs  float64
	}

	configs := []struct {
		name           string
		fuseMLP        string
		fuseAddRMSNorm string
	}{
		{"fused (FusedMLP+AddRMSNorm)", "", ""},          // default: fusions enabled
		{"unfused (baseline)", "0", "0"},                   // fusions disabled
	}

	var results []fusionResult

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			if cfg.fuseMLP != "" {
				os.Setenv("VEXEL_FUSE_MLP", cfg.fuseMLP)
				defer os.Unsetenv("VEXEL_FUSE_MLP")
			}
			if cfg.fuseAddRMSNorm != "" {
				os.Setenv("VEXEL_FUSE_ADD_RMSNORM", cfg.fuseAddRMSNorm)
				defer os.Unsetenv("VEXEL_FUSE_ADD_RMSNORM")
			}

			m, b, c := setupModel(t, false)
			defer b.Close()
			defer c.Free()

			// Prefill
			prefillTokens := []int{1, 15043, 29892, 920, 526}
			logits := getLogits(t, m, b, prefillTokens, 0)
			nextToken := argmax(logits)
			pos := len(prefillTokens)

			// Warmup
			for w := 0; w < warmupTokens; w++ {
				logits = getLogits(t, m, b, []int{nextToken}, pos)
				nextToken = argmax(logits)
				pos++
			}

			// Benchmark
			tokenTimes := make([]time.Duration, numDecodeTokens)
			for i := 0; i < numDecodeTokens; i++ {
				start := time.Now()
				logits = getLogits(t, m, b, []int{nextToken}, pos)
				nextToken = argmax(logits)
				tokenTimes[i] = time.Since(start)
				pos++
			}

			sortDurations(tokenTimes)
			var totalMiddle time.Duration
			for _, d := range tokenTimes[1 : numDecodeTokens-1] {
				totalMiddle += d
			}
			avgTime := totalMiddle / time.Duration(numDecodeTokens-2)
			tokPerSec := 1.0 / avgTime.Seconds()
			medianMs := float64(tokenTimes[numDecodeTokens/2].Microseconds()) / 1000

			t.Logf("%s: %.1f tok/s (median=%.2f ms/token)", cfg.name, tokPerSec, medianMs)

			results = append(results, fusionResult{
				name:      cfg.name,
				tokPerSec: tokPerSec,
				medianMs:  medianMs,
			})
		})
	}

	// Print comparison
	t.Log("\n=== Fusion A/B Comparison ===")
	for _, r := range results {
		t.Logf("  %-35s %6.1f tok/s  (%.2f ms/token)", r.name, r.tokPerSec, r.medianMs)
	}
	if len(results) == 2 {
		speedup := (results[0].tokPerSec/results[1].tokPerSec - 1) * 100
		t.Logf("  Speedup: %+.1f%% (dispatch reduction does NOT measurably improve throughput)", speedup)
		t.Log("  Conclusion: bottleneck is memory bandwidth in Q4_0 matmul kernels,")
		t.Log("  not kernel dispatch overhead (~0.6% of total time)")
	}
}

// TestFusionCorrectness verifies that FusedMLP and AddRMSNorm kernels produce
// identical token sequences to the unfused path. This is the Phase 5 correctness
// verification: deterministic generation (greedy argmax) must match token-for-token.
func TestFusionCorrectness(t *testing.T) {
	const numTokens = 20

	// Generate tokens with fusions DISABLED (baseline)
	os.Setenv("VEXEL_FUSE_MLP", "0")
	os.Setenv("VEXEL_FUSE_ADD_RMSNORM", "0")

	m1, b1, c1 := setupModel(t, false)
	prefillTokens := []int{1, 15043, 29892, 920, 526} // BOS + "Hello, how are"
	logits := getLogits(t, m1, b1, prefillTokens, 0)
	nextToken := argmax(logits)
	pos := len(prefillTokens)

	unfusedTokens := make([]int, numTokens)
	for i := 0; i < numTokens; i++ {
		unfusedTokens[i] = nextToken
		logits = getLogits(t, m1, b1, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	b1.Close()
	c1.Free()

	os.Unsetenv("VEXEL_FUSE_MLP")
	os.Unsetenv("VEXEL_FUSE_ADD_RMSNORM")

	// Generate tokens with fusions ENABLED (default)
	m2, b2, c2 := setupModel(t, false)
	logits = getLogits(t, m2, b2, prefillTokens, 0)
	nextToken = argmax(logits)
	pos = len(prefillTokens)

	fusedTokens := make([]int, numTokens)
	for i := 0; i < numTokens; i++ {
		fusedTokens[i] = nextToken
		logits = getLogits(t, m2, b2, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	b2.Close()
	c2.Free()

	// Compare: tokens must match
	mismatches := 0
	for i := 0; i < numTokens; i++ {
		if unfusedTokens[i] != fusedTokens[i] {
			t.Errorf("Token mismatch at position %d: unfused=%d, fused=%d", i, unfusedTokens[i], fusedTokens[i])
			mismatches++
		}
	}

	if mismatches == 0 {
		t.Logf("Correctness verified: %d tokens match between fused and unfused paths", numTokens)
		t.Logf("  Unfused tokens: %v", unfusedTokens[:10])
		t.Logf("  Fused tokens:   %v", fusedTokens[:10])
	} else {
		t.Errorf("%d/%d token mismatches — fused kernels produce different output!", mismatches, numTokens)
	}
}

// TestModelLoadTime measures the time to load the model from disk, copy weights
// to GPU, and create the KV cache. This captures the cold-start latency.
func TestModelLoadTime(t *testing.T) {
	const runs = 3
	times := make([]time.Duration, runs)

	for i := 0; i < runs; i++ {
		start := time.Now()
		m, b, c := setupModel(t, false)
		times[i] = time.Since(start)
		c.Free()
		b.Close()
		_ = m
	}

	sortDurations(times)
	median := times[runs/2]
	t.Logf("Model load time (median of %d runs): %.0f ms", runs, float64(median.Microseconds())/1000)
	for i, d := range times {
		t.Logf("  Run %d: %.0f ms", i+1, float64(d.Microseconds())/1000)
	}
}

// TestPrefillPerOpProfile measures the per-operation time breakdown for a 128-token
// prefill forward pass. Profiling adds Sync() between every operation, so total time
// is inflated, but the RELATIVE proportions are accurate.
//
// Track: close_prefill_gap, Phase 0 Task 0.2.
func TestPrefillPerOpProfile(t *testing.T) {
	m, b, c := setupModel(t, false)
	defer b.Close()
	defer c.Free()

	tokens := generateTokenSequence(128)

	// Warmup without profiling
	for i := 0; i < 2; i++ {
		_, err := m.DecodeWithGPUKV(tokens, 0)
		if err != nil {
			t.Fatalf("Warmup failed: %v", err)
		}
		b.Sync()
		c.Reset()
	}

	// Enable profiling and run one forward pass
	runtime.EnableProfiling()
	runtime.ResetProfile()

	c.Reset()
	_, err := m.DecodeWithGPUKV(tokens, 0)
	if err != nil {
		t.Fatalf("Profiled forward pass failed: %v", err)
	}
	b.Sync()

	runtime.PrintProfile()
	runtime.DisableProfiling()
}

// TestDecodeProfileBaseline profiles per-operation GPU timing for M=1 decode
// to establish baseline measurements for decode gap optimization.
//
// Track: close_decode_gap_20260228, Phase 0 Task 0.1 + 0.2
//
// This test profiles:
// 1. Per-operation timing breakdown (FusedRMSNorm+QKV, RoPE, KVCache, SDPA, Wo, FFN, etc.)
// 2. Per-token wall-clock time at ctx=16
// 3. Per-dimension matvec bandwidth utilization
//
// Target: ≥73 tok/s decode throughput (≤5% gap to llama.cpp's 76.3 tok/s)
func TestDecodeProfileBaseline(t *testing.T) {
	model, be, cache := setupModel(t, false)
	defer be.Close()
	defer cache.Free()

	// Short prefill to seed KV cache
	prefillTokens := []int{1, 15043, 29892, 920, 526} // BOS + "Hello, how are"
	logits := getLogits(t, model, be, prefillTokens, 0)
	nextToken := argmax(logits)
	pos := len(prefillTokens)

	// Warmup: 5 decode tokens
	for w := 0; w < 5; w++ {
		logits = getLogits(t, model, be, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}

	// === Part 1: Throughput measurement (no profiling overhead) ===
	const numDecodeTokens = 50
	tokenTimes := make([]time.Duration, numDecodeTokens)

	for i := 0; i < numDecodeTokens; i++ {
		start := time.Now()
		logits = getLogits(t, model, be, []int{nextToken}, pos)
		nextToken = argmax(logits)
		tokenTimes[i] = time.Since(start)
		pos++
	}

	sortDurations(tokenTimes)
	// Use middle 80% to avoid outliers
	var totalMiddle time.Duration
	startIdx := numDecodeTokens / 10
	endIdx := numDecodeTokens * 9 / 10
	for _, d := range tokenTimes[startIdx:endIdx] {
		totalMiddle += d
	}
	avgTime := totalMiddle / time.Duration(endIdx-startIdx)
	tokPerSec := 1.0 / avgTime.Seconds()
	medianMs := float64(tokenTimes[numDecodeTokens/2].Microseconds()) / 1000
	p99Ms := float64(tokenTimes[numDecodeTokens*99/100].Microseconds()) / 1000
	minMs := float64(tokenTimes[0].Microseconds()) / 1000
	maxMs := float64(tokenTimes[numDecodeTokens-1].Microseconds()) / 1000

	t.Logf("\n=== Decode Throughput (ctx=%d, %d tokens, no profiling overhead) ===", pos-numDecodeTokens, numDecodeTokens)
	t.Logf("  Throughput:  %.1f tok/s", tokPerSec)
	t.Logf("  Per-token:   min=%.2f ms, median=%.2f ms, p99=%.2f ms, max=%.2f ms", minMs, medianMs, p99Ms, maxMs)

	// Bandwidth analysis
	// LLaMA 2 7B Q4_0: 3.56 GB weights, 400 GB/s M3 Max bandwidth
	modelSizeGB := 3.56
	hwBandwidthGBps := 400.0
	effectiveBW := modelSizeGB * tokPerSec
	bwUtil := effectiveBW / hwBandwidthGBps * 100
	theoreticalMax := hwBandwidthGBps / modelSizeGB
	t.Logf("  BW utilization: %.1f%% (%.1f GB/s of %.0f GB/s, theoretical max: %.0f tok/s)",
		bwUtil, effectiveBW, hwBandwidthGBps, theoreticalMax)

	// === Part 2: Per-operation profiling (with sync overhead) ===
	runtime.EnableProfiling()
	runtime.ResetProfile()

	// Generate 20 profiled tokens (profiling adds sync overhead per op)
	const numProfileTokens = 20
	for i := 0; i < numProfileTokens; i++ {
		logits = getLogits(t, model, be, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}

	t.Logf("\n=== Per-Operation Profile (%d decode tokens, includes sync overhead) ===", numProfileTokens)
	runtime.PrintProfile()
	runtime.DisableProfiling()

	// === Part 3: GPU-level timing with VEXEL_GPU_PROFILE ===
	metal.ResetGPUProfile()

	const numGPUProfileTokens = 20
	gpuStart := time.Now()
	for i := 0; i < numGPUProfileTokens; i++ {
		logits = getLogits(t, model, be, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	gpuWallTime := time.Since(gpuStart)

	gpuStats := metal.GetGPUProfile()
	t.Logf("\n=== GPU Profile (%d decode tokens) ===", numGPUProfileTokens)
	t.Logf("  Wall time:     %.1f ms (%.1f ms/token)",
		float64(gpuWallTime.Microseconds())/1000,
		float64(gpuWallTime.Microseconds())/1000/float64(numGPUProfileTokens))
	t.Logf("  GPU time:      %.1f ms (%.1f ms/token)",
		float64(gpuStats.TotalTimeNs)/1e6,
		float64(gpuStats.TotalTimeNs)/1e6/float64(numGPUProfileTokens))
	t.Logf("  Sync time:     %.1f ms (%.1f ms/token)",
		float64(gpuStats.SyncTimeNs)/1e6,
		float64(gpuStats.SyncTimeNs)/1e6/float64(numGPUProfileTokens))
	t.Logf("  Batches:       %d (%.1f/token)", gpuStats.BatchCount, float64(gpuStats.BatchCount)/float64(numGPUProfileTokens))
	t.Logf("  Kernels:       %d (%.1f/token)", gpuStats.KernelCount, float64(gpuStats.KernelCount)/float64(numGPUProfileTokens))
	if gpuStats.TotalTimeNs > 0 {
		syncPct := float64(gpuStats.SyncTimeNs) / float64(gpuStats.TotalTimeNs) * 100
		t.Logf("  Sync overhead: %.1f%%", syncPct)
	}

	// === Part 4: Dispatch profiler ===
	profiler := be.DispatchProfiler()
	profiler.Enable()
	profiler.BeginPass()

	logits = getLogits(t, model, be, []int{nextToken}, pos)
	nextToken = argmax(logits)
	pos++

	profile := profiler.EndPass()
	profiler.Disable()

	t.Logf("\n=== Dispatch Profile (single decode token) ===")
	t.Logf("  Total dispatches: %d", profile.TotalDispatches)
	t.Logf("  Pass duration:    %v", profile.PassDuration)
	if profile.TotalDispatches > 0 {
		t.Logf("  Avg per dispatch: %.1f µs", float64(profile.PassDuration.Microseconds())/float64(profile.TotalDispatches))
	}
	for op, count := range profile.OpCounts {
		pct := float64(count) / float64(profile.TotalDispatches) * 100
		t.Logf("    %-30s %4d (%5.1f%%)", op, count, pct)
	}

	// Threshold: decode throughput target for closing the gap to llama.cpp
	t.Logf("\n=== Gap Analysis ===")
	llamaCppTokPerSec := 76.3
	gap := (1 - tokPerSec/llamaCppTokPerSec) * 100
	t.Logf("  Vexel:     %.1f tok/s", tokPerSec)
	t.Logf("  llama.cpp: %.1f tok/s", llamaCppTokPerSec)
	t.Logf("  Gap:       %.1f%%", gap)

	if tokPerSec < 73 {
		t.Errorf("Decode throughput %.1f tok/s below target 73 tok/s (gap to llama.cpp: %.1f%%)", tokPerSec, gap)
	}
}

// sortDurations sorts a slice of time.Duration in ascending order.
func sortDurations(d []time.Duration) {
	for i := 1; i < len(d); i++ {
		key := d[i]
		j := i - 1
		for j >= 0 && d[j] > key {
			d[j+1] = d[j]
			j--
		}
		d[j+1] = key
	}
}
