//go:build metal && darwin && cgo

package runtime_test

import (
	"fmt"
	"os"
	"testing"
	"time"

	"vexel/inference/backend/metal"
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
	contextLengths := []int{16, 64, 128}

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
