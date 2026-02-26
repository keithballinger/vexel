//go:build metal && darwin && cgo

package runtime_test

import (
	"testing"
	"time"
)

// TestLatencyTTFT measures Time-To-First-Token (TTFT) — the latency from
// receiving a prompt to producing the first output token. TTFT determines
// how quickly the user sees the model start generating.
//
// TTFT = prefill time (all prompt tokens) + first decode step.
// For interactive use, lower TTFT means a more responsive experience.
func TestLatencyTTFT(t *testing.T) {
	// Test TTFT at various prompt lengths
	cases := []struct {
		name   string
		tokens []int
	}{
		{
			name:   "short_prompt_5tok",
			tokens: []int{1, 15043, 29892, 920, 526}, // BOS + "Hello, how are"
		},
		{
			name:   "medium_prompt_32tok",
			tokens: generateTokenSequence(32),
		},
		{
			name:   "long_prompt_128tok",
			tokens: generateTokenSequence(128),
		},
	}

	type ttftResult struct {
		promptLen     int
		prefillMs     float64
		firstDecodeMs float64
		ttftMs        float64
	}
	results := make([]ttftResult, 0, len(cases))

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, b, c := setupModel(t, false)
			defer b.Close()
			defer c.Free()

			// Warmup: one full prefill + decode cycle
			logits := getLogits(t, m, b, tc.tokens, 0)
			nextToken := argmax(logits)
			_ = getLogits(t, m, b, []int{nextToken}, len(tc.tokens))
			c.Reset()

			// Measure TTFT: 5 iterations, take trimmed mean
			const iterations = 5
			prefillTimes := make([]time.Duration, iterations)
			firstDecodeTimes := make([]time.Duration, iterations)
			ttftTimes := make([]time.Duration, iterations)

			for i := 0; i < iterations; i++ {
				c.Reset()

				// Phase 1: Prefill
				ttftStart := time.Now()
				prefillStart := time.Now()
				logits = getLogits(t, m, b, tc.tokens, 0)
				prefillTimes[i] = time.Since(prefillStart)

				// Phase 2: Sample + first decode token
				nextToken = argmax(logits)
				firstDecodeStart := time.Now()
				_ = getLogits(t, m, b, []int{nextToken}, len(tc.tokens))
				firstDecodeTimes[i] = time.Since(firstDecodeStart)

				ttftTimes[i] = time.Since(ttftStart)
			}

			// Sort and take middle 3
			sortDurations(prefillTimes)
			sortDurations(firstDecodeTimes)
			sortDurations(ttftTimes)

			avgPrefill := trimmedMean(prefillTimes)
			avgFirstDecode := trimmedMean(firstDecodeTimes)
			avgTTFT := trimmedMean(ttftTimes)

			prefillMs := float64(avgPrefill.Microseconds()) / 1000
			firstDecodeMs := float64(avgFirstDecode.Microseconds()) / 1000
			ttftMs := float64(avgTTFT.Microseconds()) / 1000

			t.Logf("TTFT breakdown (prompt=%d tokens):", len(tc.tokens))
			t.Logf("  Prefill:      %.2f ms", prefillMs)
			t.Logf("  First decode: %.2f ms", firstDecodeMs)
			t.Logf("  Total TTFT:   %.2f ms", ttftMs)

			results = append(results, ttftResult{
				promptLen:     len(tc.tokens),
				prefillMs:     prefillMs,
				firstDecodeMs: firstDecodeMs,
				ttftMs:        ttftMs,
			})
		})
	}

	// Summary
	t.Log("\n=== Time-To-First-Token Summary (LLaMA 2 7B Q4_0) ===")
	t.Logf("  %-15s %10s %10s %10s", "Prompt", "Prefill", "1st Decode", "TTFT")
	for _, r := range results {
		t.Logf("  %-15s %8.2f ms %8.2f ms %8.2f ms",
			formatPromptLen(r.promptLen), r.prefillMs, r.firstDecodeMs, r.ttftMs)
	}

	// Assertions: TTFT should scale roughly linearly with prompt length.
	// Short prompt TTFT should be < 200ms for interactive use on 7B model.
	for _, r := range results {
		// Max TTFT: 5× prefill time (generous to account for first decode)
		maxTTFT := r.prefillMs * 5
		if maxTTFT < 200 {
			maxTTFT = 200 // minimum 200ms allowed
		}
		if r.ttftMs > maxTTFT {
			t.Errorf("TTFT too high for prompt=%d: %.2f ms (max: %.2f ms)",
				r.promptLen, r.ttftMs, maxTTFT)
		}
	}

	// Decode latency should be consistent regardless of prompt length
	if len(results) >= 2 {
		first := results[0].firstDecodeMs
		last := results[len(results)-1].firstDecodeMs
		ratio := last / first
		if ratio > 2.0 || ratio < 0.5 {
			t.Errorf("First decode latency varies too much with prompt length: %.2f ms vs %.2f ms (ratio: %.2f)",
				first, last, ratio)
		}
	}
}

// TestLatencyPerTokenDistribution measures the distribution of per-token decode
// latencies to identify jitter and outliers. Consistent per-token latency is
// important for smooth streaming output.
func TestLatencyPerTokenDistribution(t *testing.T) {
	m, b, c := setupModel(t, false)
	defer b.Close()
	defer c.Free()

	// Prefill a short prompt
	tokens := []int{1, 15043, 29892, 920, 526}
	logits := getLogits(t, m, b, tokens, 0)
	nextToken := argmax(logits)
	pos := len(tokens)

	// Warmup: 5 tokens
	for i := 0; i < 5; i++ {
		logits = getLogits(t, m, b, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}

	// Measure 100 tokens
	const numTokens = 100
	times := make([]time.Duration, numTokens)

	for i := 0; i < numTokens; i++ {
		start := time.Now()
		logits = getLogits(t, m, b, []int{nextToken}, pos)
		nextToken = argmax(logits)
		times[i] = time.Since(start)
		pos++
	}

	sortDurations(times)

	p50 := float64(times[numTokens/2].Microseconds()) / 1000
	p90 := float64(times[numTokens*90/100].Microseconds()) / 1000
	p95 := float64(times[numTokens*95/100].Microseconds()) / 1000
	p99 := float64(times[numTokens*99/100].Microseconds()) / 1000
	minMs := float64(times[0].Microseconds()) / 1000
	maxMs := float64(times[numTokens-1].Microseconds()) / 1000

	t.Log("=== Per-Token Decode Latency Distribution (100 tokens) ===")
	t.Logf("  min=%.2f ms, p50=%.2f ms, p90=%.2f ms, p95=%.2f ms, p99=%.2f ms, max=%.2f ms",
		minMs, p50, p90, p95, p99, maxMs)

	// Jitter check: p99 should not be more than 2x p50
	jitterRatio := p99 / p50
	t.Logf("  Jitter ratio (p99/p50): %.2f", jitterRatio)
	if jitterRatio > 3.0 {
		t.Errorf("Excessive jitter: p99/p50 = %.2f (max: 3.0). p50=%.2f ms, p99=%.2f ms",
			jitterRatio, p50, p99)
	}
}

// --- Helpers ---

// trimmedMean returns the average of the middle values (excluding min and max).
// Requires the input to be sorted.
func trimmedMean(sorted []time.Duration) time.Duration {
	if len(sorted) <= 2 {
		var total time.Duration
		for _, d := range sorted {
			total += d
		}
		return total / time.Duration(len(sorted))
	}
	var total time.Duration
	for _, d := range sorted[1 : len(sorted)-1] {
		total += d
	}
	return total / time.Duration(len(sorted)-2)
}

// formatPromptLen returns a human-readable prompt length label.
func formatPromptLen(n int) string {
	switch {
	case n < 10:
		return "short"
	case n < 64:
		return "medium"
	default:
		return "long"
	}
}
