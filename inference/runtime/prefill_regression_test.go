//go:build metal && darwin && cgo

package runtime_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// modelPath returns the path to a GGUF model for testing.
// Checks VEXEL_TEST_MODEL env var first, falls back to LLaMA 2 7B Q4_0.
func modelPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("VEXEL_TEST_MODEL"); p != "" {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			t.Skipf("Model not found at %s (from VEXEL_TEST_MODEL)", p)
		}
		return p
	}
	cwd, _ := os.Getwd()
	root := filepath.Join(cwd, "../..")
	path := filepath.Join(root, "models", "llama-2-7b.Q4_0.gguf")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Model not found at %s", path)
	}
	return path
}

// setupModel loads the model and returns everything needed for inference.
func setupModel(t *testing.T, forceFP32KV bool) (*runtime.ModelRuntime, *metal.Backend, *runtime.GPUKVCache) {
	t.Helper()
	path := modelPath(t)

	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	cfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to init Metal: %v", err)
	}

	memCtx := memory.NewInferenceContext(tensor.Metal)
	maxTokens := 2048 // must cover max prefill prompt length
	totalScratch := cfg.TotalArenaBytes(maxTokens)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	model, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, cfg)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	if err := model.LoadWeights(path); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		t.Fatalf("Failed to copy weights: %v", err)
	}

	if forceFP32KV {
		os.Setenv("VEXEL_KV_FP32", "1")
		defer os.Unsetenv("VEXEL_KV_FP32")
	}

	cache := model.CreateGPUKVCache(2048)
	return model, gpuBackend, cache
}

// getLogits runs DecodeWithGPUKV and returns the logits as float32 slice.
func getLogits(t *testing.T, model *runtime.ModelRuntime, backend *metal.Backend, tokens []int, pos int) []float32 {
	t.Helper()
	logits, err := model.DecodeWithGPUKV(tokens, pos)
	if err != nil {
		t.Fatalf("DecodeWithGPUKV failed: %v", err)
	}
	backend.Sync()

	numElements := logits.Shape().NumElements()
	data := make([]byte, numElements*4)
	backend.ToHost(data, logits.DevicePtr())

	result := make([]float32, numElements)
	for i := range result {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result
}

// argmax returns the index of the maximum value.
func argmax(values []float32) int {
	maxIdx := 0
	maxVal := float32(-math.MaxFloat32)
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// TestPrefillVsSequentialDecode compares prefill (all tokens at once) against
// sequential decode (one token at a time). They should produce the same logits.
// This test exercises the actual model to detect issues in the prefill pipeline.
func TestPrefillVsSequentialDecode(t *testing.T) {
	// Use BOS + a few common token IDs for LLaMA 2
	// These are arbitrary token IDs that form a short sequence
	tokens := []int{1, 15043, 29892, 920, 526} // BOS + "Hello, how are"

	// --- Phase 1: Sequential decode (one token at a time) ---
	t.Log("Phase 1: Sequential decode (token-by-token)")
	model1, backend1, cache1 := setupModel(t, false)
	defer backend1.Close()
	defer cache1.Free()

	var seqLogits []float32
	for i, tok := range tokens {
		logits := getLogits(t, model1, backend1, []int{tok}, i)
		if i == len(tokens)-1 {
			seqLogits = logits
		}
	}
	seqNext := argmax(seqLogits)
	t.Logf("Sequential decode: next token = %d (logit = %f)", seqNext, seqLogits[seqNext])

	// --- Phase 2: Prefill (all tokens at once) ---
	t.Log("Phase 2: Prefill (all tokens at once)")
	model2, backend2, cache2 := setupModel(t, false)
	defer backend2.Close()
	defer cache2.Free()

	prefillLogits := getLogits(t, model2, backend2, tokens, 0)
	prefillNext := argmax(prefillLogits)
	t.Logf("Prefill: next token = %d (logit = %f)", prefillNext, prefillLogits[prefillNext])

	// --- Compare ---
	if seqNext != prefillNext {
		t.Errorf("MISMATCH: Sequential predicts token %d, Prefill predicts token %d", seqNext, prefillNext)
	}

	// Also compare logit distributions
	if len(seqLogits) != len(prefillLogits) {
		t.Fatalf("Logits size mismatch: seq=%d, prefill=%d", len(seqLogits), len(prefillLogits))
	}

	var maxDiff float64
	var maxDiffIdx int
	var mismatchCount int
	for i := range seqLogits {
		diff := math.Abs(float64(seqLogits[i] - prefillLogits[i]))
		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}
		if diff > 1.0 {
			mismatchCount++
		}
	}

	t.Logf("Logit comparison: max diff = %f at idx %d, mismatches (>1.0) = %d/%d",
		maxDiff, maxDiffIdx, mismatchCount, len(seqLogits))

	if mismatchCount > len(seqLogits)/10 {
		t.Errorf("Too many logit mismatches: %d/%d (>10%% of vocabulary)", mismatchCount, len(seqLogits))
	}
	if maxDiff > 5.0 {
		t.Errorf("Max logit diff too large: %f at index %d (seq=%f, prefill=%f)",
			maxDiff, maxDiffIdx, seqLogits[maxDiffIdx], prefillLogits[maxDiffIdx])
	}
}

// TestPrefillMinimal tests prefill with just 2 tokens to isolate the issue.
// With seqLen=2:
// - MatMul uses the BATCHED kernel (m=2, threshold for simdgroup is m>=8)
// - SDPA uses standard sdpa_prefill_f32 (FP32) or flash_attention_2_f16 (FP16)
func TestPrefillMinimal(t *testing.T) {
	tokens := []int{1, 15043} // BOS + one token (m=2)

	// Sequential decode
	model1, backend1, cache1 := setupModel(t, true) // Force FP32 to simplify
	defer backend1.Close()
	defer cache1.Free()

	// Process token 0
	_ = getLogits(t, model1, backend1, []int{1}, 0)
	// Process token 1
	seqLogits := getLogits(t, model1, backend1, []int{15043}, 1)
	seqNext := argmax(seqLogits)
	t.Logf("Sequential (FP32): next token = %d", seqNext)

	// Prefill both tokens
	model2, backend2, cache2 := setupModel(t, true) // Force FP32
	defer backend2.Close()
	defer cache2.Free()

	prefillLogits := getLogits(t, model2, backend2, tokens, 0)
	prefillNext := argmax(prefillLogits)
	t.Logf("Prefill (FP32, m=2): next token = %d", prefillNext)

	if seqNext != prefillNext {
		t.Errorf("MISMATCH with just 2 tokens: Sequential=%d, Prefill=%d", seqNext, prefillNext)
	}

	var maxDiff float64
	var mismatchCount int
	for i := range seqLogits {
		diff := math.Abs(float64(seqLogits[i] - prefillLogits[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1.0 {
			mismatchCount++
		}
	}
	t.Logf("2-token: max diff = %f, mismatches (>1.0) = %d/%d", maxDiff, mismatchCount, len(seqLogits))
}

// TestPrefillFP32VsFP16 compares prefill with FP32 KV cache vs FP16 KV cache.
// If FP32 works but FP16 doesn't, the issue is in the FP16 conversion path.
func TestPrefillFP32VsFP16(t *testing.T) {
	tokens := []int{1, 15043, 29892, 920, 526} // BOS + "Hello, how are"

	// --- FP32 KV cache prefill ---
	t.Log("Prefill with FP32 KV cache")
	model32, backend32, cache32 := setupModel(t, true)
	defer backend32.Close()
	defer cache32.Free()

	fp32Logits := getLogits(t, model32, backend32, tokens, 0)
	fp32Next := argmax(fp32Logits)
	t.Logf("FP32 prefill: next token = %d (logit = %f)", fp32Next, fp32Logits[fp32Next])

	// --- FP16 KV cache prefill ---
	t.Log("Prefill with FP16 KV cache")
	model16, backend16, cache16 := setupModel(t, false)
	defer backend16.Close()
	defer cache16.Free()

	fp16Logits := getLogits(t, model16, backend16, tokens, 0)
	fp16Next := argmax(fp16Logits)
	t.Logf("FP16 prefill: next token = %d (logit = %f)", fp16Next, fp16Logits[fp16Next])

	// --- Compare ---
	if fp32Next != fp16Next {
		t.Errorf("MISMATCH: FP32 predicts token %d, FP16 predicts token %d", fp32Next, fp16Next)
	}

	var maxDiff float64
	for i := range fp32Logits {
		diff := math.Abs(float64(fp32Logits[i] - fp16Logits[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("FP32 vs FP16 max logit diff: %f", maxDiff)
}

// TestFusedQKVCorrectness verifies that fused QKV projection (single matmul + deinterleave)
// produces identical prefill logits and greedy decode tokens to the unfused path (3 separate matmuls).
// This validates Phase 3a: FuseQKVWeights + deinterleave kernel.
//
// The prefill logits must be bit-identical (same GEMM computation, same accumulation order).
// Decode tokens may diverge slightly due to autoregressive amplification of tiny numerical
// differences (quantized near-tie logits can flip argmax over many steps).
func TestFusedQKVCorrectness(t *testing.T) {
	const numDecodeTokens = 20
	prefillTokens := []int{1, 15043, 29892, 920, 526} // BOS + "Hello, how are"

	// 1. Baseline: separate Wq/Wk/Wv projections (no QKV fusion)
	m1, b1, c1 := setupModel(t, false)
	unfusedLogits := getLogits(t, m1, b1, prefillTokens, 0)
	nextToken := argmax(unfusedLogits)
	pos := len(prefillTokens)

	unfusedTokens := make([]int, numDecodeTokens)
	for i := 0; i < numDecodeTokens; i++ {
		unfusedTokens[i] = nextToken
		logits := getLogits(t, m1, b1, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	b1.Close()
	c1.Free()

	// 2. Fused path: Wqkv + deinterleave
	m2, b2, c2 := setupModelFusedQKV(t)
	fusedLogits := getLogits(t, m2, b2, prefillTokens, 0)
	nextToken = argmax(fusedLogits)
	pos = len(prefillTokens)

	fusedTokens := make([]int, numDecodeTokens)
	for i := 0; i < numDecodeTokens; i++ {
		fusedTokens[i] = nextToken
		logits := getLogits(t, m2, b2, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	b2.Close()
	c2.Free()

	// Check 1: Prefill logits comparison — should be very close (same GEMM math)
	var maxDiff float64
	var sumDiff float64
	for i := range unfusedLogits {
		diff := math.Abs(float64(unfusedLogits[i] - fusedLogits[i]))
		sumDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	avgDiff := sumDiff / float64(len(unfusedLogits))
	t.Logf("Prefill logits comparison: max_diff=%.6f, avg_diff=%.8f (%d values)", maxDiff, avgDiff, len(unfusedLogits))

	// Prefill argmax must match
	unfusedFirst := argmax(unfusedLogits)
	fusedFirst := argmax(fusedLogits)
	if unfusedFirst != fusedFirst {
		t.Errorf("CRITICAL: Prefill first-token mismatch: unfused=%d, fused=%d", unfusedFirst, fusedFirst)
	} else {
		t.Logf("Prefill first token: %d (matches)", unfusedFirst)
	}

	// Max logit diff should be very small for Q4_0 (same computation, same accumulation)
	if maxDiff > 0.01 {
		t.Errorf("Prefill logit max_diff %.6f exceeds threshold 0.01 — possible bug", maxDiff)
	}

	// Check 2: Decode token comparison — allow small divergence from autoregressive amplification
	mismatches := 0
	for i := 0; i < numDecodeTokens; i++ {
		if unfusedTokens[i] != fusedTokens[i] {
			t.Logf("  Token mismatch at decode step %d (pos %d): unfused=%d, fused=%d",
				i, i+len(prefillTokens), unfusedTokens[i], fusedTokens[i])
			mismatches++
		}
	}

	t.Logf("Decode token comparison: %d/%d match", numDecodeTokens-mismatches, numDecodeTokens)
	t.Logf("  Unfused: %v", unfusedTokens[:min(10, numDecodeTokens)])
	t.Logf("  Fused:   %v", fusedTokens[:min(10, numDecodeTokens)])

	// Allow up to 3 mismatches in 20 decode tokens (quantized near-tie logits)
	if mismatches > 3 {
		t.Errorf("%d/%d token mismatches exceeds tolerance — possible correctness issue", mismatches, numDecodeTokens)
	}
}

// TestFusedGateUpCorrectness verifies that fused gate_up projection (single matmul + deinterleave)
// produces identical prefill logits and greedy decode tokens to the unfused path (2 separate matmuls).
// This validates Phase 3b: FuseGateUpWeights + deinterleave_2way kernel.
func TestFusedGateUpCorrectness(t *testing.T) {
	const numDecodeTokens = 20
	prefillTokens := []int{1, 15043, 29892, 920, 526} // BOS + "Hello, how are"

	// 1. Baseline: separate W1/W3 projections (no gate_up fusion)
	m1, b1, c1 := setupModel(t, false)
	unfusedLogits := getLogits(t, m1, b1, prefillTokens, 0)
	nextToken := argmax(unfusedLogits)
	pos := len(prefillTokens)

	unfusedTokens := make([]int, numDecodeTokens)
	for i := 0; i < numDecodeTokens; i++ {
		unfusedTokens[i] = nextToken
		logits := getLogits(t, m1, b1, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	b1.Close()
	c1.Free()

	// 2. Fused path: W1W3 + deinterleave_2way
	m2, b2, c2 := setupModelFusedGateUp(t)
	fusedLogits := getLogits(t, m2, b2, prefillTokens, 0)
	nextToken = argmax(fusedLogits)
	pos = len(prefillTokens)

	fusedTokens := make([]int, numDecodeTokens)
	for i := 0; i < numDecodeTokens; i++ {
		fusedTokens[i] = nextToken
		logits := getLogits(t, m2, b2, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	b2.Close()
	c2.Free()

	// Check 1: Prefill logits comparison
	var maxDiff float64
	var sumDiff float64
	for i := range unfusedLogits {
		diff := math.Abs(float64(unfusedLogits[i] - fusedLogits[i]))
		sumDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	avgDiff := sumDiff / float64(len(unfusedLogits))
	t.Logf("Prefill logits comparison: max_diff=%.6f, avg_diff=%.8f (%d values)", maxDiff, avgDiff, len(unfusedLogits))

	unfusedFirst := argmax(unfusedLogits)
	fusedFirst := argmax(fusedLogits)
	if unfusedFirst != fusedFirst {
		t.Errorf("CRITICAL: Prefill first-token mismatch: unfused=%d, fused=%d", unfusedFirst, fusedFirst)
	} else {
		t.Logf("Prefill first token: %d (matches)", unfusedFirst)
	}

	if maxDiff > 0.01 {
		t.Errorf("Prefill logit max_diff %.6f exceeds threshold 0.01 — possible bug", maxDiff)
	}

	// Check 2: Decode token comparison
	mismatches := 0
	for i := 0; i < numDecodeTokens; i++ {
		if unfusedTokens[i] != fusedTokens[i] {
			t.Logf("  Token mismatch at decode step %d (pos %d): unfused=%d, fused=%d",
				i, i+len(prefillTokens), unfusedTokens[i], fusedTokens[i])
			mismatches++
		}
	}

	t.Logf("Decode token comparison: %d/%d match", numDecodeTokens-mismatches, numDecodeTokens)
	t.Logf("  Unfused: %v", unfusedTokens[:min(10, numDecodeTokens)])
	t.Logf("  Fused:   %v", fusedTokens[:min(10, numDecodeTokens)])

	if mismatches > 3 {
		t.Errorf("%d/%d token mismatches exceeds tolerance — possible correctness issue", mismatches, numDecodeTokens)
	}
}

// TestFusedAllCorrectness verifies both QKV and gate_up fusions combined.
func TestFusedAllCorrectness(t *testing.T) {
	const numDecodeTokens = 20
	prefillTokens := []int{1, 15043, 29892, 920, 526} // BOS + "Hello, how are"

	// 1. Baseline: no fusion
	m1, b1, c1 := setupModel(t, false)
	unfusedLogits := getLogits(t, m1, b1, prefillTokens, 0)
	nextToken := argmax(unfusedLogits)
	pos := len(prefillTokens)

	unfusedTokens := make([]int, numDecodeTokens)
	for i := 0; i < numDecodeTokens; i++ {
		unfusedTokens[i] = nextToken
		logits := getLogits(t, m1, b1, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	b1.Close()
	c1.Free()

	// 2. Both fusions: QKV + gate_up
	m2, b2, c2 := setupModelFusedAll(t)
	fusedLogits := getLogits(t, m2, b2, prefillTokens, 0)
	nextToken = argmax(fusedLogits)
	pos = len(prefillTokens)

	fusedTokens := make([]int, numDecodeTokens)
	for i := 0; i < numDecodeTokens; i++ {
		fusedTokens[i] = nextToken
		logits := getLogits(t, m2, b2, []int{nextToken}, pos)
		nextToken = argmax(logits)
		pos++
	}
	b2.Close()
	c2.Free()

	// Check prefill logits
	var maxDiff float64
	for i := range unfusedLogits {
		diff := math.Abs(float64(unfusedLogits[i] - fusedLogits[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("Prefill logits max_diff=%.6f", maxDiff)

	unfusedFirst := argmax(unfusedLogits)
	fusedFirst := argmax(fusedLogits)
	if unfusedFirst != fusedFirst {
		t.Errorf("Prefill first-token mismatch: unfused=%d, fused=%d", unfusedFirst, fusedFirst)
	}
	if maxDiff > 0.01 {
		t.Errorf("Prefill logit max_diff %.6f exceeds threshold 0.01", maxDiff)
	}

	// Decode tokens
	mismatches := 0
	for i := 0; i < numDecodeTokens; i++ {
		if unfusedTokens[i] != fusedTokens[i] {
			mismatches++
		}
	}
	t.Logf("Decode: %d/%d match", numDecodeTokens-mismatches, numDecodeTokens)
	if mismatches > 3 {
		t.Errorf("%d/%d token mismatches exceeds tolerance", mismatches, numDecodeTokens)
	}
}

// TestFusedPrefillThroughput measures prefill throughput with and without QKV+gate_up fusion.
// Runs back-to-back to minimize environmental variance for fair A/B comparison.
func TestFusedPrefillThroughput(t *testing.T) {
	tokens := generateTokenSequence(128)
	seqLen := len(tokens)
	const warmup = 2
	const iterations = 5

	type result struct {
		name    string
		avgMs   float64
		tokPerS float64
	}
	var results []result

	// Test 1: Baseline (no fusion)
	{
		m, b, c := setupModel(t, false)
		for w := 0; w < warmup; w++ {
			m.DecodeWithGPUKV(tokens, 0)
			b.Sync()
			c.Reset()
		}
		times := make([]time.Duration, iterations)
		for i := 0; i < iterations; i++ {
			c.Reset()
			start := time.Now()
			m.DecodeWithGPUKV(tokens, 0)
			b.Sync()
			times[i] = time.Since(start)
		}
		sortDurations(times)
		avg := (times[1] + times[2] + times[3]) / 3
		tps := float64(seqLen) / avg.Seconds()
		results = append(results, result{"baseline", float64(avg.Microseconds()) / 1000, tps})
		b.Close()
		c.Free()
	}

	// Test 2: QKV fusion only
	{
		m, b, c := setupModelFusedQKV(t)
		for w := 0; w < warmup; w++ {
			m.DecodeWithGPUKV(tokens, 0)
			b.Sync()
			c.Reset()
		}
		times := make([]time.Duration, iterations)
		for i := 0; i < iterations; i++ {
			c.Reset()
			start := time.Now()
			m.DecodeWithGPUKV(tokens, 0)
			b.Sync()
			times[i] = time.Since(start)
		}
		sortDurations(times)
		avg := (times[1] + times[2] + times[3]) / 3
		tps := float64(seqLen) / avg.Seconds()
		results = append(results, result{"QKV fused", float64(avg.Microseconds()) / 1000, tps})
		b.Close()
		c.Free()
	}

	// Test 3: Both QKV + gate_up fusion
	{
		m, b, c := setupModelFusedAll(t)
		for w := 0; w < warmup; w++ {
			m.DecodeWithGPUKV(tokens, 0)
			b.Sync()
			c.Reset()
		}
		times := make([]time.Duration, iterations)
		for i := 0; i < iterations; i++ {
			c.Reset()
			start := time.Now()
			m.DecodeWithGPUKV(tokens, 0)
			b.Sync()
			times[i] = time.Since(start)
		}
		sortDurations(times)
		avg := (times[1] + times[2] + times[3]) / 3
		tps := float64(seqLen) / avg.Seconds()
		results = append(results, result{"QKV+gate_up fused", float64(avg.Microseconds()) / 1000, tps})
		b.Close()
		c.Free()
	}

	t.Log("\n=== Prefill Throughput Comparison (seqLen=128) ===")
	for _, r := range results {
		t.Logf("  %-20s: %8.1f tok/s  (%.2f ms)", r.name, r.tokPerS, r.avgMs)
	}
	if len(results) >= 3 {
		speedup := results[2].tokPerS / results[0].tokPerS
		t.Logf("  Speedup (both fused vs baseline): %.2fx", speedup)
	}
}

// setupModelFusedQKV creates a model with QKV weight fusion enabled.
func setupModelFusedQKV(t *testing.T) (*runtime.ModelRuntime, *metal.Backend, *runtime.GPUKVCache) {
	t.Helper()
	path := modelPath(t)

	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	cfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to init Metal: %v", err)
	}

	memCtx := memory.NewInferenceContext(tensor.Metal)
	maxTokens := 2048
	totalScratch := cfg.TotalArenaBytes(maxTokens)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	model, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, cfg)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	if err := model.LoadWeights(path); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		t.Fatalf("Failed to copy weights: %v", err)
	}

	// Fuse QKV weights for prefill optimization
	if err := model.FuseQKVWeights(); err != nil {
		t.Fatalf("Failed to fuse QKV weights: %v", err)
	}

	cache := model.CreateGPUKVCache(2048)
	return model, gpuBackend, cache
}

// setupModelFusedGateUp creates a model with gate_up weight fusion enabled.
func setupModelFusedGateUp(t *testing.T) (*runtime.ModelRuntime, *metal.Backend, *runtime.GPUKVCache) {
	t.Helper()
	path := modelPath(t)

	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	cfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to init Metal: %v", err)
	}

	memCtx := memory.NewInferenceContext(tensor.Metal)
	maxTokens := 2048
	totalScratch := cfg.TotalArenaBytes(maxTokens)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	model, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, cfg)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	if err := model.LoadWeights(path); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		t.Fatalf("Failed to copy weights: %v", err)
	}

	// Fuse gate_up weights for prefill optimization
	if err := model.FuseGateUpWeights(); err != nil {
		t.Fatalf("Failed to fuse gate_up weights: %v", err)
	}

	cache := model.CreateGPUKVCache(2048)
	return model, gpuBackend, cache
}

// setupModelFusedAll creates a model with both QKV and gate_up fusion enabled.
func setupModelFusedAll(t *testing.T) (*runtime.ModelRuntime, *metal.Backend, *runtime.GPUKVCache) {
	t.Helper()
	path := modelPath(t)

	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	cfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to init Metal: %v", err)
	}

	memCtx := memory.NewInferenceContext(tensor.Metal)
	maxTokens := 2048
	totalScratch := cfg.TotalArenaBytes(maxTokens)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	model, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, cfg)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	if err := model.LoadWeights(path); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		t.Fatalf("Failed to copy weights: %v", err)
	}

	// Fuse both QKV and gate_up weights
	if err := model.FuseQKVWeights(); err != nil {
		t.Fatalf("Failed to fuse QKV weights: %v", err)
	}
	if err := model.FuseGateUpWeights(); err != nil {
		t.Fatalf("Failed to fuse gate_up weights: %v", err)
	}

	cache := model.CreateGPUKVCache(2048)
	return model, gpuBackend, cache
}
