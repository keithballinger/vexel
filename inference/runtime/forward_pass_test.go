//go:build metal && darwin && cgo

package runtime

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/sampler"
	"vexel/inference/tensor"
)

// loadTestModel creates a fully initialized ModelRuntime with GPU KV cache.
// Returns the model, backend, and a cleanup function.
// Skips the test if the model file is not found.
func loadTestModel(t *testing.T, maxTokens int) (*ModelRuntime, *metal.Backend, func()) {
	t.Helper()

	modelPath := getModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model file not found: %s (set VEXEL_TEST_MODEL env var)", modelPath)
	}

	be, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create Metal backend: %v", err)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		be.Close()
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	modelCfg := ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	memCtx := memory.NewInferenceContext(tensor.Metal)
	totalScratch := modelCfg.TotalArenaBytes(maxTokens)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), be.Alloc)

	model, err := NewModelRuntime(be, memCtx, nil, modelCfg)
	if err != nil {
		be.Close()
		t.Fatalf("Failed to create model runtime: %v", err)
	}

	t.Log("Loading weights from GGUF...")
	loadStart := time.Now()
	if err := model.LoadWeights(modelPath); err != nil {
		be.Close()
		t.Fatalf("Failed to load weights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		be.Close()
		t.Fatalf("Failed to copy weights to device: %v", err)
	}
	t.Logf("Weights loaded in %v", time.Since(loadStart))

	model.CreateGPUKVCache(2048)

	cleanup := func() {
		be.Close()
	}
	return model, be, cleanup
}

// readLogits reads logits from GPU device pointer back to host float32 slice.
func readLogits(be *metal.Backend, logits tensor.Tensor, vocabSize int) []float32 {
	buf := make([]byte, vocabSize*4)
	be.Sync()
	be.ToHost(buf, logits.DevicePtr())
	result := make([]float32, vocabSize)
	for i := range result {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return result
}

// TestForwardPass_LLaMA2_7B verifies that a single forward pass through
// LLaMA 2 7B produces valid logits with correct shape and no NaN/Inf values.
//
// Track 5: Multi-Model Validation, Phase 1 Task 1.
func TestForwardPass_LLaMA2_7B(t *testing.T) {
	model, be, cleanup := loadTestModel(t, 64)
	defer cleanup()

	vocabSize := model.config.VocabSize

	// BOS token = 1 for LLaMA 2
	tokens := []int{1}
	logits, err := model.DecodeWithGPUKV(tokens, 0)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Verify shape: [1, vocabSize]
	dims := logits.Shape().Dims()
	if len(dims) != 2 || dims[0] != 1 || dims[1] != vocabSize {
		t.Fatalf("Logits shape: got %v, want [1, %d]", dims, vocabSize)
	}

	logitData := readLogits(be, logits, vocabSize)

	// Check no NaN or Inf
	nanCount, infCount := 0, 0
	for i, v := range logitData {
		if math.IsNaN(float64(v)) {
			nanCount++
			if nanCount <= 3 {
				t.Errorf("NaN at index %d", i)
			}
		}
		if math.IsInf(float64(v), 0) {
			infCount++
			if infCount <= 3 {
				t.Errorf("Inf at index %d", i)
			}
		}
	}
	if nanCount > 0 || infCount > 0 {
		t.Fatalf("Logits contain %d NaN and %d Inf values out of %d", nanCount, infCount, vocabSize)
	}

	// Check logits are not all zeros (embedding worked)
	allZero := true
	for _, v := range logitData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("All logits are zero — embedding or forward pass is broken")
	}

	// Argmax should produce a valid token ID
	topToken := sampler.Argmax(logitData)
	if topToken < 0 || topToken >= vocabSize {
		t.Fatalf("Argmax token %d out of vocab range [0, %d)", topToken, vocabSize)
	}

	// Log top-5 tokens for manual inspection
	type tokenProb struct {
		id    int
		logit float32
	}
	top5 := make([]tokenProb, 5)
	for i := range top5 {
		top5[i].logit = float32(-math.MaxFloat32)
	}
	for i, v := range logitData {
		if v > top5[4].logit {
			top5[4] = tokenProb{i, v}
			// Bubble sort to keep top 5
			for j := 4; j > 0 && top5[j].logit > top5[j-1].logit; j-- {
				top5[j], top5[j-1] = top5[j-1], top5[j]
			}
		}
	}

	t.Logf("Forward pass OK: vocabSize=%d, argmax=%d", vocabSize, topToken)
	for i, tp := range top5 {
		t.Logf("  Top-%d: token=%d logit=%.4f", i+1, tp.id, tp.logit)
	}
}

// TestForwardPass_Deterministic verifies that running the same input twice
// produces identical logits (GPU computation is deterministic).
//
// Track 5: Multi-Model Validation, Phase 1 Task 1.
func TestForwardPass_Deterministic(t *testing.T) {
	model, be, cleanup := loadTestModel(t, 64)
	defer cleanup()

	vocabSize := model.config.VocabSize
	tokens := []int{1} // BOS

	// First forward pass
	logits1, err := model.DecodeWithGPUKV(tokens, 0)
	if err != nil {
		t.Fatalf("First forward pass failed: %v", err)
	}
	data1 := readLogits(be, logits1, vocabSize)
	top1 := sampler.Argmax(data1)

	// Reset KV cache for second run
	model.CreateGPUKVCache(2048)

	// Second forward pass with same input
	logits2, err := model.DecodeWithGPUKV(tokens, 0)
	if err != nil {
		t.Fatalf("Second forward pass failed: %v", err)
	}
	data2 := readLogits(be, logits2, vocabSize)
	top2 := sampler.Argmax(data2)

	// Argmax must be identical
	if top1 != top2 {
		t.Errorf("Determinism failed: argmax1=%d, argmax2=%d", top1, top2)
	}

	// All logits should be bitwise identical
	maxDiff := float32(0)
	diffCount := 0
	for i := range data1 {
		diff := float32(math.Abs(float64(data1[i] - data2[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 0 {
			diffCount++
		}
	}

	if maxDiff > 1e-6 {
		t.Errorf("Logits differ: maxDiff=%e, diffCount=%d/%d", maxDiff, diffCount, vocabSize)
	}

	t.Logf("Determinism verified: argmax=%d, maxDiff=%e, diffCount=%d/%d",
		top1, maxDiff, diffCount, vocabSize)
}

// TestGreedyDecode_LLaMA2_7B runs greedy decoding from BOS for 16 tokens
// and verifies the output is stable and produces valid token IDs.
//
// Track 5: Multi-Model Validation, Phase 1 Task 1.
func TestGreedyDecode_LLaMA2_7B(t *testing.T) {
	model, be, cleanup := loadTestModel(t, 64)
	defer cleanup()

	vocabSize := model.config.VocabSize

	// Generate 16 tokens starting from BOS
	numGenerate := 16
	generated := make([]int, 0, numGenerate+1)
	generated = append(generated, 1) // BOS

	start := time.Now()
	for step := 0; step < numGenerate; step++ {
		// Decode current token
		token := generated[len(generated)-1]
		logits, err := model.DecodeWithGPUKV([]int{token}, step)
		if err != nil {
			t.Fatalf("Decode step %d failed: %v", step, err)
		}

		logitData := readLogits(be, logits, vocabSize)

		// Check for NaN/Inf
		for i, v := range logitData {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("Step %d: NaN/Inf at logit index %d", step, i)
			}
		}

		nextToken := sampler.Argmax(logitData)
		if nextToken < 0 || nextToken >= vocabSize {
			t.Fatalf("Step %d: argmax=%d out of range [0, %d)", step, nextToken, vocabSize)
		}

		generated = append(generated, nextToken)

		// Stop if EOS (token 2 for LLaMA 2)
		if nextToken == 2 {
			t.Logf("EOS reached at step %d", step)
			break
		}
	}
	elapsed := time.Since(start)

	// Log the generated sequence
	t.Logf("Generated %d tokens in %v (%.1f tok/s):",
		len(generated)-1, elapsed, float64(len(generated)-1)/elapsed.Seconds())
	t.Logf("  Token IDs: %v", generated)

	// Verify no token is repeated endlessly (basic sanity)
	if len(generated) > 5 {
		allSame := true
		for i := 2; i < len(generated); i++ {
			if generated[i] != generated[2] {
				allSame = false
				break
			}
		}
		if allSame {
			t.Errorf("Degenerate output: all generated tokens are %d (possible broken forward pass)", generated[2])
		}
	}
}

// TestPrefill_LLaMA2_7B verifies that prefill (processing multiple tokens at once)
// produces valid logits for the last token.
//
// Track 5: Multi-Model Validation, Phase 1 Task 1.
func TestPrefill_LLaMA2_7B(t *testing.T) {
	model, be, cleanup := loadTestModel(t, 256)
	defer cleanup()

	vocabSize := model.config.VocabSize

	// A short prompt sequence (BOS + a few tokens).
	// These are common LLaMA 2 token IDs for a simple English prompt.
	// Token 1 = BOS, and we use some mid-range IDs that exist in the vocab.
	prompt := []int{1, 450, 29871, 1576, 4996, 310, 3064, 338} // "The city of Paris is"

	start := time.Now()
	logits, err := model.DecodeWithGPUKV(prompt, 0)
	if err != nil {
		t.Fatalf("Prefill failed: %v", err)
	}
	elapsed := time.Since(start)

	logitData := readLogits(be, logits, vocabSize)

	// Verify no NaN/Inf
	for i, v := range logitData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Prefill: NaN/Inf at logit index %d", i)
		}
	}

	topToken := sampler.Argmax(logitData)
	if topToken < 0 || topToken >= vocabSize {
		t.Fatalf("Prefill argmax=%d out of range", topToken)
	}

	t.Logf("Prefill OK: %d tokens in %v (%.0f tok/s), next_token=%d",
		len(prompt), elapsed, float64(len(prompt))/elapsed.Seconds(), topToken)
}

// TestPrefillThenDecode_LLaMA2_7B verifies the prefill+decode pipeline:
// first processes a multi-token prompt, then generates continuation tokens.
//
// Track 5: Multi-Model Validation, Phase 1 Task 1.
func TestPrefillThenDecode_LLaMA2_7B(t *testing.T) {
	model, be, cleanup := loadTestModel(t, 256)
	defer cleanup()

	vocabSize := model.config.VocabSize

	// Prefill prompt
	prompt := []int{1, 450, 29871, 1576, 4996, 310, 3064, 338}

	logits, err := model.DecodeWithGPUKV(prompt, 0)
	if err != nil {
		t.Fatalf("Prefill failed: %v", err)
	}

	logitData := readLogits(be, logits, vocabSize)
	nextToken := sampler.Argmax(logitData)

	generated := []int{nextToken}

	// Continue decoding for 8 more tokens
	for step := 0; step < 8; step++ {
		pos := len(prompt) + step
		logits, err = model.DecodeWithGPUKV([]int{nextToken}, pos)
		if err != nil {
			t.Fatalf("Decode step %d failed: %v", step, err)
		}

		logitData = readLogits(be, logits, vocabSize)

		// Verify no NaN
		for i, v := range logitData {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("Decode step %d: NaN/Inf at logit index %d", step, i)
			}
		}

		nextToken = sampler.Argmax(logitData)
		generated = append(generated, nextToken)

		if nextToken == 2 { // EOS
			break
		}
	}

	t.Logf("Prefill(%d tokens) + Decode(%d tokens) OK", len(prompt), len(generated))
	t.Logf("  Prompt tokens: %v", prompt)
	t.Logf("  Generated tokens: %v", generated)
}

// TestForwardPassTiming_LLaMA2_7B measures prefill and decode throughput
// at various sequence lengths.
//
// Track 5: Multi-Model Validation, Phase 1 Task 1.
func TestForwardPassTiming_LLaMA2_7B(t *testing.T) {
	model, be, cleanup := loadTestModel(t, 256)
	defer cleanup()

	vocabSize := model.config.VocabSize

	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║      FORWARD PASS TIMING — LLaMA 2 7B Q4_0                 ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	// Prefill at various lengths
	prefillLengths := []int{1, 8, 32, 64, 128}

	fmt.Printf("\n┌─── Prefill ──────────────────────────────────────────────────┐\n")
	fmt.Printf("│ %-8s │ %12s │ %10s │ %10s │\n", "SeqLen", "Time", "tok/s", "NextToken")
	fmt.Printf("│ %-8s │ %12s │ %10s │ %10s │\n", "------", "--------", "-----", "---------")

	for _, seqLen := range prefillLengths {
		// Reset KV cache for each test
		model.CreateGPUKVCache(2048)

		// Build a prompt of seqLen tokens (BOS + padding)
		tokens := make([]int, seqLen)
		tokens[0] = 1 // BOS
		for i := 1; i < seqLen; i++ {
			tokens[i] = 450 + i // Arbitrary valid token IDs
		}

		start := time.Now()
		logits, err := model.DecodeWithGPUKV(tokens, 0)
		if err != nil {
			t.Fatalf("Prefill(seqLen=%d) failed: %v", seqLen, err)
		}
		elapsed := time.Since(start)

		logitData := readLogits(be, logits, vocabSize)
		nextToken := sampler.Argmax(logitData)

		tokPerSec := float64(seqLen) / elapsed.Seconds()
		fmt.Printf("│ %-8d │ %12v │ %10.0f │ %10d │\n", seqLen, elapsed, tokPerSec, nextToken)
	}
	fmt.Println("└──────────────────────────────────────────────────────────────┘")

	// Decode throughput: generate 32 tokens and measure
	model.CreateGPUKVCache(2048)

	// Warm up with BOS
	logits, err := model.DecodeWithGPUKV([]int{1}, 0)
	if err != nil {
		t.Fatalf("Warmup failed: %v", err)
	}
	logitData := readLogits(be, logits, vocabSize)
	nextToken := sampler.Argmax(logitData)

	numDecode := 32
	start := time.Now()
	for step := 0; step < numDecode; step++ {
		logits, err = model.DecodeWithGPUKV([]int{nextToken}, step+1)
		if err != nil {
			t.Fatalf("Decode step %d failed: %v", step, err)
		}
		logitData = readLogits(be, logits, vocabSize)
		nextToken = sampler.Argmax(logitData)
	}
	decodeElapsed := time.Since(start)

	fmt.Printf("\n┌─── Decode (M=1) ─────────────────────────────────────────────┐\n")
	fmt.Printf("│ Tokens: %d, Time: %v, Throughput: %.1f tok/s               │\n",
		numDecode, decodeElapsed, float64(numDecode)/decodeElapsed.Seconds())
	fmt.Println("└──────────────────────────────────────────────────────────────┘")

	t.Logf("Timing complete: decode=%.1f tok/s", float64(numDecode)/decodeElapsed.Seconds())
}
