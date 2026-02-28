//go:build metal && darwin && cgo

package runtime_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

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
