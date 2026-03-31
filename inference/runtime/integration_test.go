//go:build metal && darwin && cgo

package runtime

import (
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/tensor"
)

// knownModelPaths lists model files to probe for integration tests.
// Each key is a human-readable label; the value is the expected file path.
// Add or adjust paths to match your local setup.
var knownModelPaths = map[string]string{
	"qwen2.5-0.5b": "/Users/qeetbastudio/projects/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
	"phi-3.5":      "/Users/qeetbastudio/projects/llama.cpp/models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
	"llama-8b":     "/Users/qeetbastudio/projects/llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
	"gemma-2-2b":   "/Users/qeetbastudio/projects/llama.cpp/models/gemma-2-2b-it-Q4_K_M.gguf",
}

// availableModels returns a map of label → path for models that exist on disk.
// When VEXEL_TEST_MODEL is set, only that model is returned (labeled "env").
func availableModels(t *testing.T) map[string]string {
	t.Helper()
	if p := os.Getenv("VEXEL_TEST_MODEL"); p != "" {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("VEXEL_TEST_MODEL=%s not found: %v", p, err)
		}
		return map[string]string{"env": p}
	}
	found := make(map[string]string)
	for label, path := range knownModelPaths {
		if _, err := os.Stat(path); err == nil {
			found[label] = path
		}
	}
	if len(found) == 0 {
		t.Skip("No model files found. Set VEXEL_TEST_MODEL or place a GGUF at one of the known paths.")
	}
	return found
}

// loadIntegrationModel initialises a ModelRuntime on Metal for a given GGUF file.
// maxContextLen controls KV cache size. Caller must invoke cleanup().
func loadIntegrationModel(t *testing.T, modelPath string, maxContextLen int) (*ModelRuntime, *metal.Backend, func()) {
	t.Helper()

	be, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("[%s] Failed to create Metal backend: %v", modelPath, err)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		be.Close()
		t.Fatalf("[%s] Failed to open GGUF: %v", modelPath, err)
	}
	modelCfg := ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	memCtx := memory.NewInferenceContext(tensor.Metal)
	totalScratch := modelCfg.TotalArenaBytes(maxContextLen)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), be.Alloc)

	model, err := NewModelRuntime(be, memCtx, nil, modelCfg)
	if err != nil {
		be.Close()
		t.Fatalf("[%s] Failed to create model runtime: %v", modelPath, err)
	}

	if err := model.LoadWeights(modelPath); err != nil {
		be.Close()
		t.Fatalf("[%s] Failed to load weights: %v", modelPath, err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		be.Close()
		t.Fatalf("[%s] Failed to copy weights to device: %v", modelPath, err)
	}
	model.CreateGPUKVCache(maxContextLen)

	cleanup := func() { be.Close() }
	return model, be, cleanup
}

// greedyGenerate runs a prompt through the model and returns the decoded text
// for the next maxNewTokens greedy tokens. Returns empty string on error.
func greedyGenerate(t *testing.T, label, modelPath string, prompt string, maxNewTokens int) string {
	t.Helper()

	const maxContext = 256
	model, be, cleanup := loadIntegrationModel(t, modelPath, maxContext)
	defer cleanup()

	tok, err := tokenizer.LoadFromGGUF(modelPath)
	if err != nil {
		t.Skipf("[%s] Could not load tokenizer: %v", label, err)
	}

	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Skipf("[%s] Could not encode prompt: %v", label, err)
	}
	if len(ids) == 0 {
		t.Skipf("[%s] Encoded prompt has zero tokens", label)
	}

	vocabSize := model.config.VocabSize

	// readHostLogits copies logits from GPU to a float32 slice.
	readHostLogits := func(logitsTensor tensor.Tensor) []float32 {
		buf := make([]byte, vocabSize*4)
		be.Sync()
		be.ToHost(buf, logitsTensor.DevicePtr())
		out := make([]float32, vocabSize)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
		return out
	}

	// Prefill.
	logitsTensor, err := model.DecodeWithGPUKV(ids, 0)
	if err != nil {
		t.Fatalf("[%s] Prefill failed: %v", label, err)
	}
	data := readHostLogits(logitsTensor)

	// Validate no NaN/Inf immediately after prefill.
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("[%s] Prefill produced NaN/Inf at logit index %d", label, i)
		}
	}

	// Decode loop.
	generated := make([]int, 0, maxNewTokens)
	pos := len(ids)
	nextToken := sampler.Argmax(data)

	for step := 0; step < maxNewTokens; step++ {
		// EOS check: token IDs 1 and 2 are common EOS/BOS markers; also stop on 0.
		if nextToken == 0 || nextToken == 1 || nextToken == 2 {
			break
		}
		generated = append(generated, nextToken)

		logitsTensor, err = model.DecodeWithGPUKV([]int{nextToken}, pos)
		if err != nil {
			t.Fatalf("[%s] Decode step %d failed: %v", label, step, err)
		}
		data = readHostLogits(logitsTensor)

		for i, v := range data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("[%s] Decode step %d: NaN/Inf at logit index %d", label, step, i)
			}
		}

		nextToken = sampler.Argmax(data)
		pos++
	}

	text, _ := tok.Decode(generated)
	return text
}

// TestGenerateCountSequence verifies that each model can continue a counting
// sequence.  It prefills "1 2 3 4 5" and expects "6" to appear in the output.
// This catches matmul bugs, KV cache corruption, and tokenisation regressions.
func TestGenerateCountSequence(t *testing.T) {
	models := availableModels(t)
	for label, path := range models {
		label, path := label, path
		t.Run(label, func(t *testing.T) {
			t.Parallel()
			deadline := time.After(30 * time.Second)
			done := make(chan string, 1)
			go func() {
				out := greedyGenerate(t, label, path, "1 2 3 4 5", 10)
				done <- out
			}()
			select {
			case out := <-done:
				t.Logf("[%s] output: %q", label, out)
				if !strings.Contains(out, "6") {
					t.Errorf("[%s] Expected output to contain '6', got: %q", label, out)
				}
			case <-deadline:
				t.Errorf("[%s] Test timed out after 30s", label)
			}
		})
	}
}

// TestGenerateCoherent verifies factual knowledge by prompting with
// "The capital of France is" and expecting "Paris" in the output.
// This catches SDPA bugs, attention softcap bugs, and RoPE regressions.
func TestGenerateCoherent(t *testing.T) {
	models := availableModels(t)
	for label, path := range models {
		label, path := label, path
		t.Run(label, func(t *testing.T) {
			t.Parallel()
			deadline := time.After(30 * time.Second)
			done := make(chan string, 1)
			go func() {
				out := greedyGenerate(t, label, path, "The capital of France is", 15)
				done <- out
			}()
			select {
			case out := <-done:
				t.Logf("[%s] output: %q", label, out)
				if !strings.Contains(strings.ToLower(out), "paris") {
					t.Errorf("[%s] Expected output to contain 'Paris', got: %q", label, out)
				}
			case <-deadline:
				t.Errorf("[%s] Test timed out after 30s", label)
			}
		})
	}
}

// TestNoCrashOnShortPrompt feeds a single-token prompt to each model and verifies
// that inference completes without panics, NaN logits, or errors.
// This catches embedding table OOB, KV cache init bugs, and scratch size bugs.
func TestNoCrashOnShortPrompt(t *testing.T) {
	models := availableModels(t)
	for label, path := range models {
		label, path := label, path
		t.Run(label, func(t *testing.T) {
			t.Parallel()
			deadline := time.After(30 * time.Second)
			done := make(chan struct{}, 1)
			go func() {
				// Use a single known-safe token (space character "▁") which
				// exists in nearly every SentencePiece/BPE vocabulary.
				out := greedyGenerate(t, label, path, " ", 5)
				t.Logf("[%s] short-prompt output: %q", label, out)
				done <- struct{}{}
			}()
			select {
			case <-done:
				// pass — no crash means success
			case <-deadline:
				t.Errorf("[%s] Test timed out after 30s", label)
			}
		})
	}
}

// TestNoNaNOnPrefill checks that prefill over a multi-token prompt produces
// finite logits for all vocabulary entries.  This catches buffer overruns
// and incorrect kernel dispatch for larger batch sizes.
func TestNoNaNOnPrefill(t *testing.T) {
	models := availableModels(t)
	const prompt = "The quick brown fox jumps over the lazy dog."
	const maxContext = 256

	for label, path := range models {
		label, path := label, path
		t.Run(label, func(t *testing.T) {
			t.Parallel()

			model, be, cleanup := loadIntegrationModel(t, path, maxContext)
			defer cleanup()

			tok, err := tokenizer.LoadFromGGUF(path)
			if err != nil {
				t.Skipf("[%s] Could not load tokenizer: %v", label, err)
			}

			ids, err := tok.Encode(prompt)
			if err != nil || len(ids) == 0 {
				t.Skipf("[%s] Failed to encode prompt", label)
			}

			vocabSize := model.config.VocabSize
			done := make(chan error, 1)
			go func() {
				logitsTensor, ferr := model.DecodeWithGPUKV(ids, 0)
				if ferr != nil {
					done <- ferr
					return
				}

				buf := make([]byte, vocabSize*4)
				be.Sync()
				be.ToHost(buf, logitsTensor.DevicePtr())

				nanCount, infCount := 0, 0
				for i := 0; i < vocabSize; i++ {
					v := math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
					if math.IsNaN(float64(v)) {
						nanCount++
					} else if math.IsInf(float64(v), 0) {
						infCount++
					}
				}
				if nanCount > 0 || infCount > 0 {
					t.Errorf("[%s] Prefill logits: %d NaN, %d Inf out of %d", label, nanCount, infCount, vocabSize)
				} else {
					t.Logf("[%s] Prefill OK: %d tokens → %d vocab logits, all finite", label, len(ids), vocabSize)
				}
				done <- nil
			}()

			select {
			case err := <-done:
				if err != nil {
					t.Fatalf("[%s] Prefill failed: %v", label, err)
				}
			case <-time.After(30 * time.Second):
				t.Errorf("[%s] TestNoNaNOnPrefill timed out", label)
			}
		})
	}
}
