//go:build metal && darwin && cgo

package train_test

import (
	"os"
	"path/filepath"
	"testing"

	"vexel/inference/lora"
	"vexel/inference/lora/train"
)

// knownModelPaths are candidate paths checked when VEXEL_TEST_MODEL is not set.
var knownModelPaths = []string{
	"/Users/qeetbastudio/projects/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
}

// findTestModel returns a path to a model file or "" if none is available.
func findTestModel(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("VEXEL_TEST_MODEL"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		t.Logf("VEXEL_TEST_MODEL=%q not found; falling back to known paths", p)
	}
	for _, p := range knownModelPaths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// TestTrainingE2E validates the full training pipeline without running GPU
// training:
//  1. Data loading pipeline (LoadData + BuildLossMask)
//  2. Adapter weight initialisation (InitAdapter)
//  3. Checkpoint round-trip (SaveAdapter → LoadAdapter)
func TestTrainingE2E(t *testing.T) {
	modelPath := findTestModel(t)
	if modelPath == "" {
		t.Skip("no test model available; set VEXEL_TEST_MODEL or place a model at a known path")
	}
	t.Logf("test model: %s", modelPath)

	// ── 1. Create a small JSONL training file ────────────────────────────────
	tmpDir := t.TempDir()
	jsonlPath := filepath.Join(tmpDir, "train.jsonl")

	jsonlContent := `{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "In the beginning was the Word, and the Word was with God."}
{"text": "To be or not to be, that is the question."}
`
	if err := os.WriteFile(jsonlPath, []byte(jsonlContent), 0o644); err != nil {
		t.Fatalf("write JSONL: %v", err)
	}

	// ── 2. Validate the data loading pipeline ───────────────────────────────
	examples, err := train.LoadData(jsonlPath)
	if err != nil {
		t.Fatalf("LoadData: %v", err)
	}
	if len(examples) != 3 {
		t.Fatalf("LoadData: got %d examples, want 3", len(examples))
	}
	for i, ex := range examples {
		if ex.Format != train.FormatText {
			t.Errorf("example %d: format=%v, want FormatText", i, ex.Format)
		}
		if ex.Text == "" {
			t.Errorf("example %d: empty text", i)
		}
	}
	t.Logf("loaded %d training examples", len(examples))

	// Verify BuildLossMask on a synthetic token sequence.
	tokens := []int32{1, 2, 3, 4, 5}
	mask := train.BuildLossMask(tokens, train.FormatText, 0)
	if len(mask) != len(tokens) {
		t.Fatalf("BuildLossMask: got len %d, want %d", len(mask), len(tokens))
	}
	// For FormatText every position except the last should be 1.
	for i := 0; i < len(mask)-1; i++ {
		if mask[i] != 1 {
			t.Errorf("BuildLossMask: mask[%d]=%v, want 1", i, mask[i])
		}
	}
	if mask[len(mask)-1] != 0 {
		t.Errorf("BuildLossMask: last element=%v, want 0", mask[len(mask)-1])
	}
	t.Logf("loss mask validated (len=%d)", len(mask))

	// ── 3. Create a LoRA adapter (Qwen 0.5B realistic dimensions) ───────────
	const (
		rank       = 8
		alpha      = float32(16)
		numLayers  = 24
		hiddenSize = 896 // Qwen 0.5B hidden dim
		qDim       = 896 // Q projection output dim
		vDim       = 128 // V projection output dim (GQA head count × head dim)
	)

	cfg := lora.AdapterConfig{
		Rank:          rank,
		Alpha:         alpha,
		TargetModules: []string{"q_proj", "v_proj"},
		BaseModel:     "Qwen/Qwen2.5-0.5B-Instruct",
	}

	adapter := train.InitAdapter(cfg, numLayers, hiddenSize, qDim, vDim)

	if len(adapter.Layers) != numLayers {
		t.Fatalf("InitAdapter: got %d layers, want %d", len(adapter.Layers), numLayers)
	}
	wantScale := alpha / float32(rank) // 2.0
	if adapter.Scale != wantScale {
		t.Errorf("InitAdapter: Scale=%v, want %v", adapter.Scale, wantScale)
	}

	// Spot-check first and last layer.
	for _, idx := range []int{0, numLayers - 1} {
		la := adapter.Layers[idx]
		if !la.HasQ() {
			t.Errorf("layer %d: HasQ() false", idx)
		}
		if !la.HasV() {
			t.Errorf("layer %d: HasV() false", idx)
		}
		if la.QAShape != [2]int64{rank, hiddenSize} {
			t.Errorf("layer %d: QAShape=%v", idx, la.QAShape)
		}
		if la.QBShape != [2]int64{qDim, rank} {
			t.Errorf("layer %d: QBShape=%v", idx, la.QBShape)
		}
		if la.VAShape != [2]int64{rank, hiddenSize} {
			t.Errorf("layer %d: VAShape=%v", idx, la.VAShape)
		}
		if la.VBShape != [2]int64{vDim, rank} {
			t.Errorf("layer %d: VBShape=%v", idx, la.VBShape)
		}
	}
	t.Logf("adapter initialised: %d layers, rank=%d, alpha=%.1f, scale=%.2f",
		numLayers, rank, alpha, adapter.Scale)

	// ── 4. Checkpoint round-trip ─────────────────────────────────────────────
	adapterDir := filepath.Join(tmpDir, "adapter")

	if err := lora.SaveAdapter(adapter, adapterDir); err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	// Verify the expected files were created.
	for _, name := range []string{"adapter_config.json", "adapter_model.safetensors"} {
		p := filepath.Join(adapterDir, name)
		if _, err := os.Stat(p); err != nil {
			t.Errorf("SaveAdapter: missing file %s: %v", name, err)
		}
	}

	loaded, err := lora.LoadAdapter(adapterDir)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	// Config round-trip.
	if loaded.Config.Rank != rank {
		t.Errorf("round-trip Rank: got %d, want %d", loaded.Config.Rank, rank)
	}
	if loaded.Config.Alpha != alpha {
		t.Errorf("round-trip Alpha: got %v, want %v", loaded.Config.Alpha, alpha)
	}
	if loaded.Scale != wantScale {
		t.Errorf("round-trip Scale: got %v, want %v", loaded.Scale, wantScale)
	}
	if len(loaded.Layers) != numLayers {
		t.Fatalf("round-trip Layers: got %d, want %d", len(loaded.Layers), numLayers)
	}

	// Weight round-trip: compare first layer QA values element-by-element.
	origQA := adapter.Layers[0].QA
	loadedQA := loaded.Layers[0].QA
	if len(loadedQA) != len(origQA) {
		t.Fatalf("round-trip QA length: got %d, want %d", len(loadedQA), len(origQA))
	}
	for i, v := range origQA {
		if loadedQA[i] != v {
			t.Errorf("round-trip QA[%d]: got %v, want %v", i, loadedQA[i], v)
			break
		}
	}

	// B matrices must still be zero after round-trip.
	for _, v := range loaded.Layers[0].QB {
		if v != 0 {
			t.Errorf("round-trip QB: expected all zeros, got non-zero value %v", v)
			break
		}
	}

	t.Logf("checkpoint round-trip OK: %d layers saved and reloaded", len(loaded.Layers))
	t.Logf("TestTrainingE2E PASSED: data loading, adapter init, and checkpoint round-trip all verified")
}
