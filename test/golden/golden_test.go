package golden_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"vexel/inference/backend/cpu"
	"vexel/inference/kv"
	"vexel/inference/memory"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
	"vexel/test/golden/npy"
)

const (
	goldenDataDir = "./data"
	modelDir      = "../../models"
	// Tolerances for different test types
	// Kernel-level tests use strict tolerance
	kernelTolerance = 1e-5
	// Full forward pass accumulates FP32 rounding errors across 22 layers
	forwardPassTolerance = 1e-3
)

// Metadata matches the structure of metadata.json
type Metadata struct {
	ModelName         string   `json:"model_name"`
	InputTokens       []int    `json:"input_tokens"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	HeadDim           int      `json:"head_dim"`
	VocabSize         int      `json:"vocab_size"`
	RopeTheta         float64  `json:"rope_theta"`
	RMSNormEPS        float64  `json:"rms_norm_eps"`
	LayersCaptured    []int    `json:"layers_captured"`
	Outputs           []string `json:"outputs"`
}

func loadMetadata(t *testing.T) Metadata {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(goldenDataDir, "metadata.json"))
	if err != nil {
		t.Fatalf("Failed to load metadata: %v", err)
	}
	var m Metadata
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("Failed to parse metadata: %v", err)
	}
	return m
}

func loadGolden(t *testing.T, name string) []float32 {
	t.Helper()
	path := filepath.Join(goldenDataDir, name+".npy")
	data, _, err := npy.LoadFloat32(path)
	if err != nil {
		t.Fatalf("Failed to load golden data %s: %v", name, err)
	}
	return data
}

// compareSlices checks if two float32 slices are equal within tolerance.
// Returns the max absolute difference and the index where it occurred.
func compareSlices(a, b []float32, tol float32) (maxDiff float32, maxIdx int, equal bool) {
	if len(a) != len(b) {
		return 0, -1, false
	}
	equal = true
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > maxDiff {
			maxDiff = diff
			maxIdx = i
		}
		if diff > tol {
			equal = false
		}
	}
	return maxDiff, maxIdx, equal
}

func setupRuntime(t *testing.T, meta Metadata) *runtime.ModelRuntime {
	t.Helper()

	cfg := runtime.ModelConfig{
		HiddenSize:        meta.HiddenSize,
		IntermediateSize:  meta.IntermediateSize,
		NumHiddenLayers:   meta.NumHiddenLayers,
		NumAttentionHeads: meta.NumAttentionHeads,
		NumKeyValueHeads:  meta.NumKeyValueHeads,
		VocabSize:         meta.VocabSize,
		MaxSeqLen:         2048,
		RoPETheta:         meta.RopeTheta,
		RMSNormEPS:        meta.RMSNormEPS,
		DType:             tensor.Float32,
	}

	backend := cpu.NewBackend()
	ctx := memory.NewInferenceContext(tensor.CPU)
	scratchSize := cfg.ScratchBytes(1)
	logitsSize := int64(cfg.VocabSize) * 4
	ctx.AddArena(memory.Scratch, int(scratchSize+logitsSize*2))
	cache := &kv.KVCache{}

	rt, err := runtime.NewModelRuntime(backend, ctx, cache, cfg)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	weightsPath := filepath.Join(modelDir, "tiny_model.safetensors")
	if err := rt.LoadWeights(weightsPath); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	return rt
}

func TestEmbeddingMatchesGolden(t *testing.T) {
	meta := loadMetadata(t)
	golden := loadGolden(t, "embedding")

	rt := setupRuntime(t, meta)
	backend := cpu.NewBackend()

	// Get embedding table
	table := tensor.ToFloat32Slice(rt.Embedding)
	if table == nil {
		t.Fatal("Embedding table is nil")
	}

	// Perform lookup for input tokens
	hiddenSize := meta.HiddenSize
	tokens := meta.InputTokens
	result := make([]float32, len(tokens)*hiddenSize)
	backend.Embedding(tokens, table, result, hiddenSize)

	// Compare - embedding is a simple lookup, should be exact
	maxDiff, maxIdx, equal := compareSlices(result, golden, kernelTolerance)
	if !equal {
		t.Errorf("Embedding mismatch: max diff = %e at index %d (tolerance = %e)",
			maxDiff, maxIdx, kernelTolerance)
		t.Errorf("  Got:    %v", result[:min(10, len(result))])
		t.Errorf("  Golden: %v", golden[:min(10, len(golden))])
	} else {
		t.Logf("Embedding matches golden (max diff = %e)", maxDiff)
	}
}

func TestLogitsMatchGolden(t *testing.T) {
	meta := loadMetadata(t)
	golden := loadGolden(t, "logits")

	rt := setupRuntime(t, meta)

	// Run full forward pass
	inputs := runtime.NewBatchRuntimeInputs(meta.InputTokens, nil)
	logits, err := rt.DecodeStep(inputs)
	if err != nil {
		t.Fatalf("DecodeStep failed: %v", err)
	}

	result := tensor.ToFloat32Slice(logits)
	if result == nil {
		t.Fatal("Logits are nil")
	}

	// Compare - full forward pass uses relaxed tolerance due to FP32 accumulation
	maxDiff, maxIdx, equal := compareSlices(result, golden, forwardPassTolerance)
	if !equal {
		t.Errorf("Logits mismatch: max diff = %e at index %d (tolerance = %e)",
			maxDiff, maxIdx, forwardPassTolerance)

		// Show some context around the max diff
		start := max(0, maxIdx-5)
		end := min(len(result), maxIdx+5)
		t.Errorf("  Got[%d:%d]:    %v", start, end, result[start:end])
		t.Errorf("  Golden[%d:%d]: %v", start, end, golden[start:end])
	} else {
		t.Logf("Logits match golden (max diff = %e)", maxDiff)
	}
}

func TestRMSNormMatchesGolden(t *testing.T) {
	meta := loadMetadata(t)

	// Test layer 0 input_layernorm
	goldenInput := loadGolden(t, "embedding")
	goldenOutput := loadGolden(t, "layer_0.input_layernorm")

	rt := setupRuntime(t, meta)
	backend := cpu.NewBackend()

	// Get layer 0 norm weights
	layer0 := rt.Layer(0)
	if layer0 == nil {
		t.Fatal("Layer 0 is nil")
	}
	normWeights := tensor.ToFloat32Slice(layer0.AttnNorm)
	if normWeights == nil {
		t.Fatal("AttnNorm weights are nil")
	}

	// Run RMSNorm
	result := make([]float32, len(goldenInput))
	backend.RMSNorm(goldenInput, normWeights, result, 1, meta.HiddenSize, float32(meta.RMSNormEPS))

	// Compare - kernel-level test uses strict tolerance
	maxDiff, maxIdx, equal := compareSlices(result, goldenOutput, kernelTolerance)
	if !equal {
		t.Errorf("RMSNorm mismatch: max diff = %e at index %d (tolerance = %e)",
			maxDiff, maxIdx, kernelTolerance)
		t.Errorf("  Got:    %v", result[:min(10, len(result))])
		t.Errorf("  Golden: %v", goldenOutput[:min(10, len(goldenOutput))])
	} else {
		t.Logf("RMSNorm matches golden (max diff = %e)", maxDiff)
	}
}

func TestQProjectionMatchesGolden(t *testing.T) {
	meta := loadMetadata(t)

	// Use output of input_layernorm as input to Q projection
	goldenInput := loadGolden(t, "layer_0.input_layernorm")
	goldenOutput := loadGolden(t, "layer_0.q_proj")

	rt := setupRuntime(t, meta)
	backend := cpu.NewBackend()

	// Get layer 0 Wq weights
	layer0 := rt.Layer(0)
	if layer0 == nil {
		t.Fatal("Layer 0 is nil")
	}
	wQ := tensor.ToFloat32Slice(layer0.Wq)
	if wQ == nil {
		t.Fatal("Wq weights are nil")
	}

	// Get output dimension from weight shape
	qDim := layer0.Wq.Shape().Dims()[0] // numHeads * headDim

	// Run MatmulTransposeB: input [1, hidden] x W^T [hidden, qDim] -> [1, qDim]
	result := make([]float32, qDim)
	backend.MatmulTransposeB(goldenInput, wQ, result, 1, qDim, meta.HiddenSize)

	// Compare - kernel-level test uses strict tolerance
	maxDiff, maxIdx, equal := compareSlices(result, goldenOutput, kernelTolerance)
	if !equal {
		t.Errorf("Q projection mismatch: max diff = %e at index %d (tolerance = %e)",
			maxDiff, maxIdx, kernelTolerance)
		t.Errorf("  Got:    %v", result[:min(10, len(result))])
		t.Errorf("  Golden: %v", goldenOutput[:min(10, len(goldenOutput))])
	} else {
		t.Logf("Q projection matches golden (max diff = %e)", maxDiff)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
