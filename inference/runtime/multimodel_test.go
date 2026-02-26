//go:build metal && darwin && cgo

package runtime

import (
	"fmt"
	"os"
	"testing"

	"vexel/inference/pkg/gguf"
)

const defaultModelPath = "../../models/llama-2-7b.Q4_0.gguf"

func getModelPath() string {
	if p := os.Getenv("VEXEL_TEST_MODEL"); p != "" {
		return p
	}
	return defaultModelPath
}

// TestGGUFConfigAutoDetection_LLaMA2 verifies that GGUF metadata is correctly
// parsed and mapped to ModelConfig for LLaMA 2 7B.
//
// Track 5: Multi-Model Validation, Phase 1.
func TestGGUFConfigAutoDetection_LLaMA2(t *testing.T) {
	modelPath := getModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model file not found: %s (set VEXEL_TEST_MODEL env var)", modelPath)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("Failed to open GGUF file: %v", err)
	}
	defer gf.Close()

	mcv := gf.GetModelConfig()
	cfg := ModelConfigFromGGUF(mcv)

	// Verify architecture detection
	if mcv.Architecture != "llama" {
		t.Errorf("Architecture: got %q, want %q", mcv.Architecture, "llama")
	}

	// Verify LLaMA-family settings
	if cfg.NormType != NormRMSNorm {
		t.Errorf("NormType: got %v, want RMSNorm", cfg.NormType)
	}
	if cfg.MLPType != MLPSwiGLU {
		t.Errorf("MLPType: got %v, want SwiGLU", cfg.MLPType)
	}
	if cfg.HasBias {
		t.Error("HasBias: got true, want false for LLaMA")
	}
	if cfg.ParallelResidual {
		t.Error("ParallelResidual: got true, want false for LLaMA")
	}
	if cfg.RoPENeox {
		t.Error("RoPENeox: got true, want false for LLaMA (uses interleaved pairs)")
	}

	// Verify LLaMA 2 7B dimensions
	if cfg.HiddenSize != 4096 {
		t.Errorf("HiddenSize: got %d, want 4096", cfg.HiddenSize)
	}
	if cfg.IntermediateSize != 11008 {
		t.Errorf("IntermediateSize: got %d, want 11008", cfg.IntermediateSize)
	}
	if cfg.NumHiddenLayers != 32 {
		t.Errorf("NumHiddenLayers: got %d, want 32", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads != 32 {
		t.Errorf("NumAttentionHeads: got %d, want 32", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads != 32 {
		t.Errorf("NumKeyValueHeads: got %d, want 32 (LLaMA 2 7B uses MHA)", cfg.NumKeyValueHeads)
	}
	if cfg.VocabSize != 32000 {
		t.Errorf("VocabSize: got %d, want 32000", cfg.VocabSize)
	}

	// Verify GQA ratio: LLaMA 2 7B = 1:1 (MHA), LLaMA 2 70B = 8:1
	gqaRatio := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if gqaRatio != 1 {
		t.Errorf("GQA ratio: got %d:1, want 1:1 for LLaMA 2 7B", gqaRatio)
	}

	headDim := cfg.HiddenSize / cfg.NumAttentionHeads
	if headDim != 128 {
		t.Errorf("HeadDim: got %d, want 128", headDim)
	}

	t.Logf("LLaMA 2 7B config verified: %dx%d, %d layers, %d heads (GQA=%d:1), vocab=%d",
		cfg.HiddenSize, cfg.IntermediateSize, cfg.NumHiddenLayers,
		cfg.NumAttentionHeads, gqaRatio, cfg.VocabSize)
}

// TestModelConfigFromGGUF_ArchitectureDetection verifies that different
// architecture strings produce correct config settings.
//
// Track 5: Multi-Model Validation, Phase 1-2.
func TestModelConfigFromGGUF_ArchitectureDetection(t *testing.T) {
	tests := []struct {
		name             string
		arch             string
		wantNorm         NormType
		wantMLP          MLPType
		wantBias         bool
		wantParallel     bool
		wantRoPENeox     bool
	}{
		{
			name:     "llama",
			arch:     "llama",
			wantNorm: NormRMSNorm, wantMLP: MLPSwiGLU,
			wantBias: false, wantParallel: false, wantRoPENeox: false,
		},
		{
			name:     "mistral",
			arch:     "mistral",
			wantNorm: NormRMSNorm, wantMLP: MLPSwiGLU,
			wantBias: false, wantParallel: false, wantRoPENeox: false,
		},
		{
			name:     "qwen2",
			arch:     "qwen2",
			wantNorm: NormRMSNorm, wantMLP: MLPSwiGLU,
			wantBias: false, wantParallel: false, wantRoPENeox: false,
		},
		{
			name:     "phi2",
			arch:     "phi2",
			wantNorm: NormLayerNorm, wantMLP: MLPGELU,
			wantBias: true, wantParallel: true, wantRoPENeox: true,
		},
		{
			name:     "phi3",
			arch:     "phi3",
			wantNorm: NormLayerNorm, wantMLP: MLPGELU,
			wantBias: true, wantParallel: true, wantRoPENeox: true,
		},
		{
			name:     "gpt2",
			arch:     "gpt2",
			wantNorm: NormLayerNorm, wantMLP: MLPGELU,
			wantBias: true, wantParallel: false, wantRoPENeox: true,
		},
		{
			name:     "gptneox",
			arch:     "gptneox",
			wantNorm: NormLayerNorm, wantMLP: MLPGELU,
			wantBias: true, wantParallel: false, wantRoPENeox: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mcv := gguf.ModelConfigValues{
				Architecture:     tt.arch,
				NumLayers:        2,
				HiddenSize:       256,
				IntermediateSize: 512,
				NumHeads:         4,
				NumKVHeads:       4,
				VocabSize:        1000,
				ContextLength:    512,
				RoPETheta:        10000,
			}

			cfg := ModelConfigFromGGUF(mcv)

			if cfg.NormType != tt.wantNorm {
				t.Errorf("NormType: got %v, want %v", cfg.NormType, tt.wantNorm)
			}
			if cfg.MLPType != tt.wantMLP {
				t.Errorf("MLPType: got %v, want %v", cfg.MLPType, tt.wantMLP)
			}
			if cfg.HasBias != tt.wantBias {
				t.Errorf("HasBias: got %v, want %v", cfg.HasBias, tt.wantBias)
			}
			if cfg.ParallelResidual != tt.wantParallel {
				t.Errorf("ParallelResidual: got %v, want %v", cfg.ParallelResidual, tt.wantParallel)
			}
			if cfg.RoPENeox != tt.wantRoPENeox {
				t.Errorf("RoPENeox: got %v, want %v", cfg.RoPENeox, tt.wantRoPENeox)
			}
		})
	}
}

// TestModelConfigFromGGUF_GQARatios verifies correct GQA head configurations.
//
// Track 5: Multi-Model Validation, Phase 3 Task 2.
func TestModelConfigFromGGUF_GQARatios(t *testing.T) {
	tests := []struct {
		name      string
		numHeads  int
		numKV     int
		wantRatio int
		model     string
	}{
		{"MHA_1:1", 32, 32, 1, "LLaMA 2 7B / Phi-2"},
		{"GQA_4:1", 32, 8, 4, "LLaMA 2 70B"},
		{"GQA_8:1", 32, 4, 8, "LLaMA 3 8B"},
		{"MQA_32:1", 32, 1, 32, "Multi-Query (theoretical)"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mcv := gguf.ModelConfigValues{
				Architecture:     "llama",
				NumLayers:        2,
				HiddenSize:       tt.numHeads * 128,
				IntermediateSize: tt.numHeads * 128 * 4,
				NumHeads:         tt.numHeads,
				NumKVHeads:       tt.numKV,
				VocabSize:        32000,
				ContextLength:    2048,
				RoPETheta:        10000,
			}

			cfg := ModelConfigFromGGUF(mcv)

			if cfg.NumAttentionHeads != tt.numHeads {
				t.Errorf("NumAttentionHeads: got %d, want %d", cfg.NumAttentionHeads, tt.numHeads)
			}
			if cfg.NumKeyValueHeads != tt.numKV {
				t.Errorf("NumKeyValueHeads: got %d, want %d", cfg.NumKeyValueHeads, tt.numKV)
			}

			ratio := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
			if ratio != tt.wantRatio {
				t.Errorf("GQA ratio: got %d:1, want %d:1", ratio, tt.wantRatio)
			}

			headDim := cfg.HiddenSize / cfg.NumAttentionHeads
			qDim := cfg.NumAttentionHeads * headDim
			kvDim := cfg.NumKeyValueHeads * headDim

			t.Logf("%s: heads=%d, kv_heads=%d, ratio=%d:1, q_dim=%d, kv_dim=%d",
				tt.model, cfg.NumAttentionHeads, cfg.NumKeyValueHeads,
				ratio, qDim, kvDim)
		})
	}
}

// TestModelConfigFromGGUF_Defaults verifies sensible defaults for missing metadata.
func TestModelConfigFromGGUF_Defaults(t *testing.T) {
	// Minimal config - missing context length and RoPE theta
	mcv := gguf.ModelConfigValues{
		Architecture: "llama",
		NumLayers:    2,
		HiddenSize:   256,
		NumHeads:     4,
		NumKVHeads:   4,
		VocabSize:    1000,
	}

	cfg := ModelConfigFromGGUF(mcv)

	if cfg.MaxSeqLen != 2048 {
		t.Errorf("Default MaxSeqLen: got %d, want 2048", cfg.MaxSeqLen)
	}
	if cfg.RoPETheta != 10000.0 {
		t.Errorf("Default RoPETheta: got %f, want 10000.0", cfg.RoPETheta)
	}
	if cfg.RMSNormEPS != 1e-5 {
		t.Errorf("Default RMSNormEPS: got %f, want 1e-5", cfg.RMSNormEPS)
	}
}

// TestModelConfigFromGGUF_SlidingWindow verifies sliding window config propagation.
func TestModelConfigFromGGUF_SlidingWindow(t *testing.T) {
	tests := []struct {
		name   string
		arch   string
		window int
	}{
		{"llama_no_window", "llama", 0},
		{"mistral_4096", "mistral", 4096},
		{"phi3_2048", "phi3", 2048},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mcv := gguf.ModelConfigValues{
				Architecture:  tt.arch,
				NumLayers:     2,
				HiddenSize:    256,
				NumHeads:      4,
				NumKVHeads:    4,
				VocabSize:     1000,
				ContextLength: 8192,
				SlidingWindow: tt.window,
			}

			cfg := ModelConfigFromGGUF(mcv)

			if cfg.SlidingWindow != tt.window {
				t.Errorf("SlidingWindow: got %d, want %d", cfg.SlidingWindow, tt.window)
			}
		})
	}
}

// TestModelConfigFromGGUF_PartialRoPE verifies partial RoPE dimension propagation (Phi-2).
func TestModelConfigFromGGUF_PartialRoPE(t *testing.T) {
	// Phi-2 uses partial RoPE: 32 dimensions out of 80 headDim
	mcv := gguf.ModelConfigValues{
		Architecture:     "phi2",
		NumLayers:        32,
		HiddenSize:       2560,
		IntermediateSize: 10240,
		NumHeads:         32,
		NumKVHeads:       32,
		VocabSize:        51200,
		ContextLength:    2048,
		RoPETheta:        10000,
		RoPEDimCount:     32, // Partial RoPE
	}

	cfg := ModelConfigFromGGUF(mcv)

	if cfg.RoPEDim != 32 {
		t.Errorf("RoPEDim: got %d, want 32 (partial RoPE for Phi-2)", cfg.RoPEDim)
	}

	headDim := cfg.HiddenSize / cfg.NumAttentionHeads
	if headDim != 80 {
		t.Errorf("HeadDim: got %d, want 80 for Phi-2", headDim)
	}

	t.Logf("Phi-2 config: headDim=%d, ropeDim=%d (%.0f%% rotated)",
		headDim, cfg.RoPEDim, float64(cfg.RoPEDim)/float64(headDim)*100)
}

// TestGGUFTensorInfo_LLaMA2 verifies that the GGUF tensor index is parsable
// and contains expected tensors for LLaMA 2 7B.
func TestGGUFTensorInfo_LLaMA2(t *testing.T) {
	modelPath := getModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model file not found: %s", modelPath)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	defer gf.Close()

	tensors := gf.Tensors

	// LLaMA 2 7B should have: token_embd + output_norm + output + 32 layers * 9 tensors
	// Expected tensors per layer: attn_q, attn_k, attn_v, attn_output, attn_norm,
	//                              ffn_gate, ffn_up, ffn_down, ffn_norm
	expectedMin := 32*9 + 3 // layers + embedding + norm + output
	if len(tensors) < expectedMin {
		t.Errorf("Expected at least %d tensors, got %d", expectedMin, len(tensors))
	}

	// Check key tensors exist
	expectedTensors := []string{
		"token_embd.weight",
		"output_norm.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
		"blk.31.attn_q.weight", // Last layer
	}

	tensorMap := make(map[string]bool)
	for _, ti := range tensors {
		tensorMap[ti.Name] = true
	}

	for _, name := range expectedTensors {
		if !tensorMap[name] {
			t.Errorf("Expected tensor %q not found in GGUF", name)
		}
	}

	// Count quantization types
	typeCounts := make(map[string]int)
	for _, ti := range tensors {
		typeCounts[ti.Type.String()]++
	}

	fmt.Printf("\n[GGUF TENSOR SUMMARY - LLaMA 2 7B]\n")
	fmt.Printf("Total tensors: %d\n", len(tensors))
	for ttype, count := range typeCounts {
		fmt.Printf("  %s: %d tensors\n", ttype, count)
	}

	t.Logf("GGUF tensor index verified: %d tensors across %d quant types",
		len(tensors), len(typeCounts))
}
