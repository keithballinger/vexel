package runtime_test

import (
	"testing"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestModelConfig(t *testing.T) {
	// Test creating a Llama-3-8B style config
	cfg := runtime.ModelConfig{
		HiddenSize:        4096,
		NumHiddenLayers:   32,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8, // GQA
		VocabSize:         128256,
		MaxSeqLen:         8192,
		RoPETheta:         500000.0,
		RMSNormEPS:        1e-5,
		DType:             tensor.BFloat16,
	}

	// Verify basic validation logic (if we add any)
	// For now, we just ensure the struct fields exist and are accessible
	if cfg.HiddenSize != 4096 {
		t.Error("HiddenSize mismatch")
	}

	// Test standard constructor/factory if applicable
	// For now, the plan mentions "Llama3_8B configuration", which implies a helper.
	llama3 := runtime.Llama3_8B()
	if llama3.NumHiddenLayers != 32 {
		t.Error("Llama3_8B preset has incorrect NumLayers")
	}
}
