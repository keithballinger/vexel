package runtime_test

import (
	"testing"

	"vexel/inference/backend/cpu"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// testConfig returns a minimal config for unit tests
func testConfig() runtime.ModelConfig {
	return runtime.ModelConfig{
		HiddenSize:        32,
		IntermediateSize:  64,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  2, // GQA: 4 Q heads share 2 KV heads
		VocabSize:         100,
		MaxSeqLen:         16,
		RoPETheta:         10000.0,
		RMSNormEPS:        1e-5,
		DType:             tensor.Float32,
	}
}

func TestBlockRuntime(t *testing.T) {
	// Create backend and config
	b := cpu.NewBackend()
	cfg := testConfig()

	// Initialize BlockRuntime
	rt := runtime.NewBlockRuntime(b, cfg)

	if rt == nil {
		t.Error("Failed to create BlockRuntime")
	}
}
