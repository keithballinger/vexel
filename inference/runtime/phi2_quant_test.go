package runtime

import (
	"fmt"
	"os"
	"testing"

	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/tensor"
)

// TestPhi2QuantizedParity reproduces the issue where Q4_K weights produce zero output for Phi-2.
func TestPhi2QuantizedParity(t *testing.T) {
	modelPath := "../../models/phi-2-q4.gguf"
	tokenizerPath := "../../models/phi2_tokenizer.json"

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model not found")
	}

	// 1. Initialize Backend
	backend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create metal backend: %v", err)
	}
	defer backend.Close()

	ctx := memory.NewInferenceContext(tensor.Metal)
	ctx.AddArenaWithBackend(memory.Scratch, 512*1024*1024, backend.Alloc)

	// 2. Load Model (Quantized)
	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	modelCfg := ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	m, err := NewModelRuntime(backend, ctx, nil, modelCfg)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	// Use LoadWeights to keep Q4_K/Q6_K weights raw on GPU
	err = m.LoadWeights(modelPath)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	// Verify quantization of a known Q4_K tensor
	fmt.Printf("Global OutputHead IsQuantized: %v\n", m.OutputHead.IsQuantized())

	err = m.CopyWeightsToDevice()
	if err != nil {
		t.Fatalf("Failed to copy weights to GPU: %v", err)
	}

	// Enable debug decoding to see tensor values
	os.Setenv("DEBUG_DECODE", "1")
	defer os.Unsetenv("DEBUG_DECODE")

	// 3. Run Inference on "Hello!"
	tok, err := tokenizer.Load(tokenizerPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}
	tokens, _ := tok.Encode("Hello!")
	
	// Use DecodeStep (prefill)
	inputs := NewBatchRuntimeInputs(tokens, nil)
	logits, err := m.DecodeStep(inputs)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	// Check logits
	logitsBytes := make([]byte, len(tokens)*modelCfg.VocabSize*4)
	backend.ToHost(logitsBytes, logits.DevicePtr())
	backend.Sync()
	logitsF32 := tensor.BytesToFloat32(logitsBytes)

	// Check for zeros (which indicates the failure)
	allZeros := true
	maxVal := float32(0.0)
	maxIdx := 0
	for j, v := range logitsF32 {
		if v != 0 {
			allZeros = false
		}
		if v > maxVal {
			maxVal = v
			maxIdx = j
		}
	}

	fmt.Printf("Max logit value: %f at index %d\n", maxVal, maxIdx)
	
	// Expected next token for "Hello!" is 314 (" I") or similar
	if maxIdx == 314 {
		t.Logf("Success! Produced correct token %d", maxIdx)
	} else {
		t.Logf("Failure? Produced token %d (expected 314)", maxIdx)
	}

	if allZeros {
		t.Log("Reproduced failure: All logits are zero with Q4_K weights")
	} 
}
