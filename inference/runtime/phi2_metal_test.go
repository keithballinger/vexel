//go:build metal && darwin && cgo

package runtime_test

import (
	"fmt"
	"os"
	"testing"
	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestPhi2MetalParity(t *testing.T) {
	os.Setenv("DEBUG_DECODE", "1")
	modelPath := "../../models/phi-2-q4.gguf"
	tokenizerPath := "../../models/phi2_tokenizer.json"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Phi-2 model not found at %s", modelPath)
	}

	// 1. Setup Metal Backend and Context
	b, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to init Metal backend: %v", err)
	}
	defer b.Close()

	ctx := memory.NewInferenceContext(tensor.Metal)
	ctx.AddArenaWithBackend(memory.Scratch, 256*1024*1024, b.Alloc)
	
	// 2. Load Config, Weights and Tokenizer
	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("Failed to open GGUF: %v", err)
	}
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	m, err := runtime.NewModelRuntime(b, ctx, nil, modelCfg)
	if err != nil {
		t.Fatalf("Failed to create model runtime: %v", err)
	}
	
	err = m.LoadWeightsF32(modelPath)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	err = m.CopyWeightsToDevice()
	if err != nil {
		t.Fatalf("Failed to copy weights to GPU: %v", err)
	}

	tok, err := tokenizer.Load(tokenizerPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	// 3. Tokenize "Hello!"
	prompt := "Hello!"
	tokens, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	fmt.Printf("Prompt: %s -> Tokens: %v\n", prompt, tokens)

	// 4. Run Inference (Greedy)
	generated := []int{}
	currentTokens := tokens
	
	for i := 0; i < 10; i++ {
		inputs := runtime.NewBatchRuntimeInputs(currentTokens, nil)
		logits, err := m.DecodeStep(inputs)
		if err != nil {
			t.Fatalf("Step %d failed: %v", i, err)
		}
		
		// Copy logits to host for sampling
		vocabSize := modelCfg.VocabSize
		batchSize := len(currentTokens)
		logitsBytes := make([]byte, batchSize*vocabSize*4)
		b.ToHost(logitsBytes, logits.DevicePtr())
		b.Sync()
		logitsF32 := tensor.BytesToFloat32(logitsBytes)

		// Argmax of the LAST row
		lastRow := logitsF32[len(logitsF32)-vocabSize:]
		maxIdx := 0
		maxVal := lastRow[0]
		for j, v := range lastRow {
			if v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}

		nextToken := maxIdx
		generated = append(generated, nextToken)
		currentTokens = append(currentTokens, nextToken)
	}

	decoded, _ := tok.Decode(generated)
	fmt.Printf("Metal Generated: %q\n", decoded)
	fmt.Printf("Token IDs: %v\n", generated)

	if nextToken := generated[0]; nextToken == 0 {
		t.Errorf("Metal produced token 0, likely failure")
	}
}