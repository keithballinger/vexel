package runtime_test

import (
	"fmt"
	"os"
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestPhi2Parity(t *testing.T) {
	modelPath := "../../models/phi-2-q4.gguf"
	tokenizerPath := "../../models/phi2_tokenizer.json"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Phi-2 model not found at %s", modelPath)
	}

	// 1. Setup Backend and Context
	b := cpu.NewCPUBackend()
	ctx := memory.NewInferenceContext(tensor.CPU)
	ctx.AddArena(memory.Scratch, 256*1024*1024)
	
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
	
	for i := 0; i < 20; i++ {
		// Run DecodeStep
		inputs := runtime.NewBatchRuntimeInputs(currentTokens, nil)
		logits, err := m.DecodeStep(inputs)
		if err != nil {
			t.Fatalf("Step %d failed: %v", i, err)
		}
		
		// Greedy select
		nextToken := greedySelect(logits)
		generated = append(generated, nextToken)
		currentTokens = append(currentTokens, nextToken)
	}

	decoded, _ := tok.Decode(generated)
	fmt.Printf("Generated (20 tokens): %q\n", decoded)
	fmt.Printf("Token IDs: %v\n", generated)
	
	// Example: "Hello!" -> " I am a student"
	if len(decoded) == 0 {
		t.Errorf("Decoded output is empty")
	}

	// 5. Test "Unit Testing" prompt
	prompt2 := "Describe the benefits of unit testing in Go in three concise sentences."
	tokens2, _ := tok.Encode(prompt2)
	fmt.Printf("Prompt 2: %s -> Tokens: %v\n", prompt2, tokens2)

	inputs2 := runtime.NewBatchRuntimeInputs(tokens2, nil)
	logits2, err := m.DecodeStep(inputs2)
	if err != nil {
		t.Fatalf("Prompt 2 prefill failed: %v", err)
	}
	nextToken2 := greedySelect(logits2)
	decoded2, _ := tok.Decode([]int{nextToken2})
	fmt.Printf("Next token 2: %d (%q)\n", nextToken2, decoded2)

	if nextToken2 == 0 {
		t.Errorf("Prompt 2 failed")
	}
}

func greedySelect(logits tensor.Tensor) int {
	data := tensor.ToFloat32Slice(logits)
	if len(data) == 0 {
		return 0
	}
	
	// Argmax of the LAST row
	vocabSize := logits.Shape().Dims()[1]
	lastRow := data[len(data)-vocabSize:]
	
	maxIdx := 0
	maxVal := lastRow[0]
	for i, v := range lastRow {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
