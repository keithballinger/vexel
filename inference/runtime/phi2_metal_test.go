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

func TestPhi2MetalParity(t *testing.T) {
	modelPath := "../../models/phi-2-q4.gguf"
	tokenizerPath := "../../models/phi2_tokenizer.json"

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model not found")
	}

	// 1. Initialize Backend and Context
	backend, err := metal.NewBackend(0)
	if err != nil {
		t.Fatalf("Failed to create metal backend: %v", err)
	}
	defer backend.Close()

	ctx := memory.NewInferenceContext(tensor.Metal)
	// Add arena for intermediate tensors
	ctx.AddArenaWithBackend(memory.Scratch, 512*1024*1024, backend.Alloc)

	// 2. Load Model Configuration and Weights
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

	// Use LoadWeights (quantized support)
	err = m.LoadWeights(modelPath)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	err = m.CopyWeightsToDevice()
	if err != nil {
		t.Fatalf("Failed to copy weights to GPU: %v", err)
	}

	// Initialize GPU KV Cache for optimized inference
	m.CreateGPUKVCache(512)

	if os.Getenv("VEXEL_GPU_PROFILE") == "1" {
		defer metal.PrintGPUProfile()
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
	
	// A. Prefill
	logits, err := m.DecodeWithGPUKV(tokens, 0)
	if err != nil {
		t.Fatalf("Prefill failed: %v", err)
	}
	
	sampleNext := func(logits tensor.Tensor) int {
		vocabSize := modelCfg.VocabSize
		logitsBytes := make([]byte, vocabSize*4)
		backend.ToHost(logitsBytes, logits.DevicePtr())
		backend.Sync()
		logitsF32 := tensor.BytesToFloat32(logitsBytes)

		maxIdx := 0
		maxVal := logitsF32[0]
		for j, v := range logitsF32 {
			if v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}
		return maxIdx
	}

	nextToken := sampleNext(logits)
	generated = append(generated, nextToken)

	// B. Incremental Decode (20 tokens)
	pos := len(tokens)
	for i := 0; i < 19; i++ {
		logits, err = m.DecodeWithGPUKV([]int{nextToken}, pos)
		if err != nil {
			t.Fatalf("Step %d failed: %v", i, err)
		}
		nextToken = sampleNext(logits)
		generated = append(generated, nextToken)
		pos++
	}

	decoded, _ := tok.Decode(generated)
	fmt.Printf("Metal Generated (20 tokens): %q\n", decoded)
	fmt.Printf("Token IDs: %v\n", generated)

	// 5. Test "Unit Testing" prompt
	prompt2 := "Describe the benefits of unit testing in Go in three concise sentences."
	tokens2, _ := tok.Encode(prompt2)
	fmt.Printf("Prompt 2: %s -> Tokens: %v\n", prompt2, tokens2)

	logits2, err := m.DecodeWithGPUKV(tokens2, 0)
	if err != nil {
		t.Fatalf("Prompt 2 prefill failed: %v", err)
	}
	
	maxIdx2 := sampleNext(logits2)
	decoded2, _ := tok.Decode([]int{maxIdx2})
	fmt.Printf("Next token 2: %d (%q)\n", maxIdx2, decoded2)

	if maxIdx2 == 0 {
		t.Errorf("Prompt 2 produced token 0")
	}
}

func BenchmarkPhi2Metal(b *testing.B) {
	modelPath := "../../models/phi-2-q4.gguf"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skip("Model not found")
	}

	backend, err := metal.NewBackend(0)
	if err != nil {
		b.Fatal(err)
	}
	defer backend.Close()
	
	ctx := memory.NewInferenceContext(tensor.Metal)
	ctx.AddArenaWithBackend(memory.Scratch, 512*1024*1024, backend.Alloc)
	
	gf, err := gguf.Open(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	modelCfg := ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	m, err := NewModelRuntime(backend, ctx, nil, modelCfg)
	if err != nil {
		b.Fatal(err)
	}
	
	// Use quantized weights for benchmark
	err = m.LoadWeights(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	
	err = m.CopyWeightsToDevice()
	if err != nil {
		b.Fatal(err)
	}

	m.CreateGPUKVCache(512)

	if os.Getenv("VEXEL_GPU_PROFILE") == "1" {
		metal.ResetGPUProfile()
		defer metal.PrintGPUProfile()
	}
	if os.Getenv("DEBUG_PROFILE") == "1" {
		ResetProfile()
		defer PrintProfile()
	}

	// One decode step (pos 10)
	nextToken := 100
	pos := 10

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.DecodeWithGPUKV([]int{nextToken}, pos)
		if err != nil {
			b.Fatal(err)
		}
		// No manual Sync here, DecodeWithGPUKV already syncs at end of token
	}
}