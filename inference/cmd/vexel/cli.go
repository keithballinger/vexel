package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"vexel/inference/backend/cpu"
	"vexel/inference/cmd/vexel/internal"
	"vexel/inference/memory"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/tensor"
)

func main() {
	modelDir := flag.String("model", "models", "Directory containing model files")
	temperature := flag.Float64("temp", 0.7, "Sampling temperature (0 = greedy)")
	topK := flag.Int("top-k", 40, "Top-K sampling (0 = disabled)")
	topP := flag.Float64("top-p", 0.9, "Top-P nucleus sampling (0 = disabled)")
	maxTokens := flag.Int("max-tokens", 256, "Maximum tokens to generate per response")
	flag.Parse()

	fmt.Println("Vexel Inference Engine - Interactive Mode")
	fmt.Println("Loading model...")

	// 1. Load Config
	configPath := filepath.Join(*modelDir, "tiny_config.json") // Adjust name if needed
	// ... (Load logic same as before) ...
	configFile, err := os.Open(configPath)
	// Fallback logic
	if err != nil {
		configPath = filepath.Join(*modelDir, "config.json")
		configFile, _ = os.Open(configPath)
	}
	if configFile != nil {
		configFile.Close()
	}

	// Hardcoded config for demo
	cfg := runtime.ModelConfig{
		HiddenSize:        2048,
		IntermediateSize:  5632, 
		NumHiddenLayers:   22,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  4,
		VocabSize:         32000,
		MaxSeqLen:         2048,
		RoPETheta:         10000.0,
		RMSNormEPS:        1e-5,
		DType:             tensor.Float32,
	}

	// 2. Load Tokenizer
	tokPath := filepath.Join(*modelDir, "tiny_tokenizer.json")
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		log.Printf("Warning: Failed to load tokenizer: %v", err)
	}

	// 3. Initialize Runtime
	backend := cpu.NewBackend()
	
	// Create context with Scratch Arena
	ctx := memory.NewInferenceContext(tensor.CPU)
	// Allocate scratch for batched prefill (up to 256 tokens at once)
	maxPrefillTokens := 256
	scratchSize := cfg.ScratchBytes(maxPrefillTokens)
	// Add buffer for logits (VocabSize * 4 bytes) plus attention scores (seqLen^2)
	logitsSize := int64(cfg.VocabSize) * 4
	attnScoresSize := int64(maxPrefillTokens * maxPrefillTokens * 4)
	totalScratch := scratchSize + logitsSize*2 + attnScoresSize
	ctx.AddArena(memory.Scratch, int(totalScratch))
	fmt.Printf("Scratch arena: %d MB (supports up to %d token prefill)\n", totalScratch/(1024*1024), maxPrefillTokens)
	
	rt, err := runtime.NewModelRuntime(backend, ctx, nil, cfg)
	if err != nil {
		log.Fatalf("Failed to create runtime: %v", err)
	}

	// Create paged KV cache
	maxBlocks := 256 // Enough for 4096 tokens with block size 16
	pagedCache := rt.CreatePagedKVCache(maxBlocks)
	fmt.Printf("Paged KV cache: %d blocks available\n", pagedCache.FreeBlocks())

	// 4. Load Weights
	weightsPath := filepath.Join(*modelDir, "tiny_model.safetensors")
	if err := rt.LoadWeights(weightsPath); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}
	fmt.Println("Model loaded.")

	// 5. Start Scheduler
	schedCfg := scheduler.Config{
		MaxBatchSize: 1,
		MaxSequences: 1,
		MaxTokens:    *maxTokens,
		SamplerConfig: sampler.Config{
			Temperature: float32(*temperature),
			TopK:        *topK,
			TopP:        float32(*topP),
		},
	}
	sched, _ := scheduler.NewScheduler(rt, tok, schedCfg)
	fmt.Printf("Sampling: temp=%.2f, top-k=%d, top-p=%.2f, max-tokens=%d\n",
		*temperature, *topK, *topP, *maxTokens)

	// 6. Run Scheduler
	go func() {
		sched.Run(context.Background())
	}()

	// 7. Interactive Loop
	internal.RunChatLoop(os.Stdin, os.Stdout, sched)
}
