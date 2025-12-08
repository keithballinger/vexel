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
	"vexel/inference/kv"
	"vexel/inference/memory"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/tensor"
)

func main() {
	modelDir := flag.String("model", "models", "Directory containing model files")
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
	scratchSize := cfg.ScratchBytes(1) // Batch size 1
	// Add buffer for logits (VocabSize * 4 bytes) plus overhead
	logitsSize := int64(cfg.VocabSize) * 4
	totalScratch := scratchSize + logitsSize*2
	ctx.AddArena(memory.Scratch, int(totalScratch))
	
	cache := &kv.KVCache{}
	
	rt, err := runtime.NewModelRuntime(backend, ctx, cache, cfg)
	if err != nil {
		log.Fatalf("Failed to create runtime: %v", err)
	}

	// 4. Load Weights
	weightsPath := filepath.Join(*modelDir, "tiny_model.safetensors")
	if err := rt.LoadWeights(weightsPath); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}
	fmt.Println("Model loaded.")

	// 5. Start Scheduler
	schedCfg := scheduler.Config{MaxBatchSize: 1, MaxSequences: 1}
	sched, _ := scheduler.NewScheduler(rt, tok, schedCfg)

	// 6. Run Scheduler
	go func() {
		sched.Run(context.Background())
	}()

	// 7. Interactive Loop
	internal.RunChatLoop(os.Stdin, os.Stdout, sched)
}
