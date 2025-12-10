package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"vexel/inference/cmd/vexel/internal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/tensor"
)

// Use the new unified Backend interface via CPUBackend

func main() {
	modelPath := flag.String("model", "", "Path to model file (.gguf or .safetensors) or directory")
	temperature := flag.Float64("temp", 0.7, "Sampling temperature (0 = greedy)")
	topK := flag.Int("top-k", 40, "Top-K sampling (0 = disabled)")
	topP := flag.Float64("top-p", 0.9, "Top-P nucleus sampling (0 = disabled)")
	maxTokens := flag.Int("max-tokens", 256, "Maximum tokens to generate per response")
	completionMode := flag.Bool("completion", false, "Use completion mode (no chat template)")
	useGPU := flag.Bool("gpu", false, "Use GPU acceleration (Metal on macOS, requires -tags metal)")
	flag.Parse()

	fmt.Println("Vexel Inference Engine - Interactive Mode")
	fmt.Println("Loading model...")

	// Determine model file and type
	weightsPath, isGGUF, err := resolveModelPath(*modelPath)
	if err != nil {
		log.Fatalf("Failed to find model: %v", err)
	}
	fmt.Printf("Model file: %s\n", weightsPath)

	// Load config - either from GGUF metadata or use defaults
	var cfg runtime.ModelConfig
	var tok *tokenizer.Tokenizer

	if isGGUF {
		// Load config from GGUF file
		gf, err := gguf.Open(weightsPath)
		if err != nil {
			log.Fatalf("Failed to open GGUF file: %v", err)
		}
		gf.PrintSummary()
		cfg = runtime.ModelConfigFromGGUF(gf.GetModelConfig())
		gf.Close()

		fmt.Printf("Config loaded from GGUF: %d layers, %d hidden, %d heads\n",
			cfg.NumHiddenLayers, cfg.HiddenSize, cfg.NumAttentionHeads)
	} else {
		// Default config for TinyLlama (SafeTensors mode)
		cfg = runtime.ModelConfig{
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
	}

	// Load tokenizer (look for tokenizer.json in same directory as model)
	modelDir := filepath.Dir(weightsPath)
	tokPaths := []string{
		filepath.Join(modelDir, "tokenizer.json"),
		filepath.Join(modelDir, "tiny_tokenizer.json"),
	}
	for _, tokPath := range tokPaths {
		tok, err = tokenizer.Load(tokPath)
		if err == nil {
			fmt.Printf("Tokenizer loaded from: %s\n", tokPath)
			break
		}
	}
	if tok == nil {
		log.Printf("Warning: No tokenizer found in %s", modelDir)
	}

	// Initialize backend and context
	backend, loc, err := createBackend(*useGPU)
	if err != nil {
		log.Fatalf("Failed to create backend: %v", err)
	}
	if *useGPU && backend == nil {
		log.Println("GPU not available (build with -tags metal). Using CPU backend.")
		backend, loc, _ = createBackend(false)
	}
	if *useGPU && gpuAvailable() {
		fmt.Println("Using GPU (Metal) backend")
	} else {
		fmt.Println("Using CPU backend")
	}
	ctx := memory.NewInferenceContext(loc)

	// Allocate scratch arena for batched prefill
	maxPrefillTokens := 256
	scratchSize := cfg.ScratchBytes(maxPrefillTokens)
	logitsSize := int64(cfg.VocabSize) * 4
	attnScoresSize := int64(maxPrefillTokens * maxPrefillTokens * 4)
	totalScratch := scratchSize + logitsSize*2 + attnScoresSize

	// Use backend-aware allocation for GPU
	if *useGPU && gpuAvailable() {
		ctx.AddArenaWithBackend(memory.Scratch, int(totalScratch), backend.Alloc)
	} else {
		ctx.AddArena(memory.Scratch, int(totalScratch))
	}
	fmt.Printf("Scratch arena: %d MB\n", totalScratch/(1024*1024))

	// Create runtime
	rt, err := runtime.NewModelRuntime(backend, ctx, nil, cfg)
	if err != nil {
		log.Fatalf("Failed to create runtime: %v", err)
	}

	// Create KV cache - GPU resident for GPU backend, paged for CPU
	if *useGPU && gpuAvailable() {
		maxSeqLen := 512 // Max sequence length for GPU KV cache
		gpuCache := rt.CreateGPUKVCache(maxSeqLen)
		fmt.Printf("GPU KV cache: max seq len %d\n", maxSeqLen)
		_ = gpuCache // Stored in runtime
	} else {
		maxBlocks := 256
		pagedCache := rt.CreatePagedKVCache(maxBlocks)
		fmt.Printf("Paged KV cache: %d blocks\n", pagedCache.FreeBlocks())
	}

	// Load weights (auto-detects format from extension)
	if err := rt.LoadWeights(weightsPath); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	// Copy weights to GPU if using GPU backend
	if *useGPU && gpuAvailable() {
		fmt.Println("Copying weights to GPU...")
		if err := rt.CopyWeightsToDevice(); err != nil {
			log.Fatalf("Failed to copy weights to GPU: %v", err)
		}
	}
	fmt.Println("Model loaded successfully.")

	// Create scheduler
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

	// Start scheduler
	go func() {
		sched.Run(context.Background())
	}()

	// Run interactive loop
	replConfig := internal.DefaultREPLConfig()
	if *completionMode {
		replConfig.ChatMode = false
	}
	internal.RunChatLoopWithConfig(os.Stdin, os.Stdout, sched, replConfig)
}

// resolveModelPath finds the model file from the given path.
// Returns the full path, whether it's GGUF format, and any error.
func resolveModelPath(path string) (string, bool, error) {
	if path == "" {
		path = "models"
	}

	// Check if path is a file
	info, err := os.Stat(path)
	if err == nil && !info.IsDir() {
		isGGUF := strings.HasSuffix(strings.ToLower(path), ".gguf")
		return path, isGGUF, nil
	}

	// Path is a directory - look for model files
	dir := path

	// Try GGUF files first (preferred)
	ggufFiles, _ := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if len(ggufFiles) > 0 {
		return ggufFiles[0], true, nil
	}

	// Try SafeTensors files
	stFiles, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if len(stFiles) > 0 {
		return stFiles[0], false, nil
	}

	// Try specific known filenames
	knownNames := []string{
		"tiny_model.safetensors",
		"model.safetensors",
		"pytorch_model.safetensors",
	}
	for _, name := range knownNames {
		p := filepath.Join(dir, name)
		if _, err := os.Stat(p); err == nil {
			return p, false, nil
		}
	}

	return "", false, fmt.Errorf("no model file found in %s", dir)
}
