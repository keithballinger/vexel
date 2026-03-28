//go:build metal && darwin && cgo

// This example demonstrates embedding the Vexel inference server into your
// own Go application. It initializes the Metal backend, loads a GGUF model,
// sets up the scheduler for continuous batching, and starts an HTTP server
// exposing /generate (blocking) and /stream (SSE) endpoints.
//
// Prerequisites:
//   - macOS with Apple Silicon (M1/M2/M3/M4)
//   - A GGUF model file (e.g., tinyllama-1.1b-chat-v1.0.Q4_0.gguf)
//
// Usage:
//
//	go run -tags metal ./examples/server -model path/to/model.gguf
//	curl -X POST http://localhost:8080/generate \
//	  -H "Content-Type: application/json" \
//	  -d '{"prompt": "Hello, world!"}'
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
	"vexel/inference/tensor"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (required)")
	tokenizerPath := flag.String("tokenizer", "", "Path to tokenizer.json (default: same dir as model)")
	port := flag.Int("port", 8080, "HTTP port to listen on")
	maxTokens := flag.Int("max-tokens", 256, "Max tokens to generate per request")
	contextLen := flag.Int("context-len", 2048, "Max context length for KV cache")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: server -model path/to/model.gguf [-port 8080]")
		os.Exit(1)
	}

	// --- 1. Initialize Metal backend ---
	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		log.Fatalf("Metal backend init failed: %v", err)
	}
	defer gpuBackend.Close()

	// --- 2. Load model configuration from GGUF ---
	gf, err := gguf.Open(*modelPath)
	if err != nil {
		log.Fatalf("Failed to open GGUF file: %v", err)
	}
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	// --- 3. Allocate GPU memory ---
	memCtx := memory.NewInferenceContext(tensor.Metal)
	scratchSize := modelCfg.ScratchBytes(*maxTokens)
	logitsSize := int64(modelCfg.VocabSize) * 4
	totalScratch := scratchSize + logitsSize*2 + int64(*maxTokens**maxTokens*4)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	// --- 4. Create model runtime and load weights ---
	model, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, modelCfg)
	if err != nil {
		log.Fatalf("Failed to create model runtime: %v", err)
	}

	log.Printf("Loading weights from %s...", *modelPath)
	if err := model.LoadWeights(*modelPath); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		log.Fatalf("Failed to copy weights to GPU: %v", err)
	}
	model.CreateGPUKVCache(*contextLen)

	// --- 5. Load tokenizer ---
	tokPath := *tokenizerPath
	if tokPath == "" {
		tokPath = filepath.Join(filepath.Dir(*modelPath), "tokenizer.json")
	}
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// --- 6. Create scheduler ---
	sched, err := scheduler.NewScheduler(model, tok, scheduler.Config{
		MaxBatchSize:  1,
		MaxSequences:  64,
		MaxTokens:     *maxTokens,
		SamplerConfig: sampler.DefaultConfig(),
	})
	if err != nil {
		log.Fatalf("Failed to create scheduler: %v", err)
	}

	// Start scheduler in background
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := sched.Run(ctx); err != nil {
			log.Printf("Scheduler stopped: %v", err)
		}
	}()

	// --- 7. Start HTTP server ---
	srv := serve.NewServer(sched)
	addr := fmt.Sprintf(":%d", *port)
	httpServer := &http.Server{Addr: addr, Handler: srv}

	// Graceful shutdown on SIGINT/SIGTERM
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("Shutting down...")
		cancel()
		httpServer.Close()
	}()

	log.Printf("Vexel server listening on http://localhost%s", addr)
	log.Printf("  POST /generate  — blocking text generation")
	log.Printf("  POST /stream    — SSE token streaming")
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}
