//go:build metal && darwin && cgo

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"vexel/inference/backend/metal"
	"vexel/inference/cmd/vexel/internal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
	"vexel/inference/tensor"
)

// initModel loads the model from a GGUF file and returns the runtime, tokenizer, and backend.
// This is shared setup used by serve, generate, and chat subcommands.
func initModel(modelPath string, maxTokens int, verbose bool) (*runtime.ModelRuntime, *tokenizer.Tokenizer, *metal.Backend, error) {
	if modelPath == "" {
		return nil, nil, nil, fmt.Errorf("--model flag is required")
	}

	gpuBackend, err := metal.NewBackend(0)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("metal backend: %w", err)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("open GGUF: %w", err)
	}
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	memCtx := memory.NewInferenceContext(tensor.Metal)
	scratchSize := modelCfg.ScratchBytes(maxTokens)
	logitsSize := int64(modelCfg.VocabSize) * 4
	attnSize := int64(maxTokens * maxTokens * 4)
	totalScratch := scratchSize + logitsSize*2 + attnSize
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	model, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, modelCfg)
	if err != nil {
		gpuBackend.Close()
		return nil, nil, nil, fmt.Errorf("create runtime: %w", err)
	}

	if verbose {
		log.Printf("Loading weights from %s...", modelPath)
	}
	if err := model.LoadWeights(modelPath); err != nil {
		gpuBackend.Close()
		return nil, nil, nil, fmt.Errorf("load weights: %w", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		gpuBackend.Close()
		return nil, nil, nil, fmt.Errorf("copy weights: %w", err)
	}
	model.CreateGPUKVCache(2048)

	tokPath := filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		gpuBackend.Close()
		return nil, nil, nil, fmt.Errorf("load tokenizer: %w", err)
	}

	return model, tok, gpuBackend, nil
}

// runServe starts the HTTP inference server.
func runServe(globals GlobalFlags, args []string) error {
	sf, err := parseServeFlags(subcommandArgs(args))
	if err != nil {
		return err
	}

	model, tok, gpuBackend, err := initModel(globals.Model, sf.MaxTokens, globals.Verbose)
	if err != nil {
		return err
	}
	defer gpuBackend.Close()

	sched, err := scheduler.NewScheduler(model, tok, scheduler.Config{
		MaxBatchSize:  sf.MaxBatchSize,
		MaxSequences:  64,
		MaxTokens:     sf.MaxTokens,
		SamplerConfig: sampler.DefaultConfig(),
	})
	if err != nil {
		return fmt.Errorf("create scheduler: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := sched.Run(ctx); err != nil {
			log.Printf("Scheduler stopped: %v", err)
		}
	}()

	srv := serve.NewServer(sched)
	addr := fmt.Sprintf(":%d", sf.Port)
	httpServer := &http.Server{Addr: addr, Handler: srv}

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
		return fmt.Errorf("server: %w", err)
	}
	return nil
}

// runGenerate runs one-shot text generation. Implemented in Phase 2.
func runGenerate(globals GlobalFlags, args []string) error {
	_, err := parseGenerateFlags(subcommandArgs(args))
	if err != nil {
		return err
	}
	return fmt.Errorf("generate subcommand not yet implemented (Phase 2)")
}

// runChat starts the interactive chat REPL. Implemented in Phase 2.
func runChat(globals GlobalFlags, args []string) error {
	cf, err := parseChatFlags(subcommandArgs(args))
	if err != nil {
		return err
	}

	model, tok, gpuBackend, err := initModel(globals.Model, 256, globals.Verbose)
	if err != nil {
		return err
	}
	defer gpuBackend.Close()

	sched, err := scheduler.NewScheduler(model, tok, scheduler.Config{
		MaxBatchSize:  1,
		MaxSequences:  1,
		MaxTokens:     256,
		SamplerConfig: sampler.DefaultConfig(),
	})
	if err != nil {
		return fmt.Errorf("create scheduler: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := sched.Run(ctx); err != nil && err != context.Canceled {
			log.Printf("Scheduler stopped: %v", err)
		}
	}()

	config := internal.REPLConfig{
		ChatMode:     !cf.NoChatTemplate,
		SystemPrompt: cf.SystemPrompt,
	}
	return internal.RunChatLoopWithConfig(os.Stdin, os.Stdout, sched, config)
}

// runBench runs scheduling benchmarks. Implemented in Phase 3.
func runBench(globals GlobalFlags, args []string) error {
	_, err := parseBenchFlags(subcommandArgs(args))
	if err != nil {
		return err
	}
	return fmt.Errorf("bench subcommand not yet implemented (Phase 3)")
}

// runTokenize tokenizes input text. Implemented in Phase 3.
func runTokenize(globals GlobalFlags, args []string) error {
	tf, err := parseTokenizeFlags(subcommandArgs(args))
	if err != nil {
		return err
	}

	tokPath := tf.Tokenizer
	if tokPath == "" && globals.Model != "" {
		tokPath = filepath.Join(filepath.Dir(globals.Model), "tokenizer.json")
	}
	if tokPath == "" {
		return fmt.Errorf("--tokenizer flag is required (or --model to infer)")
	}

	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}

	input := tf.Input
	if input == "" {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return fmt.Errorf("read stdin: %w", err)
		}
		input = string(b)
	}

	ids, err := tok.Encode(input)
	if err != nil {
		return fmt.Errorf("encode: %w", err)
	}

	return json.NewEncoder(os.Stdout).Encode(ids)
}
