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
	"time"

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

	// Arena must be sized for the maximum batch size in any forward pass.
	// During prefill, batchSize = prompt length, which can be up to maxContextLen.
	// During decode, batchSize = 1 (or small batch for concurrent requests).
	// Use the KV cache context length as the upper bound for prefill.
	maxContextLen := 2048 // matches CreateGPUKVCache below
	maxBatchSize := maxContextLen
	if maxTokens > maxBatchSize {
		maxBatchSize = maxTokens
	}

	memCtx := memory.NewInferenceContext(tensor.Metal)
	totalScratch := modelCfg.TotalArenaBytes(maxBatchSize)
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
	model.CreateGPUKVCache(maxContextLen)

	// Initialize GPU scratch allocator for decode-path bump allocation.
	// Sized for one layer's intermediates at decode (seqLen=1): normOut + Q + K + V + attnOut + gate + up,
	// plus alignment padding (7 allocations × 256 bytes).
	decodeScratchBytes := modelCfg.ScratchBytes(1) + 7*256
	if os.Getenv("VEXEL_NO_SCRATCH") != "1" {
		if err := gpuBackend.InitScratch(int(decodeScratchBytes)); err != nil {
			log.Printf("[WARNING] GPU scratch allocator init failed, falling back to pool alloc: %v", err)
		}
	}

	tokPath := filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		gpuBackend.Close()
		return nil, nil, nil, fmt.Errorf("load tokenizer: %w", err)
	}

	return model, tok, gpuBackend, nil
}

// loadDraftModel loads a separate draft model for speculative decoding.
// It shares the same GPU backend as the target model for efficient memory use.
func loadDraftModel(draftPath string, gpuBackend *metal.Backend, maxTokens int, verbose bool) (*runtime.ModelRuntime, error) {
	gf, err := gguf.Open(draftPath)
	if err != nil {
		return nil, fmt.Errorf("open draft GGUF: %w", err)
	}
	draftCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	maxContextLen := 2048
	maxBatchSize := maxContextLen
	if maxTokens > maxBatchSize {
		maxBatchSize = maxTokens
	}

	memCtx := memory.NewInferenceContext(tensor.Metal)
	totalScratch := draftCfg.TotalArenaBytes(maxBatchSize)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	draft, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, draftCfg)
	if err != nil {
		return nil, fmt.Errorf("create draft runtime: %w", err)
	}

	if verbose {
		log.Printf("Loading draft model from %s...", draftPath)
	}
	if err := draft.LoadWeights(draftPath); err != nil {
		return nil, fmt.Errorf("load draft weights: %w", err)
	}
	if err := draft.CopyWeightsToDevice(); err != nil {
		return nil, fmt.Errorf("copy draft weights: %w", err)
	}
	draft.CreateGPUKVCache(2048)

	return draft, nil
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

	schedConfig := scheduler.Config{
		MaxBatchSize:  sf.MaxBatchSize,
		MaxSequences:  64,
		MaxTokens:     sf.MaxTokens,
		SamplerConfig: sampler.DefaultConfig(),
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var baseSched *scheduler.Scheduler
	var runFunc func(context.Context) error

	// Use speculative scheduling if a draft model is provided
	if globals.DraftModel != "" {
		draft, err := loadDraftModel(globals.DraftModel, gpuBackend, sf.MaxTokens, globals.Verbose)
		if err != nil {
			return fmt.Errorf("load draft model: %w", err)
		}
		specConfig := scheduler.DefaultSpeculativeConfig()
		ss, err := scheduler.NewSpeculativeScheduler(model, draft, tok, schedConfig, specConfig)
		if err != nil {
			return fmt.Errorf("create speculative scheduler: %w", err)
		}
		log.Printf("Using speculative decoding with draft model: %s", globals.DraftModel)
		baseSched = ss.Scheduler
		runFunc = ss.Run
	} else if globals.Medusa {
		medusaCfg := scheduler.DefaultMedusaConfig()
		medusaCfg.EnableOnlineTraining = true
		medusaCfg.UseGPUTraining = true
		if globals.MedusaHeadsPath != "" {
			medusaCfg.HeadsPath = globals.MedusaHeadsPath
		}
		ms, err := scheduler.NewMedusaScheduler(model, tok, schedConfig, medusaCfg)
		if err != nil {
			return fmt.Errorf("create medusa scheduler: %w", err)
		}
		log.Printf("Using Medusa speculative decoding (online training=%v, GPU=%v)",
			medusaCfg.EnableOnlineTraining, medusaCfg.UseGPUTraining)
		baseSched = ms.BaseScheduler()
		runFunc = ms.Run
	} else {
		sched, err := scheduler.NewScheduler(model, tok, schedConfig)
		if err != nil {
			return fmt.Errorf("create scheduler: %w", err)
		}
		baseSched = sched
		runFunc = sched.Run
	}

	go func() {
		if err := runFunc(ctx); err != nil {
			log.Printf("Scheduler stopped: %v", err)
		}
	}()

	srvCfg := serve.Config{
		RequestTimeout: time.Duration(sf.RequestTimeout) * time.Second,
	}
	srv := serve.NewServerWithConfig(baseSched, srvCfg)
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
	log.Printf("  GET  /health    — health check")
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		return fmt.Errorf("server: %w", err)
	}
	return nil
}

// runGenerate runs one-shot text generation, streaming tokens to stdout.
func runGenerate(globals GlobalFlags, args []string) error {
	gf, err := parseGenerateFlags(subcommandArgs(args))
	if err != nil {
		return err
	}
	if gf.Prompt == "" {
		return fmt.Errorf("--prompt is required")
	}

	model, tok, gpuBackend, err := initModel(globals.Model, gf.MaxTokens, globals.Verbose)
	if err != nil {
		return err
	}
	defer gpuBackend.Close()

	samplerCfg := sampler.DefaultConfig()
	samplerCfg.Temperature = float32(gf.Temperature)

	schedConfig := scheduler.Config{
		MaxBatchSize:  1,
		MaxSequences:  1,
		MaxTokens:     gf.MaxTokens,
		SamplerConfig: samplerCfg,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var baseSched *scheduler.Scheduler
	var runFunc func(context.Context) error
	var specSched *scheduler.SpeculativeScheduler

	if globals.DraftModel != "" {
		draft, err := loadDraftModel(globals.DraftModel, gpuBackend, gf.MaxTokens, globals.Verbose)
		if err != nil {
			return fmt.Errorf("load draft model: %w", err)
		}
		specConfig := scheduler.DefaultSpeculativeConfig()
		ss, err := scheduler.NewSpeculativeScheduler(model, draft, tok, schedConfig, specConfig)
		if err != nil {
			return fmt.Errorf("create speculative scheduler: %w", err)
		}
		log.Printf("Using speculative decoding with draft model: %s", globals.DraftModel)
		baseSched = ss.Scheduler
		runFunc = ss.Run
		specSched = ss
	} else if globals.Medusa {
		medusaCfg := scheduler.DefaultMedusaConfig()
		medusaCfg.EnableOnlineTraining = true
		medusaCfg.UseGPUTraining = true
		if globals.MedusaHeadsPath != "" {
			medusaCfg.HeadsPath = globals.MedusaHeadsPath
		}
		ms, err := scheduler.NewMedusaScheduler(model, tok, schedConfig, medusaCfg)
		if err != nil {
			return fmt.Errorf("create medusa scheduler: %w", err)
		}
		log.Printf("Using Medusa speculative decoding (online training=%v, GPU=%v)",
			medusaCfg.EnableOnlineTraining, medusaCfg.UseGPUTraining)
		baseSched = ms.BaseScheduler()
		runFunc = ms.Run
	} else {
		sched, err := scheduler.NewScheduler(model, tok, schedConfig)
		if err != nil {
			return fmt.Errorf("create scheduler: %w", err)
		}
		baseSched = sched
		runFunc = sched.Run
	}

	go func() {
		if err := runFunc(ctx); err != nil && err != context.Canceled {
			log.Printf("Scheduler stopped: %v", err)
		}
	}()

	seqID := scheduler.SequenceID(1)
	seq := scheduler.NewSequence(seqID, gf.Prompt)
	baseSched.AddSequence(seq)

	tokenCount := 0
	for token := range seq.TokenChan() {
		fmt.Print(token)
		tokenCount++
	}
	fmt.Println()

	if globals.Verbose {
		m := baseSched.Metrics()
		log.Printf("[%d tokens | prefill: %.1f tok/s | decode: %.1f tok/s]",
			tokenCount, m.PrefillTokensPerSecond(), m.TokensPerSecond())
		if specSched != nil {
			sm := specSched.SpecMetrics()
			log.Printf("[speculative: acceptance=%.1f%% speedup=%.1fx generated=%d accepted=%d]",
				sm.AcceptanceRate()*100, sm.Speedup(), sm.DraftTokensGenerated, sm.DraftTokensAccepted)
		}
	}

	cancel()
	return nil
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

	schedConfig := scheduler.Config{
		MaxBatchSize:  1,
		MaxSequences:  1,
		MaxTokens:     256,
		SamplerConfig: sampler.DefaultConfig(),
	}

	var sched *scheduler.Scheduler
	var runFunc func(context.Context) error

	if globals.Medusa {
		medusaCfg := scheduler.DefaultMedusaConfig()
		medusaCfg.EnableOnlineTraining = true
		medusaCfg.UseGPUTraining = true
		if globals.MedusaHeadsPath != "" {
			medusaCfg.HeadsPath = globals.MedusaHeadsPath
		}
		ms, err := scheduler.NewMedusaScheduler(model, tok, schedConfig, medusaCfg)
		if err != nil {
			return fmt.Errorf("create medusa scheduler: %w", err)
		}
		log.Printf("Using Medusa speculative decoding (online training=%v, GPU=%v)",
			medusaCfg.EnableOnlineTraining, medusaCfg.UseGPUTraining)
		sched = ms.BaseScheduler()
		runFunc = ms.Run
	} else {
		s, err := scheduler.NewScheduler(model, tok, schedConfig)
		if err != nil {
			return fmt.Errorf("create scheduler: %w", err)
		}
		sched = s
		runFunc = s.Run
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := runFunc(ctx); err != nil && err != context.Canceled {
			log.Printf("Scheduler stopped: %v", err)
		}
	}()

	config := internal.REPLConfig{
		ChatMode:     !cf.NoChatTemplate,
		SystemPrompt: cf.SystemPrompt,
	}
	return internal.RunChatLoopWithConfig(os.Stdin, os.Stdout, sched, config)
}

// runBench runs scheduling benchmarks.
func runBench(globals GlobalFlags, args []string) error {
	bf, err := parseBenchFlags(subcommandArgs(args))
	if err != nil {
		return err
	}

	cfg := runtime.Llama3_8B()
	plan := cfg.MemoryPlan(bf.BatchSize, bf.SeqLen, tensor.Q4_0)
	fmt.Printf("Estimated Memory Usage:\n  Weights: %.2f GB\n  KV Cache: %.2f GB\n  Scratch: %.2f GB\n  Total: %.2f GB\n",
		toGB(plan.Weights), toGB(plan.KV), toGB(plan.Scratch), toGB(plan.Total))

	rt := &runtime.ModelRuntime{}
	sched, err := scheduler.NewScheduler(rt, nil, scheduler.Config{
		MaxBatchSize: bf.BatchSize,
		MaxSequences: bf.NumSeqs,
	})
	if err != nil {
		return fmt.Errorf("create scheduler: %w", err)
	}

	start := time.Now()
	for i := 0; i < bf.NumSeqs; i++ {
		sched.AddSequence(scheduler.NewSequence(scheduler.SequenceID(i), "Benchmark Prompt"))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	log.Println("Running benchmark...")
	if err := sched.Run(ctx); err != nil {
		log.Printf("Scheduler finished with: %v", err)
	}

	duration := time.Since(start)
	metrics := sched.Metrics()
	throughput := float64(metrics.TotalTokens) / duration.Seconds()

	fmt.Printf("\nResults:\n")
	fmt.Printf("  Duration: %v\n", duration)
	fmt.Printf("  Total Tokens: %d\n", metrics.TotalTokens)
	fmt.Printf("  Throughput: %.2f tokens/sec\n", throughput)

	return nil
}

func toGB(bytes int64) float64 {
	return float64(bytes) / 1024 / 1024 / 1024
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
