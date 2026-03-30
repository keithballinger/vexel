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
	"strings"
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
func initModel(modelPath string, maxTokens, contextLen int, verbose, usePaged bool) (*runtime.ModelRuntime, *tokenizer.Tokenizer, *metal.Backend, error) {
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
	maxContextLen := contextLen
	if maxContextLen <= 0 {
		maxContextLen = 2048
	}
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
	if usePaged {
		blockSize := 16
		maxBlocks := (maxContextLen + blockSize - 1) / blockSize
		model.CreatePagedKVCache(maxBlocks)
		log.Printf("Using paged KV cache (maxBlocks=%d, blockSize=%d, maxContext=%d)", maxBlocks, blockSize, maxContextLen)
	} else {
		model.CreateGPUKVCache(maxContextLen)
	}

	// GPU scratch allocator: bump-allocates layer intermediates from a single MTLBuffer.
	// Currently disabled — the fused decode kernels have offset handling issues that
	// cause incorrect output. The offset-aware variants for MatMulTransposed and
	// SDPAPrefill are in place, but the fused QKV+MLP decode path needs more work.
	// Set VEXEL_SCRATCH=1 to enable for testing.
	if os.Getenv("VEXEL_SCRATCH") == "1" {
		decodeScratchBytes := modelCfg.ScratchBytes(1) + 7*256
		if err := gpuBackend.InitScratch(int(decodeScratchBytes)); err != nil {
			log.Printf("[WARNING] GPU scratch allocator init failed, falling back to pool alloc: %v", err)
		}
	}

	// Load tokenizer from GGUF (preferred — guarantees vocab matches the model).
	// Falls back to tokenizer.json if GGUF tokenizer loading fails.
	tok, err := tokenizer.LoadFromGGUF(modelPath)
	if err != nil {
		if verbose {
			log.Printf("GGUF tokenizer unavailable (%v), falling back to tokenizer.json", err)
		}
		tokPath := filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
		tok, err = tokenizer.Load(tokPath)
		if err != nil {
			gpuBackend.Close()
			return nil, nil, nil, fmt.Errorf("load tokenizer: %w", err)
		}
	}

	return model, tok, gpuBackend, nil
}

// loadDraftModel loads a separate draft model for speculative decoding.
// It shares the same GPU backend as the target model for efficient memory use.
func loadDraftModel(draftPath string, gpuBackend *metal.Backend, maxTokens, contextLen int, verbose bool) (*runtime.ModelRuntime, error) {
	gf, err := gguf.Open(draftPath)
	if err != nil {
		return nil, fmt.Errorf("open draft GGUF: %w", err)
	}
	draftCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	maxContextLen := contextLen
	if maxContextLen <= 0 {
		maxContextLen = 2048
	}
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
	draft.CreateGPUKVCache(maxContextLen)

	return draft, nil
}

// createMedusaScheduler creates a MedusaScheduler from global flags and model.
func createMedusaScheduler(globals GlobalFlags, model *runtime.ModelRuntime, tok *tokenizer.Tokenizer, schedConfig scheduler.Config) (*scheduler.MedusaScheduler, error) {
	medusaCfg := scheduler.DefaultMedusaConfig()
	medusaCfg.EnableOnlineTraining = true
	medusaCfg.UseGPUTraining = true
	if globals.MedusaHeadsPath != "" {
		medusaCfg.HeadsPath = globals.MedusaHeadsPath
	}
	ms, err := scheduler.NewMedusaScheduler(model, tok, schedConfig, medusaCfg)
	if err != nil {
		return nil, fmt.Errorf("create medusa scheduler: %w", err)
	}
	log.Printf("Using Medusa speculative decoding (online training=%v, GPU=%v)",
		medusaCfg.EnableOnlineTraining, medusaCfg.UseGPUTraining)
	return ms, nil
}

// runServe starts the HTTP inference server.
func runServe(globals GlobalFlags, args []string) error {
	sf, err := parseServeFlags(subcommandArgs(args))
	if err != nil {
		return err
	}

	if globals.DraftModel != "" && globals.Medusa {
		return fmt.Errorf("--draft-model and --medusa are mutually exclusive")
	}

	// Draft-model speculation still requires GPU KV cache (VerifySpeculativeWithHidden uses it).
	// Use paged KV when explicitly requested (--context-len) or for Medusa mode.
	// Otherwise use GPU KV cache which is faster for single-client decode.
	usePaged := globals.ContextLen > 0
	model, tok, gpuBackend, err := initModel(globals.Model, sf.MaxTokens, globals.ContextLen, globals.Verbose, usePaged)
	if err != nil {
		return err
	}
	defer gpuBackend.Close()

	maxTokens := sf.MaxTokens
	if globals.Medusa && maxTokens < 700 {
		maxTokens = 700 // Ensure enough tokens for Medusa warmup (600 samples needed)
	}
	schedConfig := scheduler.Config{
		MaxBatchSize:  sf.MaxBatchSize,
		MaxSequences:  64,
		MaxTokens:     maxTokens,
		SamplerConfig: sampler.GreedyConfig(), // Greedy by default; per-request params can override
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var baseSched *scheduler.Scheduler
	var runFunc func(context.Context) error
	var medusaSched *scheduler.MedusaScheduler

	// Use speculative scheduling if a draft model is provided
	if globals.DraftModel != "" {
		draft, err := loadDraftModel(globals.DraftModel, gpuBackend, sf.MaxTokens, globals.ContextLen, globals.Verbose)
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
		var err error
		medusaSched, err = createMedusaScheduler(globals, model, tok, schedConfig)
		if err != nil {
			return err
		}
		baseSched = medusaSched.BaseScheduler()
		runFunc = medusaSched.Run
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

	// Medusa warmup: generate tokens to collect training samples before serving.
	// With WarmupSamples=500, we need ~600 tokens for Cold→Warming transition,
	// then a few seconds for training steps to reach Hot phase.
	if globals.Medusa && globals.MedusaHeadsPath == "" {
		// Only warmup when training heads from scratch — pre-trained heads skip this.
		warmupTarget := 600
		warmupPrompts := []string{
			"Once upon a time in a land far away there lived a brave knight who embarked on a quest",
			"The history of science begins with early civilizations who developed mathematics and astronomy",
			"In a small village by the sea the fishermen would gather every morning before dawn to prepare",
			"The principles of economics are based on the study of how societies allocate scarce resources",
			"A young wizard discovered an ancient spell book hidden in the depths of a forgotten library",
			"The development of modern technology has transformed every aspect of human life from communication",
			"Deep in the forest there existed a magical creature that could speak all languages of the world",
			"The study of philosophy asks fundamental questions about existence knowledge and ethics",
		}
		log.Printf("Warming Medusa heads (%d tokens)...", warmupTarget)
		time.Sleep(100 * time.Millisecond) // Let scheduler goroutine start
		count := 0
		for i, prompt := range warmupPrompts {
			if count >= warmupTarget {
				break
			}
			seqID := scheduler.SequenceID(999990 + i)
			warmupSeq := scheduler.NewSequence(seqID, prompt)
			baseSched.AddSequence(warmupSeq)
			for range warmupSeq.TokenChan() {
				count++
				if count >= warmupTarget {
					break
				}
			}
			baseSched.RemoveSequence(seqID)
		}
		log.Printf("Medusa warmup complete (%d tokens generated). Waiting for heads to train...", count)
		time.Sleep(15 * time.Second) // Allow ~15 training steps at 1s interval

		// Auto-save trained heads for faster startup next time
		if medusaSched != nil {
			headsPath := globals.Model + ".medusa-heads.bin"
			if err := medusaSched.SaveHeads(headsPath); err == nil {
				log.Printf("Saved Medusa heads to %s (use --medusa-heads to skip warmup)", headsPath)
			}
		}

		log.Printf("Medusa heads ready. Starting server.")
	}

	srvCfg := serve.Config{
		RequestTimeout: time.Duration(sf.RequestTimeout) * time.Second,
	}
	srv := serve.NewServerWithConfig(baseSched, srvCfg)

	// Apply chat template for instruct models unless disabled
	if !globals.NoChatTemplate && isInstructModel(globals.Model) {
		template := tokenizer.DetectChatTemplate(globals.Model)
		srv.SetChatTemplate(template)
		log.Printf("Auto-detected instruct model, using %s chat template (use --no-chat-template to disable)", template.Name)
	}

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
	if globals.DraftModel != "" && globals.Medusa {
		return fmt.Errorf("--draft-model and --medusa are mutually exclusive")
	}

	usePaged := globals.ContextLen > 0
	model, tok, gpuBackend, err := initModel(globals.Model, gf.MaxTokens, globals.ContextLen, globals.Verbose, usePaged)
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
	var genMedusaSched *scheduler.MedusaScheduler

	if globals.DraftModel != "" {
		draft, err := loadDraftModel(globals.DraftModel, gpuBackend, gf.MaxTokens, globals.ContextLen, globals.Verbose)
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
		ms, err := createMedusaScheduler(globals, model, tok, schedConfig)
		if err != nil {
			return err
		}
		baseSched = ms.BaseScheduler()
		runFunc = ms.Run
		genMedusaSched = ms
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

	// Apply chat template for instruct models unless disabled
	prompt := gf.Prompt
	if !globals.NoChatTemplate && isInstructModel(globals.Model) {
		template := tokenizer.DetectChatTemplate(globals.Model)
		messages := []tokenizer.ChatMessage{{Role: "user", Content: prompt}}
		prompt = template.FormatConversation(messages)
		if globals.Verbose {
			log.Printf("Applied %s chat template (use --no-chat-template to disable)", template.Name)
		}
	}

	seqID := scheduler.SequenceID(1)
	seq := scheduler.NewSequence(seqID, prompt)
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
		if totalMB, usedMB, freeMB := baseSched.GPUMemoryStats(); totalMB > 0 {
			log.Printf("[gpu memory: %.1f MB used / %.1f MB total (%.1f MB free)]",
				usedMB, totalMB, freeMB)
		}
	}

	// Save Medusa heads after generation (enables offline training via generate)
	if genMedusaSched != nil && globals.MedusaHeadsPath == "" {
		headsPath := globals.Model + ".medusa-heads.bin"
		if err := genMedusaSched.SaveHeads(headsPath); err == nil {
			log.Printf("Saved Medusa heads to %s", headsPath)
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

	samplerCfg := sampler.DefaultConfig()
	samplerCfg.Temperature = float32(cf.Temperature)

	model, tok, gpuBackend, err := initModel(globals.Model, cf.MaxTokens, globals.ContextLen, globals.Verbose, false)
	if err != nil {
		return err
	}
	defer gpuBackend.Close()

	schedConfig := scheduler.Config{
		MaxBatchSize:  1,
		MaxSequences:  1,
		MaxTokens:     cf.MaxTokens,
		SamplerConfig: samplerCfg,
	}

	var sched *scheduler.Scheduler
	var runFunc func(context.Context) error

	if globals.Medusa {
		ms, err := createMedusaScheduler(globals, model, tok, schedConfig)
		if err != nil {
			return err
		}
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
		ModelPath:    globals.Model,
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

// isInstructModel checks if the model filename indicates an instruction-tuned model
// by looking for common suffixes like "instruct", "chat", or "it" (instruction-tuned).
func isInstructModel(modelPath string) bool {
	lower := strings.ToLower(filepath.Base(modelPath))
	// Check for common instruct model indicators in the filename
	for _, keyword := range []string{"instruct", "-chat", "_chat", ".chat", "-it-", "_it_", "-it."} {
		if strings.Contains(lower, keyword) {
			return true
		}
	}
	return false
}

// runTokenize tokenizes input text. Implemented in Phase 3.
func runTokenize(globals GlobalFlags, args []string) error {
	tf, err := parseTokenizeFlags(subcommandArgs(args))
	if err != nil {
		return err
	}

	// Prefer GGUF tokenizer (matches model vocab), fall back to tokenizer.json
	var tok *tokenizer.Tokenizer
	if globals.Model != "" {
		tok, err = tokenizer.LoadFromGGUF(globals.Model)
	}
	if tok == nil {
		tokPath := tf.Tokenizer
		if tokPath == "" && globals.Model != "" {
			tokPath = filepath.Join(filepath.Dir(globals.Model), "tokenizer.json")
		}
		if tokPath == "" {
			return fmt.Errorf("--tokenizer flag is required (or --model to infer)")
		}
		tok, err = tokenizer.Load(tokPath)
		if err != nil {
			return fmt.Errorf("load tokenizer: %w", err)
		}
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
