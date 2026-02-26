//go:build metal && darwin && cgo

// Command medusa_test is an integration test for Medusa speculative decoding.
// It loads a model, runs inference with online training, and reports metrics.
//
// Usage:
//
//	go run -tags metal ./inference/cmd/medusa_test -model path/to/model.gguf
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"vexel/inference/backend/metal"
	"vexel/inference/medusa"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/tensor"
)

var seqCounter int64

// corpusPath is the path to the Wikipedia corpus file
var corpusPath = "data/wikipedia_corpus.txt"

// loadCorpus loads the Wikipedia corpus from file
func loadCorpus() (string, error) {
	// Try relative path first
	data, err := os.ReadFile(corpusPath)
	if err != nil {
		// Try from executable directory
		execPath, _ := os.Executable()
		execDir := strings.TrimSuffix(execPath, "/medusa_test")
		altPath := execDir + "/" + corpusPath
		data, err = os.ReadFile(altPath)
		if err != nil {
			return "", err
		}
	}
	return string(data), nil
}

// WikipediaSample is fallback test text if corpus file not found.
const WikipediaSample = `A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations. Modern digital electronic computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks.

The first computers were used primarily for numerical calculations. However, as any information can be encoded numerically, people soon realized that computers are capable of general-purpose information processing.`

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	maxTokens := flag.Int("max-tokens", 100, "Max tokens to generate per prompt")
	numPrompts := flag.Int("prompts", 5, "Number of prompts to process")
	warmupSamples := flag.Int("warmup", 500, "Samples before training starts")
	verbose := flag.Bool("verbose", false, "Print generated text")
	skipTraining := flag.Bool("skip-training", false, "Skip training, force hot phase (test speculation flow)")
	useGPUTraining := flag.Bool("gpu-training", true, "Use GPU-accelerated Medusa head training")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Usage: medusa_test -model path/to/model.gguf")
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║       Medusa Speculative Decoding Integration Test           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Load model
	fmt.Printf("Loading model: %s\n", *modelPath)
	gf, err := gguf.Open(*modelPath)
	if err != nil {
		log.Fatalf("Failed to open GGUF: %v", err)
	}
	cfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	fmt.Printf("Model: %d layers, %d hidden, %d vocab\n",
		cfg.NumHiddenLayers, cfg.HiddenSize, cfg.VocabSize)

	// Create backend
	gpuBackend, err := createMetalBackend()
	if err != nil {
		log.Fatalf("Failed to create Metal backend: %v", err)
	}
	// Type assert to get device name
	if mb, ok := gpuBackend.(*metal.Backend); ok {
		fmt.Printf("Backend: %s\n", mb.DeviceName())
	} else {
		fmt.Println("Backend: Metal")
	}

	// Create context and runtime
	memCtx := memory.NewInferenceContext(tensor.Metal)
	maxPrefillTokens := 2048 // must cover max prompt length
	totalScratch := cfg.TotalArenaBytes(maxPrefillTokens)
	memCtx.AddArenaWithBackend(memory.Scratch, int(totalScratch), gpuBackend.Alloc)

	rt, err := runtime.NewModelRuntime(gpuBackend, memCtx, nil, cfg)
	if err != nil {
		log.Fatalf("Failed to create runtime: %v", err)
	}

	// Create GPU KV cache
	maxSeqLen := 512
	gpuCache := rt.CreateGPUKVCache(maxSeqLen)
	fmt.Printf("KV Cache: max seq %d, precision=%v\n", maxSeqLen, gpuCache.Precision())

	// Load weights
	fmt.Println("Loading weights...")
	if err := rt.LoadWeights(*modelPath); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}
	if err := rt.CopyWeightsToDevice(); err != nil {
		log.Fatalf("Failed to copy weights to GPU: %v", err)
	}

	// Load tokenizer
	// Try common paths relative to model
	modelDir := strings.TrimSuffix(*modelPath, ".gguf")
	tokPaths := []string{
		modelDir + "/tokenizer.json",
		modelDir + ".tokenizer.json",
		*modelPath + ".tokenizer.json",
	}
	var tok *tokenizer.Tokenizer
	for _, p := range tokPaths {
		tok, _ = tokenizer.Load(p)
		if tok != nil {
			fmt.Printf("Tokenizer: %s\n", p)
			break
		}
	}
	if tok == nil {
		log.Println("Warning: No tokenizer found, using token IDs only")
	}

	// Create Medusa scheduler
	// Use greedy sampling for speculation to work - random sampling causes mismatches
	schedCfg := scheduler.Config{
		MaxBatchSize: 1,
		MaxSequences: 1,
		MaxTokens:    *maxTokens,
		SamplerConfig: sampler.Config{
			Temperature: 0, // Greedy sampling for speculation
			TopK:        0,
			TopP:        0,
		},
	}

	// Training config - very low LR to preserve lm_head initialization
	// When skip-training is set, use learning rate of 0 to disable weight updates
	batchSize := 8
	lr := float32(0.0001)
	if *skipTraining {
		lr = 0 // Disable weight updates to test pure initialization
	}
	medusaCfg := scheduler.MedusaConfig{
		EnableOnlineTraining: true,
		UseGPUTraining:       *useGPUTraining,
		NumHeads:             4,
		TrainingConfig: medusa.OnlineConfig{
			NumHeads:       4,
			BufferCapacity: 10000,
			WarmupSamples:  *warmupSamples,
			MinAccuracy:    0.2,
			BatchSize:      batchSize,
			LearningRate:   lr,
			TrainInterval:  100 * time.Millisecond,
			EvalInterval:   1 * time.Second,
		},
	}

	medusaSched, err := scheduler.NewMedusaScheduler(rt, tok, schedCfg, medusaCfg)
	if err != nil {
		log.Fatalf("Failed to create Medusa scheduler: %v", err)
	}

	// Report training mode
	if *useGPUTraining {
		fmt.Println("Training mode: GPU-accelerated (Metal)")
	} else {
		fmt.Println("Training mode: CPU")
	}

	// Force hot phase if skip-training flag is set
	if *skipTraining {
		fmt.Println("Skip-training mode: forcing Hot phase (random heads, for testing speculation flow)")
		medusaSched.ForceHot()
	}

	// Start scheduler
	runCtx, cancel := context.WithCancel(context.Background())
	go medusaSched.Run(runCtx)

	// Handle signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nInterrupted, stopping...")
		cancel()
	}()

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Starting inference with online training...")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	// Create prompts from Wikipedia
	prompts := createPromptsFromWikipedia(*numPrompts)

	// Track metrics
	startTime := time.Now()
	totalTokens := 0
	phaseChanges := make([]string, 0)
	lastPhase := ""

	// Process prompts
	for i, prompt := range prompts {
		// Check for cancellation
		select {
		case <-runCtx.Done():
			goto done
		default:
		}

		fmt.Printf("\n[Prompt %d/%d] %s...\n", i+1, len(prompts), truncate(prompt, 50))

		// Create sequence with unique ID
		seqID := scheduler.SequenceID(atomic.AddInt64(&seqCounter, 1))
		seq := scheduler.NewSequence(seqID, prompt)
		medusaSched.AddSequence(seq)

		// Wait for completion
		var generated strings.Builder
		tokenCount := 0
		promptStart := time.Now()

		for {
			select {
			case <-runCtx.Done():
				goto done
			case token, ok := <-seq.TokenChan():
				if !ok {
					goto promptDone
				}
				generated.WriteString(token)
				tokenCount++

				// Check phase changes
				metrics := medusaSched.MedusaMetrics()
				if metrics.Phase != lastPhase {
					phaseChange := fmt.Sprintf("%.1fs: %s -> %s",
						time.Since(startTime).Seconds(), lastPhase, metrics.Phase)
					phaseChanges = append(phaseChanges, phaseChange)
					fmt.Printf("  [PHASE CHANGE] %s\n", phaseChange)
					lastPhase = metrics.Phase
				}
			case <-time.After(30 * time.Second):
				fmt.Println("  [TIMEOUT]")
				goto promptDone
			}
		}

	promptDone:
		promptDuration := time.Since(promptStart)
		totalTokens += tokenCount

		if *verbose {
			fmt.Printf("  Generated: %s\n", truncate(generated.String(), 200))
		}
		fmt.Printf("  Tokens: %d, Time: %.2fs, Speed: %.1f tok/s\n",
			tokenCount, promptDuration.Seconds(),
			float64(tokenCount)/promptDuration.Seconds())

		// Print current metrics
		metrics := medusaSched.MedusaMetrics()
		fmt.Printf("  Phase: %s, Samples: %d, Acceptance: %.1f%%, Speedup: %.2fx\n",
			metrics.Phase, metrics.SamplesCollected,
			metrics.AcceptanceRate*100, metrics.EffectiveSpeedup)

		// Remove completed sequence
		medusaSched.RemoveSequence(seq.ID())

		// Reset KV cache for next prompt
		gpuCache.Reset()
	}

done:
	cancel()
	totalDuration := time.Since(startTime)

	// Final report
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("                        FINAL REPORT                           ")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	metrics := medusaSched.MedusaMetrics()

	fmt.Println("Training Metrics:")
	fmt.Printf("  Phase: %s\n", metrics.Phase)
	fmt.Printf("  Samples collected: %d\n", metrics.SamplesCollected)
	fmt.Printf("  Training steps: %d\n", metrics.TrainingSteps)
	fmt.Printf("  Current loss: %.4f\n", metrics.CurrentLoss)
	if len(metrics.HeadAccuracies) > 0 {
		fmt.Printf("  Head accuracies: ")
		for i, acc := range metrics.HeadAccuracies {
			fmt.Printf("H%d=%.1f%% ", i, acc*100)
		}
		fmt.Println()
	}

	fmt.Println()
	fmt.Println("Speculation Metrics:")
	fmt.Printf("  Draft tokens generated: %d\n", metrics.DraftTokensGenerated)
	fmt.Printf("  Draft tokens accepted: %d\n", metrics.DraftTokensAccepted)
	fmt.Printf("  Acceptance rate: %.1f%%\n", metrics.AcceptanceRate*100)
	fmt.Printf("  Effective speedup: %.2fx\n", metrics.EffectiveSpeedup)

	fmt.Println()
	fmt.Println("Overall:")
	fmt.Printf("  Total tokens: %d\n", totalTokens)
	fmt.Printf("  Total time: %.1fs\n", totalDuration.Seconds())
	fmt.Printf("  Average speed: %.1f tok/s\n", float64(totalTokens)/totalDuration.Seconds())

	if len(phaseChanges) > 0 {
		fmt.Println()
		fmt.Println("Phase Transitions:")
		for _, change := range phaseChanges {
			fmt.Printf("  %s\n", change)
		}
	}

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════")
}

func createPromptsFromWikipedia(n int) []string {
	// Try to load the large corpus first
	corpus, err := loadCorpus()
	if err != nil {
		log.Printf("Warning: Could not load corpus file: %v, using fallback", err)
		corpus = WikipediaSample
	} else {
		log.Printf("Loaded corpus: %d characters", len(corpus))
	}

	// Split corpus into paragraphs (double newline separated)
	paragraphs := strings.Split(corpus, "\n\n")

	// Filter to reasonable length paragraphs (50-500 chars)
	goodParagraphs := make([]string, 0, len(paragraphs))
	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		// Skip headers (start with ===) and very short/long paragraphs
		if len(p) >= 50 && len(p) <= 500 && !strings.HasPrefix(p, "===") {
			goodParagraphs = append(goodParagraphs, p)
		}
	}
	log.Printf("Found %d suitable paragraphs for prompts", len(goodParagraphs))

	prompts := make([]string, 0, n)

	// Sample paragraphs evenly distributed across the corpus
	if len(goodParagraphs) > 0 {
		step := len(goodParagraphs) / n
		if step < 1 {
			step = 1
		}
		for i := 0; i < n && i*step < len(goodParagraphs); i++ {
			p := goodParagraphs[i*step]
			// Use first sentence or first 100 chars as prompt
			endIdx := strings.Index(p, ".")
			if endIdx > 20 && endIdx < 150 {
				prompt := fmt.Sprintf("Continue this text: %s.", p[:endIdx])
				prompts = append(prompts, prompt)
			} else if len(p) > 100 {
				prompt := fmt.Sprintf("Continue this text: %s", p[:100])
				prompts = append(prompts, prompt)
			}
		}
	}

	// Add fallback prompts if we need more
	generalPrompts := []string{
		"Explain how computers work in simple terms.",
		"What was Alan Turing's contribution to computing?",
		"Describe the evolution of programming languages.",
		"How has the internet changed computing?",
		"What is artificial intelligence?",
	}

	for i := len(prompts); i < n && i-len(prompts) < len(generalPrompts); i++ {
		prompts = append(prompts, generalPrompts[i-len(prompts)])
	}

	return prompts
}

func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}
