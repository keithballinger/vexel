//go:build metal && darwin && cgo

// This example benchmarks Vexel inference performance by measuring prefill and
// decode throughput in tokens per second. It uses the runtime directly (no
// scheduler) for precise timing measurements.
//
// Optionally, pass --compare to run the same workload with both GPU KV cache
// and paged KV cache, printing a side-by-side comparison.
//
// Prerequisites:
//   - macOS with Apple Silicon (M1/M2/M3/M4)
//   - A GGUF model file (e.g., tinyllama-1.1b-chat-v1.0.Q4_0.gguf)
//
// Usage:
//
//	go run -tags metal ./examples/benchmark \
//	  -model path/to/model.gguf \
//	  -tokens 128 \
//	  -prompt "The quick brown fox jumps over the lazy dog"
//
//	# Compare GPU KV vs paged KV:
//	go run -tags metal ./examples/benchmark \
//	  -model path/to/model.gguf \
//	  -compare
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

// benchResult holds timing measurements for a single benchmark run.
type benchResult struct {
	mode           string
	promptTokens   int
	generatedTokens int
	prefillDur     time.Duration
	decodeDur      time.Duration
}

func (r benchResult) prefillTokPerSec() float64 {
	return float64(r.promptTokens) / r.prefillDur.Seconds()
}

func (r benchResult) decodeTokPerSec() float64 {
	return float64(r.generatedTokens) / r.decodeDur.Seconds()
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (required)")
	tokenizerPath := flag.String("tokenizer", "", "Path to tokenizer.json (default: same dir as model)")
	prompt := flag.String("prompt", "The quick brown fox jumps over the lazy dog", "Prompt for benchmarking")
	numTokens := flag.Int("tokens", 128, "Number of tokens to generate")
	contextLen := flag.Int("context-len", 2048, "Max context length for KV cache")
	compare := flag.Bool("compare", false, "Compare GPU KV cache vs paged KV cache")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: benchmark -model path/to/model.gguf [-tokens 128] [-prompt \"...\"] [-compare]")
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
		log.Fatalf("Failed to open GGUF: %v", err)
	}
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	// --- 3. Allocate GPU memory ---
	// Size the scratch arena to accommodate both prefill and decode.
	memCtx := memory.NewInferenceContext(tensor.Metal)
	maxPrefillTokens := 512
	scratchSize := modelCfg.ScratchBytes(maxPrefillTokens)
	logitsSize := int64(modelCfg.VocabSize) * 4
	attnSize := int64(maxPrefillTokens * maxPrefillTokens * 4)
	totalScratch := scratchSize + logitsSize*2 + attnSize
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

	// --- 5. Load tokenizer ---
	tokPath := *tokenizerPath
	if tokPath == "" {
		tokPath = filepath.Join(filepath.Dir(*modelPath), "tokenizer.json")
	}
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// --- 6. Tokenize prompt ---
	tokens, err := tok.Encode(*prompt)
	if err != nil {
		log.Fatalf("Failed to encode prompt: %v", err)
	}
	if tok.AddBOS() {
		tokens = append([]int{tok.BOS()}, tokens...)
	}

	log.Printf("Model: %s", filepath.Base(*modelPath))
	log.Printf("Prompt: %q (%d tokens)", *prompt, len(tokens))
	log.Printf("Decode tokens: %d", *numTokens)
	fmt.Println()

	// --- 7. Run benchmark with GPU KV cache ---
	gpuResult := benchGPUKV(model, gpuBackend, tokens, *numTokens, *contextLen)
	printResult(gpuResult)

	// --- 8. Optionally compare with paged KV cache ---
	if *compare {
		fmt.Println()
		pagedResult := benchPagedKV(model, gpuBackend, tokens, *numTokens, *contextLen)
		printResult(pagedResult)

		// Print comparison summary
		fmt.Println()
		printComparison(gpuResult, pagedResult)
	}
}

// benchGPUKV runs the benchmark using GPU-resident KV cache.
func benchGPUKV(model *runtime.ModelRuntime, be *metal.Backend, tokens []int, numTokens, contextLen int) benchResult {
	// Create a fresh GPU KV cache for this run
	cache := model.CreateGPUKVCache(contextLen)
	defer cache.Free()

	// Warmup: run one prefill to warm up GPU pipelines, then reset
	_, _ = model.DecodeWithGPUKV(tokens, 0)
	cache.Reset()

	// Prefill: process all prompt tokens at once
	prefillStart := time.Now()
	logits, err := model.DecodeWithGPUKV(tokens, 0)
	if err != nil {
		log.Fatalf("GPU KV prefill failed: %v", err)
	}
	be.Sync()
	prefillDur := time.Since(prefillStart)

	// Decode: generate tokens one at a time
	pos := len(tokens)
	nextToken := argmax(logits, be)
	generated := 0

	decodeStart := time.Now()
	for i := 0; i < numTokens; i++ {
		logits, err = model.DecodeWithGPUKV([]int{nextToken}, pos)
		if err != nil {
			log.Fatalf("GPU KV decode failed at position %d: %v", pos, err)
		}
		nextToken = argmax(logits, be)
		pos++
		generated++
	}
	be.Sync()
	decodeDur := time.Since(decodeStart)

	return benchResult{
		mode:            "GPU KV",
		promptTokens:    len(tokens),
		generatedTokens: generated,
		prefillDur:      prefillDur,
		decodeDur:       decodeDur,
	}
}

// benchPagedKV runs the benchmark using paged KV cache.
func benchPagedKV(model *runtime.ModelRuntime, be *metal.Backend, tokens []int, numTokens, contextLen int) benchResult {
	// Create paged KV cache; use enough blocks for the context length
	blockSize := 16
	maxBlocks := (contextLen + blockSize - 1) / blockSize
	pagedCache := model.CreatePagedKVCache(maxBlocks)

	// Warmup: run one prefill, then delete the sequence and create a new one
	warmupSeqID := pagedCache.CreateSequence()
	_, _ = model.PrefillWithPagedKV(tokens, warmupSeqID, 0)
	pagedCache.DeleteSequence(warmupSeqID)

	seqID := pagedCache.CreateSequence()

	// Prefill: process all prompt tokens at once
	prefillStart := time.Now()
	logits, err := model.PrefillWithPagedKV(tokens, seqID, 0)
	if err != nil {
		log.Fatalf("Paged KV prefill failed: %v", err)
	}
	be.Sync()
	prefillDur := time.Since(prefillStart)

	// Decode: generate tokens one at a time
	pos := len(tokens)
	nextToken := argmax(logits, be)
	generated := 0

	decodeStart := time.Now()
	for i := 0; i < numTokens; i++ {
		logits, err = model.DecodeWithPagedKV([]int{nextToken}, seqID, pos)
		if err != nil {
			log.Fatalf("Paged KV decode failed at position %d: %v", pos, err)
		}
		nextToken = argmax(logits, be)
		pos++
		generated++
	}
	be.Sync()
	decodeDur := time.Since(decodeStart)

	// Clean up paged cache state
	pagedCache.DeleteSequence(seqID)

	return benchResult{
		mode:            "Paged KV",
		promptTokens:    len(tokens),
		generatedTokens: generated,
		prefillDur:      prefillDur,
		decodeDur:       decodeDur,
	}
}

// printResult displays benchmark results for a single run.
func printResult(r benchResult) {
	fmt.Printf("=== %s ===\n", r.mode)
	fmt.Printf("  Prefill:  %4d tokens in %10s  (%7.1f tok/s)\n",
		r.promptTokens, r.prefillDur.Round(time.Microsecond), r.prefillTokPerSec())
	fmt.Printf("  Decode:   %4d tokens in %10s  (%7.1f tok/s)\n",
		r.generatedTokens, r.decodeDur.Round(time.Microsecond), r.decodeTokPerSec())
}

// printComparison prints a side-by-side comparison of two benchmark results.
func printComparison(a, b benchResult) {
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%-20s %12s %12s %10s\n", "", a.mode, b.mode, "Ratio")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%-20s %9.1f t/s %9.1f t/s %9.2fx\n",
		"Prefill throughput",
		a.prefillTokPerSec(), b.prefillTokPerSec(),
		a.prefillTokPerSec()/b.prefillTokPerSec())
	fmt.Printf("%-20s %9.1f t/s %9.1f t/s %9.2fx\n",
		"Decode throughput",
		a.decodeTokPerSec(), b.decodeTokPerSec(),
		a.decodeTokPerSec()/b.decodeTokPerSec())
	fmt.Println(strings.Repeat("-", 60))
}

// argmax reads logits from GPU memory and returns the index of the largest value.
func argmax(logits tensor.Tensor, backend *metal.Backend) int {
	backend.Sync()

	numElements := logits.Shape().NumElements()
	data := make([]byte, numElements*4)
	backend.ToHost(data, logits.DevicePtr())

	maxIdx := 0
	maxVal := float32(-math.MaxFloat32)
	for i := 0; i < numElements; i++ {
		v := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
