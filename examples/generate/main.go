//go:build metal && darwin && cgo

// This example demonstrates direct text generation using the Vexel runtime,
// without the HTTP server layer. It loads a GGUF model, runs prefill on a
// prompt, and generates tokens one at a time using greedy sampling.
//
// This is the lowest-level API — useful for benchmarking, embedding inference
// in custom pipelines, or understanding how the inference loop works.
//
// Prerequisites:
//   - macOS with Apple Silicon (M1/M2/M3/M4)
//   - A GGUF model file (e.g., tinyllama-1.1b-chat-v1.0.Q4_0.gguf)
//
// Usage:
//
//	go run -tags metal ./examples/generate \
//	  -model path/to/model.gguf \
//	  -prompt "The meaning of life is"
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"vexel/inference/backend/metal"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (required)")
	tokenizerPath := flag.String("tokenizer", "", "Path to tokenizer.json (default: same dir as model)")
	prompt := flag.String("prompt", "Hello! How are you?", "Input prompt")
	maxTokens := flag.Int("max-tokens", 64, "Maximum tokens to generate")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: generate -model path/to/model.gguf [-prompt \"...\"]")
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
	memCtx := memory.NewInferenceContext(tensor.Metal)
	maxPrefillTokens := 256
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

	// Create GPU KV cache (max sequence length 2048)
	cache := model.CreateGPUKVCache(2048)
	defer cache.Free()

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

	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Tokens: %v (%d tokens)\n\n", tokens, len(tokens))

	// --- 7. Prefill: process all prompt tokens at once ---
	prefillStart := time.Now()
	logits, err := model.DecodeWithGPUKV(tokens, 0)
	if err != nil {
		log.Fatalf("Prefill failed: %v", err)
	}
	prefillDur := time.Since(prefillStart)
	fmt.Printf("[Prefill: %d tokens in %v (%.1f tok/s)]\n",
		len(tokens), prefillDur, float64(len(tokens))/prefillDur.Seconds())

	// --- 8. Decode loop: generate tokens one at a time ---
	fmt.Print("\nGenerated: ")
	nextToken := argmax(logits, gpuBackend)
	pos := len(tokens)

	decodeStart := time.Now()
	generated := []int{nextToken}

	for i := 0; i < *maxTokens-1; i++ {
		// Check for end of sequence
		if nextToken == tok.EOS() {
			break
		}

		// Print token as it's generated (streaming)
		decoded, _ := tok.Decode([]int{nextToken})
		fmt.Print(decoded)

		// Decode next token
		logits, err = model.DecodeWithGPUKV([]int{nextToken}, pos)
		if err != nil {
			log.Fatalf("Decode failed at position %d: %v", pos, err)
		}

		nextToken = argmax(logits, gpuBackend)
		generated = append(generated, nextToken)
		pos++
	}

	// Print last token
	if nextToken != tok.EOS() {
		decoded, _ := tok.Decode([]int{nextToken})
		fmt.Print(decoded)
	}

	decodeDur := time.Since(decodeStart)
	fmt.Printf("\n\n[Decode: %d tokens in %v (%.1f tok/s)]\n",
		len(generated), decodeDur, float64(len(generated))/decodeDur.Seconds())
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
