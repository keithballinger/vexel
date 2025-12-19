//go:build metal && darwin && cgo && vexeldebug

// vexel-debug is a debug harness for tracing inference issues.
//
// Usage:
//
//	vexel-debug -model MODEL -prompt "Hello" -layers 15 -positions 4 -ops sdpa,wo
//	vexel-debug -model MODEL -prompt "Hello" -layers 14,15 -verbose
//	vexel-debug -model MODEL -prompt "Hello" -output trace.json
//
// This tool runs inference with targeted debug tracing, capturing tensor
// values at specified layers/positions/operations and outputting structured
// JSON for analysis.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"vexel/inference/backend/metal"
	"vexel/inference/debug"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func main() {
	// Model flags
	modelPath := flag.String("model", "", "Path to model file (.gguf)")
	tokenizerPath := flag.String("tokenizer", "", "Path to tokenizer.json")
	prompt := flag.String("prompt", "Hello", "Prompt to run")
	maxTokens := flag.Int("max-tokens", 5, "Max tokens to generate")

	// Debug targeting flags
	layers := flag.String("layers", "", "Comma-separated layer indices to trace (empty=all)")
	positions := flag.String("positions", "", "Comma-separated position indices to trace (empty=all)")
	ops := flag.String("ops", "", "Comma-separated ops to trace: input,norm,qkv,rope,kv,sdpa,wo,mlp,output (empty=all)")

	// Output flags
	output := flag.String("output", "", "Output file for JSON trace (empty=stderr summary only)")
	verbose := flag.Bool("verbose", true, "Print human-readable output during inference")
	maxValues := flag.Int("max-values", 16, "Max tensor values to include in dumps")

	// Threshold flags
	maxThreshold := flag.Float64("max-threshold", 0, "Flag tensors with max > this value (0=disabled)")
	minThreshold := flag.Float64("min-threshold", 0, "Flag tensors with min < this value (0=disabled)")

	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Must specify -model")
	}

	// Parse targeting flags
	cfg := debug.Config{
		Layers:       parseIntList(*layers),
		Positions:    parseIntList(*positions),
		Ops:          parseStringList(*ops),
		OutputPath:   *output,
		Verbose:      *verbose,
		MaxValues:    *maxValues,
		MaxThreshold: float32(*maxThreshold),
		MinThreshold: float32(*minThreshold),
	}

	// Initialize debug harness
	if err := debug.Init(cfg); err != nil {
		log.Fatalf("Failed to init debug harness: %v", err)
	}
	defer debug.Close()

	// Load model
	fmt.Fprintf(os.Stderr, "Loading model: %s\n", *modelPath)

	gf, err := gguf.Open(*modelPath)
	if err != nil {
		log.Fatalf("Failed to open GGUF: %v", err)
	}
	modelCfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	gf.Close()

	// Load tokenizer
	var tok *tokenizer.Tokenizer
	if *tokenizerPath != "" {
		tok, err = tokenizer.Load(*tokenizerPath)
	} else {
		modelDir := filepath.Dir(*modelPath)
		tok, err = tokenizer.Load(filepath.Join(modelDir, "tokenizer.json"))
	}
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Initialize backend
	backend, err := metal.NewMetalBackend()
	if err != nil {
		log.Fatalf("Failed to init Metal backend: %v", err)
	}
	defer backend.Close()

	// Load model weights
	model, err := runtime.LoadModel(*modelPath, modelCfg, backend)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	debug.SetModel(*modelPath)
	debug.SetPrompt(*prompt)

	// Tokenize prompt
	tokens, err := tok.Encode(*prompt)
	if err != nil {
		log.Fatalf("Failed to encode prompt: %v", err)
	}

	// Add BOS if needed
	if tok.AddBOS() {
		tokens = append([]int{tok.BOS()}, tokens...)
	}

	fmt.Fprintf(os.Stderr, "Prompt tokens: %v\n", tokens)
	fmt.Fprintf(os.Stderr, "Running inference with debug tracing...\n")
	fmt.Fprintf(os.Stderr, "  Layers: %v\n", cfg.Layers)
	fmt.Fprintf(os.Stderr, "  Positions: %v\n", cfg.Positions)
	fmt.Fprintf(os.Stderr, "  Ops: %v\n", cfg.Ops)
	fmt.Fprintf(os.Stderr, "\n")

	// Run inference with debug hooks
	runInference(model, backend, tok, tokens, *maxTokens)

	fmt.Fprintf(os.Stderr, "\nDebug trace complete.\n")
	if *output != "" {
		fmt.Fprintf(os.Stderr, "Trace written to: %s\n", *output)
	}
}

func runInference(model *runtime.ModelRuntime, backend *metal.MetalBackend, tok *tokenizer.Tokenizer, tokens []int, maxTokens int) {
	hiddenSize := model.Config().HiddenSize
	numLayers := model.Config().NumHiddenLayers

	// Create GPU KV cache
	gpuCache := runtime.NewGPUKVCache(
		backend,
		numLayers,
		model.Config().NumKeyValueHeads,
		hiddenSize/model.Config().NumAttentionHeads,
		2048, // maxSeqLen
		true, // useFP16
	)
	defer gpuCache.Close()

	// Prefill
	fmt.Fprintf(os.Stderr, "=== PREFILL (tokens=%v) ===\n", tokens)
	prefillOut, err := model.DecodeWithGPUKVDebug(tokens, 0, gpuCache, debugCapture)
	if err != nil {
		log.Printf("Prefill error: %v", err)
		return
	}

	// Get logits and sample
	logits := model.LMHead(prefillOut, len(tokens))
	nextToken := sampleGreedy(logits)
	decoded, _ := tok.Decode([]int{nextToken})
	fmt.Fprintf(os.Stderr, "\nPrefill -> token %d (%q)\n", nextToken, decoded)

	generatedTokens := []int{nextToken}

	// Decode loop
	for i := 0; i < maxTokens-1; i++ {
		pos := len(tokens) + i
		fmt.Fprintf(os.Stderr, "\n=== DECODE pos=%d token=%d ===\n", pos, nextToken)

		decodeOut, err := model.DecodeWithGPUKVDebug([]int{nextToken}, pos, gpuCache, debugCapture)
		if err != nil {
			log.Printf("Decode error at pos %d: %v", pos, err)
			break
		}

		logits = model.LMHead(decodeOut, 1)
		nextToken = sampleGreedy(logits)
		decoded, _ := tok.Decode([]int{nextToken})
		fmt.Fprintf(os.Stderr, "\nDecode pos=%d -> token %d (%q)\n", pos, nextToken, decoded)

		if nextToken == tok.EOS() {
			break
		}
		generatedTokens = append(generatedTokens, nextToken)
	}

	output, _ := tok.Decode(generatedTokens)
	fmt.Fprintf(os.Stderr, "\n=== OUTPUT ===\n%s\n", output)
}

// debugCapture is called by DecodeWithGPUKVDebug at each debug point.
func debugCapture(layer, position int, op, name string, ptr tensor.DevicePtr, size int, backend interface {
	Sync()
	ToHost([]byte, tensor.DevicePtr)
}) {
	if !debug.ShouldCapture(layer, position, op) {
		return
	}

	backend.Sync()

	// Read tensor data from GPU
	data := make([]byte, size*4)
	backend.ToHost(data, ptr)

	// Convert to float32
	floats := make([]float32, size)
	for i := 0; i < size; i++ {
		floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}

	// Capture with flag detection
	flagged := debug.CaptureWithFlag(layer, position, op, name, floats)
	if flagged {
		fmt.Fprintf(os.Stderr, "  ^^^ FLAGS DETECTED ^^^\n")
	}
}

func sampleGreedy(logits tensor.Tensor) int {
	data := logits.Data().([]float32)
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func parseIntList(s string) []int {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	result := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			log.Fatalf("Invalid integer in list: %q", p)
		}
		result = append(result, n)
	}
	return result
}

func parseStringList(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	result := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	return result
}
