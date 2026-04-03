//go:build metal && darwin && cgo

// eval_batch loads a model once, runs all eval questions base + LoRA.
// Usage: go run -tags metal experiments/eval_batch.go \
//          --model MODEL --adapter ADAPTER --questions QFILE --output OUTFILE
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"vexel/inference/backend/metal"
	"vexel/inference/lora"
	"vexel/inference/memory"
	"vexel/inference/pkg/gguf"
	"vexel/inference/pkg/sampler"
	"vexel/inference/pkg/tokenizer"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model")
	adapterPath := flag.String("adapter", "", "Path to LoRA adapter directory")
	questionsPath := flag.String("questions", "", "Questions file (one per line)")
	outputPath := flag.String("output", "", "Output file")
	maxTokens := flag.Int("max-tokens", 80, "Max tokens per response")
	flag.Parse()

	if *modelPath == "" || *questionsPath == "" || *outputPath == "" {
		log.Fatal("--model, --questions, and --output are required")
	}

	// Read questions
	questions := readLines(*questionsPath)
	log.Printf("Loaded %d questions", len(questions))

	// Init backend
	be, err := metal.NewBackend(0)
	if err != nil {
		log.Fatalf("NewBackend: %v", err)
	}
	defer be.Close()

	// Load model
	gf, err := gguf.Open(*modelPath)
	if err != nil {
		log.Fatalf("Open GGUF: %v", err)
	}
	cfg := runtime.ModelConfigFromGGUF(gf.GetModelConfig())
	tok, err := tokenizer.LoadFromGGUF(*modelPath)
	if err != nil {
		log.Fatalf("Load tokenizer: %v", err)
	}
	gf.Close()

	maxCtx := 2048
	memCtx := memory.NewInferenceContext(tensor.Metal)
	memCtx.AddArenaWithBackend(memory.Scratch, int(cfg.TotalArenaBytes(maxCtx)), be.Alloc)

	model, err := runtime.NewModelRuntime(be, memCtx, nil, cfg)
	if err != nil {
		log.Fatalf("NewModelRuntime: %v", err)
	}
	if err := model.LoadWeights(*modelPath); err != nil {
		log.Fatalf("LoadWeights: %v", err)
	}
	if err := model.CopyWeightsToDevice(); err != nil {
		log.Fatalf("CopyWeightsToDevice: %v", err)
	}
	model.CreateGPUKVCache(maxCtx)

	// Detect chat template for instruct models
	chatTpl := tokenizer.DetectChatTemplate(*modelPath)
	log.Printf("Model loaded, chat template: %s", chatTpl.Name)

	vocabSize := cfg.VocabSize
	eosToken := tok.EOS()

	readLogits := func(t tensor.Tensor) []float32 {
		buf := make([]byte, vocabSize*4)
		be.Sync()
		be.ToHost(buf, t.DevicePtr())
		out := make([]float32, vocabSize)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
		return out
	}

	generate := func(prompt string) string {
		// Apply chat template
		formatted := chatTpl.FormatChat("", prompt)
		ids, err := tok.Encode(formatted)
		if err != nil {
			return fmt.Sprintf("(encode error: %v)", err)
		}

		// Recreate KV cache to clear state from previous question
		model.CreateGPUKVCache(maxCtx)

		// Prefill
		logitsTensor, err := model.DecodeWithGPUKV(ids, 0)
		if err != nil {
			return fmt.Sprintf("(prefill error: %v)", err)
		}
		data := readLogits(logitsTensor)
		nextToken := sampler.Argmax(data)
		pos := len(ids)

		isStop := func(id int) bool {
			if id == eosToken || id == 0 {
				return true
			}
			for _, s := range chatTpl.ExtraStopTokenIDs {
				if id == s {
					return true
				}
			}
			return false
		}

		var generated []int
		for step := 0; step < *maxTokens; step++ {
			if isStop(nextToken) {
				break
			}
			generated = append(generated, nextToken)
			logitsTensor, err = model.DecodeWithGPUKV([]int{nextToken}, pos)
			if err != nil {
				break
			}
			data = readLogits(logitsTensor)
			nextToken = sampler.Argmax(data)
			pos++
		}

		text, _ := tok.Decode(generated)
		return strings.TrimSpace(text)
	}

	out, err := os.Create(*outputPath)
	if err != nil {
		log.Fatalf("Create output: %v", err)
	}
	defer out.Close()

	// BASE responses
	log.Printf("Generating BASE responses...")
	baseResponses := make([]string, len(questions))
	for i, q := range questions {
		baseResponses[i] = generate(q)
		log.Printf("  [%d/%d] done", i+1, len(questions))
	}

	// LORA responses
	loraResponses := make([]string, len(questions))
	if *adapterPath != "" {
		adapter, err := lora.LoadAdapter(*adapterPath)
		if err != nil {
			log.Fatalf("LoadAdapter: %v", err)
		}
		gpuAdapter, err := lora.UploadToGPU(adapter, be)
		if err != nil {
			log.Fatalf("UploadToGPU: %v", err)
		}
		model.AttachLoRA(gpuAdapter)
		log.Printf("LoRA loaded: rank=%d scale=%.2f", gpuAdapter.Rank, gpuAdapter.Scale)

		log.Printf("Generating LORA responses...")
		for i, q := range questions {
			loraResponses[i] = generate(q)
			log.Printf("  [%d/%d] done", i+1, len(questions))
		}
	}

	// Write
	for i, q := range questions {
		fmt.Fprintf(out, "Q: %s\n", q)
		fmt.Fprintf(out, "BASE: %s\n", baseResponses[i])
		if *adapterPath != "" {
			fmt.Fprintf(out, "LORA: %s\n", loraResponses[i])
		}
		fmt.Fprintln(out)
	}
	log.Printf("Results -> %s", *outputPath)
}

func readLines(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("Open %s: %v", path, err)
	}
	defer f.Close()
	var lines []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			lines = append(lines, line)
		}
	}
	return lines
}
