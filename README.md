# Vexel

Vexel is a high-performance inference engine for Large Language Models (LLMs) on Apple Silicon. It leverages Metal for hardware acceleration, optimized kernels (FlashAttention-2), and a custom scheduler to deliver state-of-the-art token generation speeds.

## Features

- **Metal Acceleration**: Fully optimized for Apple M-series chips (M1/M2/M3/M4) using custom Metal kernels.
- **FlashAttention-2**: Implements FlashAttention-2 for efficient attention computation.
- **Continuous Batching**: Event-driven scheduler for high-throughput concurrent inference.
- **Streaming Support**: Server-Sent Events (SSE) for real-time token streaming.
- **Go Client Library**: High-level Go client (`vexel/client`) for easy integration.
- **Direct Inference API**: Use the runtime directly for custom pipelines and benchmarking.
- **GGUF Support**: Compatible with GGUF model format (Q4_0 and other quantizations).
- **Multi-Architecture**: Supports LLaMA family (LLaMA 2/3, Mistral) and Phi family (Phi-2, Phi-3).

## Getting Started

### Prerequisites

- macOS 14.0+ (Sonoma) or later.
- Go 1.22+.
- Xcode Command Line Tools (for Metal compilation).

### Building

Build the unified `vexel` binary:

```bash
make build
```

This produces a single `./vexel` binary with all subcommands. Requires macOS with Metal support.

### CLI Subcommands

```
vexel [global flags] <subcommand> [flags]

Global flags:
  --model    Path to GGUF model file (required for serve/generate/chat)
  --verbose  Enable verbose logging

Subcommands:
  serve      Start the HTTP inference server
  generate   One-shot text generation from a prompt
  chat       Interactive chat REPL
  bench      Run scheduling benchmarks
  tokenize   Tokenize text to token IDs
```

### Running the Server

```bash
./vexel --model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf serve --port 8080
```

Serve flags:
- `--port`: HTTP port (default: 8080).
- `--max-tokens`: Max tokens per request (default: 256).
- `--max-batch-size`: Max batch size for scheduler (default: 1).

### Text Generation

```bash
./vexel --model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf generate \
  --prompt "Once upon a time" --max-tokens 100 --temperature 0.7
```

### Interactive Chat

```bash
./vexel --model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf chat \
  --system-prompt "You are a helpful assistant."
```

### Tokenization

```bash
./vexel --model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf tokenize \
  --input "Hello world"
# Or pipe from stdin:
echo "Hello world" | ./vexel tokenize --tokenizer path/to/tokenizer.json
```

### Benchmarking

```bash
./vexel bench --batch 128 --seq-len 128 --num-seqs 256
```

## Usage

### Go Client

Vexel provides a Go client library in `client/` for communicating with a running server.

```go
package main

import (
	"context"
	"fmt"
	"vexel/client"
)

func main() {
	c := client.New(client.Config{BaseURL: "http://localhost:8080"})

	// Blocking generation
	text, _ := c.Generate(context.Background(), "What is Go?", nil)
	fmt.Println(text)

	// Streaming generation (token-by-token)
	tokenChan, _ := c.Stream(context.Background(), "Hello, world!", nil)
	for token := range tokenChan {
		fmt.Print(token)
	}
}
```

See [`examples/client/main.go`](examples/client/main.go) for a full example with options.

### Direct Inference

For custom pipelines, you can use the runtime directly without the HTTP layer:

```go
// Load model and run prefill + decode loop
logits, _ := model.DecodeWithGPUKV(promptTokens, 0)  // prefill
nextToken := argmax(logits)

for i := 0; i < maxTokens; i++ {
	logits, _ = model.DecodeWithGPUKV([]int{nextToken}, pos)  // decode
	nextToken = argmax(logits)
	pos++
}
```

See [`examples/generate/main.go`](examples/generate/main.go) for a complete working example.

### Embedding the Server

You can embed Vexel's HTTP server into your own Go application:

```go
sched, _ := scheduler.NewScheduler(model, tok, scheduler.Config{
	MaxBatchSize:  1,
	MaxSequences:  64,
	MaxTokens:     256,
	SamplerConfig: sampler.DefaultConfig(),
})
go sched.Run(ctx)

srv := serve.NewServer(sched)
http.ListenAndServe(":8080", srv)
```

See [`examples/server/main.go`](examples/server/main.go) for a complete working example with model loading and graceful shutdown.

### HTTP API

**Generate (Blocking):**
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a joke"}'
```

**Stream (SSE):**
```bash
curl -N -X POST http://localhost:8080/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a joke"}'
```

## Examples

| Example | Description | Build Tags |
|---------|-------------|------------|
| [`examples/client/`](examples/client/main.go) | HTTP client with Generate and Stream | none |
| [`examples/server/`](examples/server/main.go) | Embed Vexel server in your Go app | `metal` |
| [`examples/generate/`](examples/generate/main.go) | Direct inference loop (no HTTP) | `metal` |

Run examples with:
```bash
# Client (requires a running server)
go run ./examples/client

# Server (Metal required)
go run -tags metal ./examples/server -model path/to/model.gguf

# Direct generation (Metal required)
go run -tags metal ./examples/generate -model path/to/model.gguf -prompt "Hello!"
```

## Architecture

```
client/              Go HTTP client library (no Metal dependency)
inference/
  cmd/vexel/         Unified CLI binary (serve, generate, chat, bench, tokenize)
  cmd/vexel/internal REPL and chat loop implementation
  backend/metal/     Metal GPU backend and kernel dispatch
  runtime/           Model loading, weight management, inference loop
  scheduler/         Continuous batching scheduler with metrics
  serve/             HTTP server (/generate, /stream endpoints)
  pkg/
    gguf/            GGUF model format parser
    tokenizer/       Tokenizer with chat template support
    sampler/         Temperature, top-k, top-p sampling
examples/            Usage examples (client, server, direct generation)
scripts/             Performance harness and benchmarking tools
```

## Performance

Benchmarks measured on Apple Silicon with LLaMA 2 7B Q4_0 (~3.7 GB):

| Metric | Value |
|--------|-------|
| Prefill throughput (128 tokens) | ~200 tok/s |
| Decode throughput | ~44 tok/s |
| Time-to-first-token (5-token prompt) | ~80 ms |
| Per-token decode latency (p50) | ~23 ms |
| Decode jitter (p99/p50) | 1.10 |

Run the benchmark suite:
```bash
# Unit-level benchmarks
go test -tags metal -run TestThroughput -v ./inference/runtime/
go test -tags metal -run TestLatency -v ./inference/runtime/

# Scheduling benchmarks via CLI
./vexel bench --batch 128 --seq-len 128 --num-seqs 256
```

## Performance & Correctness Harness

Use the provided harness to compare Vexel (Metal) against `llama.cpp` on a fixed prompt set.

### Requirements
- Model: default `models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf` (override with `MODEL_PATH`).
- Vexel binary: default `./vexel` (override with `VEXEL_BIN`).
- llama.cpp binary: default `llama-cli` (override with `LLAMA_BIN`).

### Run
```bash
bash scripts/perf_harness.sh
```

The harness uses deterministic sampling (`temp=0`) to make correctness comparisons meaningful.

Environment overrides:
- `MODEL_PATH` – GGUF model path
- `VEXEL_BIN` – path to Vexel binary
- `LLAMA_BIN` – path to llama.cpp binary
- `OUT_DIR` – output directory (default `perf_reports`)
