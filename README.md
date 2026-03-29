# Vexel

Vexel is a high-performance inference engine for Large Language Models (LLMs) on Apple Silicon. It leverages Metal for hardware acceleration, optimized kernels (FlashAttention-2), and a custom scheduler to deliver state-of-the-art token generation speeds.

## Features

- **Metal Acceleration**: Fully optimized for Apple M-series chips (M1/M2/M3/M4) using custom Metal kernels.
- **FlashAttention-2**: Implements FlashAttention-2 for efficient attention computation.
- **Continuous Batching**: Event-driven scheduler for high-throughput concurrent inference.
- **Streaming Support**: Server-Sent Events (SSE) for real-time token streaming.
- **Go Client Library**: High-level Go client (`vexel/client`) for easy integration.
- **Direct Inference API**: Use the runtime directly for custom pipelines and benchmarking.
- **Speculative Decoding**: Draft-model and Medusa-style speculative decoding for 20-50% throughput gains.
- **Batched Decode**: True multi-sequence batched decode with paged KV cache for high-throughput serving.
- **GGUF Support**: Compatible with GGUF model format (Q4_0, Q4_K_M, Q5_K, Q6_K, Q8_0, BF16 quantizations).
- **Multi-Architecture**: Supports LLaMA family (LLaMA 2/3, Mistral), Phi family (Phi-2, Phi-3), and Gemma 2.

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
  --model        Path to GGUF model file (required for serve/generate/chat)
  --draft-model  Path to draft model for speculative decoding
  --medusa       Enable Medusa-style speculative decoding
  --medusa-heads Path to pre-trained Medusa heads file (implies --medusa)
  --context-len  Max context length for KV cache (default: 2048)
  --verbose      Enable verbose logging

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
- `--max-batch-size`: Max concurrent sequences for batched decode (default: 1). Set higher (e.g., 4-8) for multi-client serving throughput.

**Paged KV cache:** Use `--context-len` to set the maximum context length for the KV cache. With `--max-batch-size > 1`, the scheduler uses paged KV cache allocation to efficiently share GPU memory across concurrent sequences, enabling multi-client serving without pre-allocating full context buffers per sequence. Paged KV is also used in Medusa mode to support speculation alongside multi-client serving.

```bash
# Multi-client serving with paged KV cache (4 concurrent sequences, 4096 context)
./vexel --model model.gguf --context-len 4096 serve --port 8080 --max-batch-size 4
```

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

### Speculative Decoding

**Draft-model speculation** uses a smaller model to draft tokens, verified by the target model:

```bash
./vexel --model models/llama-7b.Q4_0.gguf --draft-model models/tinyllama-1.1b.Q4_0.gguf \
  generate --prompt "Once upon a time"
```

**Medusa speculation** uses online-learned heads that predict multiple future tokens in parallel, with no separate draft model required:

```bash
# Medusa with online training (heads learn during inference)
./vexel --model models/llama-7b.Q4_0.gguf --medusa serve --port 8080

# Medusa with pre-trained heads (instant speculation)
./vexel --model models/llama-7b.Q4_0.gguf --medusa-heads heads.bin generate --prompt "Hello"
```

Note: `--draft-model` and `--medusa` are mutually exclusive.

#### Medusa Details

Medusa speculative decoding uses lightweight output heads that predict multiple future tokens in parallel, eliminating the need for a separate draft model.

**Online training:** When `--medusa` is enabled without `--medusa-heads`, Medusa heads are trained online during inference. The heads learn from the model's own predictions, starting with a brief auto-warmup phase (600 tokens + 3s training) to reach the Hot phase before serving begins.

**Adaptive speculation:** The number of speculated tokens adjusts based on acceptance rates observed during decoding. This maximizes throughput without wasting computation on low-confidence predictions.

**Pre-trained heads:** Save and load trained Medusa heads for instant speculation on subsequent runs:

```bash
# Heads are auto-saved to <model-path>.medusa-heads.bin after training
./vexel --model model.gguf --medusa serve --port 8080

# Load pre-trained heads on next run (skips warmup, instant speculation)
./vexel --model model.gguf --medusa-heads model.gguf.medusa-heads.bin serve --port 8080
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

**Per-request parameters:** Both `/generate` and `/stream` accept optional fields to override the server defaults:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a creative story.", "max_tokens": 100, "temperature": 0.8, "top_k": 40, "top_p": 0.9}'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Input text |
| `max_tokens` | int | server default | Maximum tokens to generate |
| `temperature` | float | 0 (greedy) | Sampling temperature (0 = deterministic) |
| `top_k` | int | 0 (disabled) | Keep top K tokens |
| `top_p` | float | 0 (disabled) | Nucleus sampling threshold |

The Go client library supports all parameters via `GenerateOptions` (see [`examples/client/main.go`](examples/client/main.go)). The gRPC API also supports these via `SamplingParams`.

## Examples

| Example | Description | Build Tags |
|---------|-------------|------------|
| [`examples/client/`](examples/client/main.go) | HTTP client with Generate and Stream | none |
| [`examples/server/`](examples/server/main.go) | Embed Vexel server in your Go app | `metal` |
| [`examples/generate/`](examples/generate/main.go) | Direct inference loop (no HTTP) | `metal` |
| [`examples/benchmark/`](examples/benchmark/main.go) | Programmatic benchmarking (GPU KV vs paged) | `metal` |

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

Benchmarks on Apple M4 Max (128 GB) with LLaMA 3.1 8B Q4_K_M:

| Metric | Vexel | llama.cpp | vs llama.cpp |
|--------|-------|-----------|--------------|
| Decode throughput | **66.6 tok/s** | 42.4 tok/s | **+57%** |
| Server throughput | **52 tok/s** | 42 tok/s | **+24%** |
| Context degradation (16→1024) | <6% | 8.1% | better |

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for detailed tuning guide and [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for full benchmark data.
| Prefill (128 tokens) | 377 tok/s | 803 tok/s | 725 tok/s | -53% |
| Model load time | ~885 ms | ~1100 ms | — | +20% faster |

**Q4_K_M (4.08 GB) — optimized kernels:**

| Metric | Vexel | vs Q4_0 | Before Optimization |
|--------|-------|---------|---------------------|
| Decode throughput | 53.5 tok/s | 0.84x | 14.2 tok/s (3.8x faster) |
| Prefill (128 tokens) | 220.4 tok/s | 0.58x | 16.6 tok/s (13.3x faster) |
| Context degradation (16→512) | -10.8% | — | — |

Vexel achieves **+57% over llama.cpp's decode throughput** on Q4_K_M and supports
Q4_K_M with custom sub-block strided decode and simdgroup tiled prefill kernels.

**Flash Attention:** Adaptive SDPA kernel dispatch with tiled split-K for long contexts and NWG multi-threadgroup for medium contexts. Context scaling degradation reduced from -24.5% to ~-10% (ctx=16 → ctx=512). The dispatch chain auto-selects: tiled split-K (kvLen > 2048) → NWG adaptive (kvLen > 64) → v3 chunk-based → v1 fallback.

See [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for detailed analysis,
context-length scaling data, and optimization roadmap.

**Where Vexel differentiates:**
- Go scheduler with continuous batching and batched decode (multi-client throughput)
- Speculative decoding: draft-model and Medusa-style online-learned heads
- gRPC + HTTP dual protocol for production serving
- Single binary deployment — no Python dependency chain
- Pure Go codebase — easy to embed, extend, and deploy
- Faster cold start than llama.cpp (~885ms vs ~1100ms)

Run the benchmark suite:
```bash
# Unit-level benchmarks
go test -tags metal -run TestThroughput -v ./inference/runtime/
go test -tags metal -run TestLatency -v ./inference/runtime/

# Competitive benchmarks (vs llama.cpp, MLX, Ollama)
./benchmarks/run_benchmark.sh --model path/to/model.gguf --engines vexel,llama

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
