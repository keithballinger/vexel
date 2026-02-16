# Vexel

Vexel is a high-performance inference engine for Large Language Models (LLMs) on Apple Silicon. It leverages Metal for hardware acceleration, optimized kernels (FlashAttention-2), and a custom scheduler to deliver state-of-the-art token generation speeds.

## Features

- **Metal Acceleration**: Fully optimized for Apple M-series chips (M1/M2/M3/M4) using custom Metal kernels.
- **FlashAttention-2**: Implements FlashAttention-2 for efficient attention computation.
- **Continuous Batching**: Supports continuous batching (via scheduler) for high throughput.
- **Streaming Support**: Server-Sent Events (SSE) for real-time token streaming.
- **Go Client Library**: High-level Go client for easy integration.
- **GGUF Support**: Compatible with GGUF model format (quantized models).

## Getting Started

### Prerequisites

- macOS 14.0+ (Sonoma) or later.
- Go 1.22+.
- Xcode Command Line Tools (for Metal compilation).

### Building

Build the `vexel` binary:

```bash
make build
```

This will compile the Go code and the Metal kernels, producing a `vexel` binary in the root directory.

### Running the Server

Start the inference server with a GGUF model:

```bash
./vexel serve --model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --port 8080
```

Flags:
- `--model`: Path to the GGUF model file.
- `--port`: HTTP port to listen on (default: 8080).
- `--gpu-layers`: Number of layers to offload to GPU (default: all).

## Usage

### Go Client

Vexel provides a Go client library in `client/`.

```go
package main

import (
	"context"
	"fmt"
	"vexel/client"
)

func main() {
	c := client.New(client.Config{BaseURL: "http://localhost:8080"})
	
	// Streaming
	tokenChan, _ := c.Stream(context.Background(), "Hello, world!", nil)
	for token := range tokenChan {
		fmt.Print(token)
	}
}
```

See `examples/client/main.go` for a full example.

### HTTP API

You can also use `curl`:

**Generate (Blocking):**
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a joke", "temperature": 0.7}'
```

**Stream (SSE):**
```bash
curl -N -X POST http://localhost:8080/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a joke"}'
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
