# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vexel is a high-performance LLM inference engine for Apple Silicon, written in Go with Metal GPU acceleration. It supports GGUF models (LLaMA 2/3, Mistral, Phi-2/3, Gemma 2) with Q4_0, Q4_K_M, Q5_K, Q6_K, Q8_0, and BF16 quantizations. Pure Go codebase with no Python dependencies.

**Requirements:** macOS 14.0+, Go 1.22+, Xcode Command Line Tools (for Metal compilation)

## Build & Test Commands

```bash
make build              # Build ./vexel binary (requires Metal)
make test               # Run non-Metal tests
make test-metal         # Run all tests including Metal GPU tests
make fmt                # go fmt ./...
make vet-metal          # go vet with Metal tags
make lint               # golangci-lint
make coverage           # Generate HTML coverage report

# Single package/test
go test -tags metal -v ./inference/runtime/
go test -tags metal -run TestName -v ./inference/backend/metal/

# Run examples
go run -tags metal ./examples/server -model path/to/model.gguf
go run -tags metal ./examples/generate -model path/to/model.gguf -prompt "Hello"
go run ./examples/client  # No Metal needed, requires running server
```

## Build Tags

Metal code requires `-tags metal` (with `CGO_ENABLED=1`). The constraint is `//go:build metal && darwin && cgo`. Non-Metal packages (client, tokenizer, GGUF parser, sampler, scheduler) build without special tags.

## Architecture

```
CLI (cmd/vexel)  →  Scheduler  →  Runtime  →  Backend (Metal/CPU/CUDA)
      ↕                ↕
   Server          Sampler
  (HTTP/gRPC)
```

- **`inference/cmd/vexel/`** — Unified CLI: `serve`, `generate`, `chat` (interactive multi-turn conversation), `bench`, `tokenize` subcommands
- **`inference/serve/`** — HTTP (`/generate`, `/stream`, `/health`) + gRPC server with TLS
- **`inference/scheduler/`** — Event-driven continuous batching scheduler. Manages sequence state transitions: Pending → Prefill → Decoding → Finished. Includes speculative decoding via draft models and Medusa heads, and batched decode with paged KV cache.
- **`inference/runtime/`** — Model loading, forward pass execution
  - `model.go` — Model struct, weight loading from GGUF
  - `decode.go` — Forward pass: prefill and decode loops
  - `block.go` — Per-layer transformer block execution (attention + FFN)
  - `plan.go` — Adaptive execution plan: selects kernel strategies based on model config, quantization, and head dimensions ("regime" system)
- **`inference/backend/`** — Compute backend abstraction. `backend.go` defines the core `Backend` interface plus optional capability interfaces (`QuantizedMatMul`, `FusedOps`, `FP16Ops`, `PagedKVOps`, etc.)
- **`inference/backend/metal/`** — Metal GPU kernels and dispatch (~45 test files). `metal_bridge_darwin.m` / `.h` for Obj-C bridge.
- **`inference/pkg/`** — `gguf/` (GGUF parser), `tokenizer/` (with chat templates), `sampler/` (temperature/top-k/top-p)
- **`client/`** — Go HTTP client library (no Metal dependency)

### Key Design Patterns

**Backend capability detection:** Runtime uses Go interface type assertions to check optional operations: `if qmm, ok := m.backend.(backend.QuantizedMatMul); ok { ... }`. Falls back gracefully when unimplemented.

**Execution plan system** (`runtime/plan.go`): Selects kernel strategies (SDPA variant, FFN approach, fused vs unfused) based on model architecture. Override with env vars: `VEXEL_FORCE_REGIME`, `VEXEL_FORCE_SDPA`, `VEXEL_FORCE_FFN`, `VEXEL_FORCE_FFN_GATEUP`, `VEXEL_FORCE_SDPA_PREFILL`.

**Scratch allocator:** All intermediate activations are bump-allocated from a single pre-allocated buffer via `ScratchReset()`/`ScratchAlloc()`, avoiding per-layer allocation overhead.

**Inference flow:** Prefill (batch of prompt tokens → logits) → Decode loop (single token + KV cache → next logits) → Sampling (temperature → top-k → top-p → sample).

**Adaptive Medusa speculation:** Starts off, probes every 8 decode steps, enables when acceptance >= 0.5.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `VEXEL_GPU_PROFILE=1` | GPU profiling (disables command batching) |
| `VEXEL_DECODE_TIMING=1` | Print decode timing |
| `VEXEL_KV_FP32=1` | Force FP32 KV cache |
| `VEXEL_FORCE_REGIME` | Override execution plan regime |
| `VEXEL_FORCE_SDPA` | Override SDPA kernel selection |
| `VEXEL_FORCE_FFN` | Override FFN kernel selection |
| `VEXEL_SDPA_TILED_THRESHOLD` | Override tiled split-K kvLen threshold (default: 2048) |
| `VEXEL_SDPA_NWG_THRESHOLD` | Override NWG kvLen threshold (default: 64) |
| `VEXEL_TEST_MODEL` | Path to model for integration tests |
| `DEBUG_DECODE=1` | Verbose decode loop output |
| `DEBUG_MATMUL=1` | Debug matrix multiplication |
| `DEBUG_PROFILE=1` | Enable runtime profiling |
| `VEXEL_REGRESSION_THRESHOLD` | Override regression threshold % for benchmark comparison (default: 5) |

## CLI Flags

| Flag | Purpose |
|------|---------|
| `--model` | Path to GGUF model file (required for serve/generate/chat) |
| `--draft-model` | Path to draft model for speculative decoding |
| `--context-len` | Max context length for KV cache (default: 2048) |
| `--medusa` | Enable Medusa-style speculative decoding |
| `--medusa-heads` | Path to pre-trained Medusa heads file |
| `--verbose` | Enable verbose logging |
| `--max-batch-size` | Max batch size for scheduler (serve only, default 1) |

## Commit Convention

Prefix: `feat()`, `fix()`, `perf()`, `chore()`, `docs()`, `build()`
Example: `perf(metal): add Q4_K/Q6_K quantization and Flash Attention 2`

## Dependencies

Minimal: `google/uuid`, `google.golang.org/grpc`, `google.golang.org/protobuf`. No Python dependencies.
