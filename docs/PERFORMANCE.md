# Performance Tuning Guide

This guide covers practical tuning for Vexel on Apple Silicon. All numbers are from an M4 Max (128 GB, 40 GPU cores) with LLaMA 3.1 8B Instruct Q4_K_M unless noted otherwise.

## Table of Contents

1. [Decode Performance](#decode-performance)
2. [Server Mode](#server-mode)
3. [Multi-Client Scaling](#multi-client-scaling)
4. [Medusa Speculative Decoding](#medusa-speculative-decoding)
5. [Context Length](#context-length)
6. [Memory Usage](#memory-usage)
7. [Environment Variables](#environment-variables)
8. [Profiling](#profiling)

---

## Decode Performance

Vexel offers two KV cache modes. The choice affects throughput and feature availability.

### GPU KV Cache (Default)

The default mode pre-allocates a contiguous KV cache on GPU. This is the fastest path for single-client inference.

```bash
# Single-client generation (GPU KV, greedy)
./vexel --model model.gguf generate --prompt "Hello" --max-tokens 128
```

**LLaMA 3.1 8B Q4_K_M decode:** ~65 tok/s (29% faster than llama.cpp at 51 tok/s).

GPU KV is limited to the default context length of 2048 tokens. For longer contexts or multi-client serving, use paged KV.

### Paged KV Cache

Enabled automatically when `--context-len` is specified. Allocates KV cache in blocks from a shared pool, enabling multi-client serving and longer contexts.

```bash
# Paged KV with 4096 context
./vexel --model model.gguf --context-len 4096 generate --prompt "Hello" --max-tokens 128
```

**LLaMA 3.1 8B Q4_K_M decode:** ~58 tok/s with paged KV (about 11% slower than GPU KV).

### When to Use Each

| Mode | Throughput | Use Case |
|------|-----------|----------|
| GPU KV (default) | ~65 tok/s | Single-client, short context, maximum speed |
| Paged KV (`--context-len`) | ~58 tok/s | Multi-client serving, context > 2048, Medusa mode |

**Recommendation:** Use GPU KV for single-client workloads. Switch to paged KV when you need multi-client serving or contexts beyond 2048 tokens.

---

## Server Mode

### Default Greedy Sampling

The server uses greedy sampling (temperature=0) by default for maximum throughput.

```bash
./vexel --model model.gguf serve --port 8080
```

**Single-client server decode:** ~52 tok/s with GPU KV (greedy).

### Impact of Sampling Parameters

Setting `temperature > 0` enables stochastic sampling (temperature, top-k, top-p), which runs on the CPU and reduces throughput.

```bash
# Request with temperature (slower due to CPU sampling)
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "temperature": 0.7}'
```

**With temperature > 0:** throughput drops to ~30 tok/s due to CPU-side sampling overhead.

### Per-Request Parameters

Sampling parameters can be set per-request in the JSON body without affecting other clients:

```bash
# Greedy (fastest)
curl -X POST http://localhost:8080/generate \
  -d '{"prompt": "Summarize this.", "max_tokens": 50}'

# Creative (slower, CPU sampling)
curl -X POST http://localhost:8080/generate \
  -d '{"prompt": "Write a poem.", "temperature": 0.9, "top_p": 0.95, "max_tokens": 200}'
```

**Recommendation:** Leave temperature at 0 (greedy) for tasks like summarization or code generation where deterministic output is acceptable. Use temperature only when creative variation is needed.

---

## Multi-Client Scaling

For concurrent client serving, configure paged KV cache and batch size.

### Setup

```bash
# 4 concurrent clients, 4096 context, paged KV
./vexel --model model.gguf --context-len 4096 serve \
  --port 8080 --max-batch-size 4
```

- `--context-len 4096` enables paged KV cache with a shared block pool.
- `--max-batch-size 4` allows up to 4 sequences to be decoded concurrently in a single batched forward pass.

### Testing Concurrency

```bash
# Fire 4 concurrent requests
for i in 1 2 3 4; do
  curl -s -X POST http://localhost:8080/generate \
    -d "{\"prompt\": \"Client $i: Hello\", \"max_tokens\": 64}" &
done
wait
```

All concurrent clients produce correct, independent output. The scheduler handles sequence isolation through the paged KV block allocator.

### Tuning `--max-batch-size`

| Batch Size | Behavior |
|-----------|----------|
| 1 (default) | Single sequence at a time, requests queue |
| 2-4 | Good concurrency for most workloads |
| 8+ | Higher throughput but more GPU memory pressure |

**Recommendation:** Start with `--max-batch-size 4` and increase if you observe request queuing. Monitor GPU memory with `--verbose`.

---

## Medusa Speculative Decoding

Medusa uses lightweight prediction heads to draft multiple future tokens in parallel, without a separate draft model.

### Online Training (Default)

When `--medusa` is enabled, heads are trained online during inference.

```bash
./vexel --model model.gguf --medusa serve --port 8080
```

**Warmup phase:** The server automatically runs a 600-token warmup before accepting requests. During warmup, heads train for approximately 3 seconds to reach the "Hot" phase.

The auto-warmup ensures `--max-tokens` is large enough to generate the warmup tokens. If your `--max-tokens` is too low, Vexel adjusts it automatically.

### Pre-Trained Heads (Instant Startup)

For production use, save and reload trained heads to skip the warmup entirely.

```bash
# First run: heads auto-save to <model>.medusa-heads.bin after training
./vexel --model model.gguf --medusa serve --port 8080

# Subsequent runs: load pre-trained heads (no warmup delay)
./vexel --model model.gguf --medusa-heads model.gguf.medusa-heads.bin serve --port 8080
```

Pre-trained heads provide instant speculation from the first request.

### Adaptive Speculation

Medusa speculation is not always beneficial. The scheduler adapts automatically:

1. **Starts off** -- normal decode, no speculation overhead.
2. **Probes every 8 decode steps** -- runs speculative candidates and measures acceptance.
3. **Enables at >= 50% acceptance** -- only activates when heads are accurate enough to provide a net speedup.
4. **Falls back if acceptance drops** -- reverts to normal decode if head quality degrades.

This prevents wasted compute on low-confidence predictions and ensures output correctness.

### Training Quality

Online training uses several techniques for stable head learning:

- **Learning rate warmup** -- gradual ramp to avoid early instability.
- **Gradient clipping** -- prevents divergence from large gradients.
- **Per-head noise injection** -- encourages diversity across heads so they specialize on different prediction horizons.

Current online training achieves moderate acceptance rates. For best results, use pre-trained heads or allow extended warmup time.

### Medusa with Multi-Client

Medusa works with paged KV, so it can be combined with multi-client serving:

```bash
./vexel --model model.gguf --medusa --context-len 4096 serve \
  --port 8080 --max-batch-size 4
```

Note: `--medusa` and `--draft-model` are mutually exclusive. Choose one speculation strategy.

---

## Context Length

### Default (2048, GPU KV)

Without `--context-len`, Vexel uses a 2048-token GPU KV cache. This is the fastest configuration for short-to-medium prompts.

### Extended Context (Paged KV)

Use `--context-len N` for longer contexts. Tested up to 8192 tokens.

```bash
# 8192-token context
./vexel --model model.gguf --context-len 8192 generate \
  --prompt "$(cat long_document.txt)" --max-tokens 256
```

### Context Scaling Performance

Vexel maintains strong throughput as context grows, thanks to adaptive SDPA kernel dispatch:

| Context Length | Decode tok/s | Degradation |
|---------------|-------------|-------------|
| 16 | 66.6 | baseline |
| 64 | 66.7 | -0.2% |
| 256 | 64.1 | 3.8% |
| 512 | 63.7 | 4.3% |
| 1024 | 62.9 | 5.5% |

For comparison, llama.cpp degrades 16.2% at context 2048 and 30.5% at context 4096.

The SDPA dispatch chain auto-selects the best kernel per context length:
- **Tiled split-K** for kvLen > 2048 (long context)
- **NWG adaptive tiling** for kvLen 64-2048 (medium context)
- **v3 chunk-based** for shorter contexts
- **v1 fallback** for edge cases

Override thresholds with environment variables if needed (see [Environment Variables](#environment-variables)).

---

## Memory Usage

### GPU Memory Reporting

Use `--verbose` to see GPU memory allocation details at startup:

```bash
./vexel --model model.gguf --verbose serve --port 8080
```

This reports model weight buffer sizes, KV cache allocation, scratch arena size, and block pool sizing for paged KV.

### Block Pool Sizing (Paged KV)

When using `--context-len`, the block pool is sized based on context length and `--max-batch-size`. Larger values consume more GPU memory.

**Rough memory budget:**
- Model weights: depends on quantization (see below)
- KV cache: scales with context length x number of layers x head dimensions
- Scratch arena: pre-allocated for intermediate activations (typically 100-300 MB)

### Model Quantization Impact

Choose quantization based on your quality/speed/memory tradeoff:

| Quantization | Model Size (7-8B) | Decode tok/s | Notes |
|-------------|-------------------|-------------|-------|
| Q4_0 | ~3.6 GB | ~65 tok/s | Fastest decode, lower quality |
| Q4_K_M | ~4.1 GB | ~54 tok/s | Best quality/speed balance |
| Q5_K | ~4.7 GB | — | Higher quality, moderate speed |
| Q6_K | ~5.5 GB | — | Near-FP16 quality |
| Q8_0 | ~7.2 GB | — | Highest quantized quality |
| BF16 | ~14 GB | — | Full precision, most memory |

Q4_K_M is recommended for most workloads -- it offers better quality than Q4_0 with optimized kernels (sub-block strided decode, simdgroup tiled prefill).

---

## Environment Variables

All `VEXEL_*` variables for performance tuning:

### Kernel Override Variables

Force specific kernel strategies when debugging or benchmarking:

```bash
# Force a specific execution plan regime
VEXEL_FORCE_REGIME=2 ./vexel --model model.gguf generate --prompt "Hello"

# Force a specific SDPA kernel
VEXEL_FORCE_SDPA=tiled ./vexel --model model.gguf generate --prompt "Hello"

# Force a specific FFN kernel
VEXEL_FORCE_FFN=fused ./vexel --model model.gguf generate --prompt "Hello"

# Force FFN gate-up strategy
VEXEL_FORCE_FFN_GATEUP=split ./vexel --model model.gguf generate --prompt "Hello"

# Force prefill SDPA kernel
VEXEL_FORCE_SDPA_PREFILL=v3 ./vexel --model model.gguf generate --prompt "Hello"
```

### SDPA Threshold Variables

Tune the context-length thresholds for SDPA kernel dispatch:

```bash
# Use tiled split-K for contexts above 1024 (default: 2048)
VEXEL_SDPA_TILED_THRESHOLD=1024 ./vexel --model model.gguf generate --prompt "Hello"

# Use NWG multi-threadgroup for contexts above 32 (default: 64)
VEXEL_SDPA_NWG_THRESHOLD=32 ./vexel --model model.gguf generate --prompt "Hello"
```

### KV Cache Variables

```bash
# Force FP32 KV cache (higher precision, more memory, slightly slower)
VEXEL_KV_FP32=1 ./vexel --model model.gguf generate --prompt "Hello"
```

### Debug and Timing Variables

```bash
# Print per-token decode timing
VEXEL_DECODE_TIMING=1 ./vexel --model model.gguf generate --prompt "Hello"

# Verbose decode loop output
DEBUG_DECODE=1 ./vexel --model model.gguf generate --prompt "Hello"

# Debug matrix multiplication
DEBUG_MATMUL=1 ./vexel --model model.gguf generate --prompt "Hello"
```

### Full Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `VEXEL_GPU_PROFILE` | off | Enable GPU profiling (disables command batching) |
| `VEXEL_DECODE_TIMING` | off | Print per-token decode timing |
| `VEXEL_KV_FP32` | off | Force FP32 KV cache |
| `VEXEL_FORCE_REGIME` | auto | Override execution plan regime |
| `VEXEL_FORCE_SDPA` | auto | Override SDPA kernel selection |
| `VEXEL_FORCE_FFN` | auto | Override FFN kernel selection |
| `VEXEL_FORCE_FFN_GATEUP` | auto | Override FFN gate-up strategy |
| `VEXEL_FORCE_SDPA_PREFILL` | auto | Override prefill SDPA kernel |
| `VEXEL_SDPA_TILED_THRESHOLD` | 2048 | kvLen threshold for tiled split-K |
| `VEXEL_SDPA_NWG_THRESHOLD` | 64 | kvLen threshold for NWG adaptive |
| `VEXEL_TEST_MODEL` | — | Path to model for integration tests |
| `VEXEL_REGRESSION_THRESHOLD` | 5 | Regression threshold % for benchmark comparison |
| `DEBUG_DECODE` | off | Verbose decode loop output |
| `DEBUG_MATMUL` | off | Debug matrix multiplication |
| `DEBUG_PROFILE` | off | Enable runtime profiling |

---

## Profiling

### GPU Profiling

Enable GPU profiling to get per-kernel timing. This disables command buffer batching, so throughput will be lower than normal -- use it for analysis only.

```bash
VEXEL_GPU_PROFILE=1 ./vexel --model model.gguf generate \
  --prompt "Hello" --max-tokens 32
```

This prints per-dispatch timing information, showing which Metal kernels consume the most time. Useful for identifying bottlenecks in specific model architectures or quantizations.

**Important:** GPU profiling disables command buffer batching (the optimization responsible for the 8.4x decode speedup), so profiled throughput will be significantly lower than production throughput.

### Decode Timing

For production-representative timing without disabling batching:

```bash
VEXEL_DECODE_TIMING=1 ./vexel --model model.gguf generate \
  --prompt "Once upon a time" --max-tokens 128
```

This prints wall-clock time per token, giving you actual decode latency at each step.

### Runtime Profiling

For Go-level profiling:

```bash
DEBUG_PROFILE=1 ./vexel --model model.gguf generate \
  --prompt "Hello" --max-tokens 64
```

### Benchmark Suite

Run the built-in benchmarks for systematic measurement:

```bash
# Decode throughput benchmark
go test -tags metal -run TestThroughput -v ./inference/runtime/

# Decode latency benchmark
go test -tags metal -run TestLatency -v ./inference/runtime/

# Scheduling benchmarks
./vexel bench --batch 128 --seq-len 128 --num-seqs 256

# Competitive comparison (requires llama.cpp installed)
./benchmarks/run_benchmark.sh --model model.gguf --engines vexel,llama

# Full comparison suite
cd benchmarks && ./full_comparison.sh all
```

### Performance Harness

Compare Vexel against llama.cpp on a fixed prompt set with deterministic sampling:

```bash
# Default paths
bash scripts/perf_harness.sh

# Custom paths
MODEL_PATH=models/llama-8b.Q4_K_M.gguf \
VEXEL_BIN=./vexel \
LLAMA_BIN=llama-cli \
OUT_DIR=perf_reports \
bash scripts/perf_harness.sh
```

Results are written to `perf_reports/` (or the directory specified by `OUT_DIR`).
