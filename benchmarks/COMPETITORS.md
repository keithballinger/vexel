# Competitive Landscape: Apple Silicon LLM Inference Engines

> Research date: February 2026
> Benchmark machine reference: Apple Silicon (M2 Ultra / M3 Max / M4 Max)
> Focus: memory-bandwidth-bound single-stream and batched inference

## Executive Summary

Apple Silicon LLM inference is **memory-bandwidth bound** at small batch sizes.
The primary differentiator metric is memory bandwidth utilization
(`bytes_read / time / theoretical_bandwidth`). At decode time, every token
requires reading the full model weights once, so throughput scales linearly
with memory bandwidth until compute becomes the bottleneck (large batches or
prefill).

**Current ranking (single-stream decode, M2 Ultra, 8B-class Q4):**

| Rank | Engine     | ~tok/s | Notes                                    |
|------|------------|--------|------------------------------------------|
| 1    | MLX        | ~230   | Apple-optimized Metal kernels            |
| 2    | MLC-LLM    | ~190   | Paged KV, best long-context scaling      |
| 3    | llama.cpp  | ~150   | Ecosystem standard, GGUF originator      |
| 4    | Ollama     | 20-40  | Wraps llama.cpp, UX-focused              |

**Batched throughput leader:** vllm-mlx — 4.3x aggregate scaling at 16 concurrent.

**Vexel current numbers (LLaMA 2 7B Q4_0, Apple Silicon):**
- Decode: ~44 tok/s
- Prefill (128 tokens): ~200 tok/s
- TTFT (5-token prompt): ~80 ms

---

## 1. MLX (Apple)

**Repository:** github.com/ml-explore/mlx
**Transport:** Python (`mlx-lm`), Swift, C++
**Status:** Throughput leader on Apple Silicon. Apple's first-party framework.

### Performance Numbers

| Hardware       | Mem BW    | 8B Q4 tok/s | Notes                          |
|----------------|-----------|-------------|--------------------------------|
| M4 Max 40-core | 546 GB/s  | 95-110      | Fastest consumer Apple Silicon |
| M3 Max 40-core | 400 GB/s  | 66-80       | —                              |
| M2 Ultra 76-core| 800 GB/s | ~230        | Sustained (arXiv:2511.05502)   |
| M4 Pro 20-core | 273 GB/s  | ~40-50      | Community reports              |

### Larger Models

| Model                    | Hardware         | tok/s  | Quant  |
|--------------------------|------------------|--------|--------|
| Qwen 3 30B-A3B (MoE)    | M4 Max           | 100+   | 4-bit  |
| Llama 3.3 70B            | M4 Max 128GB     | 8-15   | Q4_K_M |
| Qwen 2.5 72B            | M2 Max 64GB      | 15-25  | 4-bit  |

### Key Characteristics
- 20-30% faster than llama.cpp consistently across model sizes.
- Metal-native kernels with quantized matmul specialization.
- M5 Neural Accelerator support (Nov 2025): up to 4x TTFT speedup,
  but only 1.2-1.3x decode improvement (bandwidth-limited).
- No built-in continuous batching — single-stream only.

### Vexel vs MLX Gap
MLX achieves ~230 tok/s on M2 Ultra vs Vexel's ~44 tok/s on comparable hardware.
The gap is primarily in Metal kernel optimization (MLX has Apple's internal
expertise) and model format (MLX uses its own optimized weight layout).

---

## 2. llama.cpp (Metal Backend)

**Repository:** github.com/ggerganov/ggml / ggml-org/llama.cpp
**Transport:** C/C++, extensive bindings
**Status:** Ecosystem standard. GGUF format originator.

### Performance Numbers (LLaMA 7B, full Metal offload)

| Chip         | GPU Cores | BW      | Q4_0 TG tok/s | F16 PP tok/s |
|--------------|-----------|---------|----------------|--------------|
| M4 Max       | 32        | 546 GB/s| 69.24          | 747.59       |
| M4 Pro       | 20        | 273 GB/s| 50.74          | 464.48       |
| M3 Max       | 40        | 400 GB/s| 66.31          | 779.17       |
| M2 Ultra     | 76        | 800 GB/s| 94.27          | 1401.85      |

(Source: github.com/ggml-org/llama.cpp/discussions/4167)

### Key Characteristics
- Universal model support via GGUF format.
- Strong community, frequent optimization PRs.
- No paged KV cache — performance degrades at long context.
- No built-in continuous batching.
- ~20-30% slower than MLX on Apple Silicon due to cross-platform abstractions.

### Vexel vs llama.cpp
llama.cpp's M3 Max Q4_0 decode (~66 tok/s) vs Vexel's ~44 tok/s.
Gap is in Metal kernel matmul throughput and memory access patterns.

---

## 3. MLC-LLM (TVM/Apache)

**Repository:** github.com/mlc-ai/mlc-llm
**Transport:** Python, REST API, C++ runtime
**Status:** Best long-context scaling due to paged KV cache.

### Performance Numbers (M2 Ultra, Qwen-2.5)

| Metric                  | Value     | vs MLX  |
|-------------------------|-----------|---------|
| Sustained decode tok/s  | ~190      | ~83%    |
| P99 latency             | ~13 ms    | Lowest  |
| Long-context (64K+)     | Stable    | Winner  |

### Key Characteristics
- **Paged KV cache** is the standout feature — all other frameworks degrade
  significantly at 64K+ context, MLC-LLM maintains throughput.
- TVM-compiled Metal kernels — good but not Apple-optimized like MLX.
- TTFT consistently lower than competitors for moderate prompt sizes.
- Less active community than llama.cpp or MLX.

### Relevance to Vexel
MLC-LLM's paged KV design is directly relevant to Vexel's Track 2
(Paged KV Batching). If Vexel implements paged KV properly, it should
match MLC-LLM's long-context behavior while potentially exceeding its
decode throughput via Go-scheduled continuous batching.

---

## 4. vllm-mlx

**Repository:** github.com/waybarrios/vllm-mlx
**Paper:** arXiv:2601.19139 (accepted EuroMLSys '26)
**Status:** First production-grade continuous batching on Apple Silicon.

### Single-Request Throughput

| Model              | tok/s | vs llama.cpp |
|--------------------|-------|--------------|
| Qwen3-0.6B        | ~525  | 1.87x        |
| Qwen3-8B          | —     | 1.21-1.40x   |
| Nemotron-30B (MoE) | —    | 1.43x        |

### Batched Throughput (M4 Max)

| Concurrent Requests | Aggregate Scaling |
|---------------------|-------------------|
| 1                   | 1.0x              |
| 4                   | ~2.5x             |
| 8                   | ~3.5x             |
| 16                  | 4.3x              |

### Prefix Caching

| Scenario                    | Speedup |
|-----------------------------|---------|
| Text prefix (shared prompts)| 5.8x   |
| Repeated image queries      | 28x    |
| Video analysis (64 frames)  | 24.7x  |
| Cached multimodal TTFT      | 21.7s → 0.78s |

### Key Characteristics
- OpenAI/Anthropic API compatible server.
- Continuous batching with prefix caching.
- Multimodal support (text, vision, audio, embeddings).
- Built on MLX backend — inherits MLX's Metal kernel performance.
- Tested on M4 Max 128GB.

### Relevance to Vexel
vllm-mlx is the most directly comparable competitor to Vexel's vision:
both aim to be production serving engines with continuous batching on
Apple Silicon. Vexel's Go scheduler + Metal backend vs vllm-mlx's Python +
MLX backend. Key differentiators for Vexel: Go's lower overhead, direct
Metal control, and simpler deployment (single binary).

---

## 5. vllm-metal

**Repository:** github.com/vllm-project/vllm-metal
**Status:** Official vLLM community plugin for Apple Silicon.

### Key Characteristics
- Under the vllm-project GitHub org (official).
- Continuous batching support, but text-only (no multimodal).
- Docker Model Runner integration for containerized macOS inference.
- Less performant than vllm-mlx in benchmarks.
- Newer and less mature than vllm-mlx.

---

## 6. Ollama

**Repository:** github.com/ollama/ollama
**Status:** UX baseline, not a performance target.

- Wraps llama.cpp with a user-friendly CLI and API.
- 20-40 tok/s on consumer Apple Silicon.
- Focus is on ease-of-use, not raw performance.
- Useful as a "floor" benchmark — any serious engine should exceed this.

---

## Memory Bandwidth Reference

| Chip         | Memory BW  | Max Unified Memory |
|--------------|------------|-------------------|
| M2 Ultra     | 800 GB/s   | 192 GB            |
| M4 Max       | 546 GB/s   | 128 GB            |
| M3 Max       | 400 GB/s   | 128 GB            |
| M4 Pro       | 273 GB/s   | 48 GB             |
| M5 (base)    | 153 GB/s   | TBD               |
| M4 (base)    | 120 GB/s   | 32 GB             |

**Theoretical decode ceiling** for 8B Q4 model (~4.3 GB weights):
- M2 Ultra: `800 / 4.3 ≈ 186 tok/s` (MLX achieves ~230, suggesting optimization beyond naive bandwidth)
- M4 Max: `546 / 4.3 ≈ 127 tok/s` (MLX achieves 95-110, ~80% utilization)
- M3 Max: `400 / 4.3 ≈ 93 tok/s` (MLX achieves 66-80, ~75% utilization)

---

## Sources

1. arXiv:2511.05502 — "Production-Grade Local LLM Inference on Apple Silicon" (Nov 2025)
2. arXiv:2601.19139 — "Native LLM and MLLM Inference at Scale on Apple Silicon" (Jan 2026)
3. github.com/ggml-org/llama.cpp/discussions/4167 — Apple Silicon performance tracking
4. machinelearning.apple.com/research/exploring-llms-mlx-m5 — M5 Neural Accelerator benchmarks
5. github.com/waybarrios/vllm-mlx — vllm-mlx repository
6. github.com/vllm-project/vllm-metal — vllm-metal repository
7. arXiv:2510.18921 — "Benchmarking On-Device ML on Apple Silicon with MLX"
8. arXiv:2508.08531 — "Profiling LLM Inference on Apple Silicon: A Quantization Perspective"

---

## Benchmark Plan

Based on this research, the benchmark harness should measure:

1. **Decode tok/s** — primary metric, memory-bandwidth-bound
2. **Prefill tok/s** — compute-bound, benefits from batched quantized matmul
3. **TTFT** — user-perceived latency
4. **ITL P50/P99** — consistency/jitter
5. **Memory bandwidth utilization %** — `(model_bytes / per_token_time) / hw_bandwidth`
6. **Peak RSS** — memory efficiency
7. **Concurrent throughput scaling** — Vexel vs vllm-mlx at 1/2/4/8/16 clients
8. **Long-context degradation** — 4K/8K/16K/32K context lengths

Engines to benchmark: **Vexel, MLX (mlx-lm), llama.cpp (Metal), vllm-mlx**.
Optional: MLC-LLM (if long-context comparison needed), Ollama (UX baseline).
