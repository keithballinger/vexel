# Benchmark Results & Optimization Roadmap

> Hardware: Apple M3 Max, 128 GB Unified Memory, 400 GB/s bandwidth
> Model: LLaMA 2 7B Q4_0 (3.56 GB)
> Last updated: 2026-02-26 (post P0+P1)

## Decode Throughput

| Engine     | tok/s | ±stddev | BW Util % | vs llama.cpp |
|------------|-------|---------|-----------|--------------|
| llama.cpp  | 78.13 | 0.60    | 84.0%     | baseline     |
| Ollama     | 81.23 | 2.78    | 87.3%     | +4.0%        |
| MLX        | ~80*  | —       | ~86%      | ~+2%         |
| **Vexel**  | **8.3** | **0.04** | **8.9%** | **-89.4%** |

*MLX estimated from published benchmarks (HF auth required for direct test)

**Note:** Vexel 8.3 tok/s is measured with verbose timing which uses greedy sampling.
Previous measurement of 43.38 tok/s was from the original benchmark harness which
measures wall-clock inclusive of all overhead. The llama.cpp numbers are pure eval
time reported by the engine itself. Vexel's internal decode loop includes per-token
overhead from Go runtime, pool management, and synchronization that llama.cpp avoids
by batching C++ internally. The gap is primarily in the matmul kernel efficiency and
memory bandwidth utilization, not allocation overhead.

## Prefill Throughput

| Engine     | ~12 tok  | ~124 tok  | ~385 tok  |
|------------|----------|-----------|-----------|
| llama.cpp  | 225      | 792       | 822       |
| **Vexel**  | **41**   | **152**   | **107**   |

Post-P0+P1: Prefill works at all lengths without OOM. Vexel prefill is
5.2-7.7x slower than llama.cpp. The gap widens at longer sequences,
suggesting Vexel's matmul kernels are less efficient at larger batch sizes.

Historical comparison:
- Pre-P0: OOM'd on prompts >20 tokens
- Post-P0: ~137 tok/s at 20 tokens (estimated)
- Post-P0+P1: 41-152 tok/s (measured directly, varies by sequence length)

## Model Load Time

| Engine     | Cold start |
|------------|-----------|
| llama.cpp  | ~275 ms   |
| Vexel      | ~1335 ms  |

Vexel model load is ~4.9x slower than llama.cpp. Likely due to individual
tensor allocation vs mmap.

---

## Root Cause Analysis

The 44.7% decode throughput gap traces to six architectural bottlenecks
in Vexel's Metal execution path.

### 1. ~~Per-Layer GPU Allocation Instead of Scratch Sub-allocation~~ ✅ FIXED (commit 675b962)

GPU decode path now uses bump allocation from a single pre-allocated MTLBuffer.
13 offset-aware C bridge functions added for all hot-path Metal kernels.
Auto-detect dispatches to offset-aware path when `DevicePtr.Offset() != 0`.

**Measured impact:** within noise (~0-3%) for single-stream decode. The pool
allocator was already efficient. Primary benefit is reduced fragmentation
under concurrent load.

### 2. Split KV Cache with Double Scatter

**File:** `inference/runtime/gpu_kv_cache.go:25-45`

KV cache uses separate K and V buffers per layer (64 total for 32 layers).
Each decode step dispatches **2 scatter kernels per layer** (one for K, one
for V) instead of a single fused scatter.

**Impact:** 64 extra kernel dispatches per token. Each dispatch has ~0.5-1µs
of command buffer overhead on M3 Max.

**Fix:** Interleave K/V in a single buffer per layer: `[K0, V0, K1, V1, ...]`.
Single scatter kernel writes both K and V in one dispatch.

### 3. No Kernel Fusion

**File:** `inference/runtime/block.go:556-569`

Attention, RoPE, layer norm, and residual add are separate kernel dispatches.
llama.cpp fuses attention+norm+residual into fewer dispatches, reducing
memory I/O by ~30%.

**Impact:** Each separate kernel reads/writes intermediate results to GPU
memory. Fused kernels keep intermediates in registers/threadgroup memory.

**Fix:** Create fused Metal kernels:
- `fused_sdpa_rope` — attention with rotary embeddings
- `fused_norm_residual` — RMSNorm + residual add

### 4. Command Buffer Serialization

**File:** `inference/runtime/block.go:835` — "Batching disabled - causes
incorrect output due to Metal memory hazards"

Vexel relies on Metal's implicit command serialization. Each of the 7
operations per layer is a separate command buffer submission. llama.cpp
batches operations into fewer, larger command buffers.

**Impact:** At 32 layers × 7 ops, minimum 224 command submissions per token.
M3 Max command submission latency ≈ 0.5µs → ~112µs overhead per token
(~0.5% of the 23ms per-token budget, but cascading with allocation overhead).

**Fix:** Use Metal compute command encoders with explicit resource
dependencies (MTLFence) instead of separate command buffers. Encode all
ops for a layer in a single command buffer.

### 5. ~~Scratch Arena Sizing Bug~~ ✅ FIXED (commit 2966edd)

The arena budget formula in `initModel` did not account for token ID and
hidden-state allocations that `DecodeWithGPUKV` makes from the arena.
With default max-tokens=64, any prompt longer than ~9 tokens would OOM.

**Fix:** Added `TotalArenaBytes(maxBatchSize)` to `ModelConfig` that correctly
budgets all four arena allocations (tokens + hidden state + layer scratch +
logits) with 10% headroom. Arena now sized for `max(maxContextLen=2048, maxTokens)`.

### 6. F32→F16 Conversion Overhead

**File:** `inference/runtime/gpu_kv_cache.go:163-177`

When the KV cache uses FP16 but the model computes in F32, an extra
conversion kernel (`ScatterKVF32ToF16`) runs per layer. This is a separate
dispatch that could be fused into the scatter kernel itself.

---

## Optimization Roadmap (Priority Order)

| Priority | Fix | Expected Impact | Status |
|----------|-----|----------------|--------|
| ~~P0~~ | ~~Fix scratch arena sizing~~ | Prefill works at all lengths | ✅ DONE (2966edd) |
| ~~P1~~ | ~~GPU scratch sub-allocation~~ | Within noise (pool was efficient) | ✅ DONE (675b962) |
| **P2** | Fused KV scatter (single dispatch) | +5-8% decode throughput | Open |
| **P3** | Fused attention+norm kernels | +10-15% decode throughput | Open |
| **P4** | Command buffer batching | +3-5% decode throughput | Open |
| **P5** | F32→F16 fusion in scatter | +1-2% decode throughput | Open |

**Key finding:** The allocation overhead (P1) was not the primary bottleneck.
The dominant gap is in the matmul kernel efficiency and memory bandwidth
utilization — Vexel achieves ~9% of M3 Max theoretical bandwidth vs
llama.cpp's ~84%. The remaining P2-P5 optimizations target kernel dispatch
overhead, but the core performance gap requires matmul kernel tuning:
- Quantized matmul kernel tuning (tiling, threadgroup sizing for M3 Max)
- Memory access pattern optimization (ensure coalesced reads)
- Pipeline overlap (prefetch next layer's weights while current layer computes)

## Vexel's Competitive Advantages (Not Captured in Single-Stream)

The decode throughput gap is real, but Vexel has architectural advantages
that don't show up in single-stream benchmarks:

1. **Go scheduler with continuous batching** — multi-client throughput
   scaling that llama.cpp can't do natively
2. **gRPC + HTTP dual protocol** — production serving with streaming
3. **Single binary deployment** — no Python dependency chain
4. **Paged KV cache** — long-context efficiency (once OOM is fixed)

The path forward: fix the P0-P2 issues to close the single-stream gap,
then leverage the batching architecture for the real competitive advantage.

## Raw Data

- Decode: `benchmarks/results/decode_20260226/`
- Prefill: `benchmarks/results/prefill_20260226/`
- Post-P0 combined: `benchmarks/results/post_p0_benchmark.json`
- Analysis: `benchmarks/results/decode_20260226/summary.json`
