# Benchmark Results & Optimization Roadmap

> Hardware: Apple M3 Max, 128 GB Unified Memory, 400 GB/s bandwidth
> Model: LLaMA 2 7B Q4_0 (3.56 GB)
> Date: 2026-02-26

## Decode Throughput

| Engine     | tok/s | ±stddev | BW Util % | vs llama.cpp |
|------------|-------|---------|-----------|--------------|
| llama.cpp  | 78.45 | 0.19    | 84.3%     | baseline     |
| Ollama     | 81.23 | 2.78    | 87.3%     | +3.5%        |
| MLX        | ~80*  | —       | ~86%      | ~+2%         |
| **Vexel**  | **43.38** | **0.30** | **46.6%** | **-44.7%** |

*MLX estimated from published benchmarks (HF auth required for direct test)

## Prefill Throughput

| Engine     | 20 tok  | 128 tok | 512 tok |
|------------|---------|---------|---------|
| llama.cpp  | ~480    | 700     | 835     |
| **Vexel**  | **137** | **hangs** | **hangs** |

Vexel OOMs on prompts >20 tokens: `GPU prefill failed: OOM: requested 128000 bytes, only 53232 remaining`

## Model Load Time

| Engine     | Cold start |
|------------|-----------|
| llama.cpp  | ~288 ms   |
| Vexel      | blocked (OOM on single-token gen) |

---

## Root Cause Analysis

The 44.7% decode throughput gap traces to six architectural bottlenecks
in Vexel's Metal execution path.

### 1. Per-Layer GPU Allocation Instead of Scratch Sub-allocation

**File:** `inference/runtime/block.go:507-531`

Vexel allocates **7 separate Metal buffers per layer** during GPU execution:
```
normOut, Q, K, V, attnOut, gate, up = 7 individual backend.Alloc() calls
```

A `ScratchAllocator` exists (`backend/metal/scratch_allocator.go`) that does
single-buffer sub-allocation with 256-byte alignment — but it's **only used
for CPU**. The GPU path falls through to individual allocations because
"kernels can't take buffer+offset" (comment at block.go:740).

**Impact:** 32 layers × 7 allocs = 224 Metal buffer allocations per token.
Each allocation requires driver overhead and breaks memory coalescing.
llama.cpp uses a single pre-allocated scratch with sub-offsets.

**Fix:** Port the offset-based buffer binding to Metal kernels. Metal
`setBuffer:offset:atIndex:` supports offsets natively — this is a kernel
argument change, not an architectural one.

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

### 5. Scratch Arena Sizing Bug (Causes OOM)

**File:** `inference/runtime/config.go:273-307`

The scratch buffer scores component is sized as `seqLen × seqLen`, which
is correct for prefill but oversized for decode (where seqLen=1). Combined
with a fixed arena that doesn't account for the difference, longer prompts
exhaust the scratch arena.

**Error:** `GPU prefill failed: OOM: requested 128000 bytes, only 53232 remaining`

**Fix:** Dynamic scratch sizing: `scores = seqLen × kvLen` for prefill,
`scores = 1 × kvLen` for decode. Pre-allocate for the max case at init.

### 6. F32→F16 Conversion Overhead

**File:** `inference/runtime/gpu_kv_cache.go:163-177`

When the KV cache uses FP16 but the model computes in F32, an extra
conversion kernel (`ScatterKVF32ToF16`) runs per layer. This is a separate
dispatch that could be fused into the scatter kernel itself.

---

## Optimization Roadmap (Priority Order)

| Priority | Fix | Expected Impact | Effort |
|----------|-----|----------------|--------|
| **P0** | Fix scratch arena sizing (OOM bug) | Unblocks prefill >20 tokens | Small |
| **P1** | Enable GPU scratch sub-allocation | +15-20% decode throughput | Medium |
| **P2** | Fused KV scatter (single dispatch) | +5-8% decode throughput | Medium |
| **P3** | Fused attention+norm kernels | +10-15% decode throughput | Large |
| **P4** | Command buffer batching | +3-5% decode throughput | Medium |
| **P5** | F32→F16 fusion in scatter | +1-2% decode throughput | Small |

**Estimated total recovery: 34-50%** → would bring Vexel from 43.4 to
~58-65 tok/s, closing the gap to within 15-20% of llama.cpp.

To match or exceed llama.cpp (78.5 tok/s), additional work on:
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
- Analysis: `benchmarks/results/decode_20260226/summary.json`
