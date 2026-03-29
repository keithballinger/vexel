# Benchmark Results & Optimization Roadmap

> Hardware: Apple M4 Max, 128 GB Unified Memory, 40 GPU cores, Metal 4
> Model: LLaMA 3.1 8B Instruct Q4_K_M (4.9 GB) and Qwen 2.5 0.5B Q4_K_M (0.4 GB)
> llama.cpp: b8534 (e99d77fa4)
> Last updated: 2026-03-29 (Updated after Medusa fixes, paged KV bugfixes, adaptive speculation)

## Current Results (2026-03-29, M4 Max)

### Standard Decode (LLaMA 3.1 8B Q4_K_M, 128 gen tokens, greedy)

| Engine | Decode tok/s | vs llama.cpp |
|--------|-------------|--------------|
| **Vexel** | **65.9** | **+29%** |
| llama.cpp | 51.0 | baseline |

### Standard Decode (TinyLlama 1.1B Q4_0, 128 gen tokens, greedy)

| Engine | Decode tok/s | vs llama.cpp |
|--------|-------------|--------------|
| **Vexel** | **343.8** | **+6%** |
| llama.cpp | 323.3 | baseline |

### Context Scaling (LLaMA 3.1 8B, decode tok/s)

| Context | Vexel | Vexel Degrad. | llama.cpp | llama.cpp Degrad. |
|---------|-------|--------------|-----------|------------------|
| 16 | 66.6 | baseline | 51.5 | baseline |
| 64 | 66.7 | -0.2% | 48.6 | 5.7% |
| 256 | 64.1 | 3.8% | 49.8 | 3.3% |
| 512 | 63.7 | 4.3% | 44.8 | 12.9% |
| 1024 | 62.9 | 5.5% | 47.3 | 8.1% |
| 2048 | — | (KV limit) | 43.1 | 16.2% |
| 4096 | — | (KV limit) | 35.8 | 30.5% |

Vexel maintains <6% degradation up to ctx=1024 thanks to adaptive SDPA dispatch
(tiled split-K for kvLen>2048, NWG adaptive tiling for 64-2048, v3 chunk-based fallback).

### Medusa Speculative Decode (LLaMA 3.1 8B)

Medusa heads are trained online during inference. The scheduler uses adaptive
speculation: starts with normal decode, probes head accuracy every 32 steps,
and enables speculation only when heads achieve >= 50% acceptance rate.

With online-trained heads (600-token warmup), current acceptance rate is low
(~0-5%) due to limited training (10-17 steps, loss 6-9 vs random 10.4). The
server falls back to non-speculative decode for reliable output.

Pre-trained Medusa heads (loaded via `--medusa-heads`) are expected to achieve
higher acceptance and provide actual speedup.

### Server Decode (LLaMA 3.1 8B, single client)

| Engine | Decode tok/s | Notes |
|--------|-------------|-------|
| Vexel (GPU KV) | ~29 | Default for single/few clients |
| Vexel (paged KV) | ~10 | With `--context-len`, needed for Medusa |
| llama.cpp | ~43 | |

Server mode has inherent overhead (HTTP, scheduler loop, token streaming)
vs raw generate mode (66 tok/s). Vexel uses GPU KV cache by default for
serve; paged KV is used when `--context-len` or `--medusa` is specified.

Multi-client batched throughput (with paged KV, `--context-len 2048`)
scales with concurrency but at lower per-token rates due to the paged
attention overhead. Optimizing the paged KV decode path is a priority.

---

## Historical Results (2026-02-28, M3 Max, LLaMA 2 7B Q4_0)

### Decode Throughput

| Engine     | tok/s | BW Util % | vs llama.cpp |
|------------|-------|-----------|--------------|
| llama.cpp  | 76.30 | 82.0%     | baseline     |
| **Vexel**  | **64.8** | **69.7%** | **-15.1%** |

Historical decode performance:
- Pre-P0+P1: 8.3 tok/s (per-dispatch sync, no batching)
- Pre-batching: 7.7 tok/s (per-layer sync removed, same kernel)
- **Post-batching: 64.8 tok/s (8.4x speedup, memory barriers)**

### Context Scaling (Before/After Flash SDPA)

| Context | Before Flash SDPA | After Flash SDPA | llama.cpp | Degrad (before) | Degrad (after) |
|---------|-------------------|------------------|-----------|----------------|----------------|
| 16      | 64.8 tok/s        | 64.8 tok/s       | 77.3      | baseline       | baseline       |
| 64      | 62.0              | —                | 76.7      | -4.3%          | ~-9%           |
| 128     | 59.7              | —                | 77.2      | -7.9%          | ~-4%           |
| 256     | 55.2              | —                | 75.4      | -14.8%         | ~-10%          |
| 512     | 48.9              | —                | 75.2      | -24.5%         | ~-10%          |

### Prefill Throughput

| Engine     | 5 tok | 32 tok | 128 tok | 385 tok |
|------------|-------|--------|---------|---------|
| llama.cpp  | 110   | 582    | 803     | 793     |
| **Vexel**  | **97**| **400**| **717** | **637** |

Per-dimension GEMM GFLOPS (M=128):
| Dimension   | Before | After  | Improvement |
|-------------|--------|--------|-------------|
| 4096×4096   | 3,863  | 6,500  | **+68%**    |
| 11008×4096  | 6,928  | 9,000  | **+30%**    |
| 4096×11008  | 6,077  | 8,382  | **+38%**    |
| 32000×4096  | 7,924  | 10,651 | **+34%**    |

Historical comparison:
- Pre-P0: OOM'd on prompts >20 tokens
- Post-P0: ~137 tok/s at 20 tokens (estimated)
- Post-P0+P1: 41-152 tok/s (measured directly, varies by sequence length)
- Post-batching: 96-200 tok/s (2x improvement across all lengths)
- Post-P9 Phase 1: 94-377 tok/s (1.9x improvement at seqLen=128)
- **Post-prefill campaign: 97-717 tok/s (90% improvement at seqLen=128)**

## Q4_K_M Performance (LLaMA 2 7B, 4.08 GB)

Q4_K uses 256-element super-blocks with 6-bit packed scales/mins and 8
sub-blocks of 32 elements each. Q4_K_M models mix 193 Q4_K tensors with
33 Q6_K tensors (v_proj and down_proj) plus F32 norms.

### Q4_K Kernel Optimization Results

Custom sub-block strided decode kernel and simdgroup tiled prefill GEMM
closed the Q4_K performance gap from 3-9x slower than Q4_0 to within 16-21%.

**Decode (sub-block strided matvec):**

| Engine | Q4_0 | Q4_K_M | Q4_K/Q4_0 | Before Optimization |
|--------|------|--------|-----------|---------------------|
| Vexel  | 62.9 tok/s | 52.8 tok/s | 0.84x | 14.2 tok/s (0.36x) |

Key techniques: each SIMD lane independently processes 32-element sub-blocks
with stride-32 (128 sub-blocks for K=4096, 4 per lane), hardware fp16 via
`as_type<half>`, selective scale/min extraction, uint vector loads. **3.7x
speedup** over scalar baseline.

**Prefill (simdgroup tiled GEMM):**

| seqLen | Q4_0 | Q4_K_M | Q4_K/Q4_0 | Before Optimization |
|--------|------|--------|-----------|---------------------|
| 5      | 94.8 tok/s  | 26.5 tok/s  | 0.28x | 16.6 tok/s (0.32x) |
| 32     | 339.1 tok/s | 217.9 tok/s | 0.64x | — |
| 128    | 377.6 tok/s | 220.4 tok/s | 0.58x | 16.6 tok/s (0.11x) |
| 385    | 223.7 tok/s | 157.9 tok/s | 0.71x | 15.9 tok/s (0.15x) |

Key techniques: half-precision shared memory (`threadgroup half*`) with
TILE_K=64 (2 Q4_K sub-blocks per iteration), block-based B dequant with
correct interleaved nibble addressing, 8 simdgroups in 2×4 layout, 12KB
threadgroup memory, hardware 8×8 matrix multiply via `simdgroup_multiply_accumulate`.
**13.3x speedup** at 128 tokens over batched scalar baseline.

**Context scaling (Q4_K_M decode):**

| Context | tok/s | Degradation |
|---------|-------|-------------|
| 16      | 53.5  | baseline    |
| 64      | 53.3  | -0.4%       |
| 128     | 52.5  | -1.9%       |
| 256     | 50.8  | -5.0%       |
| 512     | 47.7  | -10.8%      |

Note: seqLen=5 prefill uses batched NR2 kernel (M<8 threshold), not the
simdgroup kernel. The seqLen=5 Q4_K gap is a known limitation of the
scalar NR2 fallback path.

## Model Load Time

| Engine     | Cold start |
|------------|-----------|
| llama.cpp  | ~1100 ms  |
| Vexel      | ~885 ms   |

Vexel model load (GGUF parse → GPU alloc → weight copy → KV cache init)
is ~20% faster than llama.cpp full process start (process + Metal init +
mmap + warmup). Both measured as wall-clock time for a complete cold start
to first-token readiness.

---

## Root Cause Analysis

### 1. ~~Per-Layer GPU Allocation~~ ✅ FIXED (commit 675b962)

GPU decode path now uses bump allocation from a single pre-allocated MTLBuffer.
**Measured impact:** within noise (~0-3%) — pool allocator was already efficient.

### 2. ~~Split KV Cache~~ — DEPRIORITIZED

64 extra scatter dispatches per token. Post-batching, each costs ~0.5µs
encoding overhead = 32µs total. Below noise floor.

### 3. ~~Kernel Fusion~~ — DEPRIORITIZED

`TestFusionABComparison` proved dispatch reduction (451→387, 14%) has NO
measurable throughput impact. Two fusions already implemented (AddRMSNorm,
FusedMLP). Remaining fusion targets (RoPE into matmul) save only ~4MB
memory I/O vs ~3.56GB weight reads per token.

### 4. ~~Command Buffer Serialization~~ ✅ FIXED (commit 4fc581c)

Command buffer batching with `memoryBarrierWithScope:MTLBarrierScopeBuffers`
between dependent dispatches. **8.4x decode speedup (7.7 → 64.8 tok/s).**

### 5. ~~Scratch Arena Sizing~~ ✅ FIXED (commit 2966edd)

Arena budget formula fixed. Prefill works at all sequence lengths.

### 6. F32→F16 Conversion Overhead — LOW PRIORITY

Extra `ScatterKVF32ToF16` dispatch per layer. Could be fused into scatter
kernel. Expected impact: +1-2%.

---

## Optimization Roadmap (Priority Order)

| Priority | Fix | Expected Impact | Status |
|----------|-----|----------------|--------|
| ~~P0~~ | ~~Fix scratch arena sizing~~ | Prefill works at all lengths | ✅ DONE (2966edd) |
| ~~P1~~ | ~~GPU scratch sub-allocation~~ | Within noise (pool was efficient) | ✅ DONE (675b962) |
| ~~P4~~ | ~~Command buffer batching~~ | **8.4x decode speedup** | ✅ DONE (4fc581c) |
| ~~P2~~ | ~~Fused KV scatter~~ | ~~+5-8%~~ Negligible post-batching | ⏭️ SKIPPED |
| ~~P3~~ | ~~Fused attention+norm~~ | ~~+10-15%~~ No measurable impact | ⏭️ SKIPPED |
| **P5** | F32→F16 fusion in scatter | +1-2% decode throughput | Open |
| ~~P6~~ | ~~Q4_0 matmul kernel optimization~~ | **90% prefill speedup** | ✅ DONE (cd22b51) |
| ~~P7~~ | ~~SDPA/KV access for long context~~ | **-24.5% → ~-10% ctx degradation** | ✅ DONE (334a491, a33e9c8) |
| ~~P8~~ | ~~Q4_K kernel optimization~~ | **3.7x decode, 9.5x prefill** | ✅ DONE (3967f85, b9916aa) |
| ~~P9~~ | ~~Prefill pipeline optimization~~ | **1.9x Q4_0, 1.4x Q4_K prefill** | ✅ DONE (f4dd160, eca90d3) |
| ~~P10~~ | ~~Close prefill gap~~ | **377→717 tok/s (1.12× gap)** | ✅ DONE (987c252, cd22b51) |

**Key finding:** The dominant bottleneck was per-dispatch synchronization.
`finish_encode()` called `[cmdBuf waitUntilCompleted]` on every kernel
dispatch (~320 per token). Command buffer batching reduced this to ~32
waits per token (one per layer). This single change closed the decode gap
from -89.4% to -15.1%.

**Remaining gaps and known issues:**
- **Paged KV decode speed:** Paged KV attention path (~10 tok/s) is significantly
  slower than GPU KV cache (~29 tok/s serve, ~66 tok/s generate). Needed for Medusa
  and multi-client serving. Likely needs a dedicated paged decode Metal kernel.
- **Qwen 0.5B paged KV crash:** Known SIGSEGV with Qwen 0.5B in paged KV mode.
  Not a production target but blocks benchmarking.
- **Medusa speculation quality:** Online-trained heads need more training time or
  pre-trained weights to achieve useful acceptance rates. Adaptive speculation
  prevents garbage output but provides no speedup currently.
- ~~**Server batching:**~~ **FIXED.** HTTP server now uses GPU KV cache by default
  for fast single-client decode. Paged KV only used when explicitly requested.
- ~~**Context scaling:**~~ **FIXED.** Adaptive SDPA dispatch. 5.5% at ctx=1024.
- ~~**Q4_K kernels:**~~ **FIXED.** Sub-block strided decode + simdgroup GEMM prefill.
- ~~**Prefill vs llama.cpp:**~~ **CLOSED.** 7% faster on 8B model.

## Vexel's Competitive Advantages

On the M4 Max, Vexel is now **28% faster** than llama.cpp on 8B model decode,
with better context scaling (<6% degradation at ctx=1024 vs llama.cpp's 8%+).

1. **28% faster decode** on LLaMA 3.1 8B Q4_K_M (66.0 vs 51.6 tok/s)
2. **Superior context scaling** — 5.5% degradation at ctx=1024 vs llama.cpp's 8.1%
3. **Speculative decoding** — Medusa (online-learned) and draft-model speculation
4. **Batched decode with paged KV** — true multi-sequence batched forward pass
5. **Go scheduler with continuous batching** — multi-client serving
6. **gRPC + HTTP dual protocol** — production serving with streaming
7. **Single binary deployment** — no Python dependency chain
8. **Faster cold start** — model loads ~20% faster than llama.cpp

## MLX Comparison

| Metric | Vexel Q4_0 | Vexel Q4_K_M | MLX (Mistral 4-bit) | Gap (Q4_K_M vs MLX) |
|--------|-----------|-------------|---------------------|-----|
| Decode (short ctx) | 64.8 tok/s | 53.5 tok/s | 83.5 tok/s | -36% |
| Context degradation (16→512) | ~-10% | -10.8% | -2.5% | ~8pp |
| Prefill 128 tok | 717 tok/s | 220.4 tok/s | 725 tok/s | -70% (Q4_K_M) |

With Q4_K_M kernel optimization, Vexel now supports the higher-quality
quantization format (256-element super-blocks with 6-bit scales/mins).
Q4_K_M decode is currently 16% slower than Q4_0 due to the 33 Q6_K
tensors using unoptimized NR2 kernels (v_proj, down_proj, lm_head).
The Q4_K matmul kernels themselves are well-optimized — the remaining
gap is in Q6_K dispatch and the larger block overhead.

## Raw Data

- **2026-03-28 comprehensive:** `benchmarks/results/2026-03-28/` (standard, speculative, batched, context scaling)
- 2026-02-26 decode: `benchmarks/results/decode_20260226/`
- 2026-02-26 prefill: `benchmarks/results/prefill_20260226/`
- Post-P0 combined: `benchmarks/results/post_p0_benchmark.json`
- Kernel benchmarks: `inference/runtime/*_test.go` and `inference/backend/metal/*_test.go`

Run the full comparison suite:
```bash
cd benchmarks && ./full_comparison.sh all
```
