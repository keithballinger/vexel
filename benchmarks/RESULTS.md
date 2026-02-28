# Benchmark Results & Optimization Roadmap

> Hardware: Apple M3 Max, 128 GB Unified Memory, 400 GB/s bandwidth
> Model: LLaMA 2 7B Q4_0 (3.56 GB)
> llama.cpp: b8140 (39fb81f87)
> Last updated: 2026-02-27 (P9 Phase 1: half-precision GEMM with block dequant)

## Decode Throughput

| Engine     | tok/s | ±stddev | BW Util % | vs llama.cpp |
|------------|-------|---------|-----------|--------------|
| llama.cpp  | 76.30 | 0.26    | 82.0%     | baseline     |
| **Vexel**  | **64.8** | **0.7** | **69.7%** | **-15.1%** |

Measured at short context (~8 tokens prefilled). Vexel generates 50 tokens;
llama.cpp generates 200 tokens. Both use greedy decoding (temperature=0).

**Post-batching update:** Command buffer batching with memory barriers
eliminated ~320 per-dispatch `waitUntilCompleted` CPU-GPU roundtrips per
token. The remaining ~15% gap to llama.cpp is primarily in Q4_0 matmul
kernel bandwidth utilization (69.7% vs 82.0%).

Historical decode performance:
- Pre-P0+P1: 8.3 tok/s (per-dispatch sync, no batching)
- Pre-batching: 7.7 tok/s (per-layer sync removed, same kernel)
- **Post-batching: 64.8 tok/s (8.4x speedup, memory barriers)**

## Decode Throughput vs Context Length

| Context | Before Flash SDPA | After Flash SDPA | llama.cpp | Degradation (before) | Degradation (after) |
|---------|-------------------|------------------|-----------|---------------------|---------------------|
| 16      | 64.8 tok/s        | 64.8 tok/s*      | 77.3      | baseline            | baseline            |
| 64      | 62.0              | —                | 76.7      | -4.3%               | ~-9%                |
| 128     | 59.7              | —                | 77.2      | -7.9%               | ~-4%                |
| 256     | 55.2              | —                | 75.4      | -14.8%              | ~-10%               |
| 512     | 48.9              | —                | 75.2      | -24.5%              | ~-10%               |

*After Flash SDPA absolute numbers not yet measured in a clean environment.
Degradation percentages from contended run (relative measurements are reliable).

**Flash Attention SDPA (commits 334a491, a33e9c8):** Replaced the materialized
attention kernel (`sdpa_decode_f16`) with a tiled Flash Attention implementation
using split-KV processing and online softmax. Threadgroup memory usage dropped
from O(kvLen) to O(headDim). Context degradation improved from **-24.5% to ~-10%**
(ctx=16 → ctx=512). Enabled at all context lengths (no minimum threshold).

## Prefill Throughput

| Engine     | 5 tok | 32 tok | 128 tok | 385 tok |
|------------|-------|--------|---------|---------|
| llama.cpp  | 110   | 582    | 803     | 793     |
| **Vexel**  | **94**| **337**| **377** | **223** |

**P9 Phase 1 (commit f4dd160):** Half-precision GEMM with block-based
dequantization delivered **1.9-2.3x prefill speedup** at batch sizes ≥32.
Gap to llama.cpp narrowed from 4-5x to **1.7-3.6x**. Isolated GEMM
throughput improved 2.62x (2033→5327 GFLOPS at M=128, N=K=4096).

Key techniques: `threadgroup half*` shared memory, TILE_K=64 (2 Q4_0 blocks
per iteration), block-based B dequant (one scale load per 32 values),
`simdgroup_half8x8` loads with `simdgroup_float8x8` accumulators.

Historical comparison:
- Pre-P0: OOM'd on prompts >20 tokens
- Post-P0: ~137 tok/s at 20 tokens (estimated)
- Post-P0+P1: 41-152 tok/s (measured directly, varies by sequence length)
- Post-batching: 96-200 tok/s (2x improvement across all lengths)
- **Post-P9 Phase 1: 94-377 tok/s (1.9x improvement at seqLen=128)**

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
| 5      | 96.2 tok/s  | 26.7 tok/s  | 0.28x | 16.6 tok/s (0.32x) |
| 32     | 145.2 tok/s | 132.3 tok/s | 0.91x | — |
| 128    | 200.2 tok/s | 157.6 tok/s | 0.79x | 16.6 tok/s (0.11x) |
| 385    | 151.7 tok/s | 123.5 tok/s | 0.81x | 15.9 tok/s (0.15x) |

Key techniques: ported `matmul_q4_0_simdgroup_f32` to Q4_K with TILE_K=32
aligned to sub-block size, 8 simdgroups in 2×4 layout, 12KB threadgroup
memory, hardware 8×8 matrix multiply via `simdgroup_multiply_accumulate`.
**9.5x speedup** at 128 tokens over batched scalar baseline.

**Context scaling (Q4_K_M decode):**

| Context | tok/s | Degradation |
|---------|-------|-------------|
| 16      | 54.3  | baseline    |
| 64      | 53.9  | -0.7%       |
| 128     | 53.2  | -2.0%       |
| 256     | 51.8  | -4.6%       |
| 512     | 48.4  | -10.9%      |

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
| **P6** | Q4_0 matmul kernel optimization | Close remaining ~15% gap | Investigated (NR4 not beneficial) |
| ~~P7~~ | ~~SDPA/KV access for long context~~ | **-24.5% → ~-10% ctx degradation** | ✅ DONE (334a491, a33e9c8) |
| ~~P8~~ | ~~Q4_K kernel optimization~~ | **3.7x decode, 9.5x prefill** | ✅ DONE (3967f85, b9916aa) |
| **P9** | Prefill pipeline optimization | **1.9x prefill (Phase 1)** | Phase 1 ✅ (f4dd160) |

**Key finding:** The dominant bottleneck was per-dispatch synchronization.
`finish_encode()` called `[cmdBuf waitUntilCompleted]` on every kernel
dispatch (~320 per token). Command buffer batching reduced this to ~32
waits per token (one per layer). This single change closed the decode gap
from -89.4% to -15.1%.

**Remaining gaps:**
- **Decode (-15%):** Q4_0 matmul kernel bandwidth utilization (69.7% vs
  llama.cpp's 82.0%). NR4 kernel investigated but not beneficial due to
  Q4_0's 18-byte block layout causing L1 cache pressure.
- ~~**Context scaling:**~~ **FIXED.** Flash Attention SDPA reduced context
  degradation from -24.5% to ~-10% (ctx=16→512). Uses split-KV tiled
  processing with online softmax, O(headDim) threadgroup memory.
- ~~**Q4_K kernels:**~~ **FIXED.** Sub-block strided decode (3.7x) and
  simdgroup tiled GEMM prefill (9.5x) closed the Q4_K gap from 3-9x
  slower to within 16-21% of Q4_0. Q4_K_M decode: 52.8 tok/s, prefill
  128: 157.6 tok/s.
- **Prefill vs llama.cpp (2-4x):** P9 Phase 1 narrowed the gap from
  4-5x to 1.7-3.6x via half-precision GEMM with block dequant (2.62x
  isolated GEMM speedup). Remaining gap likely in cross-layer CB overhead,
  FA2 prefill utilization, and fused QKV projections.

## Vexel's Competitive Advantages (Not Captured in Single-Stream)

The remaining decode gap (~15%) puts Vexel in competitive range for
production use cases where serving architecture matters more than
single-stream throughput:

1. **Go scheduler with continuous batching** — multi-client throughput
   scaling that llama.cpp can't do natively
2. **gRPC + HTTP dual protocol** — production serving with streaming
3. **Single binary deployment** — no Python dependency chain
4. **Paged KV cache** — long-context efficiency
5. **Faster cold start** — model loads ~20% faster than llama.cpp

## MLX Comparison

| Metric | Vexel Q4_0 | Vexel Q4_K_M | MLX (Mistral 4-bit) | Gap (Q4_K_M vs MLX) |
|--------|-----------|-------------|---------------------|-----|
| Decode (short ctx) | 64.8 tok/s | 52.8 tok/s | 83.5 tok/s | -37% |
| Context degradation (16→512) | ~-10% | -10.9% | -2.5% | ~8pp |
| Prefill 128 tok | 377 tok/s | 157.6 tok/s* | 725 tok/s | -78%* |

With Q4_K_M kernel optimization, Vexel now supports the higher-quality
quantization format (256-element super-blocks with 6-bit scales/mins).
Q4_K_M decode is currently 16% slower than Q4_0 due to the 33 Q6_K
tensors using unoptimized NR2 kernels (v_proj, down_proj, lm_head).
The Q4_K matmul kernels themselves are well-optimized — the remaining
gap is in Q6_K dispatch and the larger block overhead.

## Raw Data

- Decode: `benchmarks/results/decode_20260226/`
- Prefill: `benchmarks/results/prefill_20260226/`
- Post-P0 combined: `benchmarks/results/post_p0_benchmark.json`
- Analysis: `benchmarks/results/decode_20260226/summary.json`
- Final benchmarks: measured via `inference/runtime/*_test.go` and `llama-bench`
