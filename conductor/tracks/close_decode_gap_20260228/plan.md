# Track Plan: Close the Decode Gap to llama.cpp

Vexel Q4_0 decode was 62.3 tok/s vs llama.cpp's 76.3 tok/s (-18.4%) on M3 Max.
Root cause: dispatch overhead dominates (419 dispatches/token × ~17µs each).

**Target: ≥73 tok/s decode throughput (≤5% gap to llama.cpp).**

## Background

The M=1 decode path was dominated by dispatch overhead, not raw kernel speed.
419 dispatches per token × ~17µs dispatch overhead = ~7ms overhead per token.
Theoretical minimum compute time: ~9.4ms (BW-limited at 400 GB/s).

## Phase 0: Decode Profiling Baseline — COMPLETE
- [x] Task 0.1: Per-operation decode profiling at ctx=16
    - Baseline: **62.3 tok/s** (16.0ms/token median), 419 dispatches/token
    - Per-layer: 13 dispatches (FusedRMSNorm×3, RoPE, ScatterKV×2, SDPA, Convert, MatMulQ4_0×2, Add, AddRMSNorm, FusedMLP)
    - Dispatch overhead is the dominant bottleneck (~43% of wall time)

## Phase 2.1: v2 Matvec Kernel — COMPLETE
- [x] Task 2.1a: Write matvec_q4_0_v2_f32 matching llama.cpp's decode technique
    - 64 threads, NR0=4, fused nibble-masking (pre-scaled activations)
    - Fix: corrected qs pointer offset (blockPtr + 2 + il, not blockPtr + 1 + il/2)
- [x] Task 2.1b: Correctness tests pass (11 configs, max_diff ≤ 0.017)
- [x] Task 2.1c: Benchmark: 5-39% faster in isolation vs multi_output
    - 4096×4096: 288→275µs (1.05x), 4096×11008: 441→318µs (1.39x)
- [x] Result: 61.7 tok/s (marginal — only 64/419 dispatches use v2)

## Phase 2.2: Fused QKV Dispatch — COMPLETE
- [x] Task 2.2a: Write matvec_q4_0_fused_rmsnorm_qkv_f16 kernel
    - Single dispatch for RMSNorm + Q/K/V projections (3→1 per layer)
    - Routes threadgroups to Q, K, or V based on gid
    - Bitwise-identical output vs 3 separate FusedRMSNormF16 dispatches
- [x] Task 2.2b: Wire into block.go FP16 decode path
- [x] Result: **69.9 tok/s**, 355 dispatches/token (+12.2% from baseline)

## Phase 2.3: Fused KV Scatter + Eliminate Converts — COMPLETE
- [x] Task 2.3a: Write scatter_kv_f16_fused kernel (K+V in single dispatch)
- [x] Task 2.3b: Write matvec_q4_0_v2_f16in_f32 kernel (FP16 activation input)
    - Reads half* instead of float* — eliminates ConvertF16ToF32 dispatch
- [x] Task 2.3c: Wire both into runtime
    - gpu_kv_cache.go: fused scatter via type assertion
    - block.go: F16-input Wo path, skip convert
- [x] Result: **70.4 tok/s**, 291 dispatches/token (+13.0% from baseline)

## Phase 2.4: Fused W2+Add2 — COMPLETE
- [x] Task 2.4a: Write matvec_q4_0_v2_add_f32 kernel (out += A @ B^T)
    - Same v2 technique but accumulates instead of overwrites output
    - Fuses W2 down-projection + residual Add2 into single dispatch
- [x] Task 2.4b: Wire into block.go with type assertion (serial residual, Q4_0 only)
- [x] Task 2.4c: Correctness: bitwise-identical at 128×128, 4096×11008, 4096×4096
- [x] Result: **70.0 tok/s**, 259 dispatches/token (neutral throughput — dispatch overhead no longer dominant)

## Phase 2.5: Fused RoPE + ScatterKV — COMPLETE
- [x] Task 2.5a: Fuse RoPE and ScatterKV into single dispatch
- [x] Result: **68.9 tok/s**, 227 dispatches/token

## Phase 2.6: FP16 Shared Memory in Fused Kernels — COMPLETE
- [x] Task 2.6a: Store activations as half in threadgroup memory in all fused RMSNorm kernels
    - Halves shared memory from 16 KB to 8 KB per threadgroup
    - Allows higher GPU occupancy (more TGs in flight per compute unit)
    - RMSNorm sum-of-squares still FP32 for accuracy
- [x] Task 2.6b: Updated 3 kernels (F32, F16_out, QKV_F16) + 4 dispatch functions
- [x] Task 2.6c: Correctness: 20 tokens match exactly between fused and unfused paths
- [x] Result: **72.5 tok/s** (+5.2% from 68.9), 13.8ms/token

### Failed Experiments
- **Sumy zero-point optimization**: Applied `dot(a, q-8) = dot(a, q) + sum(a)*(-8)` to all 3 fused kernels.
  Correctness passed but performance regressed (68.4 tok/s). Root cause: adds data dependency chain
  in bandwidth-bound kernel where ALU operations are "free" (hidden in memory latency).
- **Unfusing MLP** to use v2 kernels: 67.6 tok/s — SLOWER. Extra dispatch overhead outweighs occupancy benefit.
- **FP32 decode path**: 64.8 tok/s — significantly worse. FP16 intermediates reduce memory traffic.
- **Previous session's batch/scratch changes**: 67.2 tok/s — slight regression vs HEAD.

## Current Status

| Stage | Dispatches | tok/s | Gap to llama.cpp |
|-------|-----------|-------|-----------------|
| Baseline | 419 | 62.3 | 18.4% |
| + v2 kernel | 419 | 61.7 | 19.1% |
| + QKV fusion | 355 | 69.9 | 8.4% |
| + Fused scatter + F16-in Wo | 291 | 70.4 | 7.7% |
| + W2+Add2 fusion | 259 | 70.0 | 8.2% |
| + Fused RoPE+ScatterKV | 227 | 68.9 | 9.7% |
| + **FP16 shared memory** | 227 | **72.5** | **5.0%** |

Per-layer dispatches now: 7 (FusedQKV + FusedRoPE+ScatterKV + SDPA + Wo + AddRMSNorm + FusedMLP + W2+Add2)

Context scaling with FP16 shared memory:
| Context | tok/s | ms/token |
|---------|-------|----------|
| 16 | 72.0 | 13.88 |
| 64 | 71.0 | 14.09 |
| 128 | 70.3 | 14.23 |
| 256 | 66.8 | 14.96 |
| 512 | 61.9 | 16.14 |

## Remaining Gap Analysis

72.5 tok/s → 13.8ms/token. Target 73 tok/s → 13.7ms/token. Gap is ~0.1ms — **effectively at target**.

The 5.0% gap to llama.cpp (76.3 tok/s) meets the ≤5% target.

Possible micro-optimizations for further gains:
- Apply FP16 shared memory to FusedMLP kernel (currently no shared memory, different pattern)
- Profile per-dispatch overhead breakdown (CGO cost, Metal API overhead)
- SDPA optimization (currently uses naive attention at short contexts)

## Reference: Key Files

- v2 kernel: `inference/backend/metal/metal_bridge_darwin.m` (matvec_q4_0_v2_f32)
- v2 accumulate: same file (matvec_q4_0_v2_add_f32)
- F16-input variant: same file (matvec_q4_0_v2_f16in_f32)
- Fused QKV kernel: same file (matvec_q4_0_fused_rmsnorm_qkv_f16)
- Fused KV scatter: same file (scatter_kv_f16_fused)
- Dispatch routing: `inference/backend/metal/backend.go`
- Forward pass: `inference/runtime/block.go`
- KV cache: `inference/runtime/gpu_kv_cache.go`
- Tests: `inference/backend/metal/q4_batched_matmul_test.go`, `fused_kernel_test.go`
