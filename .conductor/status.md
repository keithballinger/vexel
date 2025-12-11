# Project Status

**Last Updated:** 2025-12-11
**Status:** 🟢 On Track

## Current Phase
Phase 9: Performance Optimization (Target: Match llama.cpp)

## Current Task
Simdgroup_matrix kernel implemented for prefill. Prefill improved ~48% (226→336 tok/s).

## Latest Performance Metrics
| Metric | Vexel | llama.cpp | Gap |
|--------|-------|-----------|-----|
| Prefill | **336 tok/s** | ~1000 tok/s | **3.0x** |
| Decode | **101 tok/s** | ~266 tok/s | 2.6x |

**Model:** TinyLlama 1.1B Q4_0
**Hardware:** M4 Pro

**Progress since start of optimization:**
- Prefill: 70 → 336 tok/s (4.8x improvement)
- Decode: 49 → 101 tok/s (2.1x improvement)

## Recent Progress
- [x] Fixed Q4_0 kernel bug causing garbage output
- [x] Created comprehensive Q4_0 kernel test suite (9 tests)
- [x] Implemented SIMD vectorized Q4_0 kernels
- [x] Implemented multi-output Q4_0 matvec (8 outputs/threadgroup)
- [x] **Implemented simdgroup_matrix kernel for prefill (M>=8)**
  - 32x32 tiled matrix multiplication using 8x8 simdgroup_float8x8
  - Cooperative loading to threadgroup memory with dequantization
  - B matrix stored transposed for correct C = A @ B^T computation
- [x] Profiled memory bandwidth: achieving ~30 GB/s of ~273 GB/s (11%)

## Next Actions
1. Implement Q6_K kernel for lm_head (currently using F32)
2. Optimize memory bandwidth utilization
3. Profile and tune simdgroup tile sizes

## Blockers
None

## Notes
- Simdgroup kernel: 4 active simdgroups per threadgroup, each computes 16x16 output
- Uses simdgroup_multiply_accumulate for 8x8 tiles
- M >= 8 threshold for using simdgroup kernel (prefill phase)

---

## Status History

### 2025-12-11: Simdgroup Matrix Kernel for Prefill
**Status:** Major prefill performance improvement

Implemented simdgroup_matrix kernel for Q4_0 matmul during prefill (M >= 8). Uses Apple Silicon's hardware matrix multiply-accumulate units.

**Key Changes:**
- New `matmul_q4_0_simdgroup_f32` kernel using simdgroup_float8x8
- 32x32 tile sizes matching Q4_0 block size (32 elements)
- Cooperative loading to threadgroup memory with on-the-fly dequantization
- B matrix stored transposed for correct C = A @ B^T computation
- 4 active simdgroups per threadgroup, each computing 16x16 output

**Results:**
- Prefill: 226 → 336 tok/s (+48%)
- Decode: unchanged at ~101 tok/s (uses different kernel)
- Gap to llama.cpp reduced from 4.4x to 3.0x for prefill

### 2025-12-11: Multi-Output Q4_0 Kernel Optimization
**Status:** Major decode performance improvement

Implemented multi-output Q4_0 matvec kernel that computes 8 outputs per threadgroup instead of 1. Each simdgroup (32 threads) handles one output element, with all 8 simdgroups working in parallel.

**Key Changes:**
- New `matvec_q4_0_multi_output_f32` kernel with 8 simdgroups per threadgroup
- Reduced threadgroup count 8x (2048 → 256 for [1,2048]×[2048,2048])
- Vectorized float4 loads with dot() operations
- Simple simd_sum reduction (no shared memory needed)

**Results:**
- Decode: 55 → 96 tok/s (+75%)
- Prefill: 179 → 226 tok/s (+26%)
- Gap to llama.cpp reduced from 4.8x to 2.8x for decode

**Attempted Optimizations That Didn't Help:**
- Threadgroup memory for A vector caching (regressed due to scalar loads)
- Sequential processing of multiple outputs per simdgroup (loop overhead)
- Command buffer batching (minimal impact)

### 2025-12-10: Q4_0 Kernel Bug Fix
**Status:** Fixed critical GPU inference bug

GPU inference was producing garbage output. Root cause identified as buggy "optimized" Q4_0 kernels introduced in commit `c51d6eb`. The vectorized kernels had subtle bugs in nibble mapping (float4/uchar4 loads with 4-row-per-simdgroup batching).

**Fix Applied:**
- Reverted `matvec_q4_0_transposed_f32` to simple 256-thread version
- Reverted `matmul_q4_0_batched_f32` to simple loop-based version
- Added `q4_f16_to_f32` helper for fp16 scale conversion
- Updated dispatch code: 256 threads, 1 output per threadgroup

**Result:** GPU inference produces coherent text. Q4_0 matmul max diff: 0.000001

### 2025-12-08: Real Inference Working
**Status:** Basic end-to-end inference operational

- All CPU kernels implemented and tested
- GGUF loading functional
- Tokenizer integrated
- Basic chat template support
- Metal backend operational (with performance gap)

### 2025-12-07: Project Foundation
**Status:** Initial architecture and structure

- Project structure established
- Core types defined (Tensor, Arena, KV Cache)
- Backend interface defined
- Block IR system implemented
- Scheduler framework in place
