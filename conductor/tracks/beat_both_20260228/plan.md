# Track Plan: Beat llama.cpp AND MLX on Apple Silicon

Close the decode throughput gap against llama.cpp (76.3 tok/s) and MLX (83.5 tok/s).
Target: exceed both on decode, flatten context scaling, close prefill gap.

**Current state:** Vexel 72.5 tok/s (Q4_0, LLaMA 2 7B, M3 Max 128GB).

## Phase 1: Fix F32 SDPA Decode Context Scaling
- [x] Task 1.1: Write correctness tests for sdpa_flash_decode_f32_v2 (TDD Red)
    - 14 subtests: kvLen=1,4,7,16,64,128,256,512,1024 × GQA configs (4:4, 32:8, 8:8, 32:4, 32:32)
    - CPU reference from existing cpuSDPADecodeF32
    - Tolerance: 1e-4 (full F32 path, tighter than F16)
    - All pass with max_diff=0.000000 (bit-exact!)
- [x] Task 1.2: Write sdpa_flash_decode_f32_v2 kernel (Metal)
    - Adapted sdpa_flash_decode_f16 algorithm to float* I/O
    - Online softmax with split-KV across 8 simdgroups (FLASH_F32_V2_NUM_SG=8)
    - O(headDim) shared memory instead of O(kvLen)
    - Shared mem: (2*8 + 8*headDim) * sizeof(float) = 4160 bytes for headDim=128
    - Old kernel: (kvLen + 8) * sizeof(float) = ~4K for ctx=1024
- [x] Task 1.3: Write C dispatch functions (regular + offset)
    - metal_sdpa_flash_decode_f32_v2
    - metal_sdpa_flash_decode_f32_v2_offset
- [x] Task 1.4: Register pipeline, wire into Go dispatch (backend.go)
    - New pipeline: sdpaFlashDecodeF32V2Pipeline
    - Route SDPA and SDPAOffset to v2 kernel (preferred over old F32 flash)
    - Gate: headDim % 32 == 0 (required for split-KV lane layout)
- [x] Task 1.5: Context scaling benchmark
    - Q4_0 fused decode uses F16 SDPA path (not F32), so full-model numbers unchanged
    - F32 v2 kernel's context scaling benefit applies to Q4_K models (Track 2)
    - Kernel correctness verified: bit-exact at ctx=16 through ctx=1024

## Phase 2: Q4_K Fused Decode Pipeline
- [~] Task 2.1: Write Q4_K fused RMSNorm+Matvec kernels (Metal)
- [ ] Task 2.2: Write Q4_K v2-style kernels (F16-in, accumulate)
- [ ] Task 2.3: Register pipelines, wire into Go dispatch
- [ ] Task 2.4: Expand canFuseAttn gate to Q4_K
- [ ] Task 2.5: Correctness tests (fused vs unfused Q4_K)
- [ ] Task 2.6: Benchmark Q4_K fused pipeline

## Phase 3: Q4_K Decode Kernel Micro-Optimization
- [ ] Task 3.1: Write matvec_q4k_nr4_f32 kernel
- [ ] Task 3.2: Benchmark NR4 vs multi_output
- [ ] Task 3.3: Wire winning kernel into dispatch

## Phase 4: Q4_K Prefill GEMM Optimization
- [ ] Task 4.1: Audit Q4_K simdgroup GEMM tile config
- [ ] Task 4.2: Apply optimizations (vectorized loads, barriers)
- [ ] Task 4.3: FP16 activation pipeline variant

## Phase 5: Tiled KV Decode SDPA for Long Context
- [ ] Task 5.1: Write sdpa_flash_decode_f16_tiled kernel
- [ ] Task 5.2: Wire into dispatch, benchmark at ctx=512,1024,2048
