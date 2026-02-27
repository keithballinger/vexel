# Track Plan: Beat MLX on Apple Silicon

Close the decode throughput gap against MLX (83.5 tok/s) and match its context
scaling behavior. Target: exceed MLX's decode performance at all context lengths.

**Current state:** Vexel 64.8 tok/s vs MLX 83.5 tok/s (-22.4% gap).
Context scaling: Vexel -24.5% at ctx=512 vs MLX -2.5%.

## Competitive Landscape (M3 Max 128GB, 7B 4-bit models)

| Metric | Vexel (LLaMA2 Q4_0) | llama.cpp (LLaMA2 Q4_0) | MLX (Mistral 4-bit) |
|--------|---------------------|--------------------------|---------------------|
| Model size | 3.56 GB | 3.56 GB | 3.8 GB |
| Decode (short ctx) | 64.8 tok/s | 76.3 tok/s | 83.5 tok/s |
| Decode ctx=16 | 64.8 | 77.3 | 88.7 |
| Decode ctx=512 | 48.9 | 75.2 | 86.5 |
| Context degradation | -24.5% | -2.7% | -2.5% |
| Prefill 128tok | 200 | 803 | 725 |
| Prefill 385tok | 153 | 793 | 821 |
| BW utilization | ~70% | ~82% | ~85% |

## Root Cause Analysis

### 1. SDPA Kernel: No Flash Attention (Context Scaling)

`sdpa_decode_f16` (metal_bridge_darwin.m:3626) materializes ALL kvLen attention
weights in threadgroup memory, then iterates twice over full KV length:
- Phase 1a: Q dot K for all positions → writes `weights[0..kvLen-1]`
- Phase 1b: exp(score - max) over all positions → updates `weights[0..kvLen-1]`
- Phase 2: weighted V sum → sequential loop `for pos in 0..kvLen`

Problems:
- Threadgroup memory `shared[kvLen]` scales linearly with context length
- No tiling: reads entire KV cache every decode step
- No online softmax: requires two full passes over attention weights
- Phase 2 has zero parallelism over KV positions (sequential per dimension)

MLX/llama.cpp use Flash Attention with tiled KV processing and online softmax,
keeping threadgroup memory O(tile_size) instead of O(kvLen).

**Expected impact:** Recover 24.5% context scaling degradation → flat scaling to ctx=512+.
At ctx=512 alone this would lift 48.9 → ~63+ tok/s.

### 2. Q4_0 Matmul: NR2 vs NR4+ (Short-Context Decode Gap)

`matvec_q4_0_nr2_f32` (metal_bridge_darwin.m:329) computes 2 outputs per
simdgroup (16 per threadgroup). Each simdgroup loads activations once and
computes 2 dot products.

MLX's `qmv_impl` computes 4+ outputs per simdgroup. Key differences:
- MLX: NR=4 → loads activations once, computes 4 dots → 2x activation reuse
- MLX: affine quantization (scale + bias) vs Q4_0 (scale + offset=8)
- MLX: uses `simd_sum` reduction patterns optimized for M3
- MLX: group_size=64 (vs Q4_0 block_size=32) → fewer scale loads

Vexel reads activations as `float4` vectors (line 365-367) which is correct,
but reading weights as individual `ushort` values (lines 374-376) may be
suboptimal vs packed vector loads.

**Expected impact:** +15-20% decode throughput (69.7% → ~82-85% BW utilization).

### 3. Prefill Matmul: No Tiled GEMM (Prefill Gap)

Vexel's prefill path reuses the M=1 matvec kernel in a loop or uses
`matmul_q4_0_batched_f32` which isn't optimized for large M.

MLX uses STEEL (Simdgroup Tiled Efficient Execution Layout) for M>1:
- `simdgroup_matrix<float, 8, 8>` hardware-accelerated 8x8 matrix multiply
- Tiled computation with shared memory blocking
- Cooperative loading across simdgroups

**Expected impact:** 3-5x prefill improvement (200 → 600-800 tok/s at 128 tokens).

---

## Phase 1: Flash Attention SDPA Kernel

Replace `sdpa_decode_f16` with a tiled Flash Attention implementation using
online softmax. This is the highest-priority fix because it affects ALL context
lengths and is the primary reason Vexel degrades 24.5% from ctx=16 to ctx=512.

**Algorithm (Flash Attention v2 decode):**
```
For each Q head:
  m_prev = -inf, l_prev = 0, O = 0
  For each KV tile of size TILE_K:
    S_tile = Q @ K_tile^T * scale          // [1, TILE_K]
    m_new = max(m_prev, max(S_tile))
    P_tile = exp(S_tile - m_new)           // [1, TILE_K]
    l_new = exp(m_prev - m_new) * l_prev + sum(P_tile)
    O = exp(m_prev - m_new) * O + P_tile @ V_tile  // rescale + accumulate
    m_prev = m_new, l_prev = l_new
  O = O / l_new                            // final normalize
```

Key design choices:
- TILE_K = 32 or 64 (fits in threadgroup memory regardless of kvLen)
- Threadgroup memory: O(TILE_K * headDim) instead of O(kvLen)
- Online softmax: single pass, no materialization of full attention weights
- Vectorized Q dot K using half4 loads
- Cooperative V accumulation across simdgroup lanes

- [x] Task 1.1: Write Flash Attention SDPA kernel (Metal)
    - New kernel `sdpa_flash_decode_f16` with tiled KV iteration
    - Online softmax (running max + running sum)
    - TILE_K=32 for M3 Max threadgroup memory budget (32KB)
    - half4 vectorized dot products for Q dot K
    - Cooperative V tile accumulation using simd_sum
    - Support GQA head mapping (headsPerKV ratio)
    - Accept same buffer layout as existing sdpa_decode_f16
- [x] Task 1.2: Wire Flash SDPA into Go dispatch path
    - New pipeline state: `sdpaFlashDecodePipeline`
    - Route decode SDPA to new kernel (keep old kernel for fallback)
    - Threadgroup memory sized to TILE_K * headDim * sizeof(float) + reduction scratch
    - Same Go-side API: `backend.SDPADecode(Q, K, V, out, kvLen, ...)`
- [x] Task 1.3: Correctness tests — token-exact match vs reference
    - Test against existing sdpa_decode_f16 output at ctx=16, 64, 128, 256, 512, 1024
    - Test GQA configurations: 32Q/8KV (LLaMA2 7B), 32Q/32KV (no GQA), 32Q/4KV
    - Test edge cases: kvLen < TILE_K, kvLen = exact multiple of TILE_K
    - Verify numerical stability of online softmax (compare max/sum values)
- [x] Task 1.4: Context scaling benchmark
    - Run TestThroughputContextScaling at ctx=16, 64, 128, 256, 512
    - Target: <5% degradation from ctx=16 to ctx=512 (currently 24.5%)
    - Compare against MLX's 2.5% degradation
    - Measure absolute throughput improvement at each context length

## Phase 2: Q4_0 Matmul NR4 Kernel

Increase outputs per simdgroup from 2 (NR2) to 4 (NR4) in the quantized
matvec kernel. This doubles activation data reuse, directly improving
memory bandwidth utilization from ~70% toward ~85%.

**Design:**
- 4 outputs per simdgroup → 32 outputs per threadgroup (8 simdgroups)
- Load activations once per block, compute 4 dot products
- Weight loads: 4 row pointers instead of 2
- Same Q4_0 block format (32 values, 18 bytes: 2B scale + 16B nibbles)
- Activation loads remain float4 vectors (lines 365-367 pattern)
- Consider packed uint4 weight loads for better bandwidth

```
// NR4 structure per simdgroup:
int base_output = gid * 32 + simd_group * 4;
float sum0=0, sum1=0, sum2=0, sum3=0;

for (block = simd_lane; block < numBlocks; block += 32) {
    float4 a0..a7 = load_activations(A, block);   // shared across 4 outputs
    dequant_and_dot(B_row0[block], a0..a7, &sum0); // 4 weight rows
    dequant_and_dot(B_row1[block], a0..a7, &sum1);
    dequant_and_dot(B_row2[block], a0..a7, &sum2);
    dequant_and_dot(B_row3[block], a0..a7, &sum3);
}
```

Register pressure: 4 accumulators + 8 activation float4s + 4 sets of weight
temporaries ≈ 48 registers. Well within M3 Max's register file (~128 per thread).

- [ ] Task 2.1: Write NR4 Q4_0 matvec kernel
    - New kernel `matvec_q4_0_nr4_f32`
    - 4 outputs per simdgroup, 32 per threadgroup
    - Shared activation loads, 4 independent weight streams
    - Same Q4_0 dequantization (scale * (nibble - 8))
    - Grid sizing: ceil(N / 32) threadgroups
- [ ] Task 2.2: Write offset-aware NR4 variant for scratch allocator
    - `matvec_q4_0_nr4_f32_offset` with A/C byte offsets
    - Same kernel logic, offset-adjusted buffer pointers
- [ ] Task 2.3: Wire NR4 into dispatch path with fallback
    - New pipeline state: `matmulQ4NR4Pipeline`
    - Route M=1 Q4_0 dispatches to NR4 kernel
    - Keep NR2 as fallback for N not divisible by 32
    - Update both `MatMulQ4_0` and `MatMulQ4_0Offset` dispatch functions
- [ ] Task 2.4: Correctness tests — bit-exact match vs NR2
    - Test all LLaMA 2 7B layer sizes: [4096, 11008, 14336, 32000]
    - Test N not divisible by 32 (fallback to NR2)
    - Test with K not divisible by Q4_BLOCK_SIZE (partial blocks)
    - Verify full model decode produces identical tokens
- [ ] Task 2.5: Benchmark decode throughput
    - Run TestThroughputDecode (5 runs, median)
    - Target: >75 tok/s at short context (currently 64.8)
    - Measure BW utilization improvement via TestThroughputGPUProfile
    - Compare against llama.cpp 76.3 tok/s and MLX 83.5 tok/s

## Phase 3: Tiled Prefill GEMM

Replace the naive prefill matmul with a tiled GEMM using `simdgroup_matrix`
hardware instructions. This targets the 4-5x prefill gap vs MLX/llama.cpp.

**Design (STEEL-inspired tiled GEMM for quantized weights):**
- Use `simdgroup_matrix<float, 8, 8>` for 8x8 tiles
- Block sizes: BM=32, BN=32, BK=32 (tunable for M3 Max L1 cache)
- Cooperative loading: simdgroups load A tiles and dequantized B tiles into
  threadgroup memory, then compute using hardware matrix multiply
- A (activations): FP32, loaded as tiles into shared memory
- B (weights): Q4_0, dequantized on-the-fly into FP32 shared memory tiles
- C (output): FP32, accumulated in registers then written to device memory

Key implementation notes:
- Q4_0 dequantization during load: read 18-byte blocks, expand to 32 floats
- M3 Max has 32KB threadgroup memory → fits BM*BK + BK*BN = 32*32*2 = 8KB
- simdgroup_matrix multiply: `C_tile = A_tile * B_tile` (hardware 8x8 FMA)
- Need separate kernel from M=1 path — dispatched when M > threshold (e.g., M>=4)

- [ ] Task 3.1: Write tiled Q4_0 GEMM kernel with simdgroup_matrix
    - New kernel `gemm_q4_0_tiled_f32`
    - Tile sizes BM=32, BN=32, BK=32 (configurable via constants)
    - Cooperative A and B tile loading into threadgroup memory
    - Q4_0 dequantization during B tile load
    - simdgroup_matrix<float,8,8> multiply-accumulate
    - Handle edge tiles where M/N/K not divisible by block size
- [ ] Task 3.2: Wire tiled GEMM into dispatch path
    - Dispatch condition: use tiled GEMM when M >= 4
    - Keep existing batched kernel for M=1 (matvec) and M=2-3 (small batch)
    - Threadgroup memory: (BM*BK + BK*BN) * sizeof(float)
    - Grid: ceil(M/BM) * ceil(N/BN) threadgroups
- [ ] Task 3.3: Correctness tests
    - Compare output vs existing batched kernel at M=4, 8, 16, 32, 64, 128, 256, 385
    - Test all weight matrix sizes in LLaMA 2 7B
    - Verify prefill generates correct logits (full model test)
- [ ] Task 3.4: Prefill benchmark
    - Run TestThroughputPrefill at 5, 32, 128, 385 tokens
    - Target: >500 tok/s at 128 tokens (currently 200)
    - Compare against MLX (725) and llama.cpp (803)

## Phase 4: Integration Benchmarks & Results Update

Run comprehensive competitive benchmarks with all three optimizations applied,
update RESULTS.md and README.md.

- [ ] Task 4.1: Full benchmark suite
    - Decode throughput at ctx=16, 64, 128, 256, 512
    - Prefill throughput at 5, 32, 128, 385 tokens
    - Model load time (should be unchanged)
    - Re-run MLX benchmarks for apples-to-apples comparison
    - Run llama.cpp benchmarks for three-way comparison
- [ ] Task 4.2: Update RESULTS.md
    - Update all performance tables with new numbers
    - Add MLX comparison column
    - Update optimization roadmap (P6/P7/P8 status)
    - Document any remaining gaps and next priorities
- [ ] Task 4.3: Update README.md
    - Update performance section headline numbers
    - Add MLX to comparison table if beating it
    - Update competitive advantages narrative

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Decode (short ctx) | 64.8 tok/s | >80 tok/s | >85 tok/s |
| Decode ctx=512 | 48.9 tok/s | >75 tok/s | >82 tok/s |
| Context degradation | -24.5% | <5% | <3% |
| Prefill 128tok | 200 tok/s | >500 tok/s | >700 tok/s |
| BW utilization | ~70% | >82% | >85% |

**Beat MLX threshold:** Decode >83.5 tok/s at short context AND <5% context
degradation at ctx=512. This would make Vexel the fastest single-stream
inference engine on Apple Silicon for 7B models.
