# Vexel Performance Investigation - Kernel Analysis

**Date**: 2025-12-13
**Current Performance**: ~145 tok/s decode (TinyLlama Q4_0)
**Target**: ~244-258 tok/s (llama.cpp baseline)

## Summary of Findings

### 1. Kernel Performance is Good - Layer Computation

Microbenchmark results show kernels achieve good bandwidth:

| Operation | Time (µs) | Weight Size | Bandwidth |
|-----------|-----------|-------------|-----------|
| Wo (Q4_0, 2048×2048) | 13.0 | 2.36 MB | 180 GB/s |
| W2 (Q4_0, 2048×5632) | 31.0 | 6.49 MB | 391 GB/s* |

\* Including input activation reloading (8× per threadgroup)

**Full layer simulation (22 layers, simplified):**
- Time per token: **4.24 ms = 236 tok/s**
- This is close to llama.cpp's ~244 tok/s!

### 2. Major Bottleneck: LM Head (Q6_K)

The LM head is a significant bottleneck:

| Metric | Value |
|--------|-------|
| Time per call | **1.49 ms** |
| Weight size | 53.76 MB |
| Effective bandwidth | **36 GB/s** (18% of peak) |

This single operation accounts for ~21% of total decode time!

### 3. Breakdown of Token Time

| Component | Time | Percentage |
|-----------|------|------------|
| 22 layers (kernels) | 4.24 ms | 62% |
| LM head (Q6_K) | 1.49 ms | 22% |
| Framework overhead | ~1.12 ms | 16% |
| **Total** | **~6.85 ms** | 100% |

### 4. Why Q6_K is Slow

Comparing our Q6_K kernel to llama.cpp:

| Feature | Vexel | llama.cpp |
|---------|-------|-----------|
| Outputs per simdgroup | 1 | 2 (nr0=2) |
| Activation vectorization | None | yl[16] array |
| Thread collaboration | None | 2 threads/block |
| Inner loop unrolling | None | FOR_UNROLL |
| Bandwidth achieved | 36 GB/s | ~120+ GB/s |

### 5. Other Observations

**Fused kernels work well:**
- FusedRMSNorm+QKV and FusedRMSNorm+GateUp already use shared memory optimization
- The standalone "optimized" kernel was slower due to occupancy/barrier overhead

**Mid-layer sync is needed:**
- Skipping VEXEL_SKIP_MID_SYNC=1 causes *slower* decode (133 vs 147 tok/s)
- The sync helps with GPU scheduling/pipelining

**Command batching works:**
- GPU profile shows 2200 batches for 50 tokens
- Batch mode is active and functioning

## Recommendations

### High Priority (Biggest Impact)

1. **Optimize Q6_K kernel** - Expected gain: 0.5-1.0 ms/token (15-30% speedup)
   - Implement nr0=2 (2 outputs per simdgroup)
   - Vectorize activation loads (yl[16] array)
   - Add thread collaboration (2 threads per block)
   - Unroll inner loops

2. **Consider Q4_0 for LM head** - Alternative approach
   - Q4_0 achieves 180+ GB/s vs Q6_K's 36 GB/s
   - Would need to re-quantize output head weights

### Medium Priority

3. **Reduce framework overhead** (~1.1 ms)
   - Profile Go allocations during inference loop
   - Minimize CGo transition overhead
   - Pre-allocate all scratch buffers

4. **Profile context length scaling**
   - Test at 256, 2k, 8k tokens
   - Identify if SDPA becomes bottleneck at long context

### Low Priority

5. **Investigate mid-layer sync behavior**
   - Why does skipping it slow things down?
   - May reveal GPU scheduling insights

## Appendix: Microbenchmark Commands

```bash
# Run kernel microbenchmark
CGO_ENABLED=1 go test -v -tags=metal -run TestW2KernelMicrobench ./inference/runtime/

# Run full layer microbenchmark
CGO_ENABLED=1 go test -v -tags=metal -run TestFullLayerMicrobench ./inference/runtime/

# Run LM head microbenchmark
CGO_ENABLED=1 go test -v -tags=metal -run TestLMHeadMicrobench ./inference/runtime/

# Profile with per-operation timing
DEBUG_PROFILE=1 ./vexel -model ./models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -max-tokens 50 -gpu -temp 0 -completion
```
