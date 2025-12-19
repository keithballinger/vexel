# llama.cpp Q4_0 Metal Kernel Analysis

**Date:** 2025-12-13
**Goal:** Understand why llama.cpp achieves 258 tok/s decode vs Vexel's 167 tok/s (1.5x gap)

## Summary

After analyzing llama.cpp's Metal kernel implementation at `ggml/src/ggml-metal/ggml-metal.metal`, the performance gap appears to stem from different thread collaboration patterns, not arithmetic optimizations.

## llama.cpp Kernel Configuration

```
File: ggml-metal-impl.h
#define N_R0_Q4_0 4    // 4 rows per simdgroup
#define N_SG_Q4_0 2    // 2 simdgroups per threadgroup
```

- **Total outputs per threadgroup:** 8 (4 rows × 2 simdgroups)
- **Threads per threadgroup:** 64 (2 simdgroups × 32 threads)
- **NQ = 16:** Blocks processed per iteration before striding

## Key Implementation Details

### Thread Organization

```metal
// From mul_vec_q_n_f32_impl (line 3230)
const short ix = (tiisg/(NW/NQ));  // NW=32, NQ=16 → ix = tiisg/2 (0-15)
const short il = (tiisg%(NW/NQ))*8; // il = (tiisg%2)*8 → 0 or 8

const int ib0 = ix;  // Starting block index
for (int ib = ib0; ib < nb; ib += NQ) {  // Stride by 16 blocks
    // Process block ib
}
```

- Pairs of threads collaborate on each block
- Thread with il=0 handles one half, il=8 handles the other
- All 32 threads process 16 consecutive blocks together
- Then stride by 16 to next chunk

### Activation Caching with Pre-scaling

```metal
float yl[16]; // Cached activation values

FOR_UNROLL (short i = 0; i < 8; i += 2) {
    sumy[0]  += yb[i +  0] + yb[i +  1];
    yl[i + 0] = yb[i +  0];
    yl[i + 1] = yb[i +  1]/256.f;   // Pre-scaled for 0x0F00 mask

    sumy[1]  += yb[i + 16] + yb[i + 17];
    yl[i + 8] = yb[i + 16]/16.f;    // Pre-scaled for 0x00F0 mask
    yl[i + 9] = yb[i + 17]/4096.f;  // Pre-scaled for 0xF000 mask
}
```

### Dot Product with uint16 Masks

```metal
// From block_q_n_dot_y (line 3100)
device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

for (int i = 0; i < 8; i += 2) {
    acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);  // Low nibble, low byte
    acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);  // Low nibble, high byte (×256)
    acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);  // High nibble, low byte (×16)
    acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);  // High nibble, high byte (×4096)
}

return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
```

The pre-scaling compensates for the bit positions of the masks, avoiding shifts in the inner loop.

## Vexel Kernel Configuration

```
Outputs per threadgroup: 8 (8 simdgroups × 1 output each)
Threads per threadgroup: 256 (8 simdgroups × 32 threads)
```

### Thread Organization

```metal
// Each thread handles blocks with stride 32
for (int block = simd_lane; block < numBlocks; block += 32) {
    // Thread 0: blocks 0, 32, 64, ...
    // Thread 1: blocks 1, 33, 65, ...
    // ...
}
```

## Critical Differences

### 1. Memory Access Pattern

| Aspect | llama.cpp | Vexel |
|--------|-----------|-------|
| Block iteration | 16 consecutive, then stride | Stride 32 from start |
| Threads per block | 2 (collaborating) | 1 |
| Spatial locality | High (consecutive blocks) | Low (scattered) |

**Impact:** llama.cpp's pattern likely achieves better cache utilization for weight data.

### 2. Activation Handling

| Aspect | llama.cpp | Vexel |
|--------|-----------|-------|
| Caching | 16 floats in registers (`yl[]`) | On-demand float4 loads |
| Pre-scaling | Yes (avoids inner loop math) | No |
| Reuse | Across all 4 nibble extractions | Per dot product |

### 3. Weight Loading

| Aspect | llama.cpp | Vexel |
|--------|-----------|-------|
| Load granularity | uint16 (2 bytes) | uchar (1 byte) |
| Nibble extraction | Masks on uint16 | Shift/mask on uchar |

## Optimizations Tested

| Optimization | Result | Notes |
|-------------|--------|-------|
| Pre-scaling Y values | No improvement | GPU hides arithmetic latency |
| uchar4 vectorized loads | Broken output | Alignment or indexing issue |
| commandBufferWithUnretainedReferences | +1-2 tok/s | Already batching |
| Remove mid-layer sync | Broken output | Still needed for correctness |
| **llama.cpp collab kernel (4 rows/simdgroup)** | **-10% slower (143 tok/s)** | Cache thrashing from 4-row access |
| **llama.cpp 64-thread config** | **-6% slower (151 tok/s)** | Still slower than 256-thread multi_output |

## Hypothesis (DISPROVED)

~~The 1.5x performance gap is primarily due to **memory access patterns**, not arithmetic efficiency:~~

1. ~~**Weight cache efficiency:** llama.cpp's consecutive block processing keeps weight data in cache longer~~
2. ~~**Memory coalescing:** Pairs of threads accessing adjacent uint16 values vs scattered byte accesses~~
3. ~~**Bandwidth utilization:** llama.cpp likely achieves ~150 GB/s vs Vexel's ~100 GB/s~~

**Experimental Result:** Implementing llama.cpp-style kernel was **SLOWER** (143-151 tok/s vs 161 tok/s original).

### Why llama.cpp-style Failed in Vexel

The key insight is about **weight row locality**:

| Kernel | Rows per simdgroup | Cache behavior |
|--------|-------------------|----------------|
| Vexel multi_output | 1 | Each simdgroup reads from ONE row - excellent cache locality |
| llama.cpp collab | 4 | Each simdgroup reads from 4 different rows - cache thrashing |

In llama.cpp, the 4-row pattern works because their entire threadgroup is smaller (64 threads) and they iterate in a way that keeps weight data in cache. In Vexel's context with 256-thread threadgroups, each simdgroup reading 4 different rows causes cache line evictions.

**The original Vexel kernel's 1-row-per-simdgroup pattern is actually optimal for cache locality.**

## GPU Profiling Results (2025-12-13)

Implemented GPU profiling infrastructure to understand where time is spent:

### Metrics (29 tokens generated)
| Metric | With Mid-layer Sync | Without Sync |
|--------|---------------------|--------------|
| GPU compute time | 128.97 ms | 129.53 ms |
| CPU sync wait time | 169.70 ms | 173.27 ms |
| Batch count | 1320 | 660 |
| Avg per batch | 97.71 µs | 196.25 µs |
| Decode speed | 163 tok/s | 162 tok/s |

### Key Findings

1. **Mid-layer sync doesn't significantly impact GPU time** - 128ms vs 130ms is negligible
2. **Batch count doubles with sync** (1320 vs 660) but per-batch overhead is low
3. **Sync time dominates** - the 170ms sync time comes from explicit `Sync()` calls when reading results back to CPU for sampling, not from the mid-layer sync
4. **Removing mid-layer sync breaks output** - produces consistently different (wrong) results, likely due to Metal encoder ordering semantics

### Mid-layer Sync Analysis

The mid-layer sync (EndBatch + BeginBatch) exists between:
- First half: RMSNorm, Q/K/V projections, RoPE
- Second half: F32→F16 conversion, KV cache copy, SDPA

Without this sync, the output is deterministic but different (incorrect), suggesting a Metal synchronization issue between compute operations and subsequent reads.

Attempts to fix:
- **Memory barriers** (`memoryBarrierWithScope:`) - didn't help
- **Compute copy kernel** (avoid blit encoder) - didn't help
- **Wait for completion** (`waitUntilCompleted`) - fixes but too slow

The root cause remains unclear - within a single command buffer, Metal should guarantee operation ordering.

## Recommended Next Steps

1. ~~**Profile GPU utilization** to confirm memory-bound hypothesis~~ ✅ Done
2. ~~**Rewrite kernel** with NQ-style consecutive block iteration~~ ❌ Slower (143-151 tok/s)
3. **Test uint16 loads** with correct alignment handling
4. **Investigate mid-layer sync requirement** - may be a bug in encoder handling or buffer aliasing
5. ~~**Profile individual kernel times**~~ ✅ Done - see below
6. **Compare framework overhead** - llama.cpp may have lower CPU-side overhead

## Performance Breakdown (2025-12-13)

### GPU Time vs Total Time
| Metric | Vexel | llama.cpp (est.) |
|--------|-------|------------------|
| GPU time (29 tok) | 136ms | ~113ms |
| Total time | 185ms | 112ms |
| CPU overhead | 49ms | ~0ms |
| Decode speed | 157 tok/s | 258 tok/s |

The 1.5x gap breaks down as:
- **GPU efficiency**: 1.2x slower (136ms vs 113ms)
- **CPU overhead**: 49ms additional time

### Per-Operation Breakdown (DEBUG_PROFILE=1, with syncs)
Note: Percentages valid for relative comparison, absolute times affected by sync overhead.

| Operation | Time % | Notes |
|-----------|--------|-------|
| FusedRMSNorm+GateUp | 13.4% | MLP gate/up projections |
| W2 | 11.7% | MLP down projection (largest matrix) |
| FusedRMSNorm+QKV | 10.4% | Attention projections |
| KVCache | 9.7% | GPU-to-GPU copy (blit encoder) |
| Wo | 9.4% | Output projection |
| SDPA | 9.2% | Scaled dot-product attention |
| RoPE | 8.5% | Rotary embeddings |
| Add1+Add2 | 15.4% | Residual additions |
| SiLUMul | 7.7% | MLP activation |

### Identified CPU Overhead Sources

1. **Double Sync in scheduler** (`scheduler.go:618`): `getLogitsOnCPU()` calls `Sync()` after `DecodeWithGPUKV()` already synced
2. **Logits transfer**: 128KB per token (32000 × 4 bytes) copied from GPU to CPU
3. **Memory allocation**: 256KB allocated per token in `getLogitsOnCPU()`
4. **Byte-to-float conversion**: Manual loop instead of unsafe cast

### Proposed Optimizations

1. ~~**Remove redundant Sync** in `getLogitsOnCPU()` - already synced in decode~~ ✅ Implemented
2. **GPU-side argmax** for temp=0: Transfer 4 bytes instead of 128KB
3. ~~**Unsafe float32 slice**: Use `unsafe.Slice` instead of manual byte conversion~~ ✅ Implemented
4. **Reuse buffers**: Pool allocations for logits transfer

### Optimization Results (2025-12-13)

| Change | Before | After | Improvement |
|--------|--------|-------|-------------|
| Remove redundant Sync + unsafe cast | 157 tok/s | 162 tok/s | +3% |
| Sync time | 170ms | 166ms | -4ms |

The remaining gap (~162 vs 258 tok/s) likely requires:
- GPU-side argmax for temp=0 (eliminate 128KB transfer)
- Fix mid-layer sync issue (reduce batch count from 1320 to 660)

## References

- llama.cpp kernel: `ggml/src/ggml-metal/ggml-metal.metal:3230` (`mul_vec_q_n_f32_impl`)
- llama.cpp config: `ggml/src/ggml-metal/ggml-metal-impl.h:11-12`
- Vexel kernel: `inference/backend/metal/metal_bridge_darwin.m:237` (`matvec_q4_0_multi_output_f32`)
