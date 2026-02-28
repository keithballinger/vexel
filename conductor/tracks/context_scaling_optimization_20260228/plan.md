# Track Plan: Context Scaling Optimization

Vexel decode degrades ~10% from ctx=16→512 vs llama.cpp's -2.7% and MLX's -2.5%.
Prefill degrades ~11% from seqLen=128→385 (717→637 tok/s).

**Target: ≤3% decode degradation from ctx=16→512.**

## Background

After Flash Attention optimization (commit 334a491), decode degradation improved
from -24.5% to ~-10%. The remaining gap is structural:

1. **Per-simdgroup KV serialization**: Decode SDPA splits KV across 8 simdgroups,
   each iterating sequentially over kvLen/8 positions. No tiling/cooperation.
2. **Head-major KV cache layout**: `[numKVHeads, maxSeqLen, headDim]` means
   position strides grow with maxSeqLen (128KB at ctx=512 vs 4KB at ctx=16).
3. **Cross-simdgroup merge overhead**: Single-threaded merge in simdgroup 0.

The prefill FA2v2 kernel already demonstrates the fix: cooperative tile loading
where all 256 threads load KV tiles to shared memory, then all simdgroups compute.

## Phase 0: Context Scaling Baseline
- [ ] Task 0.1: Decode throughput vs context length sweep
    - Measure at ctx=16, 32, 64, 128, 256, 512, 1024
    - Record both tok/s and SDPA kernel time per token
- [ ] Task 0.2: Prefill throughput vs sequence length sweep
    - Measure at seqLen=32, 64, 128, 256, 385, 512
    - Record SDPA time as % of total forward pass
- [ ] Task 0.3: SDPA kernel timing isolation
    - Profile SDPA decode kernel alone at various kvLen values
    - Determine whether degradation is linear, quadratic, or stepwise

## Phase 1: Tiled KV Decode Kernel
- [ ] Task 1.1: Write `sdpa_flash_decode_f16_tiled` kernel
    - DECODE_TILE_KV=64 (or 32): KV positions per cooperative tile
    - All 256 threads cooperatively load K tile + V tile to shared memory
    - All 8 simdgroups compute Q·K and accumulate V on same tile
    - Online softmax with per-tile max/sum updates (same as FA2v2 prefill)
    - Single merge at end (no per-simdgroup interim merge)
    - Shared memory: 2 × TILE_KV × headDim × 2 bytes = 32KB for TILE=64
- [ ] Task 1.2: Wire tiled kernel into decode dispatch
    - Replace sdpa_flash_decode_f16 when kvLen ≥ threshold
    - Keep old kernel for very short contexts if tiled has overhead
- [ ] Task 1.3: Correctness tests
    - Test against CPU reference at kvLen=16, 64, 128, 256, 512, 1024
    - Test GQA configurations (32q/8kv, 32q/4kv)
    - Verify numerical stability of tiled online softmax
- [ ] Task 1.4: Benchmark context scaling
    - Decode throughput sweep ctx=16→1024
    - Target: ≤3% degradation at ctx=512 (matching llama.cpp)
    - Compare SDPA kernel time before/after

## Phase 2: Prefill SDPA Scaling (if Phase 1 confirms pattern)
- [ ] Task 2.1: Profile prefill SDPA at long sequences
    - seqLen=256, 385, 512: measure SDPA as % of forward pass
    - Determine if seqLen² scaling is the bottleneck
- [ ] Task 2.2: Block-wise Q tiling in FA2v2 (if needed)
    - Tile Q positions as well as KV positions
    - Reduces quadratic attention computation to tiled blocks
    - Implementation complexity: high (multiple online softmax merges)

## Phase 3: Integration & Verification
- [ ] Task 3.1: End-to-end decode benchmark at all context lengths
    - Measure with temperature=0, 50 generated tokens, ctx=16→1024
    - Compare degradation curve to llama.cpp and MLX
- [ ] Task 3.2: End-to-end prefill benchmark at all sequence lengths
    - seqLen=5, 32, 64, 128, 256, 385, 512
    - Verify no regression at short sequences
- [ ] Task 3.3: Update tracking docs

## Reference: Key Files

- Decode SDPA: `inference/backend/metal/metal_bridge_darwin.m` (lines 4624-4747)
- FA2v2 prefill (tiling reference): same file (lines 5766-5897)
- KV scatter kernel: same file (lines 4043-4127)
- KV cache management: `inference/backend/metal/backend.go`
- Paged KV (existing): same file, search for ReshapePagedKV
