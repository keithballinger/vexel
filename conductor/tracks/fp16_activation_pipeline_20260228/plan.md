# Track Plan: FP16 Activation Pipeline

Currently Vexel stores activations as float32 throughout the forward pass.
The GEMM kernel reads float32 from device, converts to half in shared memory.
Device reads for A (activations) are 2× larger than necessary.

**Target: ≥800 tok/s prefill at seqLen=128 via halved activation bandwidth.**

## Background

In the current TILE_K=32 GEMM kernel (commit cd22b51):
- A (activations): float32 device → 2× float4 device loads → 2× half4 shared stores
- B (weights): Q4_0 device → dequant to half in shared memory
- Shared memory: sa[32×32]=2048 bytes (half), sb[64×32]=4096 bytes (half)

If activations were stored as float16 in device memory:
- A loading halves: 2× half4 device loads instead of 2× float4
- Device bandwidth per tile: 1024 bytes (A) + 4224 bytes (B) = 5248 bytes
  (vs current 2048 + 4224 = 6272 bytes) → **16% less bandwidth per tile**
- Or: keep tile sizes, use freed bandwidth for larger tiles / higher throughput

The challenge is numerical stability: accumulating 32 layers of float16 matmul
results can degrade output quality. Key insight: only inter-layer activations
need to be FP16. Accumulators remain FP32 (simdgroup_float8x8).

## Phase 0: Numerical Analysis
- [ ] Task 0.1: Measure baseline float32 output quality
    - Run prefill with float32 activations, record output logits for
      reference prompts (5, 32, 128 tokens)
    - Save reference logits for comparison
- [ ] Task 0.2: Simulate FP16 activations in CPU
    - Add a test that truncates all inter-layer activations to float16
    - Compare output logits to float32 reference
    - Measure max absolute error, mean squared error, perplexity impact
    - Key dimensions to test: attention output, MLP output, residual stream
- [ ] Task 0.3: Identify precision-critical paths
    - RMSNorm: requires FP32 for variance computation (keep FP32 internally)
    - Residual additions: FP16 + FP16 may lose bits in the residual stream
    - Softmax: already FP32 in FA2v2 kernel
    - Decision point: full FP16 pipeline or selective FP16 (matmul inputs only)

## Phase 1: FP16 Scratch Allocator
- [ ] Task 1.1: Add FP16 scratch allocation support
    - Extend ScratchAlloc to allocate half-precision buffers
    - Add DevicePtr metadata for element type (FP32 vs FP16)
    - Ensure alignment requirements for half* access
- [ ] Task 1.2: FP16 format conversion kernels
    - Write f32_to_f16 and f16_to_f32 conversion kernels
    - These run at layer boundaries to convert activation format
    - Benchmark conversion overhead (should be <0.1ms per layer)
- [ ] Task 1.3: Selective FP16 activation storage
    - Store matmul outputs (Q, K, V, attn_out, gate, up, down) as FP16
    - Keep residual stream in FP32 (critical for numerical stability)
    - RMSNorm inputs: convert FP16→FP32 at start of each layer

## Phase 2: FP16-Native GEMM Kernel
- [ ] Task 2.1: Write matmul_q4_0_simdgroup_f16_in kernel variant
    - Same tile dimensions (32×64×32, 128 threads, 4 SGs)
    - A loading: direct half4 device loads (no float4→half4 conversion)
    - Accumulators: remain FP32 (simdgroup_float8x8)
    - Output: write as half* instead of float*
- [ ] Task 2.2: FP16 dispatch routing
    - Add MatMulQ4_0_F16 method to backend
    - Route based on activation precision flag
    - Wire into forward pass for matmul-output activations
- [ ] Task 2.3: Correctness tests
    - Compare FP16 GEMM output to FP32 reference
    - Max acceptable error: 1e-3 (typical for FP16 matmul)
    - Test all key dimensions: 4096×4096, 11008×4096, 4096×11008, 32000×4096
- [ ] Task 2.4: Throughput benchmarks
    - Per-dimension GFLOPS with FP16 activations
    - End-to-end prefill at seqLen=128
    - Target: ≥800 tok/s (up from 717)

## Phase 3: Integration & Quality Verification
- [ ] Task 3.1: Full forward pass with FP16 pipeline
    - Enable FP16 activations throughout inference
    - Run standard test prompts, verify output quality
    - Perplexity test on reference dataset if available
- [ ] Task 3.2: End-to-end benchmarks
    - Prefill: seqLen=5, 32, 128, 385
    - Decode: tok/s at ctx=16, 128, 512
    - Memory usage comparison (should decrease ~30-40% for activations)
- [ ] Task 3.3: Update tracking docs

## Reference: Key Files

- GEMM kernel: `inference/backend/metal/metal_bridge_darwin.m` (lines 757-890)
- GEMM dispatch: `inference/backend/metal/backend.go` (lines 595-631)
- Scratch allocator: `inference/backend/metal/backend.go` (search ScratchAlloc)
- Forward pass: `inference/runtime/block.go`
- RMSNorm kernel: `inference/backend/metal/metal_bridge_darwin.m` (search rms_norm)
