# Track Spec: Verify SDPAF16 Correctness

## Overview
Vexel supports `SDPAF16` (Attention in FP16), which is faster than F32. However, Phi-2 uses a Head Dimension (`headDim`) of **80**. Standard kernels often optimize for 64 or 128. We need to verify if the existing `SDPAF16` kernel handles `headDim=80` correctly (e.g., via padding or generic loops) and enable it for Phi-2 if not already used.

## Goals
1.  **Verify Kernel:** Check if `metal_sdpa_f16` supports non-power-of-2 head dimensions like 80.
2.  **Enable Optimization:** If supported, update `BlockRuntime` to use `SDPAF16` for Phi-2 (currently it might be using F32 SDPA due to `useFP16Path` logic).
3.  **Performance:** Small boost in attention computation speed.

## Technical Details
*   Phi-2: 32 heads, headDim 80.
*   Kernel: `sdpa_decode_f16`. Does it assume `headDim` is multiple of 32? (80 is not multiple of 32? 80 = 32*2 + 16).
*   If kernel uses SIMD shuffle reduction, it might expect 32 threads per headDim chunk.

## Acceptance Criteria
*   `TestPhi2MetalParity` passes with `SDPAF16` enabled.
