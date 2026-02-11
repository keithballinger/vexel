# Track Spec: Metal Q5_K Support

## Overview
Phi-2 models (and others using standard GGUF quantization) often use `Q5_K` quantization for the attention output or QKV projection tensors. Currently, Vexel lacks Metal kernel support for `Q5_K`, forcing a fallback to FP32 dequantization during model loading. This fallback significantly increases memory bandwidth requirements (4 bytes/param vs ~0.6 bytes/param), limiting inference performance.

## Goals
1.  **Implement Q5_K Kernel:** Create an efficient Metal `MatVec` kernel for `Q5_K` quantized tensors (targeting M=1 decode performance).
2.  **Loader Support:** Update `loader.go` to recognize and load `Q5_K` tensors as raw quantized data instead of dequantizing to F32.
3.  **Performance Improvement:** Achieve >30 tokens/sec on M4 Pro for Phi-2 (up from ~19 tok/s) by reducing QKV loading bandwidth.

## Technical Details
*   **Format:** `Q5_K` uses super-blocks containing blocks of values.
    *   Typically uses 6-bit quantization for scales and mins? Or is it the complex K-quant format?
    *   Need to reference `llama.cpp`'s `k_quants.h` / `ggml-metal.metal` implementation for exact layout.
*   **Kernel:** `kernel_matvec_q5_k_f32` (similar to existing `q4_k`).

## Acceptance Criteria
*   `TestPhi2MetalParity` passes with `LoadWeights` using `Q5_K` tensors loaded raw.
*   `BenchmarkPhi2Metal` shows significant speedup (>30 tok/s).
