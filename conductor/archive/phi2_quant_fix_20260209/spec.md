# Track Spec: Phi-2 Metal Quantization Fix

## Overview
Currently, the Phi-2 Metal backend falls back to FP32 weights because enabling `Q4_K` (4-bit quantization) results in incorrect output (all zeros or garbage). This fallback significantly limits performance due to memory bandwidth saturation. This track aims to debug and fix the quantized inference path for Phi-2 to unlock the expected 30-50 tokens/sec throughput on Apple Silicon.

## Goals
1.  **Enable Q4_K Support:** Successfully run `TestPhi2MetalParity` using `Q4_K` weights (via `LoadWeights`) with correct output.
2.  **Performance Boost:** Achieve >30 tokens/sec throughput on M4 Pro by reducing memory bandwidth usage (from 10.8GB/token to ~2.5GB/token).
3.  **Correctness:** Maintain exact (or high-fidelity) parity with CPU reference outputs when using quantized weights.

## Scope
*   **Investigation:** Determine why `Q4_K` kernels produce zeros/garbage for Phi-2 (biases, layout, or kernel bug).
*   **Kernel Fixes:** Modify `matvec_q4k` or `AddBias` kernels if necessary.
*   **Integration:** Ensure `BlockRuntime` correctly dispatches quantized operations for Phi-2.

## Key Hypotheses
*   **Bias Addition Issue:** The `Q4_K` matmul kernel might be working, but the subsequent `AddBias` (which expects FP32 inputs?) might be failing or receiving data in a format it doesn't expect (e.g., if the kernel output stride is different).
*   **Scale/Min Mismatch:** Phi-2's GGUF quantization parameters might differ from what the `Q4_K` kernel expects (though unit tests pass, integration fails).
*   **Synchronization:** A missing sync between the quantized matmul and the bias addition.

## Acceptance Criteria
*   `TestPhi2MetalParity` passes with `m.LoadWeights(modelPath)` (quantized mode).
*   `BenchmarkPhi2Metal` shows >25 tok/s throughput.
