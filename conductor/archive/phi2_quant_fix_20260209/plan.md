# Track Plan: Phi-2 Metal Quantization Fix

## Phase 1: Investigation & Reproduction
- [x] Task: Reproduce Q4_K Failure
    - Create a minimal reproduction test case in `inference/runtime/phi2_quant_test.go` that loads the model with `LoadWeights` (quantized) and asserts failure (all zeros).
    - Inspect intermediate tensors (output of MatMul vs output of AddBias) to pinpoint where the values go to zero.
- [x] Task: Debug Bias Integration
    - Verify if `AddBias` is receiving valid inputs from `MatMulQ4_K`.
    - Check if `MatMulQ4_K` output is compatible with `AddBias` (strides, data types).

## Phase 2: Implementation & Fixes
- [x] Task: Fix Q4_K/Q6_K Kernels or Integration
    - Implement the necessary fixes in `metal_bridge_darwin.m` or `block.go`.
    - Ensure `Q6_K` (used for LM Head) is also working.
- [x] Task: Verify Correctness with Quantization
    - Run `TestPhi2MetalParity` with quantized weights.
    - Ensure output matches (or is close enough to) reference.

## Phase 3: Performance Validation
- [x] Task: Benchmark Quantized Performance
    - Run `BenchmarkPhi2Metal` with quantized weights.
    - Compare against the 8.4 tok/s FP32 baseline.
- [x] Task: Conductor - User Manual Verification (Protocol in workflow.md)
