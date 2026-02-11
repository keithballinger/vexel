# Track Plan: Metal Q5_K Support

## Phase 1: Kernel Implementation
- [x] Task: Implement Q5_K MatVec Kernel
    - Port `Q5_K` matvec logic from `llama.cpp` (or reference) to `inference/backend/metal/metal_bridge_darwin.m`.
    - Ensure correct handling of `Q5_K` block layout and dequantization.
- [x] Task: Expose Kernel to Go
    - Update `metal_bridge.h` and `backend.go` to add `MatMulQ5_K` support.
    - Create `matvecQ5KPipeline`.

## Phase 2: Loader & Runtime Integration
- [x] Task: Update Loader for Q5_K
    - Modify `loader.go` to allow `Q5_K` tensors to be loaded as `NewQuantTensor(..., tensor.Q5_K)`.
    - Ensure `isWeightMatrix` returns true for these tensors.
- [x] Task: Update BlockRuntime
    - Update `BlockRuntime.Execute` to dispatch `MatMulQ5_K` when `w.QuantProfile() == Q5_K`.

## Phase 3: Verification
- [x] Task: Verify Correctness
    - Run `TestPhi2MetalParity` to ensure `qkv_proj` (which uses Q5_K) produces correct output.
- [x] Task: Benchmark
    - Run `BenchmarkPhi2Metal` to confirm performance gains.
