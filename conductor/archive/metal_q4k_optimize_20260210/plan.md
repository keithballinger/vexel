# Track Plan: Optimize Q4_K Kernels (NR2)

## Phase 1: Implementation
- [x] Task: Implement `matvec_q4k_nr2_f32` Kernel
    - Port logic from `matvec_q5k_nr2_f32` but adapted for `Q4_K` block format (144 bytes, no `qh`).
    - Use SIMD shuffle or threadgroup reduction for partial sums.
- [x] Task: Expose to Go
    - Update `metal_bridge.h` and `backend.go` to add `matvecQ4KNR2Pipeline`.

## Phase 2: Integration & Verification
- [x] Task: Update Dispatch
    - Modify `Backend.MatMulQ4_K` to use the `NR2` kernel for `m == 1`.
- [x] Task: Correctness Check
    - Run `TestPhi2MetalParity` to ensure no regression in quality.
- [x] Task: Benchmark
    - Run `BenchmarkPhi2Metal` to measure speedup.
