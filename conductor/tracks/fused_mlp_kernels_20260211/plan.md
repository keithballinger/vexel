# Track Plan: Fused MLP Kernels

## Phase 1: Implementation
- [ ] Task: Create `metal_fused_mlp_kernel`
    - Combine W1 (Gate) and W3 (Up) projections.
    - Fuse with SiLU/GELU activation function.
- [ ] Task: Backend Exposure
    - Expose `fusedMLP` in `backend.go`.

## Phase 2: Verification
- [ ] Task: Functional Tests
    - Verify output matches non-fused path.
- [ ] Task: Benchmark
    - Measure throughput gain for the MLP block.
