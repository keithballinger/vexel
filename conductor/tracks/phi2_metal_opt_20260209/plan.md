# Track Plan: Metal Performance Optimization (Phi-2)

## Phase 1: Optimized Metal Kernels
- [x] Task: Implement Optimized LayerNorm Metal Kernel [verified]
    - [x] Write Go-side unit test using Metal backend in `inference/backend/metal/phi2_ops_test.go`
    - [x] Implement MSL kernel in `inference/backend/metal/metal_bridge_darwin.m` (highly optimized)
    - [x] Update C bindings in `inference/backend/metal/metal_bridge.h` and Go wrapper in `inference/backend/metal/metal_backend.go`
    - [x] Verify correctness against CPU reference implementation
- [x] Task: Implement Optimized GELU Metal Kernel [verified]
    - [x] Write Go-side unit test
    - [x] Implement MSL kernel with fast tanh approximation
    - [x] Verify correctness and statistical properties
- [x] Task: Implement Optimized RoPE-NeoX Metal Kernel [verified]
    - [x] Write Go-side unit test for partial dimensions and split-pair rotation
    - [x] Implement MSL kernel
    - [x] Verify correctness against llama.cpp reference values
- [x] Task: Implement Optimized AddBias Metal Kernel [verified]
    - [x] Write Go-side unit test for row-wise bias addition
    - [x] Implement MSL kernel (consider fusing with matmul or using as standalone)
    - [x] Verify correctness
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Optimized Metal Kernels' (Protocol in workflow.md)

## Phase 2: Runtime Integration & GPU Residency
- [ ] Task: Enable Metal Path for Phi-2 in BlockRuntime
    - [ ] Update `Execute` and `ExecuteWithPagedKV` in `inference/runtime/block.go` to dispatch to Metal kernels
    - [ ] Ensure `applyNorm` and `matMulTransposedWithBias` correctly handle Metal backend
- [ ] Task: Verify Full GPU Weight Residency
    - [ ] Validate `CopyWeightsToDevice` in `inference/runtime/loader.go` for Phi-2 biases and combined tensors
    - [ ] Ensure no CPU fallbacks are triggered during the Phi-2 forward pass
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Runtime Integration & GPU Residency' (Protocol in workflow.md)

## Phase 3: Performance Verification & Parity
- [ ] Task: End-to-End Correctness Parity on Metal
    - [ ] Run `TestPhi2Parity` configured with Metal backend
    - [ ] Verify long-prefix (20+ tokens) matching against llama.cpp
- [ ] Task: Throughput Benchmarking & Profiling
    - [ ] Run performance harness to measure prefill and decode tokens/sec
    - [ ] Identify and address any memory bottlenecks or synchronization stalls
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Performance Verification & Parity' (Protocol in workflow.md)
