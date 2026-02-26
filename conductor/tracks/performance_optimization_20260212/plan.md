# Track Plan: Performance Optimization

## Phase 1: Investigation & Profiling
- [x] Task: Baseline Profile
    - Run detailed benchmarks/traces to identify current bottlenecks.
- [x] Task: Kernel Analysis
    - Analyze `metal_bridge_darwin.m` kernel implementations for optimization potential (e.g., register pressure, occupancy).

## Phase 2: Implementation
- [x] Task: FlashAttention Tune
    - Tune `flash_attention` kernel block sizes/SIMD group usage for M-series.
    - Implemented single-pass loop optimization to avoid recomputing Q*K dot product.
    - **Note:** Reverted single-pass optimization due to numerical instability/garbage output.
- [x] Task: Scheduler Refactor
    - Optimize `Scheduler.Run` loop and overhead for high concurrency.
    - Implemented event-driven `signal` channel to replace busy-wait ticker.
- [x] Task: Fix Prefill Regression & Correctness (Part 1)
    - Investigate garbled output and low prefill throughput (~160 tok/s vs ~700 tok/s baseline).
    - **Found:** `ExecuteWithGPUKV` was passing uninitialized (recycled) `kF16Ptr`/`vF16Ptr` to SDPA.
    - **Fix:** Added explicit F32->F16 conversion for K/V in `inference/runtime/block.go`.
    - **Status:** Output changed from random noise ("kalutterouss...") to hallucination ("edeuthat..."). Prefill speed still slow (~160 tok/s).
- [x] Task: Fix Prefill Regression & Correctness (Part 2)
    - Investigate why `MatMul` kernels for prefill (`m > 1`) produce hallucinated output while decode (`m = 1`) works.
    - **Found (Bug 1):** `ExecuteWithGPUKV` scratch sub-allocation used `!scratch.DevicePtr().IsNil()` (always true for GPU), causing all intermediate buffers to alias offset 0 in the same MTLBuffer. Fixed by using `scratchPtr.Location() == tensor.CPU` and individual `b.backend.Alloc()` for GPU.
    - **Found (Bug 2 — root cause):** `metal_matmul_q4_0_batched_f32` created its own command buffer and committed it immediately, bypassing Metal batch mode. When batch mode is active, the preceding RMSNorm is encoded in the batch encoder but NOT committed — so the MatMul reads stale/zero data from the normOut buffer. `metal_matmul_q4_0_simdgroup_f32` had a similar issue (custom finish logic instead of `finish_encode`).
    - **Fix:** Changed both functions to use `get_encoder()`/`finish_encode()` to participate in command buffer batching, matching all other kernel dispatch functions.
    - **Verification:** All 3 prefill regression tests pass (TestPrefillVsSequentialDecode, TestPrefillMinimal, TestPrefillFP32VsFP16). Max logit diff between sequential and prefill: 0.005643.

## Phase 3: Verification
- [ ] Task: Throughput Benchmark
    - Measure `tokens/second` improvements for various sequence lengths/batches.
    - **Results:**
        - Prefill: 627.6 -> 785.3 tok/s -> **Regression to ~160 tok/s**
        - Decode: 90.9 -> 125.1 tok/s -> **138 tok/s (+51%)**
- [ ] Task: Latency Benchmark
    - Measure `time-to-first-token` improvements.
