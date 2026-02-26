# Track Plan: Performance Optimization

## Phase 1: Investigation & Profiling
- [x] Task: Baseline Profile
    - Run detailed benchmarks/traces to identify current bottlenecks.
- [x] Task: Kernel Analysis
    - Analyze `metal_bridge_darwin.m` kernel implementations for optimization potential (e.g., register pressure, occupancy).

## Phase 2: Implementation [checkpoint: 295b3cd]
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

## Phase 3: Verification [checkpoint: 6df34d5]
- [x] Task: Throughput Benchmark
    - Measure `tokens/second` improvements for various sequence lengths/batches.
    - **Original results (TinyLlama 1.1B, no longer on disk):**
        - Prefill: 627.6 -> 785.3 tok/s -> ~~Regression to ~160 tok/s~~ → **Fixed (regression resolved)**
        - Decode: 90.9 -> 125.1 tok/s -> **138 tok/s (+51%)**
    - **New benchmark results (LLaMA 2 7B Q4_0, ~3.7GB weights):**
        - Prefill throughput (tok/s): 5tok=83.2, 32tok=176.4, 128tok=202.7
        - Prefill latency: 60ms (5tok), 181ms (32tok), 631ms (128tok)
        - Decode throughput: **44.2 tok/s** (22.3ms/token median)
        - Decode scaling: stable across context lengths 16-128 (~43-44 tok/s)
        - Prefill regression **confirmed fixed**: throughput scales with seqLen (83→176→203 tok/s)
    - **Test file:** `inference/runtime/throughput_bench_test.go` (4 tests, all pass)
- [x] Task: Latency Benchmark
    - Measure `time-to-first-token` improvements.
    - **Results (LLaMA 2 7B Q4_0):**
        - TTFT (5 token prompt): **79.3 ms** (prefill=58ms + decode=21ms)
        - TTFT (32 token prompt): **205.0 ms** (prefill=182ms + decode=23ms)
        - TTFT (128 token prompt): **655.6 ms** (prefill=632ms + decode=24ms)
        - Per-token decode: p50=22.9ms, p99=25.2ms, jitter ratio=1.10
        - First decode latency **consistent** across prompt lengths (~21-24ms)
    - **Test file:** `inference/runtime/latency_bench_test.go` (2 tests, all pass)
