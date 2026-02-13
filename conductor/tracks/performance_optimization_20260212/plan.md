# Track Plan: Performance Optimization

## Phase 1: Investigation & Profiling
- [ ] Task: Baseline Profile
    - Run detailed benchmarks/traces to identify current bottlenecks.
- [ ] Task: Kernel Analysis
    - Analyze `metal_bridge_darwin.m` kernel implementations for optimization potential (e.g., register pressure, occupancy).

## Phase 2: Implementation
- [ ] Task: FlashAttention Tune
    - Tune `flash_attention` kernel block sizes/SIMD group usage for M-series.
- [ ] Task: Scheduler Refactor
    - Optimize `Scheduler.Run` loop and overhead for high concurrency.

## Phase 3: Verification
- [ ] Task: Throughput Benchmark
    - Measure `tokens/second` improvements for various sequence lengths/batches.
- [ ] Task: Latency Benchmark
    - Measure `time-to-first-token` improvements.
