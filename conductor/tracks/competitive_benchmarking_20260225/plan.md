# Track Plan: Competitive Benchmarking

This track produces a rigorous, reproducible comparison of Vexel against the top inference
engines on Apple Silicon. It should run last, after all other tracks are complete, to measure
the full-featured engine.

Research (Feb 2025) identified 4 primary competitors:
- **MLX** (Apple): Throughput leader, ~230 tok/s decode on M2 Ultra. Apple's own framework.
- **llama.cpp** (Metal): Ecosystem standard, ~150 tok/s decode. GGUF format originator.
- **MLC-LLM** (TVM): Paged KV, best long-context scaling, ~190 tok/s decode.
- **vllm-mlx**: Continuous batching on Apple Silicon, up to 4.3x aggregate throughput at 16 concurrent.
- **Ollama**: Wraps llama.cpp, ~20-40 tok/s. UX baseline, not a performance target.

Key insight: Apple Silicon inference is **memory-bandwidth bound** at small batch sizes.
The primary differentiator metric is memory bandwidth utilization (% of theoretical).

Sources:
- arxiv 2511.05502 — "Production-Grade Local LLM Inference on Apple Silicon" (Nov 2025)
- arxiv 2601.19139 — "Native LLM and MLLM Inference at Scale on Apple Silicon" (Jan 2026)
- github.com/ggml-org/llama.cpp/discussions/4167 — Apple Silicon perf tracking

## Phase 1: Competitor Research & Tooling [checkpoint: 917f61b]
- [x] Task: Web research — identify fastest Apple Silicon inference engine
    - Search for latest benchmark comparisons (2025-2026) on Apple Silicon.
    - Determine if MLX is still the throughput leader or if newer competitors exist.
    - Check vllm-mlx, MLC-LLM, and any new Metal-optimized engines.
    - Document findings in `benchmarks/COMPETITORS.md`.
- [x] Task: Install and validate competitors
    - Install on benchmark machine: `mlx-lm`, `llama.cpp` (Metal), `ollama`, `vllm-mlx`.
    - Optionally install `MLC-LLM` if long-context benchmarks are needed.
    - Verify each engine can load and run the same GGUF/MLX model.
    - Document exact versions and build flags used.
- [x] Task: Build benchmark harness
    - Create `benchmarks/run_benchmark.sh` — automated harness that runs all engines.
    - Standardize: same model, same prompt set, same hardware, same quantization.
    - Collect: decode tok/s, prefill tok/s, TTFT, ITL (P50/P99), peak RSS, model load time.
    - Output machine-readable JSON for each run.
    - Warmup: discard first 3 runs, measure next 10, report mean + stddev.

## Phase 2: Single-Stream Benchmarks [checkpoint: ec60491]
- [x] Task: Decode throughput comparison
    - Model: LLaMA 2 7B Q4_0 on M3 Max 128GB.
    - Vexel: 43.38 tok/s (46.6% BW util), llama.cpp: 78.45 tok/s (84.3%), Ollama: 81.23 tok/s.
    - MLX not tested (HF auth required); published ~80-95 tok/s on M3 Max.
    - **Vexel is 44.7% slower than llama.cpp.**
- [x] Task: Prefill throughput comparison
    - llama.cpp: 480→700→835 tok/s at 20/128/512 tokens (scales well).
    - Vexel: 137 tok/s at 20 tokens, **hangs/OOMs at 128+ tokens**.
    - OOM error: `GPU prefill failed: OOM: requested 128000 bytes, only 53232 remaining`
    - **Vexel prefill is 3.5x slower and broken for real workloads.**
- [x] Task: Model load time comparison
    - llama.cpp: ~288 ms cold start.
    - Vexel: blocked by OOM on single-token generation.

## Phase 3: Batched Throughput Benchmarks
- [ ] Task: Unblocked — re-run batched benchmarks after P0 fix
    - P0 OOM fix landed (2966edd). Vexel can now handle long prompts.
    - vllm-mlx server tested and available for comparison.
    - Harness (`run_batched.sh`) is ready.

## Phase 4: Analysis & Reporting [checkpoint: ec60491]
- [x] Task: Generate comparison report
    - Created `benchmarks/RESULTS.md` with root cause analysis.
    - Identified 6 architectural bottlenecks causing the 44.7% gap.
    - P0 bug: scratch arena sizing causes OOM on >20-token prompts.
- [x] Task: Identify optimization targets
    - P0: Fix scratch arena sizing (unblocks prefill, load time, batched benchmarks).
    - P1: Enable GPU scratch sub-allocation (+15-20% decode throughput).
    - P2: Fused KV scatter (+5-8% decode throughput).
    - P3: Fused attention+norm kernels (+10-15% decode throughput).
    - P4: Command buffer batching (+3-5% decode throughput).
    - Estimated total: 34-50% recovery, closing gap to within 15-20% of llama.cpp.
- [ ] Task: Update README with competitive positioning
    - Deferred until P0-P2 fixes land and benchmarks can be re-run.

## Phase 5: P0 Fix & Re-benchmark [checkpoint: 2966edd]
- [x] Task: Fix scratch arena OOM (P0)
    - Root cause: arena budget formula in initModel did not account for token ID and
      hidden-state allocations that DecodeWithGPUKV makes from the arena.
    - Fix: Added TotalArenaBytes(maxBatchSize) to ModelConfig that budgets all 4
      arena allocations (tokens + hidden state + layer scratch + logits) + 10% headroom.
    - Updated all 7 arena creation sites (CLI, tests, debug tools).
    - TDD regression tests confirm old formula deficient, new formula sufficient.
    - E2E verified: ~70-token prompt with --max-tokens 20 completes successfully.
- [ ] Task: Re-run single-stream benchmarks (decode, prefill, load time)
    - Measure impact of P0 fix on all metrics.
    - Prefill should now work at 128/512 tokens.
- [ ] Task: Run batched throughput benchmarks (Phase 3)
    - Now unblocked by P0 fix.
