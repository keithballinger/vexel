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

## Phase 1: Competitor Research & Tooling
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

## Phase 2: Single-Stream Benchmarks
- [ ] Task: Decode throughput comparison
    - Model: LLaMA 3.1 8B Q4_K_M (community standard), Mistral 7B Q4_K_M, LLaMA 2 7B Q4_0.
    - Prompt: 32 tokens. Generate: 512 tokens. Sampling: temp=0 (greedy).
    - Measure: tok/s, ITL P50/P90/P99, peak memory.
    - Run on all competitors: Vexel, MLX, llama.cpp, Ollama.
    - Report memory bandwidth utilization: `(bytes_read / time) / theoretical_bandwidth`.
- [ ] Task: Prefill throughput comparison
    - Same models. Prompt lengths: 128, 512, 2048, 8192 tokens.
    - Measure: prefill tok/s and TTFT at each length.
    - This is where batched quantized matmul (Track 4) should shine.
- [ ] Task: Model load time comparison
    - Measure cold-start time: process launch to first token generated.
    - Include model loading, weight transfer to GPU, KV cache allocation.
    - Report for each engine.

## Phase 3: Batched Throughput Benchmarks
- [ ] Task: Concurrent request throughput
    - Vexel's key differentiator: Go scheduler with continuous batching.
    - Compare against vllm-mlx (also does batching) and sequential engines.
    - Concurrent clients: 1, 2, 4, 8, 16 simultaneous generation requests.
    - Measure: aggregate tok/s, per-request latency P50/P99, peak memory.
    - Model: LLaMA 3.1 8B Q4_K_M.
- [ ] Task: Long-context scaling
    - Context lengths: 4K, 8K, 16K, 32K tokens.
    - Measure: TTFT, decode throughput degradation, memory usage.
    - Compare with MLC-LLM (paged KV specialist) if available.
    - This benchmarks the Paged KV Batching track (Track 2).
- [ ] Task: Prefix caching efficiency
    - Run 10 requests with shared 1K-token system prompt + unique user prompts.
    - Measure TTFT improvement from prefix cache hits vs cold prefill.
    - Compare with vllm-mlx which also implements prefix caching.

## Phase 4: Analysis & Reporting
- [ ] Task: Generate comparison report
    - Create `benchmarks/RESULTS.md` with tables and analysis.
    - Include: hardware specs, software versions, raw numbers, derived metrics.
    - Compute memory bandwidth utilization % for each engine.
    - Identify where Vexel leads and where it trails.
- [ ] Task: Visualization
    - Generate charts (bar charts for tok/s, line charts for context scaling).
    - Use Go or Python script to produce SVG/PNG from benchmark JSON.
    - Include in report and optionally in README.
- [ ] Task: Identify optimization targets
    - For each benchmark where Vexel trails a competitor, document the gap.
    - Propose concrete optimizations (kernel tuning, scheduling policy, memory layout).
    - Feed findings back into future track planning.
- [ ] Task: Update README with competitive positioning
    - Add a Performance Comparison section referencing the benchmark report.
    - Honest reporting: show where Vexel leads AND where it's still behind.
    - Link to full benchmark methodology and raw data.
