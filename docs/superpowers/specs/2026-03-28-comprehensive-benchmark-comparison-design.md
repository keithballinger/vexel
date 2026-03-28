# Comprehensive Benchmark Comparison Design

**Goal:** Create a benchmark suite that measures Vexel's performance across all modes (standard, Medusa, draft-model speculative, batched decode) against llama.cpp, across two model sizes and the full context length range.

**Audience:** Personal tracking and optimization guidance (single author).

**Hardware:** Apple M3 Max, 128 GB unified memory.

---

## Script: `benchmarks/full_comparison.sh`

Single entry point with subcommands:

```
./full_comparison.sh all           # Run everything
./full_comparison.sh decode        # Standard decode comparison
./full_comparison.sh speculative   # Speculative modes (Medusa + draft-model)
./full_comparison.sh context       # Context scaling sweep (16-4096)
./full_comparison.sh batched       # Batched multi-client throughput
```

---

## Model Management

Models stored in `benchmarks/models/` (gitignored). Script checks for required models on startup and downloads missing ones from HuggingFace.

| Model | Role | Source |
|-------|------|--------|
| Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | Target (large) | Symlink from `../llama.cpp/models/` or download |
| qwen2.5-0.5b-instruct-q4_k_m.gguf | Target (small) | Symlink from `../llama.cpp/models/` or download |
| TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf | Draft model | Download from HuggingFace |

**Safety:** Script runs `git check-ignore benchmarks/models/` at startup and aborts if the directory is not gitignored. Models are never committed.

---

## Benchmark Configurations

### Standard Decode (both engines, both models)

- Single sequence, greedy (temp=0), 128 generated tokens
- 3 warmup runs, 5 measured runs
- Metrics: decode tok/s, prefill tok/s, time-to-first-token

### Speculative Decode

| Mode | Vexel | llama.cpp |
|------|-------|-----------|
| Medusa (online) | `--medusa` with 50-token warmup | N/A |
| Draft-model | `--draft-model tinyllama-1.1b` | `llama-speculative -md tinyllama-1.1b` |

- Target model: LLaMA 3.1 8B only (speculation overhead not meaningful on 0.5B)
- Metrics: decode tok/s, acceptance rate (from verbose output), effective speedup vs standard
- Medusa warmup: generate 50 tokens first to warm online-trained heads, then measure on the next 128

### Batched Decode (8B model only)

- Both engines run in server mode
- Concurrency levels: 1, 2, 4, 8 simultaneous requests
- Each request: 64 generated tokens, greedy
- Send requests via background `curl` processes
- Metrics: aggregate tok/s (total tokens / wall time), per-request latency
- Vexel: `./vexel --model M serve --max-batch-size N`
- llama.cpp: `llama-server -m M -np N`

### Context Scaling (8B model only)

- Context lengths: 16, 64, 256, 512, 1024, 2048, 4096
- Prefill a synthetic prompt of N tokens, then decode 16 tokens
- 3 runs per context length
- Metrics: decode tok/s at each length, degradation % vs ctx=16 baseline
- Both engines measured at each length

---

## Output and Reporting

### JSONL Output

All output goes to `benchmarks/results/YYYY-MM-DD/` with one JSONL file per configuration:

- `standard_decode.jsonl`
- `speculative.jsonl`
- `batched.jsonl`
- `context_scaling.jsonl`

Each JSONL line contains: engine, model, configuration, run number, prompt tokens, generated tokens, decode tok/s, prefill tok/s, and configuration-specific fields (acceptance rate, concurrency level, context length, etc.).

### Report Generation

`analyze.py --comparison benchmarks/results/YYYY-MM-DD/` reads all JSONL from a run directory and produces a unified markdown report containing:

1. **Summary table**: all configurations x both engines, tok/s and relative performance
2. **Speculative decoding table**: acceptance rates, speedup vs standard, Medusa vs draft-model
3. **Batched throughput table**: aggregate tok/s at 1/2/4/8 concurrency
4. **Context scaling table**: degradation % at each context length, both engines side-by-side
5. **Hardware info**: machine, OS, memory (auto-detected via `sysctl`)

Report written to `benchmarks/results/YYYY-MM-DD/report.md` and printed to stdout.

`benchmarks/RESULTS.md` is updated manually after reviewing results. The script does not auto-modify the tracked file.

---

## llama.cpp Integration

### Binary Discovery

Priority order:
1. `llama-cli` / `llama-server` / `llama-speculative` in `$PATH`
2. `../llama.cpp/build/bin/` relative to vexel repo root

If not found, prints which binaries are missing and skips those comparisons. Vexel-only benchmarks still run.

### Vexel Binary

Runs `make build` if `./vexel` doesn't exist or is older than source files.

### Output Parsing

| Engine | Source | Parsing |
|--------|--------|---------|
| Vexel standard | `--verbose` stdout | `prefill: X tok/s \| decode: Y tok/s` |
| Vexel Medusa | `--verbose` stdout | Same + `acceptance=X% speedup=Yx` from Medusa metrics |
| Vexel draft-model | `--verbose` stdout | Same + speculative metrics |
| llama.cpp CLI | stderr | `llama_perf_context_print` eval/prompt timings |
| llama.cpp speculative | stderr | `speculative: accept rate` |
| llama.cpp server | HTTP response timing | Request duration / tokens generated |

---

## Full Benchmark Matrix

| Configuration | Vexel | llama.cpp | Models | Context |
|--------------|-------|-----------|--------|---------|
| Standard decode | Yes | Yes | 0.5B, 8B | default |
| Medusa speculation | Yes | N/A | 8B | default |
| Draft-model speculation | Yes | Yes | 8B + 1.1B draft | default |
| Batched (1/2/4/8 clients) | Yes | Yes | 8B | default |
| Context scaling (16-4096) | Yes | Yes | 8B | 16, 64, 256, 512, 1024, 2048, 4096 |
