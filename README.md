# Vexel

## Perf/Correctness Harness

Use the provided harness to compare Vexel (Metal) against llama.cpp on a fixed prompt set.

### Requirements
- Model: default `models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf` (override with `MODEL_PATH`).
- Vexel binary: default `./vexel_metal` (override with `VEXEL_BIN`).
- llama.cpp binary: default `llama-cli` (override with `LLAMA_BIN`).

### Run
```bash
bash scripts/perf_harness.sh
```
Environment overrides:
- `MODEL_PATH` – GGUF model path
- `VEXEL_BIN` – path to Vexel binary
- `LLAMA_BIN` – path to llama.cpp binary
- `OUT_DIR` – output directory (default `perf_reports`)
- `VEXEL_FA2_MIN_SEQ` – override Flash Attention 2 minimum seq length (default 32)

The harness writes a Markdown report with throughput numbers and raw logs to `perf_reports/report-<timestamp>.md`. Copy the latest results into `plan.md` next to the completed task and append to `.conductor/status.md` per the workflow.
