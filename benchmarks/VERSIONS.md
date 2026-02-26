# Benchmark Environment Versions

Generated: 2026-02-26
Machine: Apple M3 Max
Memory: 128 GB
macOS: $(sw_vers output needed)

## Hardware

| Spec              | Value                    |
|-------------------|--------------------------|
| Chip              | Apple M3 Max             |
| GPU Cores         | 40                       |
| Memory            | 128 GB Unified           |
| Memory Bandwidth  | 400 GB/s (theoretical)   |
| macOS             | Sequoia                  |

## Engines

| Engine     | Version   | Install Method          |
|------------|-----------|-------------------------|
| Vexel      | HEAD      | go build -tags metal    |
| mlx-lm     | 0.30.7    | pip (benchmarks/.venv)  |
| mlx-core   | 0.30.6    | pip (benchmarks/.venv)  |
| llama.cpp  | b8140     | brew install llama.cpp  |
| Ollama     | 0.13.5    | ollama.ai installer     |
| vllm-mlx   | 0.2.5     | pip (benchmarks/.venv)  |

## Python

- Python 3.x (system)
- Venv: benchmarks/.venv

## Standard Benchmark Models

The following models should be downloaded for benchmarking:

| Model               | Format | Size   | Use Case                    |
|---------------------|--------|--------|-----------------------------|
| LLaMA 3.1 8B Q4_K_M| GGUF   | ~4.7GB | Primary decode benchmark    |
| Mistral 7B Q4_K_M  | GGUF   | ~4.4GB | Secondary decode benchmark  |
| LLaMA 2 7B Q4_0    | GGUF   | ~3.8GB | Vexel native format         |

## Notes

- llama.cpp b8140 reports "tensor API disabled for pre-M5 and pre-A19 devices"
  (expected — Neural Accelerator support requires M5+)
- Ollama service must be started manually: `ollama serve`
- vllm-mlx includes mlx-vlm for multimodal benchmarks (optional)
- All Python packages installed in isolated venv at `benchmarks/.venv`
