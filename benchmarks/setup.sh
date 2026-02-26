#!/usr/bin/env bash
# Benchmark Environment Setup
# Target: Apple M3 Max 128GB
# Creates isolated Python venv for competitor engines.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_DIR="$SCRIPT_DIR/models"
VERSIONS_FILE="$SCRIPT_DIR/VERSIONS.md"

echo "=== Vexel Competitive Benchmark Setup ==="
echo "Machine: $(sysctl -n machdep.cpu.brand_string)"
echo "Memory: $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1024/1024/1024}')"
echo ""

# --- Python venv ---
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating Python venv..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/5] Python venv exists."
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# --- MLX ---
echo "[2/5] Installing mlx-lm..."
pip install mlx-lm --quiet
MLX_VERSION=$(python3 -c "import mlx_lm; print(mlx_lm.__version__)" 2>/dev/null || echo "unknown")
MLX_CORE=$(python3 -c "import mlx.core; print(mlx.core.__version__)" 2>/dev/null || echo "unknown")
echo "  mlx-lm: $MLX_VERSION, mlx-core: $MLX_CORE"

# --- llama.cpp ---
echo "[3/5] Installing llama.cpp via brew..."
if command -v brew &>/dev/null; then
    brew install llama.cpp 2>/dev/null || brew upgrade llama.cpp 2>/dev/null || true
    LLAMA_VERSION=$(llama-cli --version 2>&1 | head -1 || echo "unknown")
else
    echo "  WARNING: brew not found, install llama.cpp manually."
    LLAMA_VERSION="not installed"
fi
echo "  llama.cpp: $LLAMA_VERSION"

# --- Ollama ---
echo "[4/5] Checking Ollama..."
if command -v ollama &>/dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
else
    echo "  Install Ollama from https://ollama.ai"
    OLLAMA_VERSION="not installed"
fi
echo "  Ollama: $OLLAMA_VERSION"

# --- vllm-mlx ---
echo "[5/5] Installing vllm-mlx..."
pip install vllm-mlx --quiet 2>/dev/null || {
    echo "  WARNING: vllm-mlx install failed. Try: pip install vllm-mlx"
    echo "  Or clone from: https://github.com/waybarrios/vllm-mlx"
}
VLLM_MLX_VERSION=$(python3 -c "import vllm_mlx; print(vllm_mlx.__version__)" 2>/dev/null || echo "not installed")
echo "  vllm-mlx: $VLLM_MLX_VERSION"

# --- Models directory ---
mkdir -p "$MODELS_DIR"

# --- Write versions file ---
cat > "$VERSIONS_FILE" << VERSIONS
# Benchmark Environment Versions

Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Machine: $(sysctl -n machdep.cpu.brand_string)
Memory: $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1024/1024/1024}')
macOS: $(sw_vers -productVersion)
Xcode CLT: $(xcode-select -p 2>/dev/null && pkgutil --pkg-info=com.apple.pkg.CLTools_Executables 2>/dev/null | grep version || echo "unknown")

## Engines

| Engine     | Version        |
|------------|----------------|
| mlx-lm     | $MLX_VERSION   |
| mlx-core   | $MLX_CORE      |
| llama.cpp  | $LLAMA_VERSION |
| Ollama     | $OLLAMA_VERSION|
| vllm-mlx   | $VLLM_MLX_VERSION |

## Python
$(python3 --version)
Venv: $VENV_DIR

## Models
Standard benchmark models should be placed in: $MODELS_DIR
- LLaMA 3.1 8B Q4_K_M (GGUF)
- Mistral 7B Q4_K_M (GGUF)
- LLaMA 2 7B Q4_0 (GGUF)
VERSIONS

echo ""
echo "=== Setup complete. Versions written to $VERSIONS_FILE ==="
echo "Activate venv: source $VENV_DIR/bin/activate"
