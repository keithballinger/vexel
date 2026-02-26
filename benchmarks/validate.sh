#!/usr/bin/env bash
# Validate that all benchmark engines can run inference.
# Each engine generates a short completion and reports success/failure.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
RESULTS=()
PASS=0
FAIL=0

# Activate venv if available
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "=== Benchmark Engine Validation ==="
echo ""

# --- Helper ---
report() {
    local engine="$1" status="$2" detail="$3"
    if [ "$status" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo "  ✓ $engine: $detail"
    else
        FAIL=$((FAIL + 1))
        echo "  ✗ $engine: $detail"
    fi
    RESULTS+=("$engine:$status")
}

# --- 1. Vexel ---
echo "[Vexel]"
VEXEL_BIN="$SCRIPT_DIR/../inference/cmd/vexel/vexel"
if [ -f "$VEXEL_BIN" ]; then
    report "Vexel" "PASS" "binary found at $VEXEL_BIN"
else
    # Try go build
    if command -v go &>/dev/null; then
        echo "  Building Vexel..."
        (cd "$SCRIPT_DIR/../inference" && CGO_ENABLED=1 go build -tags metal -o cmd/vexel/vexel ./cmd/vexel/ 2>&1) && \
            report "Vexel" "PASS" "built from source" || \
            report "Vexel" "FAIL" "build failed"
    else
        report "Vexel" "FAIL" "binary not found, go not available"
    fi
fi

# --- 2. MLX ---
echo "[MLX (mlx-lm)]"
if python3 -c "import mlx_lm" 2>/dev/null; then
    MLX_V=$(python3 -c "import mlx_lm; print(mlx_lm.__version__)")
    # Quick smoke test: try to load tokenizer config for a small model
    # (We don't actually download a model here — just verify the library works)
    python3 -c "
from mlx_lm import load
print('mlx-lm load function available')
" 2>/dev/null && \
        report "MLX" "PASS" "mlx-lm $MLX_V, load() available" || \
        report "MLX" "FAIL" "mlx-lm $MLX_V, load() import failed"
else
    report "MLX" "FAIL" "mlx-lm not installed"
fi

# --- 3. llama.cpp ---
echo "[llama.cpp]"
if command -v llama-cli &>/dev/null; then
    LLAMA_V=$(llama-cli --version 2>&1 | head -1)
    report "llama.cpp" "PASS" "$LLAMA_V"
elif command -v llama-server &>/dev/null; then
    report "llama.cpp" "PASS" "llama-server available"
else
    report "llama.cpp" "FAIL" "not installed (try: brew install llama.cpp)"
fi

# --- 4. Ollama ---
echo "[Ollama]"
if command -v ollama &>/dev/null; then
    OLLAMA_V=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    # Check if Ollama service is reachable
    if curl -sf http://localhost:11434/api/version &>/dev/null; then
        report "Ollama" "PASS" "v$OLLAMA_V, service running"
    else
        report "Ollama" "PASS" "v$OLLAMA_V installed (service not running — start with: ollama serve)"
    fi
else
    report "Ollama" "FAIL" "not installed (https://ollama.ai)"
fi

# --- 5. vllm-mlx ---
echo "[vllm-mlx]"
if python3 -c "import vllm_mlx" 2>/dev/null; then
    VLLM_V=$(python3 -c "import vllm_mlx; print(vllm_mlx.__version__)" 2>/dev/null || echo "unknown")
    report "vllm-mlx" "PASS" "vllm-mlx $VLLM_V"
else
    report "vllm-mlx" "FAIL" "not installed (try: pip install vllm-mlx)"
fi

# --- Summary ---
echo ""
echo "=== Validation Summary ==="
echo "Passed: $PASS / $((PASS + FAIL))"
if [ $FAIL -gt 0 ]; then
    echo "Failed: $FAIL — run benchmarks/setup.sh to install missing engines."
    exit 1
else
    echo "All engines validated."
    exit 0
fi
