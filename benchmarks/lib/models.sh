#!/usr/bin/env bash
# models.sh — Model download and discovery for the Vexel benchmark suite.
# Sourced by full_comparison.sh; not intended to be run standalone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
MODELS_DIR="$BENCH_DIR/models"

# HuggingFace download URLs
URL_LLAMA_8B="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
URL_QWEN_05B="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
URL_TINYLLAMA="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

# Corresponding filenames
FILE_LLAMA_8B="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
FILE_QWEN_05B="qwen2.5-0.5b-instruct-q4_k_m.gguf"
FILE_TINYLLAMA="tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

# Possible symlink source directory (llama.cpp models)
LLAMA_MODELS_DIR="$REPO_ROOT/../llama.cpp/models"

###############################################################################
# ensure_models_gitignored — Abort if the models directory is not gitignored.
###############################################################################
ensure_models_gitignored() {
    if ! git -C "$REPO_ROOT" check-ignore -q "$MODELS_DIR" 2>/dev/null; then
        echo "ERROR: $MODELS_DIR is not gitignored." >&2
        echo "Add 'benchmarks/models/' to .gitignore before running benchmarks." >&2
        exit 1
    fi
    echo "[models] models directory is gitignored — OK"
}

###############################################################################
# download_if_missing <filename> <url>
#   1. If the file already exists in MODELS_DIR, do nothing.
#   2. Try to symlink from ../llama.cpp/models/.
#   3. Fall back to downloading via curl.
###############################################################################
download_if_missing() {
    local filename="$1"
    local url="$2"
    local target="$MODELS_DIR/$filename"

    if [[ -f "$target" ]]; then
        echo "[models] $filename — already present"
        return 0
    fi

    mkdir -p "$MODELS_DIR"

    # Try symlink from llama.cpp models directory
    if [[ -f "$LLAMA_MODELS_DIR/$filename" ]]; then
        ln -s "$LLAMA_MODELS_DIR/$filename" "$target"
        echo "[models] $filename — symlinked from llama.cpp/models/"
        return 0
    fi

    # Download via curl
    echo "[models] $filename — downloading from HuggingFace..."
    curl -L --progress-bar -o "$target" "$url"
    echo "[models] $filename — download complete"
}

###############################################################################
# setup_models — Ensure all benchmark models are available.
#   Sets: MODEL_LLAMA_8B, MODEL_QWEN_05B, MODEL_TINYLLAMA
###############################################################################
setup_models() {
    echo "=== Setting up models ==="
    ensure_models_gitignored

    download_if_missing "$FILE_LLAMA_8B"  "$URL_LLAMA_8B"
    download_if_missing "$FILE_QWEN_05B"  "$URL_QWEN_05B"
    download_if_missing "$FILE_TINYLLAMA" "$URL_TINYLLAMA"

    export MODEL_LLAMA_8B="$MODELS_DIR/$FILE_LLAMA_8B"
    export MODEL_QWEN_05B="$MODELS_DIR/$FILE_QWEN_05B"
    export MODEL_TINYLLAMA="$MODELS_DIR/$FILE_TINYLLAMA"

    echo "[models] MODEL_LLAMA_8B  = $MODEL_LLAMA_8B"
    echo "[models] MODEL_QWEN_05B  = $MODEL_QWEN_05B"
    echo "[models] MODEL_TINYLLAMA = $MODEL_TINYLLAMA"
    echo ""
}
