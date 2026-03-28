#!/usr/bin/env bash
# models.sh — Model download and discovery for the Vexel benchmark suite.
# Sourced by full_comparison.sh; not intended to be run standalone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
MODELS_DIR="$BENCH_DIR/models"

# HuggingFace download URLs — model weights
URL_LLAMA_8B="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
URL_QWEN_05B="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
URL_TINYLLAMA="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

# HuggingFace download URLs — tokenizer.json (Vexel needs a separate tokenizer file)
URL_TOKENIZER_LLAMA_8B="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/resolve/main/tokenizer.json"
URL_TOKENIZER_QWEN_05B="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json"
URL_TOKENIZER_TINYLLAMA="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json"

# Corresponding filenames
FILE_LLAMA_8B="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
FILE_QWEN_05B="qwen2.5-0.5b-instruct-q4_k_m.gguf"
FILE_TINYLLAMA="tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

# Subdirectories for each model (Vexel looks for tokenizer.json beside the model file)
DIR_LLAMA_8B="llama-8b"
DIR_QWEN_05B="qwen-0.5b"
DIR_TINYLLAMA="tinyllama"

# Possible symlink source directory (llama.cpp models)
LLAMA_MODELS_DIR="$REPO_ROOT/../llama.cpp/models"

###############################################################################
# ensure_models_gitignored — Abort if the models directory is not gitignored.
###############################################################################
ensure_models_gitignored() {
    # Use a relative path for check-ignore since absolute paths may not match .gitignore patterns
    local rel_path
    rel_path=$(python3 -c "import os; print(os.path.relpath('$MODELS_DIR', '$REPO_ROOT'))")
    if ! git -C "$REPO_ROOT" check-ignore -q "$rel_path/" 2>/dev/null; then
        echo "ERROR: $MODELS_DIR is not gitignored." >&2
        echo "Add 'benchmarks/models/' to .gitignore before running benchmarks." >&2
        exit 1
    fi
    echo "[models] models directory is gitignored — OK"
}

###############################################################################
# download_if_missing <subdir> <filename> <url>
#   1. If the file already exists in MODELS_DIR/<subdir>/, do nothing.
#   2. Try to symlink from ../llama.cpp/models/.
#   3. Fall back to downloading via curl.
###############################################################################
download_if_missing() {
    local subdir="$1"
    local filename="$2"
    local url="$3"
    local target_dir="$MODELS_DIR/$subdir"
    local target="$target_dir/$filename"

    mkdir -p "$target_dir"

    if [[ -f "$target" ]]; then
        echo "[models] $subdir/$filename — already present"
        return 0
    fi

    # Try symlink from llama.cpp models directory
    if [[ -f "$LLAMA_MODELS_DIR/$filename" ]]; then
        ln -s "$LLAMA_MODELS_DIR/$filename" "$target"
        echo "[models] $subdir/$filename — symlinked from llama.cpp/models/"
        return 0
    fi

    # Download via curl
    echo "[models] $subdir/$filename — downloading from HuggingFace..."
    curl -L --progress-bar -o "$target" "$url"
    echo "[models] $subdir/$filename — download complete"
}

###############################################################################
# download_tokenizer <subdir> <url>
#   Download tokenizer.json into the model subdirectory if not already present.
###############################################################################
download_tokenizer() {
    local subdir="$1"
    local url="$2"
    local target_dir="$MODELS_DIR/$subdir"
    local target="$target_dir/tokenizer.json"

    mkdir -p "$target_dir"

    if [[ -f "$target" ]]; then
        echo "[models] $subdir/tokenizer.json — already present"
        return 0
    fi

    # Try symlink from llama.cpp models directory (some setups keep tokenizers there)
    local llama_tok="$LLAMA_MODELS_DIR/$subdir/tokenizer.json"
    if [[ -f "$llama_tok" ]]; then
        ln -s "$llama_tok" "$target"
        echo "[models] $subdir/tokenizer.json — symlinked from llama.cpp/models/"
        return 0
    fi

    echo "[models] $subdir/tokenizer.json — downloading from HuggingFace..."
    curl -L --progress-bar -o "$target" "$url"
    echo "[models] $subdir/tokenizer.json — download complete"
}

###############################################################################
# setup_models — Ensure all benchmark models are available.
#   Sets: MODEL_LLAMA_8B, MODEL_QWEN_05B, MODEL_TINYLLAMA
###############################################################################
setup_models() {
    echo "=== Setting up models ==="
    ensure_models_gitignored

    # Download model weights into per-model subdirectories
    download_if_missing "$DIR_LLAMA_8B"   "$FILE_LLAMA_8B"  "$URL_LLAMA_8B"
    download_if_missing "$DIR_QWEN_05B"   "$FILE_QWEN_05B"  "$URL_QWEN_05B"
    download_if_missing "$DIR_TINYLLAMA"  "$FILE_TINYLLAMA" "$URL_TINYLLAMA"

    # Download tokenizer.json files (Vexel looks for tokenizer.json beside the GGUF)
    download_tokenizer "$DIR_LLAMA_8B"  "$URL_TOKENIZER_LLAMA_8B"
    download_tokenizer "$DIR_QWEN_05B"  "$URL_TOKENIZER_QWEN_05B"
    download_tokenizer "$DIR_TINYLLAMA" "$URL_TOKENIZER_TINYLLAMA"

    export MODEL_LLAMA_8B="$MODELS_DIR/$DIR_LLAMA_8B/$FILE_LLAMA_8B"
    export MODEL_QWEN_05B="$MODELS_DIR/$DIR_QWEN_05B/$FILE_QWEN_05B"
    export MODEL_TINYLLAMA="$MODELS_DIR/$DIR_TINYLLAMA/$FILE_TINYLLAMA"

    echo "[models] MODEL_LLAMA_8B  = $MODEL_LLAMA_8B"
    echo "[models] MODEL_QWEN_05B  = $MODEL_QWEN_05B"
    echo "[models] MODEL_TINYLLAMA = $MODEL_TINYLLAMA"
    echo ""
}
