#!/usr/bin/env bash
# engines.sh — Engine binary discovery for the Vexel benchmark suite.
# Sourced by full_comparison.sh; not intended to be run standalone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/.." && pwd)"

# Relative path to llama.cpp build binaries
LLAMA_BIN_DIR="$REPO_ROOT/../llama.cpp/build/bin"

###############################################################################
# find_vexel — Build the Vexel binary if missing, then set VEXEL_BIN.
###############################################################################
find_vexel() {
    local vexel_path="$REPO_ROOT/vexel"

    if [[ -x "$vexel_path" ]]; then
        echo "[engines] vexel binary found: $vexel_path"
    else
        echo "[engines] vexel binary not found — building via 'make build'..."
        make -C "$REPO_ROOT" build
        if [[ ! -x "$vexel_path" ]]; then
            echo "ERROR: 'make build' did not produce $vexel_path" >&2
            exit 1
        fi
        echo "[engines] vexel binary built: $vexel_path"
    fi

    export VEXEL_BIN="$vexel_path"
}

###############################################################################
# find_llama — Locate llama.cpp binaries.
#   Checks PATH first, then the known build directory.
#   Sets: LLAMA_CLI, LLAMA_SERVER, LLAMA_SPECULATIVE
#   Prints "[missing]" and continues if a binary is not found.
###############################################################################
find_llama() {
    local binaries=("llama-completion" "llama-cli" "llama-server" "llama-speculative")
    local varnames=("LLAMA_COMPLETION" "LLAMA_CLI" "LLAMA_SERVER" "LLAMA_SPECULATIVE")

    for i in "${!binaries[@]}"; do
        local bin="${binaries[$i]}"
        local var="${varnames[$i]}"
        local found=""

        # Check PATH first
        if command -v "$bin" &>/dev/null; then
            found="$(command -v "$bin")"
        # Check llama.cpp build directory
        elif [[ -x "$LLAMA_BIN_DIR/$bin" ]]; then
            found="$LLAMA_BIN_DIR/$bin"
        fi

        if [[ -n "$found" ]]; then
            export "$var"="$found"
            echo "[engines] $bin found: $found"
        else
            export "$var"="[missing]"
            echo "[engines] $bin — [missing]"
        fi
    done
}

###############################################################################
# setup_engines — Discover all engine binaries.
###############################################################################
setup_engines() {
    echo "=== Setting up engines ==="
    find_vexel
    find_llama

    echo ""
    echo "[engines] VEXEL_BIN          = $VEXEL_BIN"
    echo "[engines] LLAMA_COMPLETION   = ${LLAMA_COMPLETION:-[missing]}"
    echo "[engines] LLAMA_CLI          = $LLAMA_CLI"
    echo "[engines] LLAMA_SERVER       = $LLAMA_SERVER"
    echo "[engines] LLAMA_SPECULATIVE  = $LLAMA_SPECULATIVE"
    echo ""
}
