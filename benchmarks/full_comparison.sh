#!/usr/bin/env bash
# full_comparison.sh — Main orchestrator for the Vexel benchmark suite.
#
# Usage:
#   ./benchmarks/full_comparison.sh [all|decode|speculative|context|batched]
#
# Compares Vexel against llama.cpp across multiple benchmark dimensions.

set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"

###############################################################################
# Source libraries
###############################################################################
source "$BENCH_ROOT/lib/models.sh"
source "$BENCH_ROOT/lib/engines.sh"
source "$BENCH_ROOT/lib/parse.sh"

###############################################################################
# Defaults
###############################################################################
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-3}"
GEN_TOKENS="${GEN_TOKENS:-128}"

SUITE="${1:-all}"

###############################################################################
# Validate subcommand
###############################################################################
case "$SUITE" in
    all|decode|speculative|context|batched) ;;
    *)
        echo "Usage: $0 [all|decode|speculative|context|batched]"
        echo "  all          Run every benchmark suite (default)"
        echo "  decode       Single-stream decode throughput"
        echo "  speculative  Speculative decoding comparison"
        echo "  context      Context-length scaling"
        echo "  batched      Batched inference throughput"
        exit 1
        ;;
esac

###############################################################################
# Results directory (timestamped)
###############################################################################
TODAY="$(date +%Y-%m-%d)"
RESULTS_DIR="$BENCH_ROOT/results/$TODAY"
mkdir -p "$RESULTS_DIR"

echo "============================================="
echo " Vexel Benchmark Suite"
echo " Suite:   $SUITE"
echo " Warmup:  $WARMUP runs"
echo " Runs:    $RUNS measured runs"
echo " Tokens:  $GEN_TOKENS per generation"
echo " Results: $RESULTS_DIR"
echo "============================================="
echo ""

###############################################################################
# Hardware info
###############################################################################
echo "=== Hardware Info ==="
HW_INFO="$RESULTS_DIR/hardware.txt"
{
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Host: $(hostname)"
    echo ""
    echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
    echo "CPU Cores (physical): $(sysctl -n hw.physicalcpu 2>/dev/null || echo 'unknown')"
    echo "CPU Cores (logical):  $(sysctl -n hw.logicalcpu 2>/dev/null || echo 'unknown')"
    echo "Memory: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
    echo "macOS: $(sw_vers -productVersion 2>/dev/null || echo 'unknown')"
    echo "Chip:  $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
    echo ""
    echo "Metal GPU:"
    system_profiler SPDisplaysDataType 2>/dev/null | grep -A5 "Chipset Model" || echo "  (unable to query)"
} > "$HW_INFO"
cat "$HW_INFO"
echo ""

###############################################################################
# Setup
###############################################################################
setup_models
setup_engines

export WARMUP RUNS GEN_TOKENS RESULTS_DIR
export MODEL_LLAMA_8B MODEL_QWEN_05B MODEL_TINYLLAMA
export VEXEL_BIN LLAMA_COMPLETION LLAMA_CLI LLAMA_SERVER LLAMA_SPECULATIVE

###############################################################################
# Run suites
###############################################################################
if [[ "$SUITE" == "all" || "$SUITE" == "decode" ]]; then
    echo "=== Running: standard decode ==="
    source "$BENCH_ROOT/lib/bench_standard.sh"
    run_standard_decode
    echo ""
fi

if [[ "$SUITE" == "all" || "$SUITE" == "speculative" ]]; then
    echo "=== Running: speculative decode ==="
    source "$BENCH_ROOT/lib/bench_speculative.sh"
    run_speculative
    echo ""
fi

if [[ "$SUITE" == "all" || "$SUITE" == "context" ]]; then
    echo "=== Running: context scaling ==="
    source "$BENCH_ROOT/lib/bench_context.sh"
    run_context_scaling
    echo ""
fi

if [[ "$SUITE" == "all" || "$SUITE" == "batched" ]]; then
    echo "=== Running: batched decode ==="
    source "$BENCH_ROOT/lib/bench_batched.sh"
    run_batched
    echo ""
fi

###############################################################################
# Generate report
###############################################################################
echo "=== Generating report ==="
python3 "$BENCH_ROOT/lib/report.py" "$RESULTS_DIR"

echo ""
echo "============================================="
echo " Benchmark complete. Results in: $RESULTS_DIR"
echo "============================================="
