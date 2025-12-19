#!/bin/bash
# debug-inference.sh - Debug harness for tracing inference issues
#
# Usage:
#   ./debug-inference.sh -l 15 -p 4           # Debug layer 15 at position 4
#   ./debug-inference.sh -l 14,15 -o sdpa,wo  # Debug layers 14,15, only SDPA and Wo ops
#   ./debug-inference.sh -l 0 -p 0 -o input,output  # Compare input/output at layer 0
#   ./debug-inference.sh --help               # Show help
#
# This script provides a convenient wrapper around vexel's debug flags.

set -e

# Default values
MODEL="${MODEL:-models/phi-2-q4.gguf}"
TOKENIZER="${TOKENIZER:-models/phi2_tokenizer.json}"
PROMPT="${PROMPT:-Hi}"
MAX_TOKENS="${MAX_TOKENS:-6}"
LAYERS=""
POSITIONS=""
OPS=""
OUTPUT=""
VERBOSE="true"
TIMEOUT=30

usage() {
    cat << EOF
Debug harness for tracing Vexel inference issues.

Usage: $0 [OPTIONS]

Options:
  -l, --layers LAYERS      Comma-separated layer indices to trace (required)
  -p, --positions POS      Comma-separated position indices (default: all)
  -o, --ops OPS            Comma-separated ops: input,norm,qkv,rope,kv,sdpa,wo,mlp,output
  -m, --model PATH         Model file (default: \$MODEL or models/phi-2-q4.gguf)
  -t, --tokenizer PATH     Tokenizer file (default: \$TOKENIZER or models/phi2_tokenizer.json)
  --prompt TEXT            Input prompt (default: "Hi")
  --max-tokens N           Max tokens to generate (default: 6)
  --output FILE            Output JSON trace to file (default: stderr)
  --timeout SEC            Timeout in seconds (default: 30)
  -q, --quiet              Disable verbose output
  -h, --help               Show this help

Operations:
  input   - Layer input tensor
  norm    - After normalization
  qkv     - Q, K, V after projection
  rope    - Q, K after RoPE
  kv      - KV cache (fullK, fullV)
  sdpa    - Attention output
  wo      - After output projection
  mlp     - MLP output
  output  - Final layer output

Examples:
  # Debug NaN at layer 15, position 4
  $0 -l 15 -p 4

  # Trace SDPA across all layers at position 4
  $0 -l 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 -p 4 -o sdpa

  # Compare layers 14 and 15 output
  $0 -l 14,15 -o output

  # Full trace of layer 0 to JSON file
  $0 -l 0 --output trace.json

Environment variables:
  MODEL       - Default model path
  TOKENIZER   - Default tokenizer path
  PROMPT      - Default prompt
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--layers)
            LAYERS="$2"
            shift 2
            ;;
        -p|--positions)
            POSITIONS="$2"
            shift 2
            ;;
        -o|--ops)
            OPS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -q|--quiet)
            VERBOSE="false"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$LAYERS" ]]; then
    echo "Error: --layers is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check model exists
if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model file not found: $MODEL"
    exit 1
fi

# Check tokenizer exists
if [[ ! -f "$TOKENIZER" ]]; then
    echo "Error: Tokenizer file not found: $TOKENIZER"
    exit 1
fi

# Build command
CMD="./vexel"
CMD="$CMD -model $MODEL"
CMD="$CMD -tokenizer $TOKENIZER"
CMD="$CMD -max-tokens $MAX_TOKENS"
CMD="$CMD -gpu"
CMD="$CMD -temp 0"
CMD="$CMD -completion"
CMD="$CMD -debug-layers $LAYERS"

if [[ -n "$POSITIONS" ]]; then
    CMD="$CMD -debug-positions $POSITIONS"
fi

if [[ -n "$OPS" ]]; then
    CMD="$CMD -debug-ops $OPS"
fi

if [[ -n "$OUTPUT" ]]; then
    CMD="$CMD -debug-output $OUTPUT"
fi

CMD="$CMD -debug-verbose=$VERBOSE"

# Print command
echo "Running: timeout $TIMEOUT $CMD < <(echo \"$PROMPT\")"
echo "---"

# Execute
timeout "$TIMEOUT" bash -c "$CMD" < <(echo "$PROMPT") 2>&1
