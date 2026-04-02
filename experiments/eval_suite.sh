#!/bin/bash
# LoRA Training Evaluation Suite
# Tests a model with and without a LoRA adapter on a set of questions
# Usage: ./eval_suite.sh <model_path> <adapter_path> <questions_file> <max_tokens>
#
# Questions file format: one question per line
# Output: JSON with base and adapted responses for each question

set -e

MODEL="$1"
ADAPTER="$2"
QUESTIONS="$3"
MAX_TOKENS="${4:-60}"
VEXEL_CMD="go run -tags metal ./inference/cmd/vexel"

if [ -z "$MODEL" ] || [ -z "$QUESTIONS" ]; then
    echo "Usage: $0 <model_path> <adapter_path|none> <questions_file> [max_tokens]"
    exit 1
fi

echo "{"
echo "  \"model\": \"$(basename $MODEL)\","
echo "  \"adapter\": \"$(basename $ADAPTER 2>/dev/null || echo none)\","
echo "  \"results\": ["

first=true
while IFS= read -r question || [ -n "$question" ]; do
    # Skip empty lines and comments
    [[ -z "$question" || "$question" == \#* ]] && continue

    if [ "$first" = true ]; then
        first=false
    else
        echo "    ,"
    fi

    # Base model response
    base_response=$($VEXEL_CMD --model "$MODEL" generate --prompt "$question" --max-tokens "$MAX_TOKENS" 2>/dev/null | head -1)

    # Adapted model response (if adapter provided)
    if [ "$ADAPTER" != "none" ] && [ -d "$ADAPTER" ]; then
        adapted_response=$($VEXEL_CMD --model "$MODEL" --lora "$ADAPTER" generate --prompt "$question" --max-tokens "$MAX_TOKENS" 2>/dev/null | head -1)
    else
        adapted_response="(no adapter)"
    fi

    # Escape for JSON
    base_escaped=$(echo "$base_response" | sed 's/"/\\"/g' | tr '\n' ' ')
    adapted_escaped=$(echo "$adapted_response" | sed 's/"/\\"/g' | tr '\n' ' ')

    echo "    {"
    echo "      \"question\": \"$(echo "$question" | sed 's/"/\\"/g')\","
    echo "      \"base\": \"$base_escaped\","
    echo "      \"adapted\": \"$adapted_escaped\""
    echo -n "    }"
done < "$QUESTIONS"

echo ""
echo "  ]"
echo "}"
