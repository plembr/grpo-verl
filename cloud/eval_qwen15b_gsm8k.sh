#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATA="${DATA:-$HOME/projects/grpo-qwen/data/gsm8k_verl/test.parquet}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OUTPUT="${OUTPUT:-$HOME/projects/grpo-qwen/outputs/eval_gsm8k.jsonl}"

cd "$REPO_ROOT"
python scripts/eval_gsm8k_local.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --data "$DATA" \
  --limit "$LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output "$OUTPUT"

