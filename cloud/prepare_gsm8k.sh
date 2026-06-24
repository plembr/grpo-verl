#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/projects/grpo-qwen/data/gsm8k_verl}"
FORMAT="${FORMAT:-parquet}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
TEST_LIMIT="${TEST_LIMIT:-0}"

cd "$REPO_ROOT"
python scripts/prepare_gsm8k_verl.py \
  --output-dir "$DATA_DIR" \
  --format "$FORMAT" \
  --train-limit "$TRAIN_LIMIT" \
  --test-limit "$TEST_LIMIT"

