#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_SRC_DIR="${VERL_SRC_DIR:-$HOME/src/verl}"
INSTALL_VERL="${INSTALL_VERL:-1}"

python -m pip install -U pip wheel setuptools
python -m pip install -r "$REPO_ROOT/requirements-cloud.txt"

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("verl") else 1)
PY
then
  echo "verl is already importable."
elif [[ "$INSTALL_VERL" == "1" ]]; then
  mkdir -p "$(dirname "$VERL_SRC_DIR")"
  if [[ ! -d "$VERL_SRC_DIR/.git" ]]; then
    git clone https://github.com/verl-project/verl.git "$VERL_SRC_DIR"
  fi
  (cd "$VERL_SRC_DIR" && python -m pip install -e ".[vllm]")
else
  echo "Skipping verl installation because INSTALL_VERL=$INSTALL_VERL"
fi

echo "Environment setup finished."

