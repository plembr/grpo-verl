#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_SRC_DIR="${VERL_SRC_DIR:-$HOME/src/verl}"
INSTALL_VERL="${INSTALL_VERL:-1}"
VERL_REPO_URL="${VERL_REPO_URL:-https://github.com/verl-project/verl.git}"
VERL_REF="${VERL_REF:-main}"
VERL_CLONE_DEPTH="${VERL_CLONE_DEPTH:-1}"
VERL_CLONE_RETRIES="${VERL_CLONE_RETRIES:-3}"

clone_verl() {
  local parent_dir tmp_dir attempt

  if [[ -z "$VERL_SRC_DIR" || "$VERL_SRC_DIR" == "/" || "$VERL_SRC_DIR" == "$HOME" ]]; then
    echo "Refusing unsafe VERL_SRC_DIR=$VERL_SRC_DIR" >&2
    return 1
  fi

  parent_dir="$(dirname "$VERL_SRC_DIR")"
  tmp_dir="${VERL_SRC_DIR}.tmp"
  mkdir -p "$parent_dir"
  rm -rf "$tmp_dir"

  for ((attempt = 1; attempt <= VERL_CLONE_RETRIES; attempt++)); do
    echo "Cloning verl attempt $attempt/$VERL_CLONE_RETRIES from $VERL_REPO_URL"
    if git clone --depth "$VERL_CLONE_DEPTH" --branch "$VERL_REF" "$VERL_REPO_URL" "$tmp_dir"; then
      rm -rf "$VERL_SRC_DIR"
      mv "$tmp_dir" "$VERL_SRC_DIR"
      return 0
    fi

    rm -rf "$tmp_dir"
    sleep $((attempt * 5))
  done

  echo "Failed to clone verl after $VERL_CLONE_RETRIES attempts." >&2
  return 1
}

python -m pip install -U pip wheel setuptools
python -m pip install -r "$REPO_ROOT/requirements-cloud.txt"

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("verl") else 1)
PY
then
  echo "verl is already importable."
elif [[ "$INSTALL_VERL" == "1" ]]; then
  if [[ -d "$VERL_SRC_DIR/.git" ]]; then
    if git -C "$VERL_SRC_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "Using existing verl checkout: $VERL_SRC_DIR"
    else
      echo "Existing verl checkout is incomplete; recloning."
      clone_verl
    fi
  else
    clone_verl
  fi
  (cd "$VERL_SRC_DIR" && python -m pip install -e ".[vllm]")
else
  echo "Skipping verl installation because INSTALL_VERL=$INSTALL_VERL"
fi

echo "Environment setup finished."
