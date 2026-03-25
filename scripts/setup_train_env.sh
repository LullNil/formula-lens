#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venvs/formulalens-train-$PROFILE}"

case "$PROFILE" in
    cpu)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
        ;;
    cu121)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
        ;;
    cu124)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
        ;;
    *)
        echo "Unsupported profile: $PROFILE"
        echo "Use one of: cpu, cu121, cu124"
        exit 1
        ;;
esac

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
python -m pip install -r "$ROOT_DIR/requirements/train-common.txt"

echo "Training environment is ready at $VENV_DIR"
echo "Activate it with: source $VENV_DIR/bin/activate"
