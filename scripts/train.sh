#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
YOLOX_DIR="${YOLOX_DIR:-$ROOT_DIR/third_party/YOLOX}"
EXP_FILE="${EXP_FILE:-$ROOT_DIR/configs/train/yolox_nano.py}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$ROOT_DIR/weights/pretrained/yolox_nano.pth}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/datasets/roboflow_exports/formulas_coco_v1}"
PREPARED_DATASET_DIR="${PREPARED_DATASET_DIR:-$ROOT_DIR/datasets/prepared/formulas_coco_v1}"
GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOGGER="${LOGGER:-tensorboard}"

if [[ ! -d "$YOLOX_DIR/yolox" ]]; then
    echo "YOLOX repo not found at $YOLOX_DIR"
    echo "Set YOLOX_DIR or clone https://github.com/Megvii-BaseDetection/YOLOX into third_party/YOLOX"
    exit 1
fi

if [[ ! -f "$WEIGHTS_PATH" ]]; then
    echo "Pretrained weights not found at $WEIGHTS_PATH"
    echo "Run scripts/download_weights.sh first."
    exit 1
fi

if ! "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available()"; then
    echo "YOLOX training requires a working CUDA runtime in this environment."
    echo "Use scripts/setup_train_env.sh cu121 or cu124 on a machine with accessible GPU drivers."
    exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_dataset.py" \
    --source-root "$SOURCE_ROOT" \
    --output-dir "$PREPARED_DATASET_DIR"

cd "$YOLOX_DIR"
PYTHONPATH="$ROOT_DIR/src:$YOLOX_DIR:${PYTHONPATH:-}" \
    "$PYTHON_BIN" - "${@:1}" <<PY
import runpy
import sys
from pathlib import Path

import torch

tool_path = Path("$YOLOX_DIR") / "tools" / "train.py"
original_load = torch.load


def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_load(*args, **kwargs)


torch.load = patched_load
sys.argv = [
    str(tool_path),
    "-f",
    "$EXP_FILE",
    "-d",
    "$GPUS",
    "-b",
    "$BATCH_SIZE",
    "-c",
    "$WEIGHTS_PATH",
    "-l",
    "$LOGGER",
    *sys.argv[1:],
]
runpy.run_path(str(tool_path), run_name="__main__")
PY
