#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
YOLOX_DIR="${YOLOX_DIR:-$ROOT_DIR/third_party/YOLOX}"
EXP_FILE="${EXP_FILE:-$ROOT_DIR/configs/train/yolox_nano.py}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$ROOT_DIR/weights/pretrained/yolox_nano.pth}"
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

"$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_dataset.py" --source-root "$ROOT_DIR/datasets/roboflow_exports/formulas_coco_v1"

cd "$YOLOX_DIR"
PYTHONPATH="$ROOT_DIR/src:$YOLOX_DIR:${PYTHONPATH:-}" \
    "$PYTHON_BIN" tools/train.py \
    -f "$EXP_FILE" \
    -d "$GPUS" \
    -b "$BATCH_SIZE" \
    -c "$WEIGHTS_PATH" \
    -l "$LOGGER" \
    "${@:1}"
