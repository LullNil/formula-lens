#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
YOLOX_DIR="${YOLOX_DIR:-$ROOT_DIR/third_party/YOLOX}"
EXP_FILE="${EXP_FILE:-$ROOT_DIR/configs/train/yolox_nano.py}"
EXP_NAME="${EXP_NAME:-$(basename "$EXP_FILE" .py)}"
CKPT_PATH="${CKPT_PATH:-$ROOT_DIR/weights/finetuned/$EXP_NAME/best_ckpt.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/weights/finetuned/formula_lens_v1.onnx}"

if [[ ! -d "$YOLOX_DIR/yolox" ]]; then
    echo "YOLOX repo not found at $YOLOX_DIR"
    exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Checkpoint not found at $CKPT_PATH"
    echo "Train the model first or override CKPT_PATH."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

cd "$YOLOX_DIR"
PYTHONPATH="$ROOT_DIR/src:$YOLOX_DIR:${PYTHONPATH:-}" \
    "$PYTHON_BIN" tools/export_onnx.py \
    -f "$EXP_FILE" \
    -c "$CKPT_PATH" \
    --output-name "$OUTPUT_PATH" \
    --decode_in_inference \
    "${@:1}"
