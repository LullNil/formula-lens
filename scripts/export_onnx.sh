#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
YOLOX_DIR="${YOLOX_DIR:-$ROOT_DIR/third_party/YOLOX}"
EXP_FILE="${EXP_FILE:-$ROOT_DIR/configs/train/yolox_nano.py}"
EXP_NAME="${EXP_NAME:-$(basename "$EXP_FILE" .py)}"
CKPT_PATH="${CKPT_PATH:-$ROOT_DIR/weights/finetuned/$EXP_NAME/best_ckpt.pth}"
MODEL_VERSION="${MODEL_VERSION:-v1.0.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/weights/finetuned/v1}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/formulalens_yolox_nano_${MODEL_VERSION}.onnx}"

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
    "$PYTHON_BIN" - <<PY
import runpy
import sys
from pathlib import Path

import torch

tool_path = Path("$YOLOX_DIR") / "tools" / "export_onnx.py"
original_load = torch.load
if not hasattr(torch.onnx, "_export"):
    torch.onnx._export = torch.onnx.export


def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_load(*args, **kwargs)


torch.load = patched_load
sys.argv = [
    str(tool_path),
    "-f",
    "$EXP_FILE",
    "-c",
    "$CKPT_PATH",
    "--output-name",
    "$OUTPUT_PATH",
    "--decode_in_inference",
]
runpy.run_path(str(tool_path), run_name="__main__")
PY
