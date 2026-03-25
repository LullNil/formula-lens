#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PATH="${1:-$ROOT_DIR/weights/pretrained/yolox_nano.pth}"
YOLOX_NANO_URL="${YOLOX_NANO_URL:-https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth}"

mkdir -p "$(dirname "$TARGET_PATH")"
curl -fL "$YOLOX_NANO_URL" -o "$TARGET_PATH"
echo "Downloaded YOLOX-Nano weights to $TARGET_PATH"
