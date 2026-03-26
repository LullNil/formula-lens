#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_VERSION="${FORMULALENS_MODEL_VERSION:-v1.0.0}"
MODEL_DIR="${FORMULALENS_MODEL_DIR:-$ROOT_DIR/weights/finetuned/v1}"
MODEL_FILENAME="${FORMULALENS_MODEL_FILENAME:-formulalens_yolox_nano_${MODEL_VERSION}.onnx}"
MODEL_PATH="${FORMULALENS_MODEL_PATH:-$MODEL_DIR/$MODEL_FILENAME}"
HOST="${FORMULALENS_HOST:-0.0.0.0}"
PORT="${FORMULALENS_PORT:-8000}"

if [[ ! -f "$MODEL_PATH" ]]; then
    FORMULALENS_MODEL_PATH="$MODEL_PATH" \
        bash "$ROOT_DIR/scripts/download_weights.sh" service "$MODEL_PATH"
fi

export FORMULALENS_MODEL_PATH="$MODEL_PATH"

exec uvicorn formulalens.service:app --app-dir "$ROOT_DIR/src" --host "$HOST" --port "$PORT"
