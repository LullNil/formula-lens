#!/usr/bin/env bash
set -euo pipefail

log() {
    printf '[%s] [convert] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_VERSION="${FORMULALENS_MODEL_VERSION:-v2.0.0}"
MODEL_DIR="${FORMULALENS_MODEL_DIR:-$ROOT_DIR/weights/finetuned/v1}"
MODEL_FILENAME="${FORMULALENS_MODEL_FILENAME:-formulalens_yolox_nano_${MODEL_VERSION}.onnx}"
MODEL_PATH="${FORMULALENS_MODEL_PATH:-$MODEL_DIR/$MODEL_FILENAME}"

if [[ ! -f "$MODEL_PATH" ]]; then
    log "Model file not found at $MODEL_PATH, starting download"
    FORMULALENS_MODEL_PATH="$MODEL_PATH" \
        bash "$ROOT_DIR/scripts/download_weights.sh" service "$MODEL_PATH"
else
    log "Using cached model at $MODEL_PATH"
fi

export FORMULALENS_MODEL_PATH="$MODEL_PATH"

exec python3 "$ROOT_DIR/scripts/convert_annotations.py" "$@"
