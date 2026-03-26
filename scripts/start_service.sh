#!/usr/bin/env bash
set -euo pipefail

log() {
    printf '[%s] [service] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_VERSION="${FORMULALENS_MODEL_VERSION:-v1.0.0}"
MODEL_DIR="${FORMULALENS_MODEL_DIR:-$ROOT_DIR/weights/finetuned/v1}"
MODEL_FILENAME="${FORMULALENS_MODEL_FILENAME:-formulalens_yolox_nano_${MODEL_VERSION}.onnx}"
MODEL_PATH="${FORMULALENS_MODEL_PATH:-$MODEL_DIR/$MODEL_FILENAME}"
HOST="${FORMULALENS_HOST:-0.0.0.0}"
PORT="${FORMULALENS_PORT:-8000}"

log "Boot request received"
log "Model version: $MODEL_VERSION"
log "Expected model path: $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
    log "Model file not found locally, starting download"
    FORMULALENS_MODEL_PATH="$MODEL_PATH" \
        bash "$ROOT_DIR/scripts/download_weights.sh" service "$MODEL_PATH"
else
    log "Using cached model at $MODEL_PATH"
fi

export FORMULALENS_MODEL_PATH="$MODEL_PATH"

log "Starting uvicorn on ${HOST}:${PORT}"
exec uvicorn formulalens.service:app --app-dir "$ROOT_DIR/src" --host "$HOST" --port "$PORT"
