#!/usr/bin/env bash
set -euo pipefail

log() {
    printf '[%s] [service] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS_PATH="${FORMULALENS_SETTINGS_PATH:-$ROOT_DIR/configs/service/settings.yaml}"
SETTINGS_MODEL_VERSION="$(
    FORMULALENS_SETTINGS_PATH="$SETTINGS_PATH" python3 - <<'PY'
from pathlib import Path
import os

import yaml

settings_path = Path(os.environ["FORMULALENS_SETTINGS_PATH"])
if settings_path.is_file():
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    print(settings.get("model", {}).get("model_version", "v2.0.0"))
else:
    print("v2.0.0")
PY
)"
SETTINGS_MODEL_PATH="$(
    FORMULALENS_SETTINGS_PATH="$SETTINGS_PATH" python3 - <<'PY'
from pathlib import Path
import os

import yaml

root_dir = Path.cwd()
settings_path = Path(os.environ["FORMULALENS_SETTINGS_PATH"])
if settings_path.is_file():
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    print(settings.get("model", {}).get("onnx_path", "weights/finetuned/v2/formulalens_yolox_nano_v2.0.0.onnx"))
else:
    print("weights/finetuned/v2/formulalens_yolox_nano_v2.0.0.onnx")
PY
)"
MODEL_VERSION="${FORMULALENS_MODEL_VERSION:-$SETTINGS_MODEL_VERSION}"
SETTINGS_MODEL_PATH_ABS="$ROOT_DIR/$SETTINGS_MODEL_PATH"
SETTINGS_MODEL_DIR="$(dirname "$SETTINGS_MODEL_PATH_ABS")"
SETTINGS_MODEL_FILENAME="$(basename "$SETTINGS_MODEL_PATH_ABS")"
if [[ -n "${FORMULALENS_MODEL_PATH:-}" ]]; then
    MODEL_PATH="$FORMULALENS_MODEL_PATH"
else
    MODEL_DIR="${FORMULALENS_MODEL_DIR:-$SETTINGS_MODEL_DIR}"
    MODEL_FILENAME="${FORMULALENS_MODEL_FILENAME:-${SETTINGS_MODEL_FILENAME/$SETTINGS_MODEL_VERSION/$MODEL_VERSION}}"
    MODEL_PATH="$MODEL_DIR/$MODEL_FILENAME"
fi
MODEL_DIR="${FORMULALENS_MODEL_DIR:-$(dirname "$MODEL_PATH")}"
MODEL_FILENAME="${FORMULALENS_MODEL_FILENAME:-$(basename "$MODEL_PATH")}"
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
export FORMULALENS_MODEL_VERSION="$MODEL_VERSION"

log "Starting uvicorn on ${HOST}:${PORT}"
exec uvicorn formulalens.service:app --app-dir "$ROOT_DIR/src" --host "$HOST" --port "$PORT"
