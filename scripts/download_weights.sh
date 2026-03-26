#!/usr/bin/env bash
set -euo pipefail

log() {
    printf '[%s] [weights] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pretrained}"
MODEL_VERSION="${FORMULALENS_MODEL_VERSION:-${MODEL_VERSION:-v1.0.0}}"
YOLOX_NANO_URL="${YOLOX_NANO_URL:-https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth}"
SERVICE_RELEASE_REPO="${FORMULALENS_RELEASE_REPO:-LullNil/formula-lens}"
SERVICE_FILENAME="${FORMULALENS_MODEL_FILENAME:-formulalens_yolox_nano_${MODEL_VERSION}.onnx}"
SERVICE_TARGET="${2:-$ROOT_DIR/weights/finetuned/v1/$SERVICE_FILENAME}"

case "$MODE" in
    pretrained)
        TARGET_PATH="${2:-$ROOT_DIR/weights/pretrained/yolox_nano.pth}"
        SOURCE_URL="$YOLOX_NANO_URL"
        ;;
    service)
        TARGET_PATH="$SERVICE_TARGET"
        if [[ -n "${FORMULALENS_MODEL_URL:-}" ]]; then
            SOURCE_URL="$FORMULALENS_MODEL_URL"
        elif [[ -n "$SERVICE_RELEASE_REPO" ]]; then
            SOURCE_URL="https://github.com/$SERVICE_RELEASE_REPO/releases/download/$MODEL_VERSION/$SERVICE_FILENAME"
        else
            echo "Set FORMULALENS_MODEL_URL or FORMULALENS_RELEASE_REPO for service weight download."
            exit 1
        fi
        ;;
    *)
        echo "Unsupported mode: $MODE"
        echo "Usage: bash scripts/download_weights.sh [pretrained|service] [target_path]"
        exit 1
        ;;
esac

mkdir -p "$(dirname "$TARGET_PATH")"
if [[ -f "$TARGET_PATH" ]]; then
    log "Weights already exist at $TARGET_PATH"
    exit 0
fi

log "Downloading mode=$MODE version=$MODEL_VERSION"
log "Source: $SOURCE_URL"
log "Target: $TARGET_PATH"
curl -fL "$SOURCE_URL" -o "$TARGET_PATH"
log "Downloaded weights to $TARGET_PATH"
