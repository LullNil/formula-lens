from __future__ import annotations

import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import cv2
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from .confidence import compute_confidence_breakdown
from .inference import DEFAULT_CHECKPOINT_PATH, DEFAULT_ONNX_PATH, FormulaLensPredictor
from .postprocess import postprocess_detections
from .routing import choose_routing
from .schemas import BadCaseResponse, DetectionResponse, InferenceResult, RouteResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "configs" / "service" / "settings.yaml"
BAD_CASES_ROOT = PROJECT_ROOT / "experiments" / "bad_cases"


def get_model_version() -> str:
    settings = load_service_settings()
    return os.getenv("FORMULALENS_MODEL_VERSION", settings.get("model", {}).get("model_version", "v1"))


async def _read_upload_image(image: UploadFile) -> Image.Image:
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    try:
        return Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # pragma: no cover - FastAPI boundary
        raise HTTPException(status_code=400, detail=f"Unable to decode image: {exc}") from exc


@lru_cache(maxsize=1)
def load_service_settings() -> dict:
    if SETTINGS_PATH.is_file():
        return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    return {}


app = FastAPI(title="FormulaLens", version=get_model_version())


@lru_cache(maxsize=1)
def get_predictor() -> FormulaLensPredictor:
    settings = load_service_settings()
    model_settings = settings.get("model", {})
    configured_path = model_settings.get("onnx_path", str(DEFAULT_ONNX_PATH))
    fallback_checkpoint = model_settings.get("checkpoint_path", str(DEFAULT_CHECKPOINT_PATH))
    model_path = Path(os.getenv("FORMULALENS_MODEL_PATH", configured_path))
    if not model_path.is_file():
        model_path = Path(os.getenv("FORMULALENS_CHECKPOINT", fallback_checkpoint))
    device = os.getenv("FORMULALENS_DEVICE", "cpu")
    class_names = tuple(settings.get("classes", ()))
    input_size = tuple(model_settings.get("input_size", [416, 416]))
    return FormulaLensPredictor(
        checkpoint_path=model_path,
        device=device,
        score_threshold=float(model_settings.get("score_threshold", 0.25)),
        nms_threshold=float(model_settings.get("nms_threshold", 0.45)),
        input_size=input_size,
        class_names=class_names,
    )


def _detect_from_image(image: Image.Image):
    predictor = get_predictor()
    raw_result = predictor.predict(image)
    processed = postprocess_detections(
        raw_result.detections,
        score_threshold=float(predictor.score_threshold),
        iou_threshold=float(predictor.nms_threshold),
    )
    confidence = compute_confidence_breakdown(processed, raw_result.image_width, raw_result.image_height)
    return predictor, raw_result, processed, confidence


@app.post("/detect", response_model=DetectionResponse)
async def detect(image: UploadFile = File(...)) -> DetectionResponse:
    parsed_image = await _read_upload_image(image)
    _, _, detections, confidence = _detect_from_image(parsed_image)
    return DetectionResponse(
        ok=True,
        detections=detections,
        global_confidence=confidence.global_confidence,
        model_version=get_model_version(),
        confidence_breakdown=confidence,
    )


@app.post("/route", response_model=RouteResponse)
async def route(
    image: UploadFile = File(...),
    pix2tex_output: str | None = Form(default=None),
    pix2tex_score: float | None = Form(default=None),
) -> RouteResponse:
    parsed_image = await _read_upload_image(image)
    _, raw_result, detections, _ = _detect_from_image(parsed_image)
    decision, reason, confidence = choose_routing(
        detections=detections,
        image_width=raw_result.image_width,
        image_height=raw_result.image_height,
        pix2tex_output=pix2tex_output,
        pix2tex_score=pix2tex_score,
    )
    return RouteResponse(
        ok=True,
        decision=decision,
        reason=reason,
        detections=detections,
        global_confidence=confidence.global_confidence,
        model_version=get_model_version(),
        confidence_breakdown=confidence,
    )


@app.post("/debug/save-bad-case", response_model=BadCaseResponse)
async def save_bad_case(
    image: UploadFile = File(...),
    intermediate_outputs: str = Form(default="{}"),
    reason: str = Form(...),
) -> BadCaseResponse:
    parsed_image = await _read_upload_image(image)
    case_id = uuid4().hex
    target_dir = BAD_CASES_ROOT / case_id
    target_dir.mkdir(parents=True, exist_ok=True)

    image_path = target_dir / "input.jpg"
    parsed_image.save(image_path, format="JPEG", quality=95)

    try:
        payload = json.loads(intermediate_outputs)
    except json.JSONDecodeError:
        payload = {"raw": intermediate_outputs}

    metadata = {
        "reason": reason,
        "intermediate_outputs": payload,
        "original_filename": image.filename,
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return BadCaseResponse(ok=True, case_id=case_id, saved_dir=str(target_dir))


@app.post("/debug/detect")
async def debug_detect(image: UploadFile = File(...)) -> Response:
    parsed_image = await _read_upload_image(image)
    predictor, raw_result, detections, _ = _detect_from_image(parsed_image)
    rendered_result = InferenceResult(
        checkpoint_path=raw_result.checkpoint_path,
        image_path=raw_result.image_path,
        image_width=raw_result.image_width,
        image_height=raw_result.image_height,
        resize_ratio=raw_result.resize_ratio,
        detections=detections,
    )
    rendered = predictor.render(parsed_image, result=rendered_result)

    ok, encoded = cv2.imencode(".jpg", rendered)
    if not ok:
        raise HTTPException(status_code=500, detail="Unable to encode debug image.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
