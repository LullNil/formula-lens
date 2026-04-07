from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import cv2
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from .confidence import compute_confidence_breakdown, get_confidence_level, infer_structure_type
from .inference import DEFAULT_CHECKPOINT_PATH, DEFAULT_ONNX_PATH, FormulaLensPredictor
from .postprocess import postprocess_detections
from .render_similarity import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_DILATION_KERNEL,
    DEFAULT_FONT_SIZE,
    DEFAULT_PADDING,
    DEFAULT_RENDER_DPI,
    DEFAULT_RENDERER_BACKEND,
    compute_render_similarity,
)
from .routing import choose_routing
from .schemas import (
    BadCaseResponse,
    DetectionResponse,
    InferenceResult,
    RenderSimilarityResponse,
    RouteBatchResponse,
    RouteResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "configs" / "service" / "settings.yaml"
BAD_CASES_ROOT = PROJECT_ROOT / "experiments" / "bad_cases"
logger = logging.getLogger("formulalens.service")


def get_model_version() -> str:
    settings = load_service_settings()
    return os.getenv("FORMULALENS_MODEL_VERSION", settings.get("model", {}).get("model_version", "v1"))


def _resolve_versioned_model_path(configured_path: str | Path, model_version: str, configured_version: str) -> Path:
    configured = Path(configured_path)
    model_dir = Path(os.getenv("FORMULALENS_MODEL_DIR", str(configured.parent)))
    if configured_version and configured_version in configured.name:
        filename = configured.name.replace(configured_version, model_version, 1)
    else:
        filename = configured.name
    return model_dir / filename


async def _read_upload_image(image: UploadFile) -> Image.Image:
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    try:
        parsed = Image.open(BytesIO(payload))
        parsed.load()
        return parsed
    except Exception as exc:  # pragma: no cover - FastAPI boundary
        raise HTTPException(status_code=400, detail=f"Unable to decode image: {exc}") from exc


async def _read_upload_images(images: list[UploadFile]) -> list[Image.Image]:
    if not images:
        raise HTTPException(status_code=400, detail="At least one image upload is required.")
    return [await _read_upload_image(image) for image in images]


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _route_render_similarity_settings() -> dict:
    settings = load_service_settings().get("routing", {}).get("render_similarity", {})
    enabled = _env_flag("FORMULALENS_RENDER_SIMILARITY_ENABLED", bool(settings.get("enabled", False)))
    return {
        "enabled": enabled,
        "backend": os.getenv(
            "FORMULALENS_RENDER_SIMILARITY_BACKEND",
            str(settings.get("backend", DEFAULT_RENDERER_BACKEND)),
        ),
        "canvas_size": tuple(settings.get("canvas_size", list(DEFAULT_CANVAS_SIZE))),
        "font_size": int(settings.get("font_size", DEFAULT_FONT_SIZE)),
        "dpi": int(settings.get("dpi", DEFAULT_RENDER_DPI)),
        "padding": int(settings.get("padding", DEFAULT_PADDING)),
        "dilation_kernel": int(settings.get("dilation_kernel", DEFAULT_DILATION_KERNEL)),
        "min_score": float(settings.get("min_score", 0.82)),
        "min_pix2tex_score": float(settings.get("min_pix2tex_score", 0.75)),
        "formula_confidence_cap": float(settings.get("formula_confidence_cap", 0.78)),
        "tie_break_score": float(settings.get("tie_break_score", 0.90)),
    }


def _build_render_similarity_response(
    image: Image.Image,
    pix2tex_output: str | None,
) -> tuple[RenderSimilarityResponse | None, dict]:
    settings = _route_render_similarity_settings()
    if not settings["enabled"]:
        return (
            RenderSimilarityResponse(
                enabled=False,
                applied=False,
                reason="Feature flag is disabled.",
            ),
            settings,
        )

    if not pix2tex_output or not pix2tex_output.strip():
        return (
            RenderSimilarityResponse(
                enabled=True,
                applied=False,
                reason="pix2tex output is missing.",
            ),
            settings,
        )

    result = compute_render_similarity(
        image,
        pix2tex_output,
        backend=settings["backend"],
        canvas_size=settings["canvas_size"],
        font_size=settings["font_size"],
        dpi=settings["dpi"],
        padding=settings["padding"],
        dilation_kernel=settings["dilation_kernel"],
    )
    if not result.applied:
        logger.warning("Render similarity skipped: %s", result.reason)
    return (
        RenderSimilarityResponse(
            enabled=result.enabled,
            applied=result.applied,
            score=result.score,
            renderer=result.renderer,
            reason=result.reason,
        ),
        settings,
    )


def _parse_batch_form_field(
    raw_value: str | None,
    expected_length: int,
    field_name: str,
    *,
    cast=None,
) -> list:
    if raw_value is None:
        return [None] * expected_length

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON array.") from exc

    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON array.")
    if len(payload) != expected_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} length must match images length ({expected_length}).",
        )

    normalized: list = []
    for index, item in enumerate(payload):
        if item is None:
            normalized.append(None)
            continue
        if cast is None:
            normalized.append(item)
            continue
        try:
            normalized.append(cast(item))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}[{index}] has an invalid value.",
            ) from exc
    return normalized


@lru_cache(maxsize=1)
def load_service_settings() -> dict:
    if SETTINGS_PATH.is_file():
        return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    return {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Service startup: model_version=%s settings_path=%s", get_model_version(), SETTINGS_PATH)
    started_at = time.perf_counter()
    predictor = get_predictor()
    logger.info(
        "Model ready: backend=%s device=%s path=%s input_size=%s score_threshold=%.2f nms_threshold=%.2f classes=%s load_time_ms=%.1f",
        predictor.backend,
        predictor.device,
        predictor.model_path,
        predictor.test_size,
        predictor.score_threshold,
        predictor.nms_threshold,
        ",".join(predictor.class_names),
        (time.perf_counter() - started_at) * 1000.0,
    )
    yield
    logger.info("Service shutdown")


app = FastAPI(title="FormulaLens", version=get_model_version(), lifespan=lifespan)


@lru_cache(maxsize=1)
def get_predictor() -> FormulaLensPredictor:
    settings = load_service_settings()
    model_settings = settings.get("model", {})
    configured_version = str(model_settings.get("model_version", "v1"))
    model_version = get_model_version()
    configured_path = model_settings.get("onnx_path", str(DEFAULT_ONNX_PATH))
    fallback_checkpoint = model_settings.get("checkpoint_path", str(DEFAULT_CHECKPOINT_PATH))
    exp_file = model_settings.get("exp_file")
    explicit_model_path = os.getenv("FORMULALENS_MODEL_PATH")
    model_path = Path(explicit_model_path) if explicit_model_path else _resolve_versioned_model_path(
        configured_path,
        model_version=model_version,
        configured_version=configured_version,
    )
    if not model_path.is_file():
        fallback_path = Path(os.getenv("FORMULALENS_CHECKPOINT", fallback_checkpoint))
        logger.warning("Configured model path missing: %s. Falling back to %s", model_path, fallback_path)
        model_path = fallback_path
    device = os.getenv("FORMULALENS_DEVICE", "cpu")
    class_names = tuple(settings.get("classes", ()))
    input_size = tuple(model_settings.get("input_size", [416, 416]))
    return FormulaLensPredictor(
        checkpoint_path=model_path,
        exp_file=exp_file,
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


def _detect_many_from_images(images: list[Image.Image]):
    predictor = get_predictor()
    raw_results, batched_inference_used = predictor.predict_many_with_info(images)
    processed_results = [
        postprocess_detections(
            raw_result.detections,
            score_threshold=float(predictor.score_threshold),
            iou_threshold=float(predictor.nms_threshold),
        )
        for raw_result in raw_results
    ]
    confidences = [
        compute_confidence_breakdown(processed, raw_result.image_width, raw_result.image_height)
        for raw_result, processed in zip(raw_results, processed_results)
    ]
    return predictor, raw_results, processed_results, confidences, batched_inference_used


@app.post("/detect", response_model=DetectionResponse)
async def detect(image: UploadFile = File(...)) -> DetectionResponse:
    parsed_image = await _read_upload_image(image)
    _, _, detections, confidence = _detect_from_image(parsed_image)
    return DetectionResponse(
        ok=True,
        detections=detections,
        global_confidence=confidence.global_confidence,
        confidence_level=get_confidence_level(confidence.global_confidence),
        structure_type=infer_structure_type(detections),
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
    render_similarity, render_settings = _build_render_similarity_response(parsed_image, pix2tex_output)
    decision, reason, confidence = choose_routing(
        detections=detections,
        image_width=raw_result.image_width,
        image_height=raw_result.image_height,
        pix2tex_output=pix2tex_output,
        pix2tex_score=pix2tex_score,
        render_similarity_score=render_similarity.score if render_similarity and render_similarity.applied else None,
        render_similarity_min_score=render_settings["min_score"],
        render_similarity_min_pix2tex_score=render_settings["min_pix2tex_score"],
        render_similarity_formula_confidence_cap=render_settings["formula_confidence_cap"],
        render_similarity_tie_break_score=render_settings["tie_break_score"],
    )
    return RouteResponse(
        ok=True,
        decision=decision,
        reason=reason,
        detections=detections,
        global_confidence=confidence.global_confidence,
        confidence_level=get_confidence_level(confidence.global_confidence),
        structure_type=infer_structure_type(detections),
        model_version=get_model_version(),
        confidence_breakdown=confidence,
        render_similarity=render_similarity,
    )


@app.post("/route-batch", response_model=RouteBatchResponse)
async def route_batch(
    images: list[UploadFile] = File(...),
    pix2tex_outputs_json: str | None = Form(default=None),
    pix2tex_scores_json: str | None = Form(default=None),
) -> RouteBatchResponse:
    parsed_images = await _read_upload_images(images)
    pix2tex_outputs = _parse_batch_form_field(
        pix2tex_outputs_json,
        expected_length=len(parsed_images),
        field_name="pix2tex_outputs_json",
    )
    pix2tex_scores = _parse_batch_form_field(
        pix2tex_scores_json,
        expected_length=len(parsed_images),
        field_name="pix2tex_scores_json",
        cast=float,
    )

    _, raw_results, detections_list, _, batched_inference_used = _detect_many_from_images(parsed_images)
    route_results: list[RouteResponse] = []
    for parsed_image, raw_result, detections, pix2tex_output, pix2tex_score in zip(
        parsed_images,
        raw_results,
        detections_list,
        pix2tex_outputs,
        pix2tex_scores,
    ):
        render_similarity, render_settings = _build_render_similarity_response(parsed_image, pix2tex_output)
        decision, reason, confidence = choose_routing(
            detections=detections,
            image_width=raw_result.image_width,
            image_height=raw_result.image_height,
            pix2tex_output=pix2tex_output,
            pix2tex_score=pix2tex_score,
            render_similarity_score=render_similarity.score if render_similarity and render_similarity.applied else None,
            render_similarity_min_score=render_settings["min_score"],
            render_similarity_min_pix2tex_score=render_settings["min_pix2tex_score"],
            render_similarity_formula_confidence_cap=render_settings["formula_confidence_cap"],
            render_similarity_tie_break_score=render_settings["tie_break_score"],
        )
        route_results.append(
            RouteResponse(
                ok=True,
                decision=decision,
                reason=reason,
                detections=detections,
                global_confidence=confidence.global_confidence,
                confidence_level=get_confidence_level(confidence.global_confidence),
                structure_type=infer_structure_type(detections),
                model_version=get_model_version(),
                confidence_breakdown=confidence,
                render_similarity=render_similarity,
            )
        )

    return RouteBatchResponse(
        ok=True,
        count=len(route_results),
        batched_inference_used=batched_inference_used,
        results=route_results,
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
    parsed_image.convert("RGB").save(image_path, format="JPEG", quality=95)

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
    logger.info("Bad case saved: case_id=%s reason=%s dir=%s", case_id, reason, target_dir)

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
    rendered = predictor.render(parsed_image, result=rendered_result, show_labels=False, line_thickness=1)

    ok, encoded = cv2.imencode(".jpg", rendered)
    if not ok:
        raise HTTPException(status_code=500, detail="Unable to encode debug image.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
