from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_serializer


def _round_confidence(value: float) -> float:
    return float(round(value, 2))


def _round_score(value: float) -> float:
    return float(round(value, 3))


def _round_bbox(value: float) -> float:
    return float(round(value, 2))


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float

    @model_serializer(mode="plain")
    def serialize(self) -> list[float]:
        return [
            _round_bbox(self.x1),
            _round_bbox(self.y1),
            _round_bbox(self.x2),
            _round_bbox(self.y2),
        ]


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_id: int
    label: str = Field(serialization_alias="class")
    score: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox

    @field_serializer("score")
    def serialize_score(self, value: float) -> float:
        return _round_score(value)


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str
    image_path: str | None = None
    image_width: int
    image_height: int
    resize_ratio: float
    detections: list[Detection] = Field(default_factory=list)


class RoutingDecision(str, Enum):
    USE_PIX2TEX = "use_pix2tex"
    USE_FORMULA_LENS = "use_formula_lens"
    USE_HEURISTICS = "use_heuristics"


class BBoxFormat(str, Enum):
    XYXY = "xyxy"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_confidence: float = Field(ge=0.0, le=1.0)
    base_score: float = Field(ge=0.0, le=1.0)
    geometry_penalty: float = Field(ge=0.0, le=1.0)
    combination_penalty: float = Field(ge=0.0, le=1.0)
    detection_count: int = Field(ge=0)
    class_distribution: dict[str, int] = Field(default_factory=dict)

    @field_serializer("global_confidence", "base_score", "geometry_penalty", "combination_penalty")
    def serialize_confidence(self, value: float) -> float:
        return _round_confidence(value)


class RenderSimilarityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    applied: bool = False
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    renderer: str | None = None
    reason: str | None = None

    @field_serializer("score")
    def serialize_score(self, value: float | None) -> float | None:
        if value is None:
            return None
        return _round_confidence(value)


class DetectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bbox_format: BBoxFormat = BBoxFormat.XYXY
    detections: list[Detection] = Field(default_factory=list)
    global_confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    structure_type: str
    model_version: str
    confidence_breakdown: ConfidenceBreakdown

    @field_serializer("global_confidence")
    def serialize_global_confidence(self, value: float) -> float:
        return _round_confidence(value)


class RouteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    decision: RoutingDecision
    reason: str
    bbox_format: BBoxFormat = BBoxFormat.XYXY
    detections: list[Detection] = Field(default_factory=list)
    global_confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    structure_type: str
    model_version: str
    confidence_breakdown: ConfidenceBreakdown
    render_similarity: RenderSimilarityResponse | None = None

    @field_serializer("global_confidence")
    def serialize_global_confidence(self, value: float) -> float:
        return _round_confidence(value)


class BadCaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    case_id: str
    saved_dir: str
