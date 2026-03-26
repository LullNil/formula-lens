from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_serializer


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float

    @model_serializer(mode="plain")
    def serialize(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_id: int
    label: str = Field(serialization_alias="class")
    score: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


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


class ConfidenceBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_confidence: float = Field(ge=0.0, le=1.0)
    base_score: float = Field(ge=0.0, le=1.0)
    geometry_penalty: float = Field(ge=0.0, le=1.0)
    combination_penalty: float = Field(ge=0.0, le=1.0)


class DetectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    detections: list[Detection] = Field(default_factory=list)
    global_confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    confidence_breakdown: ConfidenceBreakdown


class RouteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    decision: RoutingDecision
    reason: str
    detections: list[Detection] = Field(default_factory=list)
    global_confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    confidence_breakdown: ConfidenceBreakdown


class BadCaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    case_id: str
    saved_dir: str
