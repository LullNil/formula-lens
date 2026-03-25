from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int
    label: str
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
