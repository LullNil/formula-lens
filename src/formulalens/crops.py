from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .schemas import Detection


@dataclass(frozen=True)
class CropRegion:
    label: str
    bbox: tuple[int, int, int, int]
    score: float
    image: np.ndarray


def _load_bgr_image(image: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image}")
        return bgr
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def crop_detections(
    image: str | Path | np.ndarray | Image.Image,
    detections: list[Detection],
    labels: set[str] | None = None,
    padding: int = 2,
) -> list[CropRegion]:
    raw = _load_bgr_image(image)
    height, width = raw.shape[:2]
    allowed = labels or {"numerator", "denominator", "system_row", "block", "text", "whole_part"}

    crops: list[CropRegion] = []
    for detection in detections:
        if detection.label not in allowed:
            continue

        x1 = max(0, int(round(detection.bbox.x1)) - padding)
        y1 = max(0, int(round(detection.bbox.y1)) - padding)
        x2 = min(width, int(round(detection.bbox.x2)) + padding)
        y2 = min(height, int(round(detection.bbox.y2)) + padding)
        if x2 <= x1 or y2 <= y1:
            continue

        crops.append(
            CropRegion(
                label=detection.label,
                bbox=(x1, y1, x2, y2),
                score=detection.score,
                image=raw[y1:y2, x1:x2].copy(),
            )
        )
    return crops
