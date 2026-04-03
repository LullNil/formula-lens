from __future__ import annotations

from collections import Counter
from itertools import combinations

from .schemas import ConfidenceBreakdown, ConfidenceLevel, Detection


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bbox_area(detection: Detection) -> float:
    return max(0.0, detection.bbox.x2 - detection.bbox.x1) * max(0.0, detection.bbox.y2 - detection.bbox.y1)


def _iou(left: Detection, right: Detection) -> float:
    inter_x1 = max(left.bbox.x1, right.bbox.x1)
    inter_y1 = max(left.bbox.y1, right.bbox.y1)
    inter_x2 = min(left.bbox.x2, right.bbox.x2)
    inter_y2 = min(left.bbox.y2, right.bbox.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter_area = inter_w * inter_h
    union_area = _bbox_area(left) + _bbox_area(right) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_confidence_breakdown(
    detections: list[Detection],
    image_width: int,
    image_height: int,
) -> ConfidenceBreakdown:
    if not detections:
        return ConfidenceBreakdown(
            global_confidence=0.0,
            base_score=0.0,
            geometry_penalty=0.0,
            combination_penalty=0.0,
            detection_count=0,
            class_distribution={},
        )

    image_area = float(max(1, image_width * image_height))
    base_score = sum(detection.score for detection in detections) / len(detections)

    geometry_penalty = 0.0
    for detection in detections:
        if _bbox_area(detection) / image_area < 0.0004:
            geometry_penalty += 0.02

    for left, right in combinations(detections, 2):
        if left.class_id == right.class_id and _iou(left, right) > 0.85:
            geometry_penalty += 0.04

    labels = [detection.label for detection in detections]
    combination_penalty = 0.0
    if ("denominator" in labels) != ("numerator" in labels):
        combination_penalty += 0.15
    if labels.count("system_row") > 1:
        combination_penalty += min(0.15, 0.05 * (labels.count("system_row") - 1))
    if labels.count("exponent") > max(1, labels.count("block") + labels.count("numerator")):
        combination_penalty += 0.08

    geometry_penalty = _clamp(geometry_penalty)
    combination_penalty = _clamp(combination_penalty)
    global_confidence = _clamp(base_score - geometry_penalty - combination_penalty)
    class_distribution = dict(sorted(Counter(labels).items()))

    return ConfidenceBreakdown(
        global_confidence=global_confidence,
        base_score=_clamp(base_score),
        geometry_penalty=geometry_penalty,
        combination_penalty=combination_penalty,
        detection_count=len(detections),
        class_distribution=class_distribution,
    )


def compute_global_confidence(detections: list[Detection], image_width: int, image_height: int) -> float:
    return compute_confidence_breakdown(detections, image_width, image_height).global_confidence


def get_confidence_level(score: float) -> ConfidenceLevel:
    if score > 0.8:
        return ConfidenceLevel.HIGH
    if score > 0.6:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def infer_structure_type(detections: list[Detection]) -> str:
    labels = {detection.label for detection in detections}
    if "numerator" in labels and "denominator" in labels:
        return "fraction"
    if "system_row" in labels:
        return "system"
    if "exponent" in labels:
        return "power"
    if labels == {"whole_part"}:
        return "whole_part"
    if labels == {"text"}:
        return "text"
    if labels == {"block"}:
        return "block"
    if not labels:
        return "unknown"
    return "mixed"
