from __future__ import annotations

from collections import defaultdict

from .schemas import Detection


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


def _is_valid(detection: Detection) -> bool:
    return detection.bbox.x2 > detection.bbox.x1 and detection.bbox.y2 > detection.bbox.y1 and detection.score > 0.0


def filter_by_score(detections: list[Detection], score_threshold: float) -> list[Detection]:
    return [detection for detection in detections if detection.score >= score_threshold and _is_valid(detection)]


def apply_classwise_nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.class_id].append(detection)

    kept: list[Detection] = []
    for class_detections in grouped.values():
        ordered = sorted(class_detections, key=lambda item: item.score, reverse=True)
        while ordered:
            current = ordered.pop(0)
            kept.append(current)
            ordered = [
                candidate
                for candidate in ordered
                if _iou(current, candidate) < iou_threshold
            ]
    return kept


def sort_detections(detections: list[Detection]) -> list[Detection]:
    return sorted(
        detections,
        key=lambda item: (
            round(item.bbox.y1 / 12.0),
            item.bbox.y1,
            item.bbox.x1,
            -item.score,
        ),
    )


def postprocess_detections(
    detections: list[Detection],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    filtered = filter_by_score(detections, score_threshold)
    suppressed = apply_classwise_nms(filtered, iou_threshold)
    return sort_detections(suppressed)
