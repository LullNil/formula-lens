from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formulalens.confidence import get_confidence_level, infer_structure_type
from formulalens.schemas import BoundingBox, ConfidenceBreakdown, ConfidenceLevel, Detection, DetectionResponse


def test_detection_response_rounds_confidence_fields_to_two_decimals():
    breakdown = ConfidenceBreakdown(
        global_confidence=0.7523557233719668,
        base_score=0.8523557233719667,
        geometry_penalty=0.0,
        combination_penalty=0.1,
        detection_count=2,
        class_distribution={"denominator": 1, "numerator": 1},
    )
    detections = [
        Detection(
            class_id=3,
            label="numerator",
            score=0.93648,
            bbox=BoundingBox(x1=5.82, y1=4.31, x2=87.98, y2=33.04),
        ),
        Detection(
            class_id=1,
            label="denominator",
            score=0.88884,
            bbox=BoundingBox(x1=6.02, y1=34.01, x2=88.11, y2=61.99),
        ),
    ]
    payload = DetectionResponse(
        ok=True,
        detections=detections,
        global_confidence=breakdown.global_confidence,
        confidence_level=get_confidence_level(breakdown.global_confidence),
        structure_type=infer_structure_type(detections),
        model_version="v1.0.0",
        confidence_breakdown=breakdown,
    ).model_dump(mode="json", by_alias=True)

    assert payload["bbox_format"] == "xyxy"
    assert payload["structure_type"] == "fraction"
    assert payload["global_confidence"] == 0.75
    assert payload["confidence_breakdown"]["global_confidence"] == 0.75
    assert payload["confidence_breakdown"]["base_score"] == 0.85
    assert payload["confidence_breakdown"]["geometry_penalty"] == 0.0
    assert payload["confidence_breakdown"]["combination_penalty"] == 0.1
    assert payload["confidence_breakdown"]["detection_count"] == 2
    assert payload["confidence_breakdown"]["class_distribution"] == {"denominator": 1, "numerator": 1}
    assert payload["detections"][0]["score"] == 0.936
    assert payload["detections"][0]["bbox"] == [5.82, 4.31, 87.98, 33.04]


def test_confidence_level_thresholds():
    assert get_confidence_level(0.81) == ConfidenceLevel.HIGH
    assert get_confidence_level(0.8) == ConfidenceLevel.MEDIUM
    assert get_confidence_level(0.61) == ConfidenceLevel.MEDIUM
    assert get_confidence_level(0.6) == ConfidenceLevel.LOW


def test_detection_fields_round_only_in_serialized_output():
    detection = Detection(
        class_id=3,
        label="numerator",
        score=0.93648,
        bbox=BoundingBox(x1=0.05543870192307693, y1=0.9472, x2=95.3764, y2=69.0),
    )

    payload = detection.model_dump(mode="json", by_alias=True)

    assert detection.score == 0.93648
    assert detection.bbox.x1 == 0.05543870192307693
    assert payload["score"] == 0.936
    assert payload["bbox"] == [0.06, 0.95, 95.38, 69.0]


def test_infer_structure_type():
    detections = [
        Detection(
            class_id=3,
            label="numerator",
            score=0.9,
            bbox=BoundingBox(x1=1, y1=1, x2=2, y2=2),
        ),
        Detection(
            class_id=1,
            label="denominator",
            score=0.9,
            bbox=BoundingBox(x1=1, y1=3, x2=2, y2=4),
        ),
    ]

    assert infer_structure_type(detections) == "fraction"
