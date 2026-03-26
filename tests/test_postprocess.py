from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formulalens.postprocess import postprocess_detections
from formulalens.schemas import BoundingBox, Detection


def make_detection(label: str, class_id: int, score: float, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(
        class_id=class_id,
        label=label,
        score=score,
        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
    )


def test_postprocess_filters_suppresses_and_sorts():
    detections = [
        make_detection("block", 0, 0.95, (50, 50, 120, 120)),
        make_detection("block", 0, 0.40, (52, 52, 122, 122)),
        make_detection("numerator", 3, 0.90, (10, 10, 40, 30)),
        make_detection("denominator", 1, 0.91, (10, 40, 40, 60)),
        make_detection("text", 5, 0.10, (5, 5, 8, 8)),
    ]

    processed = postprocess_detections(detections, score_threshold=0.25, iou_threshold=0.45)

    assert [item.label for item in processed] == ["numerator", "denominator", "block"]
    assert all(item.score >= 0.25 for item in processed)
    assert len(processed) == 3
