from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formulalens.routing import choose_routing
from formulalens.schemas import BoundingBox, Detection, RoutingDecision


def make_detection(label: str, class_id: int, score: float, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(
        class_id=class_id,
        label=label,
        score=score,
        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
    )


def test_choose_formula_lens_when_structure_is_confident():
    detections = [
        make_detection("numerator", 3, 0.92, (10, 10, 60, 30)),
        make_detection("denominator", 1, 0.91, (10, 35, 60, 55)),
    ]

    decision, reason, confidence = choose_routing(
        detections=detections,
        image_width=100,
        image_height=100,
        pix2tex_output="x/y",
        pix2tex_score=0.5,
    )

    assert decision == RoutingDecision.USE_FORMULA_LENS
    assert confidence.global_confidence >= 0.65
    assert "Structural detections" in reason


def test_choose_pix2tex_when_formula_lens_is_weaker():
    detections = [
        make_detection("block", 0, 0.35, (10, 10, 20, 20)),
    ]

    decision, _, confidence = choose_routing(
        detections=detections,
        image_width=200,
        image_height=200,
        pix2tex_output="E = mc^2",
        pix2tex_score=0.97,
    )

    assert decision == RoutingDecision.USE_PIX2TEX
    assert confidence.global_confidence < 0.6


def test_choose_formula_lens_for_whole_part_structure():
    detections = [
        make_detection("whole_part", 6, 0.88, (15, 12, 90, 70)),
    ]

    decision, reason, confidence = choose_routing(
        detections=detections,
        image_width=120,
        image_height=100,
        pix2tex_output="x",
        pix2tex_score=0.62,
    )

    assert decision == RoutingDecision.USE_FORMULA_LENS
    assert confidence.global_confidence >= 0.65
    assert "Structural detections" in reason


def test_choose_pix2tex_when_render_similarity_is_strong():
    detections = [
        make_detection("block", 0, 0.72, (10, 10, 120, 90)),
    ]

    decision, reason, confidence = choose_routing(
        detections=detections,
        image_width=200,
        image_height=120,
        pix2tex_output="\\frac{x}{y}",
        pix2tex_score=0.81,
        render_similarity_score=0.91,
    )

    assert decision == RoutingDecision.USE_PIX2TEX
    assert confidence.global_confidence < 0.78
    assert "matches the source image strongly enough" in reason


def test_keep_formula_lens_when_its_confidence_is_still_dominant():
    detections = [
        make_detection("numerator", 3, 0.95, (10, 10, 90, 40)),
        make_detection("denominator", 1, 0.94, (10, 55, 90, 95)),
    ]

    decision, reason, confidence = choose_routing(
        detections=detections,
        image_width=120,
        image_height=120,
        pix2tex_output="\\frac{x}{y}",
        pix2tex_score=0.81,
        render_similarity_score=0.95,
    )

    assert decision == RoutingDecision.USE_FORMULA_LENS
    assert confidence.global_confidence >= 0.78
    assert "Structural detections" in reason
