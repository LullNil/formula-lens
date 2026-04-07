from __future__ import annotations

from .confidence import compute_confidence_breakdown
from .schemas import ConfidenceBreakdown, Detection, RoutingDecision


STRUCTURAL_LABELS = {"numerator", "denominator", "exponent", "system_row", "whole_part"}


def choose_routing(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    pix2tex_output: str | None,
    pix2tex_score: float | None,
    render_similarity_score: float | None = None,
    render_similarity_min_score: float = 0.82,
    render_similarity_min_pix2tex_score: float = 0.75,
    render_similarity_formula_confidence_cap: float = 0.78,
    render_similarity_tie_break_score: float = 0.9,
) -> tuple[RoutingDecision, str, ConfidenceBreakdown]:
    confidence = compute_confidence_breakdown(detections, image_width, image_height)
    pix_score = max(0.0, min(1.0, float(pix2tex_score or 0.0)))
    structure_count = sum(1 for detection in detections if detection.label in STRUCTURAL_LABELS)

    if (
        pix2tex_output
        and render_similarity_score is not None
        and pix_score >= render_similarity_min_pix2tex_score
        and render_similarity_score >= render_similarity_min_score
        and confidence.global_confidence < render_similarity_formula_confidence_cap
    ):
        return (
            RoutingDecision.USE_PIX2TEX,
            "pix2tex rendering matches the source image strongly enough to override weaker FormulaLens confidence.",
            confidence,
        )

    if structure_count >= 1 and confidence.global_confidence >= 0.65 and pix_score < 0.85:
        return (
            RoutingDecision.USE_FORMULA_LENS,
            "Structural detections are confident enough to override pix2tex.",
            confidence,
        )

    if pix2tex_output and pix_score >= 0.9 and confidence.global_confidence < 0.6:
        return (
            RoutingDecision.USE_PIX2TEX,
            "pix2tex is confident while FormulaLens confidence is weaker.",
            confidence,
        )

    if pix2tex_output and not detections and pix_score >= 0.6:
        return (
            RoutingDecision.USE_PIX2TEX,
            "FormulaLens produced no reliable detections, so pix2tex wins by default.",
            confidence,
        )

    if (
        pix2tex_output
        and render_similarity_score is not None
        and render_similarity_score >= render_similarity_tie_break_score
        and pix_score >= 0.6
    ):
        return (
            RoutingDecision.USE_PIX2TEX,
            "Rendered pix2tex output matches the source image strongly enough to break the tie.",
            confidence,
        )

    return (
        RoutingDecision.USE_HEURISTICS,
        "Neither branch is decisive enough, fallback to heuristics.",
        confidence,
    )
