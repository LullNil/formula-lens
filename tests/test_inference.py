from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CHECKPOINT_PATH = PROJECT_ROOT / "weights" / "finetuned" / "yolox_nano" / "best_ckpt.pth"
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "datasets" / "prepared" / "formulas_coco_v1" / "val2017" / "0-124_png.rf.da5fccdb9aed8a966ca079f509a6a0bb.jpg"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(scope="module")
def predictor():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")

    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not SAMPLE_IMAGE_PATH.is_file():
        pytest.skip(f"Sample image not found: {SAMPLE_IMAGE_PATH}")

    from formulalens.inference import FormulaLensPredictor

    return FormulaLensPredictor(checkpoint_path=CHECKPOINT_PATH, device="cpu")


def test_predict_returns_structured_detections(predictor):
    result = predictor.predict(SAMPLE_IMAGE_PATH)

    assert result.image_path == str(SAMPLE_IMAGE_PATH)
    assert result.image_width > 0
    assert result.image_height > 0
    assert result.resize_ratio > 0
    assert len(result.detections) >= 1

    scores = [detection.score for detection in result.detections]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_render_writes_visualization(predictor, tmp_path):
    result = predictor.predict(SAMPLE_IMAGE_PATH)
    output_path = tmp_path / "prediction.jpg"

    rendered = predictor.render(SAMPLE_IMAGE_PATH, result=result, output_path=output_path)

    assert output_path.is_file()
    assert rendered.shape[0] == result.image_height
    assert rendered.shape[1] == result.image_width
