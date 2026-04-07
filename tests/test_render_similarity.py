from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formulalens.render_similarity import compute_mask_similarity, normalize_foreground_mask


def test_normalize_foreground_mask_centers_content():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:70] = 255

    normalized = normalize_foreground_mask(mask, canvas_size=(64, 64), padding=0)

    assert normalized.shape == (64, 64)
    assert normalized.sum() > 0
    foreground_columns = np.where(normalized.sum(axis=0) > 0)[0]
    foreground_rows = np.where(normalized.sum(axis=1) > 0)[0]
    assert foreground_columns[0] > 0
    assert foreground_columns[-1] < 63
    assert foreground_rows[0] > 0
    assert foreground_rows[-1] < 63


def test_compute_mask_similarity_prefers_matching_shapes():
    reference = np.zeros((64, 64), dtype=np.uint8)
    reference[12:52, 28:36] = 255
    reference[28:36, 12:52] = 255

    shifted = np.zeros((64, 64), dtype=np.uint8)
    shifted[13:53, 29:37] = 255
    shifted[29:37, 13:53] = 255

    mismatch = np.zeros((64, 64), dtype=np.uint8)
    mismatch[10:20, 10:54] = 255
    mismatch[44:54, 10:54] = 255

    similar_score = compute_mask_similarity(reference, shifted)
    mismatch_score = compute_mask_similarity(reference, mismatch)

    assert similar_score > 0.85
    assert mismatch_score < similar_score
