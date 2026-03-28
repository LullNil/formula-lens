from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.convert_annotations import format_detection_line, resolve_output_image_relative_path
from formulalens.schemas import BoundingBox, Detection


def test_resolve_output_image_relative_path_keeps_non_transparent_suffix():
    relative_path = Path("nested/formula.jpg")

    resolved = resolve_output_image_relative_path(relative_path, has_transparency=False)

    assert resolved == relative_path


def test_resolve_output_image_relative_path_converts_transparent_to_png():
    relative_path = Path("nested/formula.webp")

    resolved = resolve_output_image_relative_path(relative_path, has_transparency=True)

    assert resolved == Path("nested/formula.png")


def test_format_detection_line_exports_editable_detection_record():
    detection = Detection(
        class_id=3,
        label="numerator",
        score=0.93612,
        bbox=BoundingBox(x1=12.004, y1=8.0, x2=64.499, y2=31.001),
    )

    line = format_detection_line(detection, image_width=100, image_height=50)

    assert line == "3 0.382515 0.390010 0.524950 0.460020"
