#!/usr/bin/env python3
"""Run the current FormulaLens detection flow over a directory and export editable pseudo-labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SETTINGS_PATH = PROJECT_ROOT / "configs" / "service" / "settings.yaml"
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formulalens.inference import DEFAULT_CHECKPOINT_PATH, DEFAULT_ONNX_PATH, FormulaLensPredictor
from formulalens.postprocess import postprocess_detections
from formulalens.utils.image import pil_has_transparency, save_image_preserving_or_whitening_transparency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export FormulaLens pseudo-labels for a directory of images. "
            "Outputs copied/composited images plus .txt files with detected boxes."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Source directory with input images. Search is recursive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory. Images go to output/images, labels go to output/labels.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit model path. By default uses the service config and env overrides.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("FORMULALENS_DEVICE", "cpu"),
        help="Inference device for FormulaLensPredictor.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional score threshold override for postprocessing.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=None,
        help="Optional NMS threshold override for postprocessing.",
    )
    return parser.parse_args()


def load_service_settings() -> dict:
    if not SETTINGS_PATH.is_file():
        return {}
    return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}


def build_predictor(args: argparse.Namespace) -> FormulaLensPredictor:
    settings = load_service_settings()
    model_settings = settings.get("model", {})
    configured_path = Path(model_settings.get("onnx_path", str(DEFAULT_ONNX_PATH)))
    fallback_checkpoint = Path(model_settings.get("checkpoint_path", str(DEFAULT_CHECKPOINT_PATH)))
    model_path = Path(args.model_path or os.getenv("FORMULALENS_MODEL_PATH", configured_path))
    if not model_path.is_file():
        model_path = Path(os.getenv("FORMULALENS_CHECKPOINT", fallback_checkpoint))

    class_names = tuple(settings.get("classes", ()))
    input_size = tuple(model_settings.get("input_size", [416, 416]))
    score_threshold = float(
        args.score_threshold if args.score_threshold is not None else model_settings.get("score_threshold", 0.25)
    )
    nms_threshold = float(
        args.nms_threshold if args.nms_threshold is not None else model_settings.get("nms_threshold", 0.45)
    )

    return FormulaLensPredictor(
        checkpoint_path=model_path,
        device=args.device,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        input_size=input_size,
        class_names=class_names,
    )


def discover_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    image_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {root}")
    return image_paths


def read_transparency_flag(image_path: Path) -> bool:
    from PIL import Image

    with Image.open(image_path) as image:
        image.load()
        return pil_has_transparency(image)


def resolve_output_image_relative_path(relative_path: Path, has_transparency: bool) -> Path:
    if has_transparency:
        return relative_path.with_suffix(".png")
    return relative_path


def format_detection_line(detection, image_width: int, image_height: int) -> str:
    box_width = max(0.0, float(detection.bbox.x2) - float(detection.bbox.x1))
    box_height = max(0.0, float(detection.bbox.y2) - float(detection.bbox.y1))
    center_x = float(detection.bbox.x1) + box_width * 0.5
    center_y = float(detection.bbox.y1) + box_height * 0.5
    return " ".join(
        [
            str(int(detection.class_id)),
            f"{center_x / float(image_width):.6f}",
            f"{center_y / float(image_height):.6f}",
            f"{box_width / float(image_width):.6f}",
            f"{box_height / float(image_height):.6f}",
        ]
    )


def write_label_file(destination: Path, detections: list, image_width: int, image_height: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_detection_line(detection, image_width=image_width, image_height=image_height) for detection in detections]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_metadata(output_root: Path, class_names: tuple[str, ...]) -> None:
    (output_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    data_yaml_lines = [
        "path: .",
        "train: images",
        "val: images",
        "test: images",
        "names:",
    ]
    data_yaml_lines.extend([f"  {index}: {name}" for index, name in enumerate(class_names)])
    (output_root / "data.yaml").write_text("\n".join(data_yaml_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    predictor = build_predictor(args)
    image_paths = discover_images(args.input_dir)

    dataset_root = args.output_dir / args.input_dir.name
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    dataset_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    write_dataset_metadata(dataset_root, predictor.class_names)

    class_counts: Counter[str] = Counter()
    total_detections = 0
    empty_label_files = 0
    whitened_images = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(args.input_dir)
        has_transparency = read_transparency_flag(image_path)
        output_image_relative_path = resolve_output_image_relative_path(relative_path, has_transparency)
        output_image_path = images_root / output_image_relative_path
        output_label_path = labels_root / output_image_relative_path.with_suffix(".txt")

        raw_result = predictor.predict(image_path)
        detections = postprocess_detections(
            raw_result.detections,
            score_threshold=float(predictor.score_threshold),
            iou_threshold=float(predictor.nms_threshold),
        )

        whitened = save_image_preserving_or_whitening_transparency(image_path, output_image_path)
        if whitened:
            whitened_images += 1

        write_label_file(
            output_label_path,
            detections,
            image_width=int(raw_result.image_width),
            image_height=int(raw_result.image_height),
        )

        if not detections:
            empty_label_files += 1
        else:
            total_detections += len(detections)
            class_counts.update(detection.label for detection in detections)

    total_processing_seconds = time.perf_counter() - started_at
    average_processing_seconds = total_processing_seconds / len(image_paths)

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(dataset_root),
        "images_dir": str(images_root),
        "labels_dir": str(labels_root),
        "model_path": str(predictor.model_path),
        "backend": predictor.backend,
        "device": predictor.device,
        "input_size": list(predictor.test_size),
        "score_threshold": float(predictor.score_threshold),
        "nms_threshold": float(predictor.nms_threshold),
        "images_processed": len(image_paths),
        "images_with_white_background": whitened_images,
        "empty_label_files": empty_label_files,
        "total_detections": total_detections,
        "class_counts": dict(sorted(class_counts.items())),
        "processing_time_seconds_total": round(total_processing_seconds, 4),
        "processing_time_seconds_avg_per_image": round(average_processing_seconds, 4),
        "processing_time_ms_avg_per_image": round(average_processing_seconds * 1000.0, 3),
        "txt_format": "yolo: class_id x_center y_center width height",
    }
    (dataset_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
