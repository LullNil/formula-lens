#!/usr/bin/env python3
"""Normalize a Roboflow COCO export into the folder layout expected by YOLOX."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path("datasets/roboflow_exports/formulas_coco_v1")
DEFAULT_OUTPUT_DIR = Path("datasets/prepared/formulas_coco_v1")
SPLIT_ALIASES = (
    ("train", "train2017"),
    ("valid", "val2017"),
    ("val", "val2017"),
    ("test", "test2017"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a Roboflow COCO export into YOLOX-compatible folder names."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory that contains Roboflow train/valid/test split folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Prepared dataset root directory.",
    )
    return parser.parse_args()


def normalize_bbox(raw_bbox: list[Any], width: int, height: int) -> list[float] | None:
    if len(raw_bbox) != 4:
        return None

    x, y, w, h = [float(value) for value in raw_bbox]
    x = max(0.0, min(x, float(width)))
    y = max(0.0, min(y, float(height)))
    w = max(0.0, min(w, float(width) - x))
    h = max(0.0, min(h, float(height) - y))
    if w <= 0.0 or h <= 0.0:
        return None
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def discover_splits(source_root: Path) -> dict[str, tuple[str, Path, Path]]:
    discovered: dict[str, tuple[str, Path, Path]] = {}
    for source_name, target_name in SPLIT_ALIASES:
        split_dir = source_root / source_name
        annotations_path = split_dir / "_annotations.coco.json"
        if not split_dir.is_dir() or not annotations_path.is_file():
            continue
        if target_name in discovered:
            previous_source = discovered[target_name][0]
            raise ValueError(
                f"Both '{previous_source}' and '{source_name}' map to '{target_name}'. "
                "Keep only one source split."
            )
        discovered[target_name] = (source_name, split_dir, annotations_path)

    if "train2017" not in discovered:
        raise FileNotFoundError(
            f"No Roboflow train split found under {source_root}. Expected {source_root / 'train'}."
        )
    return discovered


def normalize_split_payload(payload: dict[str, Any]) -> dict[str, Any]:
    images = sorted(payload.get("images", []), key=lambda item: item["file_name"])
    annotations = payload.get("annotations", [])
    image_ids = {int(image["id"]) for image in images}

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_ids:
            continue
        annotations_by_image[image_id].append(annotation)

    categories = sorted(
        [
            category
            for category in payload.get("categories", [])
            if int(category["id"]) != 0
        ],
        key=lambda item: int(item["id"]),
    )

    normalized_images: list[dict[str, Any]] = []
    normalized_annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    for image in images:
        normalized_images.append(
            {
                "id": next_image_id,
                "license": image.get("license"),
                "file_name": image["file_name"],
                "height": int(image["height"]),
                "width": int(image["width"]),
                "date_captured": image.get("date_captured", ""),
            }
        )

        sorted_annotations = sorted(
            annotations_by_image.get(int(image["id"]), []),
            key=lambda item: (
                int(item["category_id"]),
                tuple(float(value) for value in item.get("bbox", [])),
                int(item.get("id", 0)),
            ),
        )
        for annotation in sorted_annotations:
            bbox = normalize_bbox(
                annotation.get("bbox", []),
                width=int(image["width"]),
                height=int(image["height"]),
            )
            if bbox is None:
                continue

            normalized_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": int(annotation["category_id"]),
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 4),
                    "segmentation": [],
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            )
            next_annotation_id += 1

        next_image_id += 1

    return {
        "info": payload.get("info", {}),
        "licenses": payload.get("licenses", []),
        "images": normalized_images,
        "annotations": normalized_annotations,
        "categories": categories,
    }


def copy_images(split_dir: Path, split_payload: dict[str, Any], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for image in split_payload["images"]:
        source_image = split_dir / image["file_name"]
        if not source_image.exists():
            raise FileNotFoundError(f"Source image not found: {source_image}")
        shutil.copy2(source_image, destination_dir / image["file_name"])


def main() -> None:
    args = parse_args()
    discovered_splits = discover_splits(args.source_root)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    annotations_dir = args.output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_root": str(args.source_root),
        "output_dir": str(args.output_dir),
        "splits": {},
    }

    for target_name, (source_name, split_dir, annotations_path) in sorted(discovered_splits.items()):
        with annotations_path.open("r", encoding="utf-8") as handle:
            source_payload = json.load(handle)

        normalized_payload = normalize_split_payload(source_payload)
        copy_images(split_dir, normalized_payload, args.output_dir / target_name)

        annotation_output = annotations_dir / f"instances_{target_name}.json"
        with annotation_output.open("w", encoding="utf-8") as handle:
            json.dump(normalized_payload, handle, indent=2)

        summary["splits"][target_name] = {
            "source_split": source_name,
            "images": len(normalized_payload["images"]),
            "annotations": len(normalized_payload["annotations"]),
            "categories": [category["name"] for category in normalized_payload["categories"]],
        }

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
