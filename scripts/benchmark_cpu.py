#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formulalens.inference import DEFAULT_ONNX_PATH, FormulaLensPredictor
from formulalens.postprocess import postprocess_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FormulaLens CPU latency on ONNX Runtime.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "prepared" / "formulas_coco_v1" / "val2017",
    )
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(args.image_dir.glob("*.jpg"))[: args.num_images]
    if not image_paths:
        raise FileNotFoundError(f"No benchmark images found in {args.image_dir}")

    predictor = FormulaLensPredictor(checkpoint_path=args.model_path, device="cpu")

    for image_path in image_paths[: args.warmup]:
        result = predictor.predict(image_path)
        postprocess_detections(result.detections, score_threshold=predictor.score_threshold, iou_threshold=predictor.nms_threshold)

    latencies_ms: list[float] = []
    for image_path in image_paths:
        started_at = time.perf_counter()
        result = predictor.predict(image_path)
        postprocess_detections(result.detections, score_threshold=predictor.score_threshold, iou_threshold=predictor.nms_threshold)
        latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

    summary = {
        "model_path": str(args.model_path),
        "images": len(image_paths),
        "mean_ms": round(statistics.mean(latencies_ms), 3),
        "median_ms": round(statistics.median(latencies_ms), 3),
        "p95_ms": round(sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1], 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
