from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = PROJECT_ROOT / "third_party" / "YOLOX"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from configs.train.yolox_nano import Exp
from .schemas import BoundingBox, Detection, InferenceResult
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess


DEFAULT_EXP_FILE = PROJECT_ROOT / "configs" / "train" / "yolox_nano.py"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "weights" / "finetuned" / "yolox_nano" / "best_ckpt.pth"
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.45
DEFAULT_CLASS_NAMES = (
    "block",
    "denominator",
    "exponent",
    "numerator",
    "system_row",
    "text",
)
DEFAULT_COLORS = (
    (39, 125, 161),
    (76, 175, 80),
    (255, 167, 38),
    (229, 57, 53),
    (126, 87, 194),
    (0, 172, 193),
)


def _resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _load_bgr_image(image: str | Path | np.ndarray | Image.Image) -> tuple[np.ndarray, str | None]:
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return bgr, str(image_path)

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR), None

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
        if image.ndim == 3 and image.shape[2] == 3:
            return np.ascontiguousarray(image.copy()), None
        raise ValueError(f"Unsupported ndarray image shape: {image.shape}")

    raise TypeError(f"Unsupported image type: {type(image)!r}")


class FormulaLensPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        exp_file: str | Path = DEFAULT_EXP_FILE,
        device: str | None = "auto",
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        fp16: bool = False,
        legacy: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.exp_file = Path(exp_file)
        self.device = _resolve_device(device)
        self.fp16 = fp16 and self.device.type == "cuda"

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.exp_file.is_file():
            raise FileNotFoundError(f"Exp file not found: {self.exp_file}")

        self.exp = Exp()
        self.exp.test_conf = float(score_threshold)
        self.exp.nmsthre = float(nms_threshold)
        self.class_names = tuple(getattr(self.exp, "class_names", DEFAULT_CLASS_NAMES))
        self.num_classes = int(self.exp.num_classes)
        self.test_size = tuple(self.exp.test_size)

        self.model = self.exp.get_model()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        if self.fp16:
            self.model.half()

        self.preproc = ValTransform(legacy=legacy)

    def predict(self, image: str | Path | np.ndarray | Image.Image) -> InferenceResult:
        raw_bgr, image_path = _load_bgr_image(image)
        image_height, image_width = raw_bgr.shape[:2]
        ratio = min(self.test_size[0] / image_height, self.test_size[1] / image_width)

        processed, _ = self.preproc(raw_bgr, None, self.test_size)
        tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
        tensor = tensor.float()
        if self.fp16:
            tensor = tensor.half()

        with torch.no_grad():
            outputs = self.model(tensor)
            outputs = postprocess(
                outputs,
                self.num_classes,
                conf_thre=self.exp.test_conf,
                nms_thre=self.exp.nmsthre,
                class_agnostic=True,
            )

        detections: list[Detection] = []
        output = outputs[0]
        if output is not None:
            output = output.detach().cpu().numpy()
            output[:, 0:4] /= ratio

            ordered_rows = sorted(output, key=lambda row: float(row[4] * row[5]), reverse=True)
            for row in ordered_rows:
                class_id = int(row[6])
                score = float(row[4] * row[5])
                x1 = float(np.clip(row[0], 0, image_width))
                y1 = float(np.clip(row[1], 0, image_height))
                x2 = float(np.clip(row[2], 0, image_width))
                y2 = float(np.clip(row[3], 0, image_height))
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=self.class_names[class_id],
                        score=score,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        return InferenceResult(
            checkpoint_path=str(self.checkpoint_path),
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            resize_ratio=float(ratio),
            detections=detections,
        )

    def predict_many(self, images: Iterable[str | Path]) -> list[InferenceResult]:
        return [self.predict(image) for image in images]

    def render(
        self,
        image: str | Path | np.ndarray | Image.Image,
        result: InferenceResult | None = None,
        output_path: str | Path | None = None,
    ) -> np.ndarray:
        rendered, _ = _load_bgr_image(image)
        current_result = result or self.predict(image)

        for detection in current_result.detections:
            color = DEFAULT_COLORS[detection.class_id % len(DEFAULT_COLORS)]
            x1 = int(round(detection.bbox.x1))
            y1 = int(round(detection.bbox.y1))
            x2 = int(round(detection.bbox.x2))
            y2 = int(round(detection.bbox.y2))
            label = f"{detection.label} {detection.score:.2f}"

            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1,
            )
            text_top = max(0, y1 - text_height - baseline - 4)
            text_bottom = text_top + text_height + baseline + 4
            cv2.rectangle(
                rendered,
                (x1, text_top),
                (x1 + text_width + 6, text_bottom),
                color,
                thickness=-1,
            )
            cv2.putText(
                rendered,
                label,
                (x1 + 3, text_bottom - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), rendered)

        return rendered


def load_default_predictor(device: str | None = "auto") -> FormulaLensPredictor:
    return FormulaLensPredictor(device=device)
