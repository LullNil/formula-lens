from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - optional backend
    ort = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional backend
    torch = None

from .schemas import BoundingBox, Detection, InferenceResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = PROJECT_ROOT / "third_party" / "YOLOX"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))


DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "weights" / "finetuned" / "yolox_nano" / "best_ckpt.pth"
DEFAULT_ONNX_PATH = PROJECT_ROOT / "weights" / "finetuned" / "v1" / "formulalens_yolox_nano_v1.0.0.onnx"
DEFAULT_EXP_FILE = PROJECT_ROOT / "configs" / "train" / "yolox_nano.py"
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.45
DEFAULT_INPUT_SIZE = (416, 416)
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


def _resolve_device(device: str | None) -> str:
    if device in (None, "auto"):
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        return "cpu"
    return device


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


def _preprocess_image(img: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    chw = padded.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(chw), float(ratio)


class FormulaLensPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        exp_file: str | Path = DEFAULT_EXP_FILE,
        device: str | None = "auto",
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        fp16: bool = False,
        input_size: tuple[int, int] | None = None,
        class_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        selected_path = Path(checkpoint_path) if checkpoint_path is not None else (
            DEFAULT_ONNX_PATH if DEFAULT_ONNX_PATH.is_file() else DEFAULT_CHECKPOINT_PATH
        )
        self.model_path = selected_path
        self.exp_file = Path(exp_file)
        self.device = _resolve_device(device)
        self.fp16 = fp16 and self.device == "cuda"
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.class_names = tuple(class_names or DEFAULT_CLASS_NAMES)
        self.num_classes = len(self.class_names)
        self.test_size = tuple(input_size or DEFAULT_INPUT_SIZE)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.backend = "onnxruntime" if self.model_path.suffix == ".onnx" else "torch"
        if self.backend == "onnxruntime":
            self._load_onnx_session()
        else:
            self._load_torch_model()

    def _load_onnx_session(self) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is required to load ONNX models.")

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _load_torch_model(self) -> None:
        if torch is None:
            raise RuntimeError("torch is required to load PyTorch checkpoints.")
        if not self.exp_file.is_file():
            raise FileNotFoundError(f"Exp file not found: {self.exp_file}")

        from configs.train.yolox_nano import Exp

        self.exp = Exp()
        self.class_names = tuple(getattr(self.exp, "class_names", self.class_names))
        self.num_classes = int(self.exp.num_classes)
        self.test_size = tuple(self.exp.test_size)
        self.score_threshold = float(self.score_threshold or self.exp.test_conf)
        self.nms_threshold = float(self.nms_threshold or self.exp.nmsthre)

        self.model = self.exp.get_model()
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        if self.fp16:
            self.model.half()

    def _run_model(self, batch: np.ndarray) -> np.ndarray:
        if self.backend == "onnxruntime":
            outputs = self.session.run(None, {self.input_name: batch})
            return np.asarray(outputs[0])

        tensor = torch.from_numpy(batch).to(self.device)
        tensor = tensor.float()
        if self.fp16:
            tensor = tensor.half()

        with torch.no_grad():
            outputs = self.model(tensor)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return outputs.detach().cpu().numpy()

    def _decode_predictions(
        self,
        predictions: np.ndarray,
        ratio: float,
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        detections: list[Detection] = []
        rows = predictions[0]

        for row in rows:
            object_conf = float(row[4])
            class_scores = row[5 : 5 + self.num_classes]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            score = object_conf * class_conf
            if score < self.score_threshold:
                continue

            cx, cy, width, height = [float(value) for value in row[:4]]
            x1 = (cx - width * 0.5) / ratio
            y1 = (cy - height * 0.5) / ratio
            x2 = (cx + width * 0.5) / ratio
            y2 = (cy + height * 0.5) / ratio

            x1 = float(np.clip(x1, 0, image_width))
            y1 = float(np.clip(y1, 0, image_height))
            x2 = float(np.clip(x2, 0, image_width))
            y2 = float(np.clip(y2, 0, image_height))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    label=self.class_names[class_id],
                    score=score,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        return sorted(detections, key=lambda item: item.score, reverse=True)

    def predict(self, image: str | Path | np.ndarray | Image.Image) -> InferenceResult:
        raw_bgr, image_path = _load_bgr_image(image)
        image_height, image_width = raw_bgr.shape[:2]
        processed, ratio = _preprocess_image(raw_bgr, self.test_size)
        batch = np.expand_dims(processed, axis=0)
        predictions = self._run_model(batch)
        detections = self._decode_predictions(predictions, ratio, image_width, image_height)

        return InferenceResult(
            checkpoint_path=str(self.model_path),
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            resize_ratio=ratio,
            detections=detections,
        )

    def predict_many(self, images: Iterable[str | Path]) -> list[InferenceResult]:
        return [self.predict(image) for image in images]

    def render(
        self,
        image: str | Path | np.ndarray | Image.Image,
        result: InferenceResult | None = None,
        output_path: str | Path | None = None,
        show_labels: bool = True,
        line_thickness: int = 2,
    ) -> np.ndarray:
        rendered, _ = _load_bgr_image(image)
        current_result = result or self.predict(image)

        for detection in current_result.detections:
            color = DEFAULT_COLORS[detection.class_id % len(DEFAULT_COLORS)]
            x1 = int(round(detection.bbox.x1))
            y1 = int(round(detection.bbox.y1))
            x2 = int(round(detection.bbox.x2))
            y2 = int(round(detection.bbox.y2))

            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, line_thickness)
            if show_labels:
                label = f"{detection.label} {detection.score:.2f}"
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
