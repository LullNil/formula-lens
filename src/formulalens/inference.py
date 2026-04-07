from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional backend
    cv2 = None

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - optional backend
    ort = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional backend
    torch = None

from .schemas import BoundingBox, Detection, InferenceResult
from .utils.image import composite_pil_on_white


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = PROJECT_ROOT / "third_party" / "YOLOX"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))


DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "weights" / "finetuned" / "yolox_nano" / "best_ckpt.pth"
DEFAULT_ONNX_PATH = PROJECT_ROOT / "weights" / "finetuned" / "v2" / "formulalens_yolox_nano_v2.0.0.onnx"
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
    "whole_part",
)
DEFAULT_COLORS = (
    (39, 125, 161),
    (76, 175, 80),
    (255, 167, 38),
    (229, 57, 53),
    (126, 87, 194),
    (0, 172, 193),
    (141, 110, 99),
)
logger = logging.getLogger("formulalens.inference")


def _composite_bgra_on_white(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected BGRA image, got shape={image.shape}")

    bgr = image[:, :, :3].astype(np.float32)
    alpha = (image[:, :, 3:4].astype(np.float32) / 255.0)
    background = np.full_like(bgr, 255.0)
    composited = bgr * alpha + background * (1.0 - alpha)
    return np.ascontiguousarray(composited.astype(np.uint8))


def _require_cv2(feature: str) -> None:
    if cv2 is None:
        raise RuntimeError(
            f"OpenCV is required for {feature}. Install dependencies from requirements/service.txt."
        )


def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, :, ::-1])


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, :, ::-1])


def _gray_to_bgr(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8, copy=False)


def _resize_bgr_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if cv2 is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    pil_image = Image.fromarray(_bgr_to_rgb(image))
    resized = pil_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return _rgb_to_bgr(np.array(resized, dtype=np.uint8))


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
        if cv2 is not None:
            raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")
            if raw.ndim == 2:
                bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            elif raw.ndim == 3 and raw.shape[2] == 4:
                bgr = _composite_bgra_on_white(raw)
            elif raw.ndim == 3 and raw.shape[2] == 3:
                bgr = np.ascontiguousarray(raw.copy())
            else:
                raise ValueError(f"Unsupported image shape: {raw.shape}")
        else:
            with Image.open(image_path) as pil_image:
                pil_image.load()
                rgb = composite_pil_on_white(pil_image)
            bgr = _rgb_to_bgr(np.array(rgb, dtype=np.uint8))
        return bgr, str(image_path)

    if isinstance(image, Image.Image):
        rgb = composite_pil_on_white(image)
        return _rgb_to_bgr(np.array(rgb, dtype=np.uint8)), None

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return _gray_to_bgr(image), None
        if image.ndim == 3 and image.shape[2] == 3:
            return np.ascontiguousarray(image.copy()), None
        if image.ndim == 3 and image.shape[2] == 4:
            return _composite_bgra_on_white(image), None
        raise ValueError(f"Unsupported ndarray image shape: {image.shape}")

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _preprocess_image(img: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    padded = np.full((input_size[0], input_size[1], 3), 255, dtype=np.uint8)
    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = _resize_bgr_image(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
    ).astype(np.uint8)
    pad_y = (input_size[0] - resized.shape[0]) // 2
    pad_x = (input_size[1] - resized.shape[1]) // 2
    padded[pad_y : pad_y + resized.shape[0], pad_x : pad_x + resized.shape[1]] = resized
    chw = padded.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(chw), float(ratio), int(pad_x), int(pad_y)


def _load_exp_class(exp_file: Path):
    if not exp_file.is_file():
        raise FileNotFoundError(f"Exp file not found: {exp_file}")

    module_name = f"formulalens_exp_{exp_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, exp_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load exp module from {exp_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Exp"):
        raise AttributeError(f"Exp class not found in {exp_file}")
    return module.Exp


class FormulaLensPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        exp_file: str | Path | None = DEFAULT_EXP_FILE,
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
        self.exp_file = Path(exp_file) if exp_file is not None else DEFAULT_EXP_FILE
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
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = tuple(input_meta.shape)

    def _load_torch_model(self) -> None:
        if torch is None:
            raise RuntimeError("torch is required to load PyTorch checkpoints.")

        exp_cls = _load_exp_class(self.exp_file)
        self.exp = exp_cls()
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

    def _decode_prediction_rows(
        self,
        rows: np.ndarray,
        ratio: float,
        pad_x: int,
        pad_y: int,
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        detections: list[Detection] = []

        for row in rows:
            object_conf = float(row[4])
            class_scores = row[5 : 5 + self.num_classes]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            score = object_conf * class_conf
            if score < self.score_threshold:
                continue

            cx, cy, width, height = [float(value) for value in row[:4]]
            x1 = (cx - width * 0.5 - pad_x) / ratio
            y1 = (cy - height * 0.5 - pad_y) / ratio
            x2 = (cx + width * 0.5 - pad_x) / ratio
            y2 = (cy + height * 0.5 - pad_y) / ratio

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

    def _decode_predictions(
        self,
        predictions: np.ndarray,
        ratio: float,
        pad_x: int,
        pad_y: int,
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        return self._decode_prediction_rows(
            predictions[0],
            ratio=ratio,
            pad_x=pad_x,
            pad_y=pad_y,
            image_width=image_width,
            image_height=image_height,
        )

    def _prepare_image_batch(
        self,
        images: list[str | Path | np.ndarray | Image.Image],
    ) -> list[dict[str, object]]:
        prepared: list[dict[str, object]] = []
        for image in images:
            raw_bgr, image_path = _load_bgr_image(image)
            image_height, image_width = raw_bgr.shape[:2]
            processed, ratio, pad_x, pad_y = _preprocess_image(raw_bgr, self.test_size)
            prepared.append(
                {
                    "image_path": image_path,
                    "image_width": image_width,
                    "image_height": image_height,
                    "processed": processed,
                    "ratio": ratio,
                    "pad_x": pad_x,
                    "pad_y": pad_y,
                }
            )
        return prepared

    def _build_inference_results(
        self,
        prepared_items: list[dict[str, object]],
        predictions: np.ndarray,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for index, item in enumerate(prepared_items):
            detections = self._decode_prediction_rows(
                predictions[index],
                ratio=float(item["ratio"]),
                pad_x=int(item["pad_x"]),
                pad_y=int(item["pad_y"]),
                image_width=int(item["image_width"]),
                image_height=int(item["image_height"]),
            )
            results.append(
                InferenceResult(
                    checkpoint_path=str(self.model_path),
                    image_path=item["image_path"],
                    image_width=int(item["image_width"]),
                    image_height=int(item["image_height"]),
                    resize_ratio=float(item["ratio"]),
                    detections=detections,
                )
            )
        return results

    def _supports_native_batching(self, batch_size: int) -> bool:
        if batch_size <= 1:
            return False
        if self.backend == "torch":
            return True

        batch_dim = self.input_shape[0] if hasattr(self, "input_shape") and self.input_shape else None
        if batch_dim is None or isinstance(batch_dim, str):
            return True
        if isinstance(batch_dim, int):
            return batch_dim == batch_size
        return False

    def predict(self, image: str | Path | np.ndarray | Image.Image) -> InferenceResult:
        raw_bgr, image_path = _load_bgr_image(image)
        image_height, image_width = raw_bgr.shape[:2]
        processed, ratio, pad_x, pad_y = _preprocess_image(raw_bgr, self.test_size)
        batch = np.expand_dims(processed, axis=0)
        predictions = self._run_model(batch)
        detections = self._decode_predictions(predictions, ratio, pad_x, pad_y, image_width, image_height)

        return InferenceResult(
            checkpoint_path=str(self.model_path),
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            resize_ratio=ratio,
            detections=detections,
        )

    def predict_many_with_info(
        self,
        images: Iterable[str | Path | np.ndarray | Image.Image],
    ) -> tuple[list[InferenceResult], bool]:
        image_list = list(images)
        if not image_list:
            return [], False

        prepared_items = self._prepare_image_batch(image_list)
        if len(prepared_items) == 1:
            batch = np.expand_dims(prepared_items[0]["processed"], axis=0)
            predictions = self._run_model(batch)
            return self._build_inference_results(prepared_items, predictions), False

        if not self._supports_native_batching(len(prepared_items)):
            results: list[InferenceResult] = []
            for item in prepared_items:
                single_batch = np.expand_dims(item["processed"], axis=0)
                predictions = self._run_model(single_batch)
                results.extend(self._build_inference_results([item], predictions))
            return results, False

        batch = np.stack([item["processed"] for item in prepared_items], axis=0)
        try:
            predictions = self._run_model(batch)
            if predictions.shape[0] != len(prepared_items):
                raise RuntimeError(
                    f"Unexpected batch output shape {predictions.shape} for {len(prepared_items)} inputs."
                )
            return self._build_inference_results(prepared_items, predictions), True
        except Exception as exc:
            logger.warning("Falling back to sequential inference for batch_size=%s: %s", len(prepared_items), exc)
            results: list[InferenceResult] = []
            for item in prepared_items:
                single_batch = np.expand_dims(item["processed"], axis=0)
                predictions = self._run_model(single_batch)
                results.extend(self._build_inference_results([item], predictions))
            return results, False

    def predict_many(self, images: Iterable[str | Path | np.ndarray | Image.Image]) -> list[InferenceResult]:
        return self.predict_many_with_info(images)[0]

    def render(
        self,
        image: str | Path | np.ndarray | Image.Image,
        result: InferenceResult | None = None,
        output_path: str | Path | None = None,
        show_labels: bool = True,
        line_thickness: int = 2,
    ) -> np.ndarray:
        _require_cv2("rendering detections")
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
