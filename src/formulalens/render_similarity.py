from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import os

import numpy as np
from PIL import Image

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional backend
    cv2 = None


DEFAULT_CANVAS_SIZE = (416, 416)
DEFAULT_FONT_SIZE = 28
DEFAULT_PADDING = 10
DEFAULT_DILATION_KERNEL = 5


@dataclass(frozen=True)
class RenderSimilarityResult:
    enabled: bool
    applied: bool
    score: float | None = None
    renderer: str | None = None
    reason: str | None = None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _require_cv2(feature: str) -> None:
    if cv2 is None:
        raise RuntimeError(f"OpenCV is required for {feature}.")


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _extract_foreground_mask(image: Image.Image) -> np.ndarray:
    _require_cv2("extracting render similarity masks")
    rgb = _to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _crop_to_foreground(mask: np.ndarray, padding: int) -> np.ndarray:
    points = cv2.findNonZero(mask)
    if points is None:
        return mask

    x, y, width, height = cv2.boundingRect(points)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(mask.shape[1], x + width + padding)
    y2 = min(mask.shape[0], y + height + padding)
    return mask[y1:y2, x1:x2]


def normalize_foreground_mask(
    mask: np.ndarray,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    _require_cv2("normalizing render similarity masks")
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={mask.shape}")

    foreground = np.where(mask > 0, 255, 0).astype(np.uint8, copy=False)
    if not np.any(foreground):
        return np.zeros(canvas_size, dtype=np.uint8)

    cropped = _crop_to_foreground(foreground, padding=padding)
    target_height, target_width = canvas_size
    ratio = min(target_height / cropped.shape[0], target_width / cropped.shape[1])
    resized_width = max(1, int(round(cropped.shape[1] * ratio)))
    resized_height = max(1, int(round(cropped.shape[0] * ratio)))
    resized = cv2.resize(cropped, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    normalized = np.zeros((target_height, target_width), dtype=np.uint8)
    offset_y = (target_height - resized_height) // 2
    offset_x = (target_width - resized_width) // 2
    normalized[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return normalized


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float32, copy=False)
    right = right.astype(np.float32, copy=False)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return _clamp(float(np.dot(left, right) / denominator))


def compute_mask_similarity(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    dilation_kernel: int = DEFAULT_DILATION_KERNEL,
) -> float:
    _require_cv2("computing render similarity")
    left = np.where(left_mask > 0, 1, 0).astype(np.uint8, copy=False)
    right = np.where(right_mask > 0, 1, 0).astype(np.uint8, copy=False)
    if not np.any(left) or not np.any(right):
        return 0.0

    kernel_size = max(1, int(dilation_kernel))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    left_dilated = cv2.dilate(left, kernel, iterations=1)
    right_dilated = cv2.dilate(right, kernel, iterations=1)

    intersection = float(np.logical_and(left_dilated > 0, right_dilated > 0).sum())
    union = float(np.logical_or(left_dilated > 0, right_dilated > 0).sum())
    iou = intersection / union if union > 0.0 else 0.0

    row_similarity = _cosine_similarity(left_dilated.sum(axis=1), right_dilated.sum(axis=1))
    column_similarity = _cosine_similarity(left_dilated.sum(axis=0), right_dilated.sum(axis=0))
    area_ratio = min(float(left.sum()), float(right.sum())) / max(float(left.sum()), float(right.sum()))

    return _clamp(0.50 * iou + 0.20 * row_similarity + 0.20 * column_similarity + 0.10 * area_ratio)


def normalize_input_image(
    image: Image.Image,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    return normalize_foreground_mask(_extract_foreground_mask(image), canvas_size=canvas_size, padding=padding)


def _normalize_latex_expression(expression: str) -> str:
    normalized = expression.strip()
    if not normalized:
        raise ValueError("pix2tex output is empty.")
    if normalized.startswith("$") and normalized.endswith("$"):
        return normalized
    return f"${normalized}$"


@lru_cache(maxsize=128)
def _render_latex_png(expression: str, font_size: int) -> bytes:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("matplotlib is not installed.") from exc

    figure = Figure(figsize=(4, 4), dpi=200, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        _normalize_latex_expression(expression),
        ha="center",
        va="center",
        fontsize=font_size,
        color="black",
    )

    buffer = BytesIO()
    canvas.print_png(buffer)
    return buffer.getvalue()


def render_latex_mask(
    expression: str,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    font_size: int = DEFAULT_FONT_SIZE,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    png_bytes = _render_latex_png(expression, font_size=font_size)
    with Image.open(BytesIO(png_bytes)) as rendered:
        rendered.load()
        return normalize_foreground_mask(_extract_foreground_mask(rendered), canvas_size=canvas_size, padding=padding)


def compute_render_similarity(
    image: Image.Image,
    pix2tex_output: str,
    *,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    font_size: int = DEFAULT_FONT_SIZE,
    padding: int = DEFAULT_PADDING,
    dilation_kernel: int = DEFAULT_DILATION_KERNEL,
) -> RenderSimilarityResult:
    try:
        input_mask = normalize_input_image(image, canvas_size=canvas_size, padding=padding)
    except Exception as exc:
        return RenderSimilarityResult(
            enabled=True,
            applied=False,
            renderer="mathtext",
            reason=f"Unable to normalize the input image: {exc}",
        )

    try:
        rendered_mask = render_latex_mask(
            pix2tex_output,
            canvas_size=canvas_size,
            font_size=font_size,
            padding=padding,
        )
    except Exception as exc:
        return RenderSimilarityResult(
            enabled=True,
            applied=False,
            renderer="mathtext",
            reason=f"Unable to render pix2tex output: {exc}",
        )

    score = compute_mask_similarity(input_mask, rendered_mask, dilation_kernel=dilation_kernel)
    return RenderSimilarityResult(
        enabled=True,
        applied=True,
        score=score,
        renderer="mathtext",
        reason="Rendered pix2tex output was compared against the normalized input mask.",
    )
