# FormulaLens

FormulaLens is a structure detector for formulas. It detects structural regions inside a rendered formula and returns boxes, labels, scores, confidence, and an inferred structure type.

FormulaLens does not perform OCR. Formula-to-text reconstruction is expected to happen downstream, for example with `pix2tex`.

## Detection classes

- `block`
- `denominator`
- `exponent`
- `numerator`
- `system_row`
- `text`
- `whole_part`

## Train

Training uses YOLOX and requires CUDA. Before training, place a Roboflow COCO export under `datasets/roboflow_exports/formulas_coco_v2/` with `train/` and optionally `valid/` and `test/`.

`scripts/train.sh` automatically runs `scripts/prepare_dataset.py`, which rewrites the export into the YOLOX-friendly layout under `datasets/prepared/formulas_coco_v2/`.

If YOLOX is not present yet:

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX third_party/YOLOX
```

Set up the training environment, activate it, download the pretrained YOLOX-Nano checkpoint, then start training:

```bash
bash scripts/setup_train_env.sh cu124
source .venvs/formulalens-train-cu124/bin/activate
bash scripts/download_weights.sh
EXP_FILE=$PWD/configs/train/yolox_nano_v2.py \
WEIGHTS_PATH=$PWD/weights/finetuned/yolox_nano/best_ckpt.pth \
SOURCE_ROOT=$PWD/datasets/roboflow_exports/formulas_coco_v2 \
PREPARED_DATASET_DIR=$PWD/datasets/prepared/formulas_coco_v2 \
bash scripts/train.sh --fp16
```

If your machine uses a different CUDA stack, use `cu121` instead of `cu124`.

Artifacts:

- pretrained checkpoint: `weights/pretrained/yolox_nano.pth`
- base checkpoint: `weights/finetuned/yolox_nano/best_ckpt.pth`
- fine-tuned checkpoints: `weights/finetuned/yolox_nano_v2/`
- main exp file: `configs/train/yolox_nano_v2.py`

## Export weights

Export the best checkpoint to ONNX:

```bash
source .venvs/formulalens-train-cu124/bin/activate
EXP_FILE=$PWD/configs/train/yolox_nano_v2.py \
CKPT_PATH=$PWD/weights/finetuned/yolox_nano_v2/best_ckpt.pth \
MODEL_VERSION=v2.0.0 \
OUTPUT_DIR=$PWD/weights/finetuned/v2 \
bash scripts/export_onnx.sh
```

Default output:

```text
weights/finetuned/v2/formulalens_yolox_nano_v2.0.0.onnx
```

You can override `MODEL_VERSION`, `OUTPUT_DIR`, `OUTPUT_PATH`, or `CKPT_PATH` through environment variables.

To export a separate dynamic-batch artifact without replacing the current service model:

```bash
source .venvs/formulalens-train-cu124/bin/activate
EXP_FILE=$PWD/configs/train/yolox_nano_v2.py \
CKPT_PATH=$PWD/weights/finetuned/yolox_nano_v2/best_ckpt.pth \
MODEL_VERSION=v2.0.0-batch \
OUTPUT_DIR=$PWD/weights/finetuned/v2 \
ONNX_BATCH_SIZE=8 \
ONNX_DYNAMIC_BATCH=1 \
ONNX_NO_SIMPLIFY=1 \
bash scripts/export_onnx.sh
```

This produces a separate file such as `weights/finetuned/v2/formulalens_yolox_nano_v2.0.0-batch.onnx`.

## Run CPU inference service

The runtime serves CPU inference through ONNX Runtime and starts from a versioned ONNX file. If the model file is missing, `scripts/start_service.sh` downloads it automatically.

### Docker

This is the simplest way to run the service:

```bash
docker compose up -d --build
```

By default the API is available at `http://localhost:18080`.

Useful overrides:

- `FORMULALENS_HOST_PORT=18081 docker compose up --build`
- `FORMULALENS_MODEL_VERSION=v2.0.0 docker compose up --build`
- `FORMULALENS_MODEL_VERSION=v2.0.0-batch docker compose up --build`
- `FORMULALENS_MODEL_URL=https://github.com/LullNil/formula-lens/releases/download/v2.0.0/formulalens_yolox_nano_v2.0.0.onnx docker compose up --build`
- `FORMULALENS_RELEASE_REPO=owner/repo docker compose up --build`

### Local

Create a small service environment and start FastAPI directly:

```bash
python3 -m venv .venvs/formulalens-service
source .venvs/formulalens-service/bin/activate
pip install -r requirements/service.txt
bash scripts/start_service.sh
```

By default the local service listens on `http://0.0.0.0:8000`. You can override host, port, or model path with `FORMULALENS_HOST`, `FORMULALENS_PORT`, and `FORMULALENS_MODEL_PATH`.

Optional routing refinement can be enabled with `FORMULALENS_RENDER_SIMILARITY_ENABLED=1`. When enabled, `/route` renders `pix2tex_output` with the configured backend (default: `mathtext_parser`, override with `FORMULALENS_RENDER_SIMILARITY_BACKEND`), normalizes both images to the same canvas, and compares their foreground masks. If rendering fails, the request falls back to the original routing logic.

## Quick checks with curl

For Docker use:

```bash
API_URL=http://localhost:18080
IMAGE=path/to/formula.png
```

For local FastAPI use:

```bash
API_URL=http://localhost:8000
IMAGE=path/to/formula.png
```

Detect structure:

```bash
curl -X POST "$API_URL/detect" \
  -F "image=@$IMAGE"
```

Run FormulaLens vs `pix2tex` routing:

```bash
curl -X POST "$API_URL/route" \
  -F "image=@$IMAGE" \
  -F "pix2tex_output=\\frac{x}{y}" \
  -F "pix2tex_score=0.93"
```

Run batched routing in one request:

```bash
curl -X POST "$API_URL/route-batch" \
  -F "images=@$IMAGE_ONE" \
  -F "images=@$IMAGE_TWO" \
  -F 'pix2tex_outputs_json=["\\\\frac{x}{y}","x^2+y^2"]' \
  -F 'pix2tex_scores_json=[0.93,0.88]'
```

Get a rendered debug image with boxes:

```bash
curl -X POST "$API_URL/debug/detect" \
  -F "image=@$IMAGE" \
  --output debug_detect.jpg
```

Save a bad case for later inspection:

```bash
curl -X POST "$API_URL/debug/save-bad-case" \
  -F "image=@$IMAGE" \
  -F 'intermediate_outputs={"pix2tex_output":"\\\\frac{x}{y}","pix2tex_score":0.93}' \
  -F "reason=manual smoke test"
```

## JSON API

All main endpoints use `multipart/form-data`.

### `POST /detect`

Request fields:

- `image`

Response:

```json
{
  "ok": true,
  "bbox_format": "xyxy",
  "detections": [
    {
      "class_id": 3,
      "class": "numerator",
      "score": 0.936,
      "bbox": [12.0, 8.0, 64.0, 31.0]
    }
  ],
  "global_confidence": 0.91,
  "confidence_level": "high",
  "structure_type": "fraction",
  "model_version": "v1.0.0",
  "confidence_breakdown": {
    "global_confidence": 0.91,
    "base_score": 0.94,
    "geometry_penalty": 0.01,
    "combination_penalty": 0.02,
    "detection_count": 2,
    "class_distribution": {
      "denominator": 1,
      "numerator": 1
    }
  }
}
```

`bbox_format: "xyxy"` means each box is `[x1, y1, x2, y2]`.

### `POST /route`

Request fields:

- `image`
- `pix2tex_output`
- `pix2tex_score`

This endpoint combines FormulaLens structural detections with downstream OCR confidence and returns one of:

- `use_formula_lens`
- `use_pix2tex`
- `use_heuristics`

If render-similarity refinement is enabled, the response also includes a `render_similarity` block with the renderer name, whether the comparison was applied, and the similarity score when available.

Response:

```json
{
  "ok": true,
  "decision": "use_formula_lens",
  "reason": "Structural detections are confident enough to override pix2tex.",
  "bbox_format": "xyxy",
  "detections": [],
  "global_confidence": 0.87,
  "confidence_level": "high",
  "structure_type": "fraction",
  "model_version": "v1.0.0",
  "render_similarity": {
    "enabled": true,
    "applied": true,
    "score": 0.91,
    "renderer": "mathtext_parser",
    "reason": "Rendered pix2tex output was compared against the normalized input mask."
  },
  "confidence_breakdown": {
    "global_confidence": 0.87,
    "base_score": 0.9,
    "geometry_penalty": 0.01,
    "combination_penalty": 0.02,
    "detection_count": 2,
    "class_distribution": {
      "denominator": 1,
      "numerator": 1
    }
  }
}
```

`batched_inference_used` shows whether the loaded model artifact accepted native batch inference. If it is `false`, the endpoint still saves HTTP and serialization overhead, but the current model was processed sequentially inside the service.

### `POST /route-batch`

Request fields:

- `images`: repeated multipart file field
- `pix2tex_outputs_json`: optional JSON array aligned with `images`
- `pix2tex_scores_json`: optional JSON array aligned with `images`

Response:

```json
{
  "ok": true,
  "count": 2,
  "batched_inference_used": true,
  "results": [
    {
      "ok": true,
      "decision": "use_formula_lens",
      "reason": "Structural detections are confident enough to override pix2tex.",
      "bbox_format": "xyxy",
      "detections": [],
      "global_confidence": 0.87,
      "confidence_level": "high",
      "structure_type": "fraction",
      "model_version": "v2.0.0",
      "render_similarity": {
        "enabled": true,
        "applied": true,
        "score": 0.91,
        "renderer": "mathtext_parser",
        "reason": "Rendered pix2tex output was compared against the normalized input mask."
      },
      "confidence_breakdown": {
        "global_confidence": 0.87,
        "base_score": 0.9,
        "geometry_penalty": 0.01,
        "combination_penalty": 0.02,
        "detection_count": 2,
        "class_distribution": {
          "denominator": 1,
          "numerator": 1
        }
      }
    }
  ]
}
```

## Debug endpoints

- `POST /debug/detect`: returns `image/jpeg` with rendered boxes
- `POST /debug/save-bad-case`: saves the input image and metadata for later inspection

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
