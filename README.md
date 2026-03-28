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

## Train

Training uses YOLOX and requires CUDA. Before training, place a Roboflow COCO export under `datasets/roboflow_exports/formulas_coco_v1/` with `train/` and optionally `valid/` and `test/`.

`scripts/train.sh` automatically runs `scripts/prepare_dataset.py`, which rewrites the export into the YOLOX-friendly layout under `datasets/prepared/formulas_coco_v1/`.

If YOLOX is not present yet:

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX third_party/YOLOX
```

Set up the training environment, activate it, download the pretrained YOLOX-Nano checkpoint, then start training:

```bash
bash scripts/setup_train_env.sh cu124
source .venvs/formulalens-train-cu124/bin/activate
bash scripts/download_weights.sh
bash scripts/train.sh --fp16
```

If your machine uses a different CUDA stack, use `cu121` instead of `cu124`.

Artifacts:

- pretrained checkpoint: `weights/pretrained/yolox_nano.pth`
- fine-tuned checkpoints: `weights/finetuned/yolox_nano/`
- main exp file: `configs/train/yolox_nano.py`

## Export weights

Export the best checkpoint to ONNX:

```bash
source .venvs/formulalens-train-cu124/bin/activate
bash scripts/export_onnx.sh
```

Default output:

```text
weights/finetuned/v1/formulalens_yolox_nano_v1.0.0.onnx
```

You can override `MODEL_VERSION`, `OUTPUT_DIR`, `OUTPUT_PATH`, or `CKPT_PATH` through environment variables.

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
- `FORMULALENS_MODEL_VERSION=v1.0.0 docker compose up --build`
- `FORMULALENS_MODEL_URL=... docker compose up --build`
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

## Debug endpoints

- `POST /debug/detect`: returns `image/jpeg` with rendered boxes
- `POST /debug/save-bad-case`: saves the input image and metadata for later inspection

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
