# FormulaLens

FormulaLens is a structure detector for formulas. It predicts structural regions such as `block`, `numerator`, `denominator`, `exponent`, `system_row`, and `text`, but it does not perform OCR itself. Formula-to-text reconstruction is expected to happen downstream, for example with `pix2tex`.

## Detection classes

- `block`
- `denominator`
- `exponent`
- `numerator`
- `system_row`
- `text`

## Repository layout

- `configs/train/` contains YOLOX exp files. The main training config is `configs/train/yolox_nano.py`.
- `configs/service/settings.yaml` contains the service runtime config, including the default `v1.0.0` ONNX path.
- `datasets/roboflow_exports/` stores the raw Roboflow export.
- `datasets/prepared/` stores the normalized COCO layout expected by YOLOX.
- `weights/pretrained/` stores the starting YOLOX weights for fine-tuning.
- `weights/finetuned/` stores fine-tuned checkpoints and exported versioned ONNX artifacts.
- `scripts/` contains dataset preparation, training, export, benchmarking and runtime startup helpers.
- `src/formulalens/` contains the service runtime: inference, postprocess, confidence, routing, crops, schemas and FastAPI.
- `docker/` contains the minimal service and training Dockerfiles.

## Architecture

Best practice in this repository is:

- container = code
- weights = runtime artifacts
- image build = no `.pth`, no `.onnx`
- model download = done at startup if the versioned file is missing

This means:

- do not commit `.pth` or `.onnx` into Git
- do not rebuild the image just to change the model
- do not hardcode weights into the container filesystem

The intended storage for deployed models is GitHub Releases.

## Data pipeline

The working training pipeline is:

1. Unpack a Roboflow COCO export into `datasets/roboflow_exports/formulas_coco_v1/`.
2. Run `python3 scripts/prepare_dataset.py`.
3. The script normalizes folder names to YOLOX conventions:

```text
datasets/prepared/formulas_coco_v1/
  annotations/
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json   # only if test exists
  train2017/
  val2017/
  test2017/
```

`prepare_dataset.py` does not invent a split. It only normalizes the Roboflow export structure, renames `valid` to `val2017`, moves annotations into `annotations/`, rewrites bbox values to stable numeric COCO JSON, and keeps a stable category list across splits.

## Training environment

Create an isolated environment:

```bash
bash scripts/setup_train_env.sh cpu
```

For a CUDA machine use one of:

```bash
bash scripts/setup_train_env.sh cu121
bash scripts/setup_train_env.sh cu124
```

Then activate it:

```bash
source .venvs/formulalens-train-cu124/bin/activate
```

## Pretrained weights

Download the official YOLOX-Nano pretrained checkpoint used for fine-tuning:

```bash
bash scripts/download_weights.sh
```

The default target is `weights/pretrained/yolox_nano.pth`.

## Train

The main exp file is `configs/train/yolox_nano.py`. It defines:

- number of classes
- dataset root
- model size
- number of epochs
- input size
- augmentations
- dataloader and evaluator settings

Run training with:

```bash
bash scripts/train.sh --fp16
```

Checkpoints are written under `weights/finetuned/yolox_nano/`.

## Export ONNX

After training, export the best checkpoint into a versioned ONNX artifact:

```bash
bash scripts/export_onnx.sh
```

By default this writes:

```text
weights/finetuned/v1/formulalens_yolox_nano_v1.0.0.onnx
```

Model versioning follows a simple layout:

```text
weights/finetuned/
  v1/
    formulalens_yolox_nano_v1.0.0.onnx
  v2/
    formulalens_yolox_nano_v2.0.0.onnx
```

The service runtime currently uses:

- `FORMULALENS_MODEL_VERSION=v1.0.0`

## CPU benchmark

Benchmark the ONNX Runtime path on local images:

```bash
python scripts/benchmark_cpu.py --num-images 24
```

Current benchmark on this machine for `v1.0.0` ONNX over `24` validation images:

- mean: `22.967 ms`
- median: `18.952 ms`
- p95: `34.455 ms`
- min: `14.691 ms`
- max: `54.943 ms`

This benchmark uses `onnxruntime` and includes preprocessing + model inference + service-side postprocess.

## Service runtime

The service runtime is split into small, reliable modules:

- `src/formulalens/inference.py`: model loading, preprocessing, raw detections; supports both `.pth` and `.onnx`
- `src/formulalens/postprocess.py`: thresholding, cleanup, NMS, sorting
- `src/formulalens/confidence.py`: `global_confidence` plus geometry / combination penalties
- `src/formulalens/crops.py`: crops for `numerator`, `denominator`, `system_row`, `block`, `text`
- `src/formulalens/routing.py`: `use_pix2tex | use_formula_lens | use_heuristics`
- `src/formulalens/schemas.py`: pydantic contracts
- `src/formulalens/service.py`: FastAPI endpoints

The service prefers the versioned ONNX model from `configs/service/settings.yaml`. If the ONNX file is missing, it can fall back to the PyTorch checkpoint for local development.

## Run service locally

```bash
source .venvs/formulalens-train-cu124/bin/activate
bash scripts/start_service.sh
```

## Model download for deployment

For deployment the service model should be downloaded at runtime, not baked into the image.

Download a service ONNX artifact into the versioned cache directory:

```bash
FORMULALENS_MODEL_URL="https://github.com/LullNil/formula-lens/releases/download/v1.0.0/formulalens_yolox_nano_v1.0.0.onnx" \
bash scripts/download_weights.sh service
```

Alternative form using only the release repository:

```bash
FORMULALENS_RELEASE_REPO="LullNil/formula-lens" FORMULALENS_MODEL_VERSION="v1.0.0" \
bash scripts/download_weights.sh service
```

If the target file already exists, the script exits immediately without downloading again.

## Docker deployment

The service container is intentionally minimal:

- base image: `python:3.12-slim`
- no training stack
- no model weights copied into the image
- startup script downloads the `v1.0.0` ONNX only if the cached file is missing

Files involved:

- `docker/Dockerfile.service`
- `docker-compose.yml`
- `scripts/start_service.sh`

### First start

Set one of these:

- `FORMULALENS_MODEL_URL`
- `FORMULALENS_RELEASE_REPO`

Then run:

```bash
docker compose up --build
```

By default the service is published on host port `18080`, so the API is available at `http://localhost:18080`.
If you need a different host port, override it with `FORMULALENS_HOST_PORT`, for example:

```bash
FORMULALENS_HOST_PORT=18081 docker compose up --build
```

On the first start:

- the container is built from code only
- the model is downloaded into the mounted weight cache
- FastAPI starts

On the second start:

- the cached `v1.0.0` ONNX file is already present
- the service starts immediately without re-downloading the model

The compose file keeps weights and bad cases in named volumes:

- `formulalens_weights`
- `formulalens_bad_cases`

## FastAPI contract

### `POST /detect`

Input: image upload

Response:

```json
{
  "ok": true,
  "detections": [
    {
      "class_id": 3,
      "class": "numerator",
      "score": 0.93,
      "bbox": [12.0, 8.0, 64.0, 31.0]
    }
  ],
  "global_confidence": 0.91,
  "model_version": "v1",
  "confidence_breakdown": {
    "global_confidence": 0.91,
    "base_score": 0.94,
    "geometry_penalty": 0.01,
    "combination_penalty": 0.02
  }
}
```

### `POST /route`

Input:

- image upload
- `pix2tex_output`
- `pix2tex_score`

Response:

```json
{
  "ok": true,
  "decision": "use_formula_lens",
  "reason": "Structural detections are confident enough to override pix2tex.",
  "detections": [],
  "global_confidence": 0.87,
  "model_version": "v1",
  "confidence_breakdown": {
    "global_confidence": 0.87,
    "base_score": 0.9,
    "geometry_penalty": 0.01,
    "combination_penalty": 0.02
  }
}
```

### `POST /debug/save-bad-case`

Input:

- image upload
- `intermediate_outputs` as JSON string
- `reason`

Response:

```json
{
  "ok": true,
  "case_id": "8f7f0f3d8c0d4b18a8716c8b8f6d7d9a",
  "saved_dir": "/app/experiments/bad_cases/8f7f0f3d8c0d4b18a8716c8b8f6d7d9a"
}
```

### `POST /debug/detect`

Input: image upload

Response: `image/jpeg` with thin color-coded bounding boxes for fast visual inspection. Labels are intentionally not drawn on the image so they do not overlap the formula itself.

Color legend for `/debug/detect`:

- `block`: ochre
- `denominator`: green
- `exponent`: bright blue
- `numerator`: red
- `system_row`: rose
- `text`: cyan

## Deployment summary

The intended production flow is:

```text
Docker container
  -> downloads model if missing
  -> caches the ONNX artifact locally
  -> starts FastAPI
  -> serves requests on CPU through onnxruntime
```

Updating the model should only require:

1. train a new model
2. export a new versioned ONNX
3. upload it to GitHub Releases
4. change `MODEL_VERSION` or the model URL

The container itself does not need to be rebuilt for a weight-only update.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
