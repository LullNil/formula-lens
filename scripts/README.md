# Scripts

Reference for the helper scripts in this repository.

## `convert_annotations.py`

Purpose: runs the current FormulaLens inference flow over a directory of images and exports a pseudo-labeled dataset.

Why: use it to generate editable predictions for manual cleanup and later fine-tuning.

Run:

```bash
python3 scripts/convert_annotations.py \
  --input-dir path/to/source_images \
  --output-dir path/to/pseudo_labels
```

Run with Docker:

```bash
docker compose run --rm \
  -v /home/memoire/formula-lens/datasets:/app/datasets \
  formulalens \
  bash /app/scripts/run_convert_annotations.sh \
  --input-dir /app/datasets/raw/6 \
  --output-dir /app/datasets/pseudo_labels
```

Output:

- `pseudo_labels/<source_dir_name>/...`: exported dataset root
- `images/...`: exported images only
- `labels/...`: YOLO `.txt` labels only
- transparent inputs are saved on a white background
- `classes.txt`: class list by index
- `data.yaml`: dataset metadata pointing to the image folders
- `summary.json`: processing summary, including total and average runtime

## `prepare_dataset.py`

Purpose: converts a Roboflow COCO export into the folder layout expected by YOLOX.

Why: use it before training to normalize split names and annotation files.

Run:

```bash
python3 scripts/prepare_dataset.py \
  --source-root datasets/roboflow_exports/formulas_coco_v1 \
  --output-dir datasets/prepared/formulas_coco_v1
```

## `benchmark_cpu.py`

Purpose: measures CPU inference latency for the current ONNX model on a set of images.

Why: use it to estimate average runtime and compare inference performance between model versions.

Run:

```bash
python3 scripts/benchmark_cpu.py \
  --image-dir datasets/prepared/formulas_coco_v1/val2017 \
  --num-images 24
```

## `setup_train_env.sh`

Purpose: creates a virtual environment for training and installs dependencies for `cpu`, `cu121`, or `cu124`.

Why: use it to prepare a clean training/export environment.

Run:

```bash
bash scripts/setup_train_env.sh cu124
```

## `download_weights.sh`

Purpose: downloads pretrained YOLOX weights or the versioned ONNX model used by the inference service.

Why: use it when required model artifacts are missing locally.

Run:

```bash
bash scripts/download_weights.sh pretrained
bash scripts/download_weights.sh service
```

## `train.sh`

Purpose: prepares the dataset and launches YOLOX training.

Why: use it as the main entrypoint for model fine-tuning.

Run:

```bash
bash scripts/train.sh --fp16
```

## `export_onnx.sh`

Purpose: exports a fine-tuned checkpoint to ONNX.

Why: use it to produce the model artifact required by the CPU inference service.

Run:

```bash
bash scripts/export_onnx.sh
```

Useful overrides:

```bash
MODEL_VERSION=v2.0.0-batch ONNX_BATCH_SIZE=8 ONNX_DYNAMIC_BATCH=1 ONNX_NO_SIMPLIFY=1 bash scripts/export_onnx.sh
```

## `start_service.sh`

Purpose: downloads the configured ONNX model if needed and starts the FastAPI/uvicorn service.

Why: use it to run the local inference API.

Run:

```bash
bash scripts/start_service.sh
```

## `collect_bad_cases.py`

Purpose: not implemented yet.

Why: reserved for future bad-case collection tooling.

Run: not available.

## `split_dataset.py`

Purpose: not implemented yet.

Why: reserved for future dataset split tooling.

Run: not available.
