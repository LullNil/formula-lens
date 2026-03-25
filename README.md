# FormulaLens

FormulaLens is a structure detector for formulas. It detects structural regions such as `block`, `numerator` and `denominator`, but it does not perform OCR itself. Text reconstruction is expected to happen downstream, for example with `pix2tex`.

## Detection classes

- `block`
- `denominator`
- `exponent`
- `numerator`
- `system_row`
- `text`

## Repository layout

- `configs/train/` contains YOLOX exp files. The main training config is `configs/train/yolox_nano.py`.
- `datasets/roboflow_exports/` stores the raw Roboflow export.
- `datasets/prepared/` stores the normalized COCO layout expected by YOLOX.
- `weights/pretrained/` stores the starting YOLOX weights.
- `weights/finetuned/` stores fine-tuned checkpoints and the exported `formula_lens_v1.onnx`.
- `scripts/` contains dataset preparation, training, export and benchmarking helpers.
- `src/formulalens/` is reserved for the service and inference layer, separate from training.

## Data pipeline

The first working pipeline is:

1. Unpack a Roboflow COCO export into `datasets/roboflow_exports/formulas_coco_v1/`.
2. Run `python3 scripts/prepare_dataset.py`.
3. The script normalizes folder names to YOLOX conventions:

```text
datasets/prepared/formulas_coco_v1/
  annotations/
    instances_train2017.json
    instances_val2017.json   # only if valid/val exists in the Roboflow export
    instances_test2017.json  # only if test exists in the Roboflow export
  train2017/
  val2017/
  test2017/
```

`prepare_dataset.py` does not invent a split. It only normalizes the Roboflow export structure, renames `valid` to `val2017`, moves annotations into `annotations/`, rewrites bbox values to stable numeric COCO JSON, and drops unused categories that would break YOLOX class indexing.

## Training environment

YOLOX training itself requires CUDA. A CPU profile is still useful for export tooling, ONNXRuntime-based service work and local development, but `scripts/train.sh` will stop early if CUDA is not available.

Create an isolated environment:

```bash
bash scripts/setup_train_env.sh cpu
```

For a CUDA machine use one of:

```bash
bash scripts/setup_train_env.sh cu121
bash scripts/setup_train_env.sh cu124
```

Then activate the environment:

```bash
source .venvs/formulalens-train-cpu/bin/activate
```

Adjust the suffix if you created a CUDA profile instead of `cpu`.

## Pretrained weights

Download YOLOX-Nano pretrained weights:

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
bash scripts/train.sh
```

The script:

1. normalizes the Roboflow export into `datasets/prepared/formulas_coco_v1`
2. checks that CUDA is available
3. starts YOLOX training from `weights/pretrained/yolox_nano.pth`

Checkpoints are written under `weights/finetuned/yolox_nano/`.

## Export ONNX

After training, export the best checkpoint to ONNX:

```bash
bash scripts/export_onnx.sh
```

By default this reads `weights/finetuned/yolox_nano/best_ckpt.pth` and writes `weights/finetuned/formula_lens_v1.onnx`.

## CPU inference service

The service layer belongs in `src/formulalens/`. At the moment this part of the repository is still scaffold-only, but the intended CPU serving path is:

1. export `weights/finetuned/formula_lens_v1.onnx`
2. configure the model path in `configs/service/settings.yaml`
3. start the FastAPI service from `src/formulalens/service.py`

Target command:

```bash
uvicorn src.formulalens.service:app --host 0.0.0.0 --port 8000
```

## JSON API

The intended API contract for the CPU service is:

```json
{
  "image_base64": "<base64-encoded-image>",
  "return_crops": false
}
```

Response shape:

```json
{
  "detections": [
    {
      "label": "numerator",
      "score": 0.97,
      "bbox": [12, 8, 64, 31]
    }
  ],
  "global_confidence": 0.93,
  "ocr": null
}
```

`ocr` is intentionally separate because FormulaLens only detects structure. Formula-to-text reconstruction should be performed downstream by `pix2tex` or another OCR model.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
