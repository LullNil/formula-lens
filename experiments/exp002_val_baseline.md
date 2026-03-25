# exp002_val_baseline

## Summary

- Date: 2026-03-26
- Model: YOLOX-Nano
- Exp file: `configs/train/yolox_nano.py`
- GPU: NVIDIA GeForce RTX 4060, 8 GB VRAM
- Environment: `.venvs/formulalens-train-cu124`
- Pretrained checkpoint: `weights/pretrained/yolox_nano.pth`
- Fine-tuned artifacts: `weights/finetuned/yolox_nano/`

## Run Command

```bash
source .venvs/formulalens-train-cu124/bin/activate
BATCH_SIZE=8 bash scripts/train.sh --fp16
```

Training started at `2026-03-26 00:24:28` and finished at `2026-03-26 00:46:37`.

## Dataset

- Prepared dataset root: `datasets/prepared/formulas_coco_v1`
- Train images: `486`
- Train annotations: `4188`
- Val images: `40`
- Val annotations: `339`
- Classes: `block`, `denominator`, `exponent`, `numerator`, `system_row`, `text`

## Validation Note

This run uses a real `val2017` split, so the AP values are now held-out validation metrics.

The current `valid` annotations contain `0` instances for class `text`.
Because of that, COCO reports `nan` for `text` AP/AR, which is expected for this export and should be fixed in the dataset rather than in code.

## Main Result

- `best_ckpt.pth`: `weights/finetuned/yolox_nano/best_ckpt.pth`
- Best AP50:95: `0.763084` (`76.31%`)
- Best checkpoint metadata: `start_epoch=130`
- Final epoch AP50:95: `0.758`
- Final epoch AP50: `0.981`
- Final epoch AP75: `0.884`

## Final Epoch Per-Class AP

| Class | AP |
| --- | ---: |
| block | 67.281 |
| denominator | 81.265 |
| exponent | 66.634 |
| numerator | 81.472 |
| system_row | 82.238 |
| text | nan |

## Produced Artifacts

- `weights/finetuned/yolox_nano/best_ckpt.pth`
- `weights/finetuned/yolox_nano/last_epoch_ckpt.pth`
- `weights/finetuned/yolox_nano/latest_ckpt.pth`
- `weights/finetuned/yolox_nano/train_log.txt`
- `weights/finetuned/yolox_nano/tensorboard/`

## Notes

- The previous train-only baseline is preserved in `weights/finetuned/yolox_nano_exp001_train_only/`.
- `prepare_dataset.py` was updated to keep a stable class list across splits, even when a class has zero annotations in `val`.
