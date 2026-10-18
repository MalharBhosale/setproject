# Experiment log

Every reported number must trace to an entry here. Large artefacts stay in `data/runs/` (git-ignored); this file records what is needed to identify and reproduce them.

## kaggle_v1: symbol detector, first training run (2026-10-17/18)

| Item | Value |
|---|---|
| Local folder | `data/runs/Kaggle1/` (duplicate copies and auto-downloaded base weights removed 2026-10-18; nothing unique lost) |
| Code | `training/train_detector.py` at commit with message "Add Kaggle detector training package" |
| Data | `data/kaggle/pid_tiles.zip`, sha256 `f41f6fa2100fc2fbc18c4418345e2804c8eb37a2f626f13d2059670a9cb62540` (18,900 train and 4,050 val tiles; no test sheets) |
| Platform | Kaggle, Tesla T4, Python 3.13.15, torch 2.11.0+cu128, Ultralytics 8.4.175 |
| Model | `yolo11n.pt` (COCO-pretrained YOLO11 nano), imgsz 1024, batch 16, seed 0, deterministic, AMP on |
| Stopping | Time limit of 10 h reached: **87 epochs in 10.01 h** (about 414 s per epoch). With `time` set, Ultralytics sizes the run to the time budget and ignores `epochs=50`; this was not anticipated when the script was written. |
| Weights | `runs/detector/weights/best.pt`, sha256 `9e7456951193029c874d9b320c0b3b1f11848acc5a7d07a55f01b623891a2d99` |
| Best epoch | 85 |

Tile-level validation (Ultralytics, 4,050 validation tiles, from `val_metrics.json`): precision 0.997, recall 0.994, mAP50 0.995, mAP50-95 0.990. Learning curve (mAP50-95): epoch 1 0.873, epoch 10 0.970, epoch 30 0.984, epoch 85 0.989.

Caveats:
- Tile-level metrics count a symbol once per tile that contains its centre; overlapping tiles repeat some symbols. Sheet-level evaluation is the reported measure (see below).
- The data is synthetic and clean; these scores say nothing about real drawings.
- Log warnings: slow reads from the Kaggle input mount and an unwritable cache directory (speed only, not correctness).

### Sheet-level evaluation on the 75 validation sheets (2026-10-18)

`python src/detect_sheets.py data/runs/Kaggle1/runs/detector/weights/best.pt data/runs/Kaggle1 val`. Each sheet is tiled as in training; boxes touching an inner tile edge are dropped (the symbol is whole in a neighbouring tile); duplicates are merged per class (NMS, IoU 0.5); predictions are matched to ground truth at IoU >= 0.5 and score >= 0.25. Raw predictions (score >= 0.01) are in `data/runs/Kaggle1/predictions_val/`; numbers in `sheet_eval_val.json`. CPU inference took about 20 s per sheet.

| Symbols | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| 8,913 | 8,912 | 3 | 1 | 0.9997 | 0.9999 |

All four errors, inspected on the drawings:
- Sheet 317: a class-17 symbol drawn rotated 90 degrees was labelled class 19 (1 FP + 1 FN).
- Sheet 138: two overlapping symbols produced a second class-19 box for one symbol (1 FP).
- Sheet 154: a class-7 symbol also received a class-10 box (1 FP). The merge step suppresses duplicates within a class only.

Interpretation: on this synthetic data the detector is near the ceiling, so detection errors alone will rarely cause false verdicts. Tag reading (OCR) is the step expected to drive REVIEW and false FAIL rates. The test sheets have not been touched.
