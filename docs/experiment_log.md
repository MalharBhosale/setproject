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

## ocr_v1: OCR engine comparison (2026-10-18)

`python src/eval_ocr.py data/runs/ocr_v1 easyocr rapidocr`. Task: read the tag words of the 23 identifier classes. Each ground-truth word box is cropped and recognised alone, so this measures reading, not finding, the text. Tall boxes are tried rotated both ways and the more confident reading kept (the dataset's rotation label misses most vertical text; see dataset card). Crop margin tuned per engine on validation sheets 3, 5, 16, 37, 38 (EasyOCR 4 px, RapidOCR 24 px); results below are on the other 70 validation sheets. CPU only.

Engines: EasyOCR 1.7.2 (Apache-2.0; PyTorch). RapidOCR 1.2.3 via onnxruntime 1.31.0 (Apache-2.0; runs PaddleOCR models). PaddleOCR itself cannot be installed: PaddlePaddle has no build for Python 3.14.

| Engine | Tags | Tag exact | Tag, ignoring case and spaces | Word exact | Character error rate | Vertical words exact | ms per word |
|---|---|---|---|---|---|---|---|
| EasyOCR | 5,654 | 0.808 | 0.835 | 0.844 | 0.045 | 0.852 (1,769 / 2,077) | 63 |
| RapidOCR | 5,654 | 0.824 | 0.825 | 0.835 | 0.074 | 0.971 (2,016 / 2,077) | 500 |

Error character (wrong words): EasyOCR 1,142, mostly same-length character confusions (762; e.g. `IJ-79437` read `1J-79437`) and case (154). RapidOCR 1,212, mostly added or lost characters (925; e.g. `QR-57978,`, `LG-` for `LG-10`) and empty output (149), plausibly from neighbouring strokes inside its wider margin.

Confidence separates right from wrong for both: mean confidence on correct vs wrong words, EasyOCR 0.897 vs 0.600, RapidOCR 0.806 vs 0.586. This supports using OCR confidence in the REVIEW decision (H3), pending calibration.

Choice: EasyOCR. Tag accuracy is within 1.6 points of RapidOCR, it is 8 times faster on CPU, and its errors are systematic confusions rather than truncation. Untested improvements: restricting EasyOCR to the characters tags use, and better handling of vertical text, where it trails RapidOCR.

Interpretation: about 19% of tags are misread even with perfect text boxes, against near-perfect detection. If H1 holds, OCR is where false FAIL verdicts will come from.

## pipeline_v1: end-to-end extraction and verification, no abstention (2026-10-18)

`python src/pipeline.py data/runs/Kaggle1/predictions_val data/runs/pipeline_v1 val`. Detector boxes (kaggle_v1, score >= 0.25) plus EasyOCR: text is found with EasyOCR's detector in a region around each symbol (bubble interior trimmed 12% for classes 26-32; 80 px around the symbol otherwise), boxes on one line are joined, each piece is read with the Step 4.7 settings (4 px margin, character allowlist, vertical text turned clockwise), and each piece of text is assigned to exactly one symbol. Design choices came from training-sheet statistics and from validation sheets 3 and 5 (tuning sheets). About 27 s per sheet on CPU.

**Tags, 75 validation sheets** (all classes that carry text): 4,888 exact of 7,875 (**62.1%**); 2,986 wrong; 1 symbol not detected. For comparison, reading perfectly located words gave 89.0% on the tuning sheets (Step 4.7 with the allowlist), so finding and assigning text costs roughly 25 points.

**Verification of the 300 validation cases with the existing deterministic verifier:**

| True verdict | Reported PASS | Reported FAIL |
|---|---|---|
| PASS (75 matching pairs) | 0 | **75** |
| FAIL (225 seeded pairs) | 0 | 225 |

**False FAIL rate on matching pairs: 100% (75 / 75).** Every clean drawing-BOM pair is declared non-conforming.

| Finding type | True found | False reported | Missed |
|---|---|---|---|
| MISSING | 142 | 5,588 | 1 |
| EXTRA | 62 | 5,626 | 35 |
| TAG | 104 | 2,096 | 41 |
| TYPE | 89 | 15 | 51 |
| QUANTITY | 61 | 7,306 | 89 |

About 20,600 false findings over 300 cases (about 69 per case) against 458 true ones. A tag misread by more than 2 characters becomes a MISSING plus an EXTRA; a misread size or code makes the counts of two (class, text) groups disagree, hence the QUANTITY flood.

Interpretation: **H1 is supported on validation data**: extraction errors, almost all from OCR, turn every matching pair into a FAIL. The FAIL verdict without abstention is uninformative here (it is FAIL for every case). This motivates the confidence-aware REVIEW outcome (H2) and shows that end-to-end tag reading is the main weakness to report. Not yet measured on the test sheets.

## decide_v1: confidence-aware PASS / REVIEW / FAIL on validation cases (2026-10-18)

`python src/decide.py data/runs/pipeline_v1/extracted_val data/runs/decide_v1 val`. A component is reliable when its detector score is >= 0.5 and, for text-bearing classes, its OCR confidence is >= tau (rule B also requires a plausible format, below). A finding is supported when no unreliable component could explain it (EXTRA/TAG/TYPE: the component itself; MISSING: no unsupported EXTRA/TAG/TYPE of the same class; QUANTITY: no unreliable component of that class). Verdict: FAIL if any finding is supported, else REVIEW if any finding exists, else PASS. tau chosen by a rule fixed before looking at results: highest coverage with risk (wrong automatic decisions) <= 5%.

**Rule A, OCR confidence only (planned rule): fails.** For tau up to 0.98 all 300 cases are decided and 25% are wrong (all 75 matching pairs FAIL); only at tau 1.0 (nothing text-bearing trusted) does risk reach 0%, at 23.7% coverage. Cause: every matching pair contains confident misreadings (at least 3 supported findings per pair at tau 0.9; 585 wrong tags have confidence 0.9-1.0, next to 3,579 right ones). Typical: truncation (`KL-80-` at 0.987), I read as 1 (`1J-74522` at 0.923), stray characters (`SDL 906)` at 0.908).

**Rule B, confidence + format (revised after seeing rule A's failure on validation; test sheets untouched):** a reading must also match a shape seen for its class on the training sheets (letters -> A, digits -> 9, other characters kept; the letters themselves are not used, so the prefix cannot reveal the class).

| tau | Coverage | Risk | PASS->FAIL | PASS->REVIEW | FAIL->FAIL | FAIL->REVIEW |
|---|---|---|---|---|---|---|
| 0.00 | 0.893 | 0.198 | 53 | 22 | 215 | 10 |
| 0.90 | 0.623 | 0.086 | 16 | 59 | 171 | 54 |
| 0.94 | 0.577 | 0.064 | 11 | 64 | 162 | 63 |
| 0.96 | 0.543 | 0.055 | 9 | 66 | 154 | 71 |
| **0.98 (chosen)** | **0.503** | **0.040** | 6 | 69 | 145 | 80 |
| 1.00 | 0.237 | 0.000 | 0 | 75 | 71 | 154 |

At the chosen tau, supported findings have precision 0.796 and recall 0.319 against the seeded discrepancies. Risk-coverage curves: `data/runs/decide_v1/risk_coverage_val_*.png`.

Interpretation:
- Against the no-abstention baseline (100% coverage, 25% risk), rule B decides half of the cases automatically with 4% of those decisions wrong. Rule A cannot get under 5% risk except by trusting no text at all. On validation data this supports H2 for rule B, not for rule A.
- **No matching pair is ever passed automatically** (PASS->PASS is 0 at every tau). All automatic decisions are FAILs; certifying a drawing as conforming still needs a human. This limits the practical claim and must be stated.
- tau and the format rule were chosen on these same validation cases, so these figures are optimistic. They must be confirmed once on the test sheets with tau frozen at 0.98.

## FROZEN CONFIGURATION for the one-time test run (recorded 2026-10-18, before any test sheet was processed)

Up to this entry no test sheet has been detected, OCR-read or verified; the detector never saw test tiles. Everything below is fixed and will not be changed after seeing test results.

| Component | Frozen setting |
|---|---|
| Detector | `data/runs/Kaggle1/runs/detector/weights/best.pt`, sha256 `9e7456951193029c874d9b320c0b3b1f11848acc5a7d07a55f01b623891a2d99` |
| Sheet detection | `src/detect_sheets.py`: tile 1024, overlap 256, edge drop 2 px, class-wise NMS 0.5, boxes kept at score >= 0.01 |
| Extraction | `src/pipeline.py`: detector score >= 0.25; MARGIN 80; INSET 0.12; READ_PAD 4; JOIN 0.6; VERTICAL_TURN 3 (clockwise); EasyOCR 1.7.2 English with the allowlist in `ALLOW` |
| Decision | `src/decide.py`: DET_MIN 0.5; rule B (OCR confidence + format shapes learned from the 350 training sheets); **tau = 0.98** |
| Benchmark | `docs/benchmark_manifest.json` test part: 300 cases on 75 sheets (75 PASS, 225 FAIL) |

Commands, run once:
1. `python src/detect_sheets.py data/runs/Kaggle1/runs/detector/weights/best.pt data/runs/Kaggle1 test --allow-test`
2. `python src/pipeline.py data/runs/Kaggle1/predictions_test data/runs/pipeline_v1 test --allow-test`
3. `python src/decide.py data/runs/pipeline_v1/extracted_test data/runs/decide_v1 test --frozen-tau=0.98`

Headline numbers to report: sheet-level detection precision and recall; end-to-end tag exact rate; false FAIL rate without abstention; coverage and risk at tau = 0.98 for rule B; rule A at tau = 0.98 for comparison. The full test risk-coverage curve is descriptive only.

H3 (calibration) was not tested before this run. Any H3 analysis afterwards is post hoc and cannot change these settings.

## TEST RESULTS: one-time run with the frozen configuration (2026-10-18)

Run after the freeze commit `16db1d2` (committed 2026-10-18 21:48:01 +0530), with the three commands listed there and no change to any setting. Intervals are 95% bootstrap intervals over the 75 test sheets (`python src/bootstrap_ci.py test 0.98`; 2,000 resamples, seed 0).

| Measure | Test | Validation (for comparison) |
|---|---|---|
| Detection, sheet level | precision 0.9999, recall 1.0000 (9,105 TP, 1 FP, 0 FN) | 0.9997 / 0.9999 |
| End-to-end tag exact | 61.6% (4,924 of 7,991) | 62.1% |
| No abstention: false FAIL rate on matching pairs | **100% (75 / 75)** | 100% |
| No abstention: coverage / risk | 100% / 25.0% | 100% / 25.0% |
| Rule A (confidence only) at tau 0.98: coverage / risk | 96.0% / 24.0% | n/a |
| **Rule B (confidence + format) at frozen tau 0.98: coverage** | **52.3% [46.3%, 58.3%]** | 50.3% [45.0%, 56.0%] |
| **Rule B at frozen tau 0.98: risk** | **6.4% [3.1%, 9.6%]** | 4.0% [1.4%, 6.9%] |
| Rule B verdicts | PASS->FAIL 10, PASS->REVIEW 65, FAIL->FAIL 147, FAIL->REVIEW 78 | 6 / 69 / 145 / 80 |
| Rule B supported findings | precision 0.728, recall 0.301 | 0.796 / 0.319 |

Findings without abstention (test): MISSING 119 true / 5,714 false; EXTRA 98 / 5,764 (50 missed); TAG 98 / 2,489 (44 missed); TYPE 79 / 20 (39 missed); QUANTITY 62 / 6,991 (86 missed).

Reading the results honestly:
- **H1 supported on test:** without abstention every matching pair is declared FAIL.
- **H2 supported on test for rule B, with a caveat:** risk falls from 25.0% to 6.4% while half the cases are still decided automatically. Rule A barely helps (24.0%). But the **5% target chosen on validation is not met on test** (6.4%, interval 3.1% to 9.6%). The validation figure of 4.0% was optimistic, as expected from choosing tau on it.
- **No matching pair is ever passed automatically** on test either: 65 of 75 go to REVIEW and 10 are wrongly failed. Every automatic decision is a FAIL.
- Coverage is close between validation and test, so the trade-off generalises; the risk estimate is uncertain with 75 sheets.
- The test sheets are now used. Nothing may be tuned on them; any further change needs a new held-out set, or must be reported as post hoc.
