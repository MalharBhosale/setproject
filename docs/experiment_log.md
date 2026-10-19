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

## graph_v1: can graph reasoning help this task? (validation, 2026-10-18; post hoc relative to the test run)

The benchmark specification carries no connectivity (the BOM lists components and tags, not connections) and no connectivity discrepancy is seeded, so a with-graph versus without-graph verification comparison has nothing to differ on. The one place connectivity could act in the current pipeline is association (which text belongs to which symbol). `python src/attribute_errors.py data/runs/pipeline_v1/extracted_val val` attributes every wrong end-to-end tag on the 75 validation sheets:

| Cause | Wrong tags | Share of errors |
|---|---|---|
| Misread (own text read wrongly) | 2,214 | 74.1% |
| No text assigned | 441 | 14.8% |
| Association (reading within 2 edits of another text on the sheet, and closer to it than to the true tag) | 331 | 11.1% |

Only 2 association errors involved a pipe's line label. Association errors concentrate in the size classes 19, 17, 18 and 20 (66, 39, 19, 17), whose texts repeat on a sheet, so some of them are misreads that happen to equal another size. 11.1% is therefore an upper bound.

Conclusion: even perfect connectivity could fix at most about 11% of tag errors here; the bottleneck is reading. A graph-reasoning ablation is not justified for the task as defined and is deferred. Graph reasoning becomes testable only with a connectivity-bearing specification (for example a line list) and connectivity discrepancies; PID2Graph's aligned graphs for all 500 sheets (`docs/pid2graph_graph_to_sheet.json`) would support building that benchmark extension.

## open100_v1: qualitative real-drawing check (2026-10-18; post hoc)

`python src/eval_open100.py data/runs/Kaggle1/runs/detector/weights/best.pt data/runs/open100_v1`. The kaggle_v1 detector, trained only on synthetic sheets, run on the 12 real OPEN100 P&IDs from PID2Graph (CC BY-SA 4.0; licence of the original drawings unconfirmed, so no images are committed). Our 32 classes are mapped to PID2Graph's coarse labels (mapping from docs/dataset_card.md). Matching at IoU >= 0.5, score >= 0.25. OPEN100 symbols are about 4 times smaller than ours (median longest side 22 px vs 88 px), so the plans were run at native size and upscaled 4x. The 4x factor was chosen from the plans' annotated box sizes, which is an advantage a deployed system would not have without knowing the drawing scale.

| Label | Symbols | Recall, native | Recall, 4x | Precision, 4x |
|---|---|---|---|---|
| instrumentation | 350 | 0.87 | 1.00 | 0.88 |
| arrow | 451 | 0.20 | 0.82 | 0.83 |
| valve | 257 | 0.47 | 0.83 | 0.64 |
| general | 234 | 0.11 | 0.62 | 0.35 |
| tank / pump / inlet-outlet | 35 / 16 / 96 | 0 | 0 | n/a (classes absent from training data) |

Over the four representable labels at 4x: 1,078 of 1,292 symbols found with the right coarse label (recall 0.834), 516 false detections (precision 0.676). Per-plan predictions are in `data/runs/open100_v1/`.

Visual inspection of plan 0: most valves, instrument bubbles and arrows are found; misses are equipment boxes and half-circle symbols unlike anything in the synthetic data; some false detections are label-mapping disagreements (a boxed `T` mapped to instrumentation is annotated `general`).

Interpretation: the synthetic-trained detector transfers partly to real drawings, and only at a matched scale; accuracy falls from 0.9999 (synthetic test) to about 0.83 recall and 0.68 precision on coarse labels. 147 of 1,439 symbols (10%) belong to classes the training data never had. Tag reading and verification could not be tested on these plans (no text annotation), so every verification result in this project remains a synthetic-data result. Twelve plans support a qualitative statement only.

## selective_v1: AURC, format-only ablation, paired comparison and review workload (2026-10-19; post hoc on the frozen outputs)

`python src/selective_metrics.py data/runs/pipeline_v1/extracted_<part> <part>`; no setting changed. Each case is scored by the highest tau at which a rule still decides it automatically, giving an exact risk-coverage ranking; AURC is the mean risk over all coverage levels. Format-only is rule B with no confidence threshold. Paired intervals: 2,000 sheet-level bootstrap resamples, seed 0.

| Rule | Val coverage / risk at tau 0.98 | Val AURC | Test coverage / risk at tau 0.98 | Test AURC |
|---|---|---|---|---|
| A: confidence only | 98.0% / 24.8% | 0.139 | 96.0% / 24.0% | 0.154 |
| Format only (no threshold) | 89.3% / 19.8% | n/a | 94.7% / 23.2% | n/a |
| **B: confidence + format** | 50.3% / 4.0% | **0.077** | 52.3% / 6.4% | **0.090** |

Paired, A minus B: test AURC difference 0.064 [0.051, 0.080]; test risk difference at tau 0.98, 17.6 points [14.5, 20.9]. Validation: 0.062 [0.050, 0.074] and 20.9 points [17.9, 23.5].

Neither signal alone makes automatic decisions trustworthy; only the combination does.

**Review workload (rule B, tau 0.98):** in REVIEW cases the median share of untrusted components is 56.7% on test (57.3% on validation), with a median of 66 flagged findings. The rule routes cases correctly but leaves a human most of each REVIEW drawing to check. **No matching pair is passed automatically under any rule** (0 of 75 on both splits). Both limits trace to end-to-end tag reading (61.6%).

## PLAN v2: domain-trained text reader (recorded 2026-10-19, before the Kaggle-trained reader has been evaluated anywhere)

Motivation: end-to-end tag reading (61.6% on test, 74% of wrong tags misreadings) caps both coverage and auto-PASS. v2 changes one component only: EasyOCR's recognition step is replaced by the CRNN in `src/train_reader.py`, trained on word crops from the 350 training sheets. Text finding (EasyOCR's detector), region rules, assignment, verifier and decision rule stay exactly as frozen for v1.

Fixed in advance:
1. **Reader selection.** The weights are the epoch with the best exact-word accuracy on validation-sheet word crops, as the training script saves them (12 epochs, seed 0). No other reader variant is tried. The file is copied to `data/reader/crnn.pt` and its sha256 recorded in the v2 freeze entry.
2. **Validation run.** `python src/pipeline.py data/runs/Kaggle1/predictions_val data/runs/pipeline_v2 val --reader=crnn`, then `python src/decide.py data/runs/pipeline_v2/extracted_val data/runs/decide_v2 val`.
3. **Threshold.** tau is chosen on validation by the unchanged rule in `decide.choose`: highest coverage with risk <= 5% (rule B). The CRNN's confidence (product of character probabilities) is on a different scale from EasyOCR's, so tau is re-chosen; the rule for choosing it is not changed.
4. **Freeze, then test once.** A v2 FROZEN entry is committed before the test run. Commands: pipeline with `--reader=crnn --allow-test`, decide with `--frozen-tau=<chosen>`, then `python src/bootstrap_ci.py test <tau> data/runs/pipeline_v2` and `python src/selective_metrics.py data/runs/pipeline_v2/extracted_test test <tau>`.
5. **Disclosure.** The test sheets were already used once for v1. The v2 test run is a second, disclosed look. Both v1 and v2 test results will be reported whatever v2 shows, and nothing is tuned after the v2 test run.

Headline numbers for v2: end-to-end tag exact rate; rule B coverage and risk at the frozen tau with sheet-level intervals; AURC; auto-PASS rate on matching pairs; review workload.

## reader_v1: domain-trained CRNN text reader (Kaggle, 2026-10-19)

Run: `training/kaggle_train_reader.ipynb` on a Tesla T4 (Python 3.13.15, torch 2.11.0+cu128), 12 epochs, batch 64, seed 0, on 78,783 training-sheet word crops. Outputs in `data/runs/reader_v1/`. The browser saved `crnn.pt` as `crnn.zip`; it was renamed, unchanged. Copied to `data/reader/crnn.pt`, sha256 `77c4592122b5f97fd4345095f58fc85572ae3736ee955c42b623095e314beac1`. The earlier one-epoch CPU trial is kept as `data/reader/crnn_cpu_epoch1.pt`.

| Epoch | 1 | 2 | 4 | 6 | 8 | 9 | 11 (kept) | 12 |
|---|---|---|---|---|---|---|---|---|
| Val word exact | 0.933 | 0.971 | 0.989 | 0.962 | 0.994 | 0.9978 | **0.9978** | 0.9978 |

`python src/train_reader.py eval`: 16,967 validation words, 99.78% exact, 37 wrong. Mean confidence 0.933 when right and 0.749 when wrong. Remaining errors are mostly I/J and dropped characters (`IQ` read as `JQ`, `GH-63446` as `GH-3446`). For comparison, EasyOCR read 80.8% of tags exactly from the same kind of perfect word boxes (ocr_v1), and Gadekar et al. (2026) report 0.995 with fine-tuned TrOCR.

Caveat: the validation crops chose the kept epoch, so 99.78% is slightly optimistic; the v2 test run is the unbiased measurement. These are perfect word boxes; end-to-end reading also depends on text finding and assignment, which v2 leaves unchanged.

## pipeline_v2 / decide_v2 on validation: domain reader end to end (2026-10-19)

As pre-registered: `python src/pipeline.py data/runs/Kaggle1/predictions_val data/runs/pipeline_v2 val --reader=crnn`, then `python src/decide.py data/runs/pipeline_v2/extracted_val data/runs/decide_v2 val`.

| Validation | v1 (EasyOCR) | v2 (CRNN) |
|---|---|---|
| End-to-end tag exact | 62.1% | 69.4% |
| Rule B, tau chosen by the unchanged rule | 0.98: coverage 50.3%, risk 4.0% | 1.00: coverage 26.0%, risk 0.0% |
| Rule B at tau 0.98 | coverage 50.3%, risk 4.0% | coverage 68.7%, risk 13.6% |
| Matching pairs passed automatically | 0 of 75 | 0 of 75 |

Error attribution (`src/attribute_errors.py`): misread 64.7%, no text 18.3%, association 17.1% of 2,410 wrong tags. Misreads of outside text are mostly truncations of a first or last character (`RV-05408` read `V-05408`, `27"` read `27`); in bubbles, digit swaps such as 5/9. Truncated readings often get confidence near 1.0, so the confidence no longer separates right from wrong.

Diagnosis: the reader is 99.78% exact on ground-truth word boxes but meets different crops in the pipeline (EasyOCR detector boxes, joined pieces, clutter). A padding probe on the 5 OCR tuning sheets confirms it is brittle to crop geometry, not short of pixels: end-to-end exact 64.2% at READ_PAD 4, 25.5% at 8, 6.5% at 12. READ_PAD stays 4.

**v2 as pre-registered does not improve the decision**: it is worse than v1 at the frozen-style operating point. This result is kept and will be reported.

## PLAN v2 AMENDMENT (recorded 2026-10-19, before any v2 test use; the test sheets remain untouched since the v1 run)

Deviation from PLAN v2 item 1 ("no other reader variant"): one further reader, v2b, is trained, because the diagnosis above shows a train/inference crop mismatch rather than a weak reader. v2b uses the same network, seed and schedule, trained on the original ground-truth crops **plus crops cut by the pipeline's own text finding** (EasyOCR detector, region rules and READ_PAD exactly as frozen) on training sheets, with true symbol boxes defining the regions and each crop labelled with the true words it covers. The epoch is selected on such pipeline crops from validation sheets. Everything else in PLAN v2 (tau rule, freeze, single disclosed test run, reporting v1 and v2 whatever they show) is unchanged. No further reader variants after v2b: if v2b fails on validation, v1 remains the reported system and v2/v2b are reported as negative results.

## reader_v2b data: pipeline-cut crops (2026-10-19)

`python src/train_reader.py build-detected val 25` and `... train 150`: 3,309 crops from the first 25 validation sheets and 19,478 from the first 150 training sheets, cut by `pipeline.read_region` with frozen settings inside true symbol regions, each labelled with the true words it covers by at least half their area. Some crops cut a word short, yet carry the full word as label; such crops are kept on purpose, so the reader can learn low confidence on them.

Baseline, the reader_v1 weights on these validation crops (`python src/train_reader.py eval val_det`): **77.9% exact** (732 wrong), against 99.78% on ground-truth word crops. Confidently wrong examples: `KL-80483` read `KL-80` at 1.0, `DV-29035` read `29035` at 1.0. This quantifies the train/inference mismatch behind the v2 result.

Kaggle package rebuilt with these files: `data/kaggle/reader_package.zip`, 280 MB, sha256 `f7509861aa63c5eafa0f276429c346b8323d6629b6e0ecfcd5576a21503d83c6`.

## reader_v2b: CRNN trained with pipeline-cut crops (Kaggle, 2026-10-19)

Same notebook and settings as reader_v1 (Tesla T4, 12 epochs, batch 64, seed 0), on 78,783 ground-truth crops plus 19,478 pipeline-cut crops; epoch chosen on 3,309 pipeline-cut crops from validation sheets. Outputs in `data/runs/reader_v2b/` (browser renamed the weights `crnn (1).zip`; renamed back, unchanged). Installed as `data/reader/crnn.pt`, sha256 `209dbe5f557f9daf877b6fda55b9475f4803f123146310b9351a1b3b31e4c0f7`; the reader_v1 weights are kept as `data/reader/crnn_v2.pt`.

| Epoch | 1 | 3 | 5 | 8 | 10 | 11 (kept) | 12 |
|---|---|---|---|---|---|---|---|
| Exact on validation pipeline crops | 0.859 | 0.899 | 0.934 | 0.943 | 0.944 | **0.947** | 0.947 |

Against 77.9% for the reader_v1 weights on the same crops. As before, these crops also chose the epoch, so 94.7% is slightly optimistic.

## open100_labels_v1: tag transcription for the 12 real OPEN100 plans (2026-10-19)

All 988 non-arrow, non-junction symbols (instrumentation 350, valve 257, general 234, inlet/outlet 96, tank 35, pump 16) were transcribed from the drawings, at the user's request, by the AI assistant (Claude) reading enlarged views of each symbol with its surroundings. **These are AI-transcribed labels, not independent human annotation**, and must be described as such; a human spot-check of a random sample is planned before they support any claim. Files (git-ignored, drawings' licence unconfirmed): `data/open100_labels/labels.csv` (all rows checked), the user's earlier partial file kept as `labels_user_partial.csv`, and `index.html` prefilled for review.

Conventions:
- **Instrument bubble:** letters, a space, the number, as printed (`PT 14086`, `TE 102B43`, `FCV XXX`); empty bubbles get an empty tag. Misprints are kept as printed (`LVC 1615`).
- **Valve:** the tag of an instrument bubble joined to it by a leader line, or a label printed beside it (`V-1-100X`, `SP-01`); plain manual valves are empty; nearby but unconnected text is not used.
- **General:** nozzle labels next to flanges (`N3`), equipment tags inside or under the symbol (`CVCS-HX-111`), detail markers (`DET A SHT 2`); fittings without text are empty.
- **Tank / pump:** the equipment tag as printed (`CND-TK-165`, `RHR-PU-120A`); two-line tags are joined with a space (`CVCS ION-114A`).
- **Inlet/outlet:** all text inside the connector, lines joined top to bottom (`TBS PID 240-1 TURBINE SYSTEM`).
- **Unsure (32 rows):** part of the label is cut off or ambiguous, the nozzle label could belong to either of two flanges, or the source graph's class is wrong (flow elements labelled pump, bubbles labelled valve, a flange labelled inlet/outlet).

Result: 754 of 988 symbols carry a tag (instrumentation 348, general 131, valve 129, inlet/outlet 95, tank 35, pump 16). The OCR suggestions pre-filled on the page were exactly right for 120 of the 754 (100 of 350 instrument bubbles). Agreement with the user's 13 earlier corrections: 12 identical; the 13th was nearby but unconnected text that the convention leaves empty.
