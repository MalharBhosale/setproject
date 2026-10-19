# Confidence-aware verification of raster P&IDs against bills of materials

Research code for verifying a raster Piping and Instrumentation Diagram (P&ID) against an independent bill of materials (BOM). The system returns **PASS**, **FAIL** (a discrepancy supported by evidence) or **REVIEW** (evidence insufficient: a human should look). The central question is how to keep extraction errors (missed symbols, misread tags) from being reported as discrepancies.

**Status:** research prototype. All verification results are on synthetic drawings with synthetically seeded discrepancies. Not certified engineering software, and not for unsupervised safety-critical use.

## Main results (held-out test sheets, configuration frozen before the single test run)

| | Result |
|---|---|
| Symbol detection, 75 test sheets | precision 0.9999, recall 1.0000 |
| Tags read exactly, end to end | 61.6% |
| Verification trusting every extraction | all 75 matching pairs wrongly declared FAIL; 25.0% of decisions wrong |
| Confidence + tag-format rule at frozen threshold | 52.3% of cases decided automatically, 6.4% of those wrong (95% CI 3.1-9.6%) |

Full tables, the protocol and every run: [`docs/experiment_log.md`](docs/experiment_log.md) and [`docs/paper/results.md`](docs/paper/results.md). What may and may not be claimed: [`docs/paper/claim_evidence_matrix.csv`](docs/paper/claim_evidence_matrix.csv).

## Data (not included)

- **Dataset-P&ID** (Paliwal et al., 2021; arXiv:2109.03794): 500 synthetic P&ID sheets. Public on Google Drive via the paper. No licence is stated beyond research use, so no drawings, crops or derived images are redistributed here. `src/download_annotations.py` fetches the annotations; the image archive `image_2.tar.gz` (924 MB, SHA-256 `BC088BA0A926DC527EAD93BAEE92B5D3ABA9E41DB2B2913A2AE0F985CBF5996E`) must be placed at `data/image_2.tar.gz`. See [`docs/dataset_card.md`](docs/dataset_card.md).
- **PID2Graph** (Stürmer et al., 2025; Zenodo 14803338; CC BY-SA 4.0): only the graphs for the 500 sheets and the 12 real OPEN100 plans are used, fetched selectively by `src/pid2graph.py fetch`.

## Setup

Python 3.14 on Windows was used locally (CPU only); detector training ran on Kaggle (Python 3.13, Tesla T4).

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Reproduce, in order

| Step | Command | Output |
|---|---|---|
| Annotations | `python src/download_annotations.py` then `python src/convert_annotations.py` | `data/annotations/<sheet>.json` + integrity report |
| Split | `python src/make_split.py` | `docs/split.json` (committed) |
| Benchmark | `python src/make_benchmark.py` | `data/benchmark/boms/`, `docs/benchmark_manifest.json` (committed) |
| Oracle check | `python src/verify.py` | verifier on perfect inputs (upper bound) |
| Tiles | `python src/make_tiles.py extract` then `python src/make_tiles.py tile` | train/val tiles only |
| Train detector | `python training/make_kaggle_package.py`, then run `training/kaggle_train_detector.ipynb` on Kaggle | `best.pt` (hash in the experiment log) |
| Detect | `python src/detect_sheets.py <best.pt> data/runs/Kaggle1 val` | sheet-level predictions and scores |
| OCR comparison | `python src/eval_ocr.py data/runs/ocr_v1 easyocr rapidocr` | |
| Extract + verify | `python src/pipeline.py data/runs/Kaggle1/predictions_val data/runs/pipeline_v1 val` | tag accuracy, H1 |
| Decide | `python src/decide.py data/runs/pipeline_v1/extracted_val data/runs/decide_v1 val` | risk-coverage sweep, H2 |
| Intervals | `python src/bootstrap_ci.py test 0.98` | sheet-level bootstrap |
| Compare frozen runs | `python src/compare_runs.py data/runs/pipeline_v1/extracted_test 0.98 data/runs/pipeline_v2b/extracted_test 0.90 test` | paired v1 vs v2b |
| Selective metrics | `python src/selective_metrics.py data/runs/pipeline_v1/extracted_val val` | AURC, format-only ablation, paired A vs B, review workload |
| Text reader (v2) | `python src/train_reader.py build`, `python training/make_reader_package.py`, then run `training/kaggle_train_reader.ipynb` on Kaggle; `--reader=crnn` in `pipeline.py` | `crnn.pt`; protocol pre-registered in the experiment log |
| Real-drawing tag labels | `python src/make_label_sheet.py`, then open `data/open100_labels/index.html` | local page with OCR suggestions; export `labels.csv` |
| Figures | `python src/make_figures.py` | `docs/paper/figures/` |
| Extras | `python src/attribute_errors.py ...`, `python src/pid2graph.py fetch|map`, `python src/eval_open100.py ...` | error attribution, graph alignment, real-drawing check |

The test split is guarded: `detect_sheets.py` and `pipeline.py` need `--allow-test`, and `decide.py` needs `--frozen-tau=<value>` on test data.

## Tests

```
for %f in (tests\test_*.py) do .venv\Scripts\python %f
```

Each test file also runs under pytest. Tests need no data and no network.

## Layout

```
src/        data preparation, benchmark, detection, OCR, verification, decision, analysis
training/   Kaggle training package for the detector
tests/      one test file per module
docs/       dataset card, benchmark specification, split, manifest, experiment log, paper material
```

## Licence

Code: GNU AGPL-3.0 (see `LICENSE`), because it builds on Ultralytics YOLO (AGPL-3.0). EasyOCR and RapidOCR are Apache-2.0. Datasets keep their own terms (see above).
