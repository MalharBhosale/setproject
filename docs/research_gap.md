# Research gap

Status: working assessment as of 2026-10-10. It rests on a targeted search, not an exhaustive review, and must be re-checked before submission.

## What already exists

| Work | What it does | Input | Checked how |
|---|---|---|---|
| Mani et al. 2020 (`mani2020digitization`) | Digitizes P&IDs: symbols, text, line connectivity | Raster image | Read in full |
| Khan et al. 2025 (`khan2025drawings`, preprint) | Extracts annotations from mechanical drawings into JSON with YOLO + VLM | Raster image | Read in full |
| Zhou et al. 2026 (`zhou2026drawingchecker`) | Compares a target drawing against a reference drawing | Raster image | Read in full |
| Youn et al. 2026 (`youn2026compliance`) | Checks P&IDs against design rules with deterministic, LLM-generated checkers | DXF (CAD) | Read in full |
| Chen & Cheng 2026 (`chen2026auditing`) | Proposes an audit architecture with deterministic rules, abstention states and human routing | Mechanical drawings | Abstract and key passages |
| Gomez-Vega et al. 2025 | Flags inconsistencies in SmartPID database records | Database | Search summary only, not in references.bib |
| Dzhusupova et al. 2023 | Finds design-error patterns in P&IDs | Raster image | Search summary only, not in references.bib |

## What these leave open

1. **No implemented, measured verification of a raster P&ID against an independent bill of materials was found.** Extraction work stops at "what does the drawing say". Comparison work needs a reference drawing or CAD data.
2. **Abstention is proposed but not measured.** Chen & Cheng state their paper "does not report an implemented end-to-end system" and list calibration and risk-coverage analysis as future requirements.
3. **Failing cases are untested in the closest P&ID checker.** In Youn et al. all 587 rule checks passed, so detection of real violations was not measured.
4. **The image route is hard.** Youn et al. report 55.9% F1 for a vision LLM on a P&ID image against 100% from the DXF file. Raster-only verification matters for legacy and scanned drawings where no CAD file exists.

## Claims that are NOT available

- First automated P&ID checker.
- First to combine deterministic checking with abstention.
- Any benefit from vision-language models or graph reasoning. Nothing found so far shows either is needed.

## Known weaknesses of this assessment

- Finding nothing in a search is weak evidence; paywalled or differently worded work may exist.
- Two rows above rest on search summaries only.
- Only 5 of the 15 original reference papers have been read beyond page 1.
- Kim et al. 2021 and Shteriyanov et al. 2024 are available as abstracts only.
