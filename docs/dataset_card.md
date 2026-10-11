# Dataset card: Dataset-P&ID (Digitize-PID)

Status: audit of 2026-10-10. Based on the paper, the hosting pages and ONE inspected sheet (sheet 0) out of 500. Rows marked "unverified" have not been checked by us.

## Source

| | |
|---|---|
| Name | Dataset-P&ID |
| Paper | Paliwal, Jain, Sharma & Vig (2021), "Digitize-PID: Automatic Digitization of Piping and Instrumentation Diagrams", PAKDD 2021 workshops, arXiv:2109.03794 |
| Original data | Google Drive folder `1gMm_YKBZtXB3qUKUpI-LF1HE_MgzwfeR` (public, no sign-in needed on 2026-10-10) |
| Version | None published. Folder items are dated 1 Sept 2021. |
| Third-party conversion | `github.com/yunwoong7/digitize-pid-yolo`, commit `bf43602` (YOLO boxes only) |

The paper is not yet in `docs/paper/references.bib`; its citation has not been registry-verified.

## Nature of the data

- **Synthetic.** Generated sheets with 32 symbol types "uniformly plotted over different graph structures", with pixelation, blur and salt-and-pepper noise added (paper, section 3).
- Not real plant drawings. The sample sheet's own notes state the content is randomly generated.
- No independent specification, BOM or discrepancy labels are provided.

## Contents

| Item | Value | Verified |
|---|---|---|
| Sheets | 500 subfolders named 0-499, none missing | Yes, folder listing |
| Images | `image_2.tar.gz`, 924 MB | Size only; not downloaded |
| Masks | `mask.tar.gz`, size not shown | Not downloaded |
| Image size | Coordinates reach about 6843 x 4366 px | Inferred from sheet 0 annotations |
| Symbol classes | 32 | Paper and YOLO `data.yaml` |
| Author split | 4:1 train-test | Paper; the sheet assignment is unverified |

## Annotation schema (from sheet 0)

Seven NumPy `.npy` files per sheet. Column meanings are our reading of the stored values and need confirming on more sheets.

| File | Rows in sheet 0 | Columns |
|---|---|---|
| `N_symbols.npy` | 90 | symbol id, bounding box [x1, y1, x2, y2], class number |
| `N_words.npy` | 237 | word id, bounding box, text string, flag (meaning unconfirmed) |
| `N_lines.npy` | 186 | line id, endpoints [x1, y1, x2, y2], line label, style (e.g. `solid`) |
| `N_linker.npy` | 90 | symbol id -> associated word id(s) and one line id |
| `N_KeyValue.npy` | 10 | title-block field, value |
| `N_Table.npy` | 5 | revision table rows |
| `N_lines2.npy` | 68 | integer line segments plus a flag; purpose unconfirmed |

`symbols`, `words`, `lines` and `linker` are pickled object arrays. They were inspected without unpickling and reference only NumPy array constructors. Convert them once to JSON so pickles never need loading again.

## What it supports for this project

| Need | Supported? |
|---|---|
| Symbol detection training and evaluation | Yes |
| OCR evaluation (text boxes with true strings) | Yes |
| Deriving a ground-truth BOM of (type, tag) records | Yes, via `symbols` + `linker` + `words` |
| Tag-mismatch discrepancies | Yes |
| Connectivity ground truth | Partial: each symbol links to one line; full symbol-to-symbol connectivity is untested |
| Real drawing-vs-specification discrepancies | No. Discrepancies must be seeded synthetically. |

## Licence and permitted use

- No licence file exists in the Drive folder or in the YOLO repository.
- The paper says the dataset is "made publicly available for the benefit of research community".
- The YOLO repository README says "provided for research purposes only" and asks for citation of the paper.
- The arXiv paper itself is CC BY-NC-ND 4.0. That covers the paper, not necessarily the data.

Working position: use for research and report results with citation. Do NOT redistribute the images or modified copies. Publish code and a manifest that rebuild any derived benchmark from the original source. Written clarification from the authors has not been requested yet.

## Known limitations

- Synthetic layout and noise; results do not establish performance on real P&IDs.
- Uniform symbol placement is unlike real plant density.
- No held-out test split is defined beyond the authors' 4:1 statement; near-duplicate sheets across splits have not been checked.
- Only one sheet's annotations and no image have been inspected by us.
