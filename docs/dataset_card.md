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
| Images | `image_2.tar.gz`, 968,795,855 bytes, 500 JPEGs under `image_2/`, 1,227 MB uncompressed. SHA-256 `BC088BA0A926DC527EAD93BAEE92B5D3ABA9E41DB2B2913A2AE0F985CBF5996E` | Yes, downloaded 2026-10-11 and read end to end |
| Masks | `mask.tar.gz`, size not shown | Not downloaded |
| Image size | 7168 x 4562 px, RGB, for `335.jpg` | One image only |
| Symbol size | Roughly 30-60 px across, under 1% of image width | Sheet 0 boxes and one viewed crop |
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

### Confirmed on sheet 335 (2026-10-11)

- `335.jpg` matches annotation folder `335`: symbol and word boxes drawn on the image land exactly on the drawn symbols and text, and the title-block values agree.
- Symbol and line boxes are [x1, y1, x2, y2]. Word boxes are [x1, y_bottom, x2, y_top].
- The word flag is the text rotation in degrees (0 or 90).
- Every symbol has exactly one linker row: [word id(s), line id]. No dangling ids.
- Quirks to handle in code:
  - The class column mixes types: the same glyph appears as the string `'21'` (7 untagged line-end instances) and the integer `21` (4 instances). Normalise to int.
  - 14 of 107 symbols have no tag.
  - For some classes the linked text is a size, not an identifier (e.g. reducer `2"x9"`), so it is not unique (`6"` occurs twice).
  - Some word rows have near-zero-size boxes.
- Class names are NOT confirmed. The YOLO repository's names are a third party's labels; under a class-minus-one offset only "Reducer" was checked against the image, and others look doubtful. Use class numbers and the paper's Figure 3 until verified.

### Whole-dataset integrity (2026-10-11, all 500 sheets)

Produced by `python src/convert_annotations.py` after `python src/download_annotations.py`.

| Measure | Value |
|---|---|
| Annotation files downloaded | 3,500 of 3,500, all valid `.npy` |
| Sheets converted to JSON | 500 of 500; 0 load failures, 0 structural problems |
| Symbols | 59,498; per sheet min 75, mean 119.0, max 200 |
| Words | 137,967; 134,673 horizontal, 3,294 rotated 90 degrees |
| Lines | 125,267; 112,330 solid, 12,937 dashed; 54,520 carry a label |
| Symbols without a linked line | 0 |
| Untagged symbols | 7,282 (12.2%) |
| Repeated tag strings within a sheet | 2,320 |

Findings:
- **Untagged symbols are exactly two classes.** Every instance of class 21 (5,625) and class 24 (1,657) is untagged; no other class ever is. Each links to a placeholder word with empty text and a near-zero box, which accounts for all 7,282 near-zero word boxes.
- **Class balance is near-uniform** at 1,620-1,840 instances per class, except class 21 with 5,625 (it is present on all 500 sheets).
- **Linked text means different things by class.** Repeats come almost entirely from classes whose text is not an identifier: class 30 (constant `STA`, 1,146 repeats), classes 17-20 (sizes such as `6"` or `1"x6"`), classes 22-23 (codes such as `CS-93`, `INS ( 13°C )`). Classes with identifier-style tags (e.g. `KL-81809`) repeat almost never.
- **Our symbol count (59,498) is 500 higher than the YOLO repository's README (58,998).** Unexplained; the difference is exactly one per sheet.
- `DRAWING NAME` has 482 distinct values with 16 repeated. Names are `SAMPLE_<4 digits>.JPG`, so some repeats are expected by chance; this is not evidence of duplicate sheets, and duplicates have not been checked.
- `lines2` has 68 rows on 393 sheets and 70-112 on the rest. Purpose still unknown.

### Duplicate check and split (2026-10-11)

Duplicate check over all 124,750 sheet pairs, on annotation content:
- No two sheets share an identical set of lines, symbols or symbol positions.
- Highest pairwise overlap (Jaccard): lines 0.267, symbol positions 0.011, identifier tags 0.056. The line overlap is consistent with a shared sheet frame and title block, but that cause was not confirmed.
- The 16 repeated `DRAWING NAME` values are coincidences: those pairs overlap no more than unrelated sheets (at most 0.18 on lines).
- Not checked: image-level similarity, or sheets that share a layout after shifting or scaling.

Split, written by `python src/make_split.py` to `docs/split.json` (seed 20261011, by whole sheet):

| Part | Sheets | Symbols | Instances per class (min / max) |
|---|---|---|---|
| train | 350 | 41,480 | 1,143 / 3,951 |
| val | 75 | 8,913 | 228 / 810 |
| test | 75 | 9,105 | 231 / 864 |

All 32 classes appear in every part. This is our own split: the authors' 4:1 assignment and the YOLO repository's 400/100 assignment are not published per sheet, so results will not be directly comparable with theirs. The test part must not be used for any tuning.

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
