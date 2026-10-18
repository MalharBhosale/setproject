# Confidence-Aware Verification of Raster P&IDs Against Bills of Materials: Separating Extraction Failure from True Discrepancy

*First draft, 2026-10-18. Citations use keys from `references.bib`. Every number comes from `experiment_log.md`. Title not yet agreed (see `outline.md`).*

## Abstract

Checking a piping and instrumentation diagram (P&ID) against its bill of materials (BOM) is still done by hand, and many plants hold drawings only as raster images. Automating the check requires extracting components and tags from the image, and extraction makes mistakes that look exactly like real discrepancies. We study this problem on a paired benchmark of 2,000 drawing-BOM cases built from a public synthetic P&ID dataset, with discrepancies seeded into the BOM. A tiled YOLO detector finds symbols almost perfectly on held-out sheets (precision 0.9999, recall 1.0), but end-to-end tag reading reaches only 61.6%. A deterministic verifier that treats every extraction as fact therefore declares all 75 matching test pairs non-conforming. We add a PASS / REVIEW / FAIL decision in which a finding counts only when no untrusted component could explain it. Trusting readings by OCR confidence alone barely helps (24.0% wrong automatic decisions against 25.0% without abstention), because the OCR makes confident, systematic errors. Requiring readings to be confident and to match a tag format learned from training sheets reduces wrong automatic decisions to 6.4% (95% interval 3.1-9.6%) while still deciding 52.3% of cases automatically, with all parameters frozen before the single test run. The system never passes a matching pair automatically: it flags problems reliably but cannot yet certify a drawing as conforming. Error analysis places the bottleneck in reading rather than in assigning text to symbols, and a check on 12 real P&IDs shows partial transfer of the detector at a matched scale.

**Keywords:** P&ID, engineering drawing verification, bill of materials, selective prediction, abstention, OCR, symbol detection

## 1 Introduction

P&IDs record the equipment, valves, instruments and piping of a process plant. A drawing must agree with related documents such as the bill of materials, and engineers confirm this agreement by reading both side by side. The work is slow and error-prone, and it is harder for legacy plants whose drawings exist only as scans or rasters, where CAD-based checking tools cannot be used.

Deep learning has made it possible to detect symbols and read text in P&IDs [@elyan2020symbols; @mani2020digitization; @yun2020recognition; @gao2020component; @kim2021highdensity], and reviews describe steady progress in digitisation [@jamieson2024review]. Most of this work stops at extraction: it converts a drawing into data but does not ask whether the data agrees with a specification. Work that does compare drawings either needs a second drawing as reference [@zhou2026drawingchecker] or checks design rules on CAD files rather than images [@youn2026compliance].

Verification from images has a property that extraction-only work can ignore. When the system fails to read a component, the result is indistinguishable from the component being absent. A verifier that trusts its extraction reports its own mistakes as discrepancies. The question is not only how accurate extraction is, but whether the system can tell when its evidence is insufficient and defer to a human. Abstention of this kind has been proposed for drawing audits [@chen2026auditing], but we found no implemented and measured system for raster P&IDs against an independent BOM.

This paper makes four contributions:
1. A reproducible paired benchmark of 2,000 drawing-BOM cases, including failing cases, generated from a public P&ID dataset.
2. A measurement showing that verification which trusts extraction fails completely on this benchmark.
3. A confidence-aware PASS / REVIEW / FAIL rule, with the finding that OCR confidence alone is not enough, while confidence combined with format plausibility makes automatic decisions substantially more reliable.
4. An analysis of where errors come from, and a check of how the detector transfers to real drawings.

## 2 Related work

**Symbol and text extraction.**
- CNN detectors recognise most symbols in industrial diagrams [@elyan2020symbols].
- Region proposals with learned negative classes reach 98% symbol recognition [@yun2020recognition].
- Faster R-CNN with text detection handles nuclear-plant P&IDs on cropped patches [@gao2020component].
- Combined symbol and text recognition on dense P&IDs is reported at industrial accuracy [@kim2021highdensity].
- Few-shot detection addresses rare symbol classes [@jamieson2024fewshot].
- Markup noise is identified as a cause of recognition errors [@shteriyanov2024density].
- OCR for mechanical drawings reports a 94% recognition F1 with an 8% character error rate [@villenatoro2023ocr].

These works establish that extraction is feasible but imperfect; none verifies the extracted content against a specification.

**Structure and graphs.** Graph search over pipe lines recovers symbol connectivity [@mani2020digitization]. Graph neural networks interpret mechanical drawings [@xie2022gnn; @zhang2023segmentation]. P&IDs can be converted into graphs with transformers [@sturmer2025pid2graph].

**Vision-language models.** Benchmarks show the abilities and limits of general vision-language models on engineering tasks [@picard2025vlm]. A hybrid YOLO plus fine-tuned VLM pipeline parses mechanical drawing annotations, with about 11% hallucinated fields [@khan2025drawings].

**Checking and comparison.**
- Kim et al. inspect the output of P&ID recognition for missed and misrecognised objects, comparing against the drawing itself [@kim2025inspection].
- Drawing-Checker compares a target drawing with a reference drawing using a multimodal LLM [@zhou2026drawingchecker].
- Youn et al. verify P&IDs against design rules with LLM-generated deterministic checks, but need DXF input; their image-based baseline reached 55.9% F1 [@youn2026compliance].
- Chen and Cheng propose an audit architecture with deterministic rules, abstention states and human routing, without an implemented end-to-end system [@chen2026auditing].

Our work differs in task (a raster drawing against an independent BOM) and in measuring how abstention trades coverage against error.

## 3 Problem and benchmark

**Task.** The input is a raster P&ID and an independent BOM. The output is a verdict and the findings behind it:
- **PASS:** drawing and BOM agree.
- **FAIL:** a discrepancy is supported by evidence.
- **REVIEW:** the evidence is insufficient for either.

REVIEW is never a ground-truth label; it is the system's choice to defer.

**Data.** We use Dataset-P&ID [@paliwal2021digitize], 500 synthetic sheets of 7168 x 4561-4562 px with 32 symbol classes, annotated with symbol boxes, word boxes with their text, and lines. We audited all 500 sheets: 59,498 symbols, 0 load failures, 0 structural problems. We split by whole sheet into 350 training, 75 validation and 75 test sheets with a fixed seed; no two sheets share identical content. The dataset states no licence beyond research use, so we publish code and a manifest, not images.

**BOM.** A BOM has two kinds of row, derived from the annotations alone:
- **Tagged rows:** class and tag, one per symbol, for 22 classes whose text is a unique identifier.
- **Counted rows:** class, text and quantity, for size labels, codes and untagged classes.

**Seeded discrepancies.** Each case edits a clean BOM. Five types are seeded:
- MISSING (add a row with an unused tag);
- EXTRA (delete a row);
- TAG (change one or two digits);
- TYPE (change the class, keep the tag);
- QUANTITY (change a count by 1 to 3).

Each sheet yields one clean case and three seeded cases with 1, 3 and 5 discrepancies, giving 2,000 cases (300 in test). The drawings are never modified.

**Known weaknesses.**
- Tag prefixes identify the class in this dataset, so TYPE discrepancies are easier than in practice.
- A one-character TAG edit is indistinguishable from an OCR misreading, by design.
- Three training cases are ambiguous by construction.

With perfect extraction, the deterministic verifier reproduces 1,997 of 2,000 answer keys and gets every verdict right.

## 4 Method

**Detection.** Sheets are cut into 1024 px tiles overlapping by 256 px, more than the largest symbol (157 px). A YOLO11 nano detector, COCO-pretrained, is trained on the training tiles: Kaggle T4, image size 1024, seed 0, 10 hours, 87 epochs. At inference, boxes touching an inner tile edge are dropped, because the symbol appears whole in a neighbouring tile, and duplicates are merged per class.

**Text.** Training-sheet statistics fix where text sits:
- inside the symbol for instrument bubbles (classes 26-32);
- none for classes 21 and 24;
- otherwise one word within 80 px outside the symbol.

EasyOCR's detector finds text in the corresponding region; for bubbles, the outline is excluded. Boxes on one line closer than 0.6 text heights are joined. Each piece is read by EasyOCR with a 4 px margin and a character allowlist. Vertical text is always turned clockwise, the convention for 495 of 495 readable vertical words on training sheets. Each piece of text is assigned to exactly one symbol, so no symbol can take a neighbour's tag.

We chose EasyOCR over RapidOCR (which runs PaddleOCR models). Tag accuracy on perfect word boxes was similar (80.8% and 82.4%), but EasyOCR is eight times faster on CPU and has a lower character error rate. PaddleOCR itself could not be installed.

**Verifier.** Extracted components are paired with BOM rows in four passes:
1. same class and tag: a match;
2. same tag, different class: TYPE;
3. same class, tags within two characters: TAG;
4. the rest: EXTRA or MISSING.

Counted rows are compared by quantity.

**Decision.** A component is *trusted* if its detector score is at least 0.5 and, for text-bearing classes, its reading passes a test:
- **Rule A:** the OCR confidence is at least tau.
- **Rule B:** additionally, the reading's format must match a format seen for its class on training sheets. The format maps letters to A and digits to 9, so the specific letters, which would reveal the class, are not used.

A finding is *supported* if no untrusted component could explain it:
- EXTRA, TAG and TYPE need a trusted component;
- MISSING needs no unsupported finding of the same class;
- QUANTITY needs every component of its class trusted.

The verdict is FAIL if any finding is supported, REVIEW if findings exist but none is supported, and PASS otherwise. With tau = 0, rule A reduces to the plain verifier.

## 5 Experimental protocol

All design choices were made on training and validation data:
- region sizes and text placement from training sheets;
- OCR margins and the allowlist from five validation sheets;
- the format rule after rule A failed on validation;
- tau = 0.98, chosen as the threshold that maximised coverage with at most 5% wrong automatic decisions on the 300 validation cases.

The complete configuration, including the detector weights' hash, was committed before any test sheet was processed. The test run was then executed once.

**Metrics:**
- detection precision and recall at IoU >= 0.5;
- exact tag match;
- coverage (share of cases decided automatically) and risk (share of automatic decisions that are wrong).

Intervals are 95% bootstrap intervals over the 75 test sheets (2,000 resamples), since the four cases on a sheet share one extraction.

## 6 Results

Full tables are in `results.md`.

**Components.** On the 75 test sheets the detector finds 9,105 of 9,105 symbols with one false detection. End-to-end tag reading is exact for 61.6% of tags (62.1% on validation), against 89.0% when reading perfectly located words, so finding and assigning text costs about 25 points.

**H1: verification that trusts extraction.** Every one of the 75 matching test pairs is declared FAIL. Across the 300 test cases the verifier raises 20,978 false findings against 456 true ones. Misread tags become MISSING plus EXTRA pairs, and misread sizes make counts disagree.

**H2: confidence-aware decisions.** The figure (`figures/risk_coverage.png`) shows risk against coverage for both rules as tau varies.
- **Rule A** stays close to the no-abstention point: at tau = 0.98 it decides 96.0% of test cases with 24.0% wrong. Every sheet contains confidently misread tags. On validation, 585 wrong readings had confidence 0.9 to 1.0, next to 3,579 correct ones, for example truncations such as `KL-80-` at 0.987 and `1J-74522` for `IJ-74522` at 0.923.
- **Rule B,** at the frozen tau = 0.98, decides 52.3% of test cases [46.3, 58.3] with 6.4% wrong [3.1, 9.6], against 25.0% without abstention. On validation the figures were 50.3% and 4.0%. Coverage carries over to test; risk is higher than on validation and exceeds the 5% target set there.
- Supported findings at this operating point have precision 0.728 and recall 0.301 on test.

**No automatic PASS.** Under every rule and threshold, no matching pair is passed automatically. On test, rule B sends 65 of 75 matching pairs to REVIEW and wrongly fails 10. All automatic decisions are FAILs.

**Where errors come from.** On validation, 74.1% of wrong tags are misreadings of the symbol's own text, 14.8% have no text assigned, and at most 11.1% are association errors in which the text of another symbol or a line label was taken. Only 2 errors involved a line label. Connectivity information could address at most the association errors, so a graph-reasoning component cannot remove the dominant error on this task. We did not pursue it further.

**Real drawings.** On 12 real P&IDs from the OPEN100 project [@sturmer2025pid2graph], whose symbols are about four times smaller than in the synthetic data, the detector at matched scale finds 83.4% of symbols with the correct coarse label at 67.6% precision. At native scale results are much worse. Tanks, pumps and inlet/outlet markers, 10% of real symbols, never occur in the training data. These plans carry no text annotation, so tag reading and verification could not be evaluated on them.

## 7 Discussion and limitations

The results separate two questions that are easily conflated: how accurate extraction is, and whether a verifier can tell when its extraction is wrong. On this benchmark detection is solved, reading is not, and confidence scores from the OCR engine do not reveal its systematic errors. A domain constraint as simple as the expected tag format provides the missing signal. The resulting system is useful in a specific way: it can raise discrepancies that are right most of the time and route the rest to an engineer. It does not replace the engineer, because it never certifies a drawing as conforming.

Limitations:
- **Synthetic data.** Drawings and discrepancies are synthetic, and tag formats in the dataset are clean and regular. Real tags vary more, which would weaken the format signal.
- **Small test set.** The risk estimate comes from 75 test sheets and is uncertain (3.1-9.6%). The 5% target chosen on validation was not met.
- **Design on validation.** Rule B was introduced after rule A failed on validation. Its test performance is an honest held-out measurement, but its design was not pre-registered.
- **No automatic PASS.** Practical value depends on how much of the REVIEW queue a human can clear quickly. We did not measure that.
- **Real drawings.** Detection transfers only partly, and only at a matched scale; verification on real drawings is untested.
- **Untested components.** Calibration (H3) and vision-language assistance were not evaluated. Graph reasoning was analysed but not built, because the benchmark has no connectivity discrepancies.

## 8 Conclusion and future work

Verifying raster P&IDs against bills of materials fails if extraction is trusted, and OCR confidence alone does not fix it. Combining confidence with format plausibility makes half of the decisions automatic with about 6% error on held-out synthetic sheets, while the remainder goes to review.

Future work:
- improve reading, which causes three quarters of the errors;
- extend the specification with connectivity (for example a line list) so that graph reasoning becomes testable, using the aligned PID2Graph graphs;
- build a real-drawing benchmark with text annotation;
- study how quickly engineers clear the REVIEW queue.

## Reproducibility

Code, the split, the benchmark manifest, the frozen configuration and all commands are in the project repository. `results.md` lists the commands that reproduce every reported number. The source drawings must be obtained from the original dataset.
