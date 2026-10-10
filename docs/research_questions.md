# Research question and hypotheses

Status: PROVISIONAL, recorded 2026-10-10. The scope was approved to proceed, but the open decisions at the end are unanswered, and the dataset audit (Phase 2) may force changes.

## Primary research question

When components extracted from a raster P&ID image are checked against an independent bill of materials, how much does a confidence-aware PASS / REVIEW / FAIL decision rule reduce wrong automatic decisions, compared with treating every extraction result as fact, and at what cost in human review?

## Hypotheses

| | Prediction | Test | Disproved if |
|---|---|---|---|
| H1 | Without abstention, extraction errors cause a measurable share of false FAIL verdicts. | Run the no-abstention pipeline on drawing-BOM pairs that truly match; count FAILs. | False FAIL rate is near zero. |
| H2 | Sending low-evidence cases to REVIEW lowers the error rate among automatic decisions at equal coverage. | Compare risk-coverage curves on a held-out test set, thresholds fixed on validation data. | The curve is no better than the baseline's. |
| H3 | Calibrated confidence separates right from wrong decisions better than the raw detector score. | Compare calibration error and area under the risk-coverage curve. | No measurable difference. |

No hypothesis has been tested. No results exist yet.

## Scope

In:
- Raster P&ID images.
- Discrepancy types that can be labelled reliably: missing, extra, wrong type, quantity; tag mismatch only if tags are legible in the chosen dataset.
- Synthetic seeded discrepancies, always labelled as synthetic.

Out:
- CAD / DXF input.
- Mechanical part drawings.
- Design-rule compliance checking.
- Any claim of industrial readiness or certification.

Deferred (ablation only, removed from the paper and title if no measured gain):
- Graph reasoning. Requires connectivity ground truth.
- Vision-language model assistance.

## Candidate contributions (each needs evidence before it is claimed)

1. A paired drawing-BOM benchmark for P&ID images that includes failing cases.
2. A measured comparison of verification with and without abstention.
3. An error analysis separating extraction failures from true discrepancies.

## Open decisions (owner: project author)

1. Working title. Proposed: "Confidence-Aware Verification of Raster P&IDs Against Bills of Materials: Separating Extraction Failure from True Discrepancy". The original title names vision-language models and graph reasoning, which are not yet justified.
2. Whether the guide or institution requires VLM or graph components to stay in scope.
3. Whether a synthetic-discrepancy benchmark is acceptable to the guide. No source of real drawing-BOM pairs with known errors has been found.

## Dependencies

- A lawfully usable public P&ID dataset with enough annotation to derive a BOM (Phase 2). If tags are not readable, tag mismatch leaves the scope.
