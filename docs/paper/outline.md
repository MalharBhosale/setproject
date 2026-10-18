# Paper outline

Status: first outline, 2026-10-18. Built from `results.md` and `claim_evidence_matrix.csv`; a section may only make claims marked supported or partially supported there.

## Title (decision for the author and guide)

- **Proposed:** Confidence-Aware Verification of Raster P&IDs Against Bills of Materials: Separating Extraction Failure from True Discrepancy
- **Registered working title:** A Confidence-Aware Multimodal Framework for Automated Engineering Blueprint Verification Using Vision-Language Models and Spatial-Semantic Graph Reasoning. It is not supported by the evidence as it stands: vision-language models were not evaluated (C12), and graph reasoning was analysed and found unable to address the dominant error (C11).

## Contributions (each tied to evidence)

1. A reproducible paired benchmark of 2,000 drawing-BOM cases built from a public P&ID dataset, including failing cases, with a generator and a ground-truth manifest (C13).
2. A measurement showing that verification which treats extraction as fact fails completely: every matching pair is declared non-conforming (C3, H1).
3. A confidence-aware PASS / REVIEW / FAIL rule, with evidence that OCR confidence alone does not make automatic decisions trustworthy (C4), while confidence combined with format plausibility cuts the error of automatic decisions from 25.0% to 6.4% at 52% coverage on held-out sheets (C5, H2).
4. An error analysis locating the bottleneck in reading rather than association (C2, C11), and a qualitative check of detector transfer to real drawings (C9).

## Structure

| Section | Content | Evidence |
|---|---|---|
| 1 Introduction | Manual drawing-BOM checking; extraction errors masquerade as discrepancies; contributions | research_gap.md |
| 2 Related work | Digitisation; drawing comparison; rule checking on CAD; abstention proposals | references.bib, research_gap.md |
| 3 Problem and benchmark | Task definition; data; split; BOM format; seeded discrepancy types; known weaknesses | dataset_card.md, benchmark_spec.md |
| 4 Method | Tiled detection; text finding, reading and assignment; deterministic verifier; decision rules A and B | src/, experiment_log |
| 5 Experimental protocol | Validation-only design; freeze; single test run; metrics; bootstrap | experiment_log FROZEN entry |
| 6 Results | Components; H1; H2 with figure; real-drawing check; graph analysis | results.md |
| 7 Discussion and limitations | No automatic PASS; 5% target missed; synthetic data; scale; class coverage | claim matrix rows C6-C12 |
| 8 Conclusion and future work | Connectivity-bearing specifications; better reading; real-drawing benchmarks | |

## Open items before submission

- Choose the title and the target venue; check the venue's scope, length limit and AI-assistance policy.
- Re-run the literature search; Khan et al. (2025) was a preprint and may now have a journal version.
- Obtain the full texts of Kim et al. (2021) and Shteriyanov et al. (2024), currently abstract-only.
- Ask the Dataset-P&ID authors for licence terms in writing.
- Decide whether to run H3 (post hoc) or leave it as future work.
