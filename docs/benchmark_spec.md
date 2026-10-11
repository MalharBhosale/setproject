# Benchmark specification: paired drawing-BOM verification

Status: DRAFT for approval, 2026-10-11. Nothing here is implemented. Numbers marked "proposed" are the author's to change.

## 1. What one benchmark case is

A case is a pair: one original Dataset-P&ID sheet image, unmodified, and one bill of materials (BOM) file. The task is to say whether they agree and, if not, where.

Discrepancies are seeded by editing the BOM only. The drawing is never altered. Reasons: the original image stays intact, no modified images need to be stored or redistributed (the dataset licence is unclear), and every change is exact and recorded.

Consequence to state in the paper: all discrepancies are synthetic and live on the specification side. Editing the BOM to add a row is equivalent to a component missing from the drawing, but no drawing in the benchmark was ever really drawn wrongly.

## 2. Class groups

Measured on all 500 sheets. Classes are referred to by number; names are not yet verified.

| Group | Classes | Linked text | In-sheet repeats | Checked by |
|---|---|---|---|---|
| Identifier | 1-16 | `AA-99999`, e.g. `KL-81809` | 1 in 28,357 | tag, one row per symbol |
| Identifier | 25 | `ERV-9-99` | 3 | tag |
| Identifier | 26-29, 31 | two words, e.g. `SDL 472` | 0-5 each | tag |
| Identifier | 32 | three words, e.g. `771 LG-10 433` | 0 | tag |
| Attribute | 17-20 | a size, e.g. `6"`, `2"x9"` | 45-312 each | count per (class, text) |
| Attribute | 22, 23 | a code, e.g. `CS-93`, `INS ( 13°C )` | 22-28 each | count per (class, text) |
| Constant | 30 | always `STA` | n/a | count per class |
| Untagged | 21, 24 | none | n/a | count per class |

22 identifier classes, 10 counted classes.

## 3. BOM format

One JSON file per case.

```json
{
  "sheet": "335",
  "tagged": [ {"cls": 13, "tag": "AV-33033"} ],
  "counted": [ {"cls": 19, "text": "6\"", "quantity": 2}, {"cls": 21, "text": "", "quantity": 11} ]
}
```

Rules:
- `tagged` has one row per symbol of an identifier class. A tag is the linked words joined by one space, exactly as annotated. The rare repeated tag appears as repeated rows.
- `counted` has one row per distinct (class, text) among counted classes, with its quantity.
- Mandatory fields are exactly those shown. The BOM carries no positions, lines, materials or ratings, so the system must never report a mismatch on a field the BOM does not contain.
- The clean BOM is derived from the ground-truth annotations alone. It is independent of any model, but not of the annotations used to train the detector on other sheets.

## 4. Discrepancy types

Each seeded discrepancy is one edit to a clean BOM.

| Type | Applies to | Edit | Meaning |
|---|---|---|---|
| MISSING | tagged | add a row with a new tag in the class's format, unused on the sheet | BOM lists a component the drawing lacks |
| EXTRA | tagged | delete a row | drawing shows a component the BOM lacks |
| TAG | tagged | change 1-2 characters of a tag, keeping its format and not colliding with another tag on the sheet | same component, wrong identifier |
| TYPE | tagged | change `cls` to another identifier class, keep the tag | same identifier, wrong component type |
| QUANTITY | counted | change `quantity` by +/-1 to +/-3, never below 0 | wrong count of an untagged or attribute item |

Excluded from version 1: attribute (size or code) mismatch, connectivity mismatch, and anything requiring material, rating or dimension data, which the dataset does not have.

## 5. Known weaknesses of these types

- **Tag prefix reveals the class.** Every tag prefix is used by exactly one class (`AB` only by class 1, `SDL` only by class 28, and so on). A TYPE discrepancy can therefore be caught from the text alone, without looking at the symbol, and a detector's class could be "corrected" from OCR. Real plants are not this tidy. TYPE results must be reported separately and described as optimistic; any method that reads the class from the prefix must be reported as doing so.
- **TAG looks like an OCR error.** A one-character BOM change and a one-character misread produce the same evidence. This is intended: it is the case the REVIEW outcome exists for.
- **TAG versus MISSING plus EXTRA.** A changed tag can also be explained as one component missing and another extra. Ground truth records it as one TAG discrepancy; how a system's alternative explanation is scored belongs in the experimental protocol and is not yet decided.
- **Uniform data.** Classes are evenly balanced and tags are clean machine text, so discrepancy rates here say nothing about real plant drawings.

## 6. Ground truth

Per case:
- sheet verdict: `PASS` if no discrepancy was seeded, otherwise `FAIL`. `REVIEW` is never a ground-truth label; it is a system's choice to abstain.
- a list of discrepancies, each with type, the BOM row before and after, and for EXTRA, TAG and TYPE the symbol id and bounding box on the drawing as the evidence location. MISSING and QUANTITY have no single drawing location.

## 7. Generation and manifest

- Cases are generated only after the split in `docs/split.json`, and each case inherits its sheet's part, so no test sheet contributes to training or validation cases.
- Proposed: per sheet, 1 clean case and 3 seeded cases with 1, 3 and 5 discrepancies, types drawn uniformly from those applicable. That gives 2,000 cases: 1,400 train, 300 val, 300 test.
- One fixed seed. The generator writes a manifest listing, for every case, the sheet, the seed, and each edit with its parameters. Clean and seeded BOMs can be regenerated from the annotations and the manifest; no drawings are redistributed.
- All cases are labelled synthetic.

## 8. Decisions needed from the author

1. Accept BOM-side seeding, or require drawing-side edits as well?
2. Accept the five types and the exclusions?
3. Accept 1 clean + 3 seeded cases per sheet with 1, 3 and 5 discrepancies?
4. Should TYPE stay in the benchmark given the prefix weakness?
