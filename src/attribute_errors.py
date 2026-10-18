"""Attribute end-to-end tag errors to their cause, to bound what graph reasoning (better association) could fix.

Usage: python src/attribute_errors.py <extracted_dir> [part]      part defaults to "val"
For each detected symbol whose extracted tag is wrong:
  no text      - nothing was assigned to it
  association  - the reading is closer to some other text on the sheet (another symbol's tag or a line label)
                 than to the symbol's own tag, within 2 edits: the right characters, given to the wrong symbol
  misread      - everything else: the symbol's own text, read wrongly
Only association errors could be fixed by knowing connectivity.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ocr import edit_distance  # noqa: E402
from pipeline import UNTAGGED, iou  # noqa: E402


def classify(pred, own, others):
    """pred: extracted tag; own: true tag; others: other texts on the sheet."""
    if pred == own:
        return "exact"
    if not pred:
        return "no text"
    d_own = edit_distance(pred, own)
    d_other = min((edit_distance(pred, o) for o in others), default=99)
    return "association" if d_other < d_own and d_other <= 2 else "misread"


def main():
    ext = Path(sys.argv[1])
    part = sys.argv[2] if len(sys.argv) > 2 else "val"
    tally, by_class, line_label = Counter(), {}, 0
    for sid in json.loads(Path("docs/split.json").read_text(encoding="utf-8"))[part]:
        comps = json.loads((ext / f"{sid}.json").read_text(encoding="utf-8"))
        sheet = json.loads(Path(f"data/annotations/{sid}.json").read_text(encoding="utf-8"))
        words = {w["id"]: w["text"] for w in sheet["words"]}
        line_labels = {l["label"] for l in sheet["lines"] if l["label"].strip()}
        for s in sheet["symbols"]:
            if s["cls"] in UNTAGGED:
                continue
            own_ids = sheet["links"][s["id"]]["words"]
            own = " ".join(words[i] for i in own_ids)
            m = [c for c in comps if c["cls"] == s["cls"] and iou(c["box"], s["box"]) >= 0.5]
            if not m:
                continue
            others = [t for i, t in words.items() if i not in own_ids and t] + sorted(line_labels)
            kind = classify(m[0]["tag"], own, others)
            tally[kind] += 1
            by_class.setdefault(s["cls"], Counter())[kind] += 1
            if kind == "association" and min(edit_distance(m[0]["tag"], l) for l in line_labels) <= 2:
                line_label += 1
    wrong = sum(v for k, v in tally.items() if k != "exact")
    print(f"{part}: tags {sum(tally.values())}, wrong {wrong}")
    for k in ("no text", "association", "misread"):
        print(f"  {k:12} {tally[k]:5}  ({tally[k] / max(1, wrong):.1%} of errors)")
    print(f"  association errors that took a line label: {line_label}")
    top = sorted(by_class.items(), key=lambda kv: -kv[1]["association"])[:5]
    print("  most association errors by class:", [(c, v["association"]) for c, v in top])
    return tally


if __name__ == "__main__":
    main()
