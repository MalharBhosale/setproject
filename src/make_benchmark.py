"""Generate the paired drawing-BOM benchmark defined in docs/benchmark_spec.md.

Usage: python src/make_benchmark.py
Reads data/annotations/ and docs/split.json. Writes one BOM per case to data/benchmark/boms/ and the
ground truth for every case to docs/benchmark_manifest.json. Drawings are never modified.
All discrepancies are SYNTHETIC edits to the BOM.
"""
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_annotations import bom as symbol_rows  # noqa: E402

SEED = 20261011
IDENTIFIER_CLASSES = frozenset(range(1, 17)) | {25, 26, 27, 28, 29, 31, 32}
TYPES = ("MISSING", "EXTRA", "TAG", "TYPE", "QUANTITY")
SEEDED_SIZES = (1, 3, 5)
ANNOTATIONS, SPLIT = Path("data/annotations"), Path("docs/split.json")
BOM_DIR, MANIFEST = Path("data/benchmark/boms"), Path("docs/benchmark_manifest.json")


def clean_bom(sheet):
    """Return (bom, evidence). evidence[i] is the drawing symbol behind bom['tagged'][i]; it is ground truth, not part of the BOM."""
    boxes = {s["id"]: s["box"] for s in sheet["symbols"]}
    rows = sorted((r for r in symbol_rows(sheet) if r["cls"] in IDENTIFIER_CLASSES), key=lambda r: (r["cls"], r["tag"], r["symbol"]))
    counts = Counter((r["cls"], r["tag"]) for r in symbol_rows(sheet) if r["cls"] not in IDENTIFIER_CLASSES)
    bom = {
        "sheet": sheet["sheet"],
        "tagged": [{"cls": r["cls"], "tag": r["tag"]} for r in rows],
        "counted": [{"cls": c, "text": t, "quantity": q} for (c, t), q in sorted(counts.items())],
    }
    return bom, [{"symbol": r["symbol"], "box": boxes[r["symbol"]]} for r in rows]


def _redigit(tag, rng, positions=None):
    """Replace digits of a tag with different digits; all of them, or only the given character positions."""
    chars = list(tag)
    for i in (positions if positions is not None else [i for i, c in enumerate(chars) if c.isdigit()]):
        chars[i] = rng.choice([d for d in "0123456789" if d != chars[i]])
    return "".join(chars)


def seed_discrepancies(bom, evidence, n, rng):
    """Apply n independent edits to a copy of a clean BOM. Returns (seeded_bom, discrepancies)."""
    out = copy.deepcopy(bom)
    tag_counts = Counter(r["tag"] for r in bom["tagged"])
    used = set(tag_counts)
    # rows with a repeated tag have no unambiguous evidence symbol, so they are never edited
    free_tagged = [i for i, r in enumerate(bom["tagged"]) if tag_counts[r["tag"]] == 1]
    free_counted = list(range(len(bom["counted"])))
    rng.shuffle(free_tagged)
    rng.shuffle(free_counted)
    delete, add, found = set(), [], []
    for _ in range(n):
        kind = rng.choice([t for t in TYPES if (free_counted if t == "QUANTITY" else free_tagged)])
        if kind == "QUANTITY":
            row = out["counted"][free_counted.pop()]
            before = dict(row)
            row["quantity"] += rng.choice([d for d in (-3, -2, -1, 1, 2, 3) if row["quantity"] + d >= 0])
            found.append({"type": kind, "before": before, "after": dict(row)})
            continue
        i = free_tagged.pop()
        row, before = out["tagged"][i], dict(out["tagged"][i])
        if kind == "MISSING":  # the popped row is only a format template; it stays in the BOM unchanged
            new = _redigit(row["tag"], rng)
            while new in used:
                new = _redigit(row["tag"], rng)
            used.add(new)
            add.append({"cls": row["cls"], "tag": new})
            found.append({"type": kind, "before": None, "after": add[-1]})
            continue
        if kind == "EXTRA":
            delete.add(i)
        elif kind == "TAG":
            digits = [k for k, c in enumerate(row["tag"]) if c.isdigit()]
            new = row["tag"]
            while new in used:
                new = _redigit(row["tag"], rng, rng.sample(digits, rng.choice((1, 2))))
            used.add(new)
            row["tag"] = new
        else:  # TYPE
            row["cls"] = rng.choice(sorted(IDENTIFIER_CLASSES - {row["cls"]}))
        found.append({"type": kind, "before": before, "after": None if kind == "EXTRA" else dict(row), **evidence[i]})
    out["tagged"] = sorted([r for i, r in enumerate(out["tagged"]) if i not in delete] + add, key=lambda r: (r["cls"], r["tag"]))
    return out, found


def cases_for_sheet(sheet, seed=SEED):
    """Yield (case_id, bom, discrepancies): one clean case, then one per SEEDED_SIZES."""
    bom, evidence = clean_bom(sheet)
    yield f"{sheet['sheet']}_clean", bom, []
    rng = random.Random(f"{seed}-{sheet['sheet']}")  # per-sheet stream: independent of processing order
    for n in SEEDED_SIZES:
        seeded, found = seed_discrepancies(bom, evidence, n, rng)
        yield f"{sheet['sheet']}_d{n}", seeded, found


def main():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    part_of = {s: part for part in ("train", "val", "test") for s in split[part]}
    BOM_DIR.mkdir(parents=True, exist_ok=True)
    cases, per_part, per_type = [], Counter(), Counter()
    for sid in sorted(part_of, key=int):
        sheet = json.loads((ANNOTATIONS / f"{sid}.json").read_text(encoding="utf-8"))
        for case_id, bom, found in cases_for_sheet(sheet):
            (BOM_DIR / f"{case_id}.json").write_text(json.dumps(bom, ensure_ascii=False), encoding="utf-8")
            cases.append({"case": case_id, "sheet": sid, "part": part_of[sid], "synthetic": True,
                          "verdict": "FAIL" if found else "PASS", "discrepancies": found})
            per_part[part_of[sid]] += 1
            per_type.update((part_of[sid], d["type"]) for d in found)
    MANIFEST.write_text(json.dumps({"seed": SEED, "spec": "docs/benchmark_spec.md", "cases": cases}, ensure_ascii=False, indent=0),
                        encoding="utf-8")
    print("cases:", len(cases), dict(per_part), " PASS:", sum(c["verdict"] == "PASS" for c in cases))
    for part in ("train", "val", "test"):
        print(f"  {part:5}", {t: per_type[(part, t)] for t in TYPES})


if __name__ == "__main__":
    main()
