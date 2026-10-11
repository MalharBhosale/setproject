"""Convert Dataset-P&ID .npy annotations (data/raw/<sheet>/) to one JSON per sheet (data/annotations/<sheet>.json).

Usage: python src/convert_annotations.py
Prints an integrity report for the whole dataset. The JSON is the only form the rest of the project reads,
so the pickled .npy files never need loading again.
"""
import json
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

RAW = Path("data/raw")
OUT = Path("data/annotations")
PARTS = ["symbols", "words", "lines", "linker", "KeyValue", "Table", "lines2"]
N_CLASSES = 32
# The dataset's object arrays are pickles; allow only what NumPy itself needs to rebuild an array.
ALLOWED_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def safe_load(path):
    """Load an .npy file without letting a pickled payload import anything but NumPy array builders."""
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        _, _, dtype = read_header(f)
        if not dtype.hasobject:
            return np.load(path, allow_pickle=False).tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # old files name numpy.core, renamed in NumPy 2
            return _NumpyOnlyUnpickler(f).load().tolist()


def _ids(ref):
    """Linker word references come as 'word_1', 'word_1,word_2', a list, or empty."""
    if ref is None:
        return []
    parts = ref if isinstance(ref, (list, tuple)) else str(ref).split(",")
    return [str(p).strip() for p in parts if str(p).strip()]


def convert_sheet(folder, sheet):
    raw = {p: safe_load(Path(folder) / f"{sheet}_{p}.npy") for p in PARTS}
    links = {}
    for row in raw["linker"]:
        sid, ref = row[0], row[1]
        word_ref, line_ref = (list(ref) + [None, None])[:2] if isinstance(ref, (list, tuple)) else (ref, None)
        links[sid] = {"words": _ids(word_ref), "line": line_ref or None}
    return {
        "sheet": str(sheet),
        # class is stored as int in some rows and str in others for the same glyph
        "symbols": [{"id": s[0], "box": [int(v) for v in s[1]], "cls": int(s[2])} for s in raw["symbols"]],
        # word boxes are stored [x1, y_bottom, x2, y_top]; normalise to [x1, y1, x2, y2]
        "words": [
            {"id": w[0], "box": [min(w[1][0], w[1][2]), min(w[1][1], w[1][3]), max(w[1][0], w[1][2]), max(w[1][1], w[1][3])],
             "text": str(w[2]), "rotation": int(w[3])}
            for w in raw["words"]
        ],
        "lines": [{"id": l[0], "points": [int(v) for v in l[1]], "label": str(l[2] or ""), "style": str(l[3])} for l in raw["lines"]],
        "links": links,
        "title_block": {str(k): str(v) for k, v in raw["KeyValue"]},
        "revisions": raw["Table"],
        "lines2": raw["lines2"],  # meaning not yet established; kept verbatim
    }


def bom(sheet):
    """Ground-truth bill of materials: one record per symbol, built from annotations only."""
    words = {w["id"]: w["text"] for w in sheet["words"]}
    rows = []
    for s in sheet["symbols"]:
        link = sheet["links"].get(s["id"], {"words": [], "line": None})
        rows.append({"symbol": s["id"], "cls": s["cls"], "tag": " ".join(words[w] for w in link["words"] if w in words),
                     "line": link["line"]})
    return rows


def check(sheet):
    """Return a list of structural problems; empty means the sheet matches the expected schema."""
    problems = []
    sym = [s["id"] for s in sheet["symbols"]]
    words = {w["id"] for w in sheet["words"]}
    lines = {l["id"] for l in sheet["lines"]}
    for name, ids in (("symbol", sym), ("word", [w["id"] for w in sheet["words"]]), ("line", [l["id"] for l in sheet["lines"]])):
        if len(set(ids)) != len(ids):
            problems.append(f"duplicate {name} ids")
    if set(sheet["links"]) != set(sym):
        problems.append("linker rows do not match symbols one-to-one")
    for sid, link in sheet["links"].items():
        if any(w not in words for w in link["words"]):
            problems.append(f"{sid}: linked word missing")
        if link["line"] and link["line"] not in lines:
            problems.append(f"{sid}: linked line missing")
    for s in sheet["symbols"]:
        x1, y1, x2, y2 = s["box"]
        if not 1 <= s["cls"] <= N_CLASSES:
            problems.append(f"{s['id']}: class {s['cls']} outside 1..{N_CLASSES}")
        if x2 <= x1 or y2 <= y1:
            problems.append(f"{s['id']}: empty box")
    return problems


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folders = sorted((d for d in RAW.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
    failed, bad = {}, {}
    totals, classes, rotations, styles = Counter(), Counter(), Counter(), Counter()
    sheets_per_class, per_sheet_symbols, title_names = Counter(), [], Counter()
    for d in folders:
        try:
            sheet = convert_sheet(d, d.name)
        except Exception as e:
            failed[d.name] = f"{type(e).__name__}: {e}"
            continue
        (OUT / f"{d.name}.json").write_text(json.dumps(sheet, ensure_ascii=False), encoding="utf-8")
        problems = check(sheet)
        if problems:
            bad[d.name] = problems
        rows = bom(sheet)
        totals.update(sheets=1, symbols=len(rows), words=len(sheet["words"]), lines=len(sheet["lines"]),
                      untagged=sum(not r["tag"] for r in rows), unlinked_line=sum(not r["line"] for r in rows),
                      labelled_lines=sum(bool(l["label"].strip()) for l in sheet["lines"]),
                      tiny_word_boxes=sum((w["box"][2] - w["box"][0]) < 3 or (w["box"][3] - w["box"][1]) < 3 for w in sheet["words"]),
                      duplicate_tags=sum(c - 1 for t, c in Counter(r["tag"] for r in rows if r["tag"]).items() if c > 1))
        classes.update(r["cls"] for r in rows)
        sheets_per_class.update({r["cls"] for r in rows})
        rotations.update(w["rotation"] for w in sheet["words"])
        styles.update(l["style"] for l in sheet["lines"])
        per_sheet_symbols.append(len(rows))
        title_names[sheet["title_block"].get("DRAWING NAME", "")] += 1

    n = totals["sheets"]
    print(f"folders found: {len(folders)}  converted: {n}  failed to load: {len(failed)}  with structural problems: {len(bad)}")
    if not n:
        return 1
    print(f"symbols: {totals['symbols']}  per sheet min/mean/max: {min(per_sheet_symbols)}/{totals['symbols'] / n:.1f}/{max(per_sheet_symbols)}")
    print(f"untagged symbols: {totals['untagged']} ({100 * totals['untagged'] / totals['symbols']:.1f}%)"
          f"  symbols without a line: {totals['unlinked_line']}  repeated tags within a sheet: {totals['duplicate_tags']}")
    print(f"words: {totals['words']}  rotations: {dict(rotations)}  near-zero boxes: {totals['tiny_word_boxes']}")
    print(f"lines: {totals['lines']}  styles: {dict(styles)}  with label: {totals['labelled_lines']}")
    print(f"distinct DRAWING NAME values: {len(title_names)}  repeated: {sum(c > 1 for c in title_names.values())}")
    print("class: instances / sheets containing it")
    for c in range(1, N_CLASSES + 1):
        print(f"  {c:2}: {classes[c]:5} / {sheets_per_class[c]:3}")
    for name, why in list(failed.items())[:20]:
        print(f"FAILED sheet {name}: {why}")
    for name, problems in list(bad.items())[:20]:
        print(f"PROBLEM sheet {name}: {problems[:3]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
