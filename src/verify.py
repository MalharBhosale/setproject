"""Deterministic drawing-versus-BOM verifier (no confidence handling yet).

verify(components, bom) compares components extracted from a drawing with a BOM and returns findings.
Run as a script, it scores the verifier on PERFECT inputs (ground-truth components) for every benchmark case:
    python src/verify.py
That is an upper bound and a check of the rules and the benchmark, not a result about any model.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_annotations import bom as symbol_rows  # noqa: E402
from make_benchmark import IDENTIFIER_CLASSES, TYPES  # noqa: E402

MAX_TAG_EDITS = 2  # a same-class pair of unmatched tags this close is reported as one TAG finding, not MISSING + EXTRA


def _distance(a, b):
    return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else MAX_TAG_EDITS + 1


def verify(components, bom):
    """components: [{'cls', 'tag', ...}] for every symbol on the drawing. Returns a list of findings.

    Each finding has 'type', 'drawing' (what the drawing shows, or None) and 'bom' (what the BOM says, or None).
    """
    findings = []
    drawn = [c for c in components if c["cls"] in IDENTIFIER_CLASSES]
    listed = list(bom["tagged"])

    def pair_off(matches, kind):
        """matches(component, row) -> sort key or None. Pair best-first, one-to-one, and remove paired items."""
        nonlocal drawn, listed
        pairs = sorted(((k, i, j) for i, c in enumerate(drawn) for j, r in enumerate(listed) if (k := matches(c, r)) is not None))
        used_c, used_r = set(), set()
        for _, i, j in pairs:
            if i in used_c or j in used_r:
                continue
            used_c.add(i)
            used_r.add(j)
            if kind:
                findings.append({"type": kind, "drawing": drawn[i], "bom": listed[j]})
        drawn = [c for i, c in enumerate(drawn) if i not in used_c]
        listed = [r for j, r in enumerate(listed) if j not in used_r]

    pair_off(lambda c, r: 0 if (c["cls"], c["tag"]) == (r["cls"], r["tag"]) else None, None)
    pair_off(lambda c, r: 0 if c["tag"] == r["tag"] else None, "TYPE")
    pair_off(lambda c, r: d if c["cls"] == r["cls"] and (d := _distance(c["tag"], r["tag"])) <= MAX_TAG_EDITS else None, "TAG")
    findings += [{"type": "EXTRA", "drawing": c, "bom": None} for c in drawn]
    findings += [{"type": "MISSING", "drawing": None, "bom": r} for r in listed]

    seen = Counter((c["cls"], c["tag"]) for c in components if c["cls"] not in IDENTIFIER_CLASSES)
    expected = {(r["cls"], r["text"]): r["quantity"] for r in bom["counted"]}
    for cls, text in sorted(set(seen) | set(expected)):
        if seen[(cls, text)] != expected.get((cls, text), 0):
            findings.append({"type": "QUANTITY", "drawing": {"cls": cls, "text": text, "quantity": seen[(cls, text)]},
                             "bom": {"cls": cls, "text": text, "quantity": expected.get((cls, text), 0)}})
    return findings


def finding_key(f):
    """Comparable identity of a finding: its type plus what the drawing shows and what the BOM says."""
    side = lambda x: None if x is None else (x["cls"], x.get("tag", x.get("text")), x.get("quantity"))
    return f["type"], side(f["drawing"]), side(f["bom"])


def truth_key(d):
    """The same identity for a manifest discrepancy ('before' is the drawing side, 'after' the BOM side)."""
    return finding_key({"type": d["type"], "drawing": d["before"], "bom": d["after"]})


def score_cases(cases, components_for):
    """Run the verifier on each case. components_for(sheet_id) gives the drawing's components.

    Returns per-type tp/fp/fn, a verdict table {part: {(true verdict, reported verdict): n}}, and the mismatched cases.
    """
    score, verdicts, wrong = defaultdict(Counter), defaultdict(Counter), []
    for case in cases:
        bom = json.loads(Path(f"data/benchmark/boms/{case['case']}.json").read_text(encoding="utf-8"))
        found = Counter(finding_key(f) for f in verify(components_for(case["sheet"]), bom))
        truth = Counter(truth_key(d) for d in case["discrepancies"])
        for key, n in (found & truth).items():
            score[key[0]]["tp"] += n
        for key, n in (found - truth).items():
            score[key[0]]["fp"] += n
        for key, n in (truth - found).items():
            score[key[0]]["fn"] += n
        verdicts[case["part"]][(case["verdict"], "FAIL" if found else "PASS")] += 1
        if found != truth:
            wrong.append((case["case"], sorted(map(str, found - truth)), sorted(map(str, truth - found))))
    return score, verdicts, wrong


def main():
    cases = json.loads(Path("docs/benchmark_manifest.json").read_text(encoding="utf-8"))["cases"]
    sheets = {}

    def truth_components(sid):
        if sid not in sheets:
            sheets[sid] = symbol_rows(json.loads(Path(f"data/annotations/{sid}.json").read_text(encoding="utf-8")))
        return sheets[sid]

    score, verdicts, wrong = score_cases(cases, truth_components)
    print("ORACLE RUN: ground-truth components, no detector, no OCR. Upper bound only.")
    print(f"cases: {len(cases)}  cases with any wrong finding: {len(wrong)}")
    for part in ("train", "val", "test"):
        v = verdicts[part]
        print(f"  {part:5} verdict correct: {v[('PASS', 'PASS')] + v[('FAIL', 'FAIL')]}/{sum(v.values())}")
    for t in TYPES:
        print(f"  {t:8} tp={score[t]['tp']:5} fp={score[t]['fp']:3} fn={score[t]['fn']:3}")
    for name, extra, missed in wrong[:10]:
        print(f"  WRONG {name}: reported-not-true={extra} true-not-reported={missed}")
    return 1 if wrong else 0

if __name__ == "__main__":
    sys.exit(main())
