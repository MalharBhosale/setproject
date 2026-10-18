"""Confidence-aware PASS / REVIEW / FAIL decision on top of the deterministic verifier (tests H2).

Usage: python src/decide.py <extracted_dir> <out_dir> [part]      part defaults to "val"
<extracted_dir> holds <sheet>.json from src/pipeline.py. Sweeps the OCR-confidence threshold tau, writes the
risk-coverage table and figure, and reports the tau chosen by the rule fixed in advance (MAX_RISK).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import UNTAGGED  # noqa: E402
from verify import finding_key, truth_key, verify  # noqa: E402

DET_MIN = 0.5    # detector score below this makes a component unreliable regardless of OCR
MAX_RISK = 0.05  # chosen tau: most cases decided automatically with at most 5% wrong automatic decisions
TAUS = [round(i * 0.02, 2) for i in range(51)]


def shape(text):
    """Letters to A, digits to 9, everything else kept: AB-12345 -> AA-99999. Ignores which letters, so it cannot reveal the class."""
    return re.sub(r"[A-Z]", "A", re.sub(r"[0-9]", "9", text))


def tag_shapes(sheet_ids):
    """Shapes each class's text takes, learned from the given (training) sheets only."""
    shapes = {}
    for sid in sheet_ids:
        sheet = json.loads(Path(f"data/annotations/{sid}.json").read_text(encoding="utf-8"))
        words = {w["id"]: w["text"] for w in sheet["words"]}
        for s in sheet["symbols"]:
            shapes.setdefault(s["cls"], set()).add(shape(" ".join(words[i] for i in sheet["links"][s["id"]]["words"])))
    return shapes


def reliable(c, tau, shapes=None):
    """Trusted when detected confidently and, for text-bearing classes, read confidently in a plausible format."""
    if c["det_score"] < DET_MIN:
        return False
    if c["cls"] in UNTAGGED:
        return True
    return c["ocr_conf"] >= tau and (shapes is None or shape(c["tag"]) in shapes.get(c["cls"], ()))


def decide(comps, bom, tau, shapes=None):
    """Return (verdict, findings), each finding marked 'supported' when no unreliable component could explain it."""
    findings = verify(comps, bom)
    shaky_classes = {c["cls"] for c in comps if not reliable(c, tau, shapes)}
    for f in findings:
        if f["type"] in ("EXTRA", "TAG", "TYPE"):
            f["supported"] = reliable(f["drawing"], tau, shapes)
        elif f["type"] == "QUANTITY":
            f["supported"] = f["bom"]["cls"] not in shaky_classes
    explaining = {f["drawing"]["cls"] for f in findings if f["type"] in ("EXTRA", "TAG", "TYPE") and not f["supported"]}
    for f in findings:
        if f["type"] == "MISSING":  # a misread component of the same class may be the row the BOM lists
            f["supported"] = f["bom"]["cls"] not in explaining
    if any(f["supported"] for f in findings):
        return "FAIL", findings
    return ("REVIEW" if findings else "PASS"), findings


def sweep(cases, comps_for, taus=TAUS, shapes=None):
    rows = []
    for tau in taus:
        table, sup = Counter(), Counter()
        for case in cases:
            bom = json.loads(Path(f"data/benchmark/boms/{case['case']}.json").read_text(encoding="utf-8"))
            verdict, findings = decide(comps_for(case["sheet"]), bom, tau, shapes)
            table[(case["verdict"], verdict)] += 1
            truth = Counter(truth_key(d) for d in case["discrepancies"])
            found = Counter(finding_key(f) for f in findings if f["supported"])
            sup["tp"] += sum((found & truth).values())
            sup["fp"] += sum((found - truth).values())
            sup["true"] += sum(truth.values())
        auto = sum(n for (t, v), n in table.items() if v != "REVIEW")
        wrong = sum(n for (t, v), n in table.items() if v != "REVIEW" and v != t)
        rows.append({"tau": tau, "coverage": auto / len(cases), "risk": wrong / auto if auto else 0.0,
                     "table": {f"{t}->{v}": n for (t, v), n in sorted(table.items())},
                     "supported_finding_precision": sup["tp"] / max(1, sup["tp"] + sup["fp"]),
                     "supported_finding_recall": sup["tp"] / max(1, sup["true"])})
    return rows


def choose(rows, max_risk=MAX_RISK):
    ok = [r for r in rows if r["coverage"] > 0 and r["risk"] <= max_risk]
    return max(ok, key=lambda r: (r["coverage"], -r["tau"])) if ok else None


def main():
    ext, out = Path(sys.argv[1]), Path(sys.argv[2])
    part = sys.argv[3] if len(sys.argv) > 3 else "val"
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in json.loads(Path("docs/benchmark_manifest.json").read_text(encoding="utf-8"))["cases"] if c["part"] == part]
    comps = {s: json.loads((ext / f"{s}.json").read_text(encoding="utf-8")) for s in {c["sheet"] for c in cases}}
    shapes = tag_shapes(json.loads(Path("docs/split.json").read_text(encoding="utf-8"))["train"])
    for name, sh in (("confidence only", None), ("confidence + format", shapes)):
        rows = sweep(cases, comps.__getitem__, shapes=sh)
        best = choose(rows)
        report(name, part, rows, best, out)


def report(name, part, rows, best, out):
    tag = name.replace(" + ", "_").replace(" ", "_")
    (out / f"sweep_{part}_{tag}.json").write_text(json.dumps({"rows": rows, "chosen": best, "max_risk": MAX_RISK}, indent=1), encoding="utf-8")
    print(f"=== {name} ===")
    for r in rows[::5] + ([best] if best else []):
        print(f"tau {r['tau']:.2f} coverage {r['coverage']:.3f} risk {r['risk']:.3f} "
              f"finding P {r['supported_finding_precision']:.3f} R {r['supported_finding_recall']:.3f} {r['table']}")
    print("chosen:", best and best["tau"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([r["coverage"] for r in rows], [r["risk"] for r in rows], marker="o", ms=3)
    ax.axhline(MAX_RISK, ls="--", lw=1, color="grey")
    ax.set_xlabel("coverage (share of cases decided automatically)")
    ax.set_ylabel("risk (share of automatic decisions that are wrong)")
    ax.set_title(f"Risk-coverage, {part} cases: {name}")
    fig.tight_layout()
    fig.savefig(out / f"risk_coverage_{part}_{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
