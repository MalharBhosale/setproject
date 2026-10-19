"""Paired comparison of two frozen pipeline runs under rule B, each at its own frozen tau (sheet-level bootstrap).

Usage: python src/compare_runs.py <extracted_dir_1> <tau_1> <extracted_dir_2> <tau_2> <part>
       e.g. python src/compare_runs.py data/runs/pipeline_v1/extracted_test 0.98 data/runs/pipeline_v2b/extracted_test 0.90 test
Reports coverage, risk and AURC for both runs and run 2 minus run 1 with 95% intervals (2,000 resamples of whole sheets, seed 0).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decide import decide, tag_shapes  # noqa: E402
from selective_metrics import aurc, case_score  # noqa: E402


def summary(rows):
    """rows: [(score, correct_when_forced, verdict_at_tau, truth)] -> coverage, risk at tau, AURC."""
    auto = [r for r in rows if r[2] != "REVIEW"]
    return {"coverage": len(auto) / len(rows), "risk": sum(r[2] != r[3] for r in auto) / max(1, len(auto)),
            "aurc": aurc([(r[0], r[1]) for r in rows])}


def main():
    runs = [(Path(sys.argv[1]), float(sys.argv[2])), (Path(sys.argv[3]), float(sys.argv[4]))]
    part = sys.argv[5]
    shapes = tag_shapes(json.loads(Path("docs/split.json").read_text(encoding="utf-8"))["train"])
    cases = [c for c in json.loads(Path("docs/benchmark_manifest.json").read_text(encoding="utf-8"))["cases"] if c["part"] == part]
    per_sheet = [defaultdict(list), defaultdict(list)]
    for c in cases:
        bom = json.loads(Path(f"data/benchmark/boms/{c['case']}.json").read_text(encoding="utf-8"))
        for k, (ext, tau) in enumerate(runs):
            comps = json.loads((ext / f"{c['sheet']}.json").read_text(encoding="utf-8"))
            score, verdict = case_score(comps, bom, shapes)
            per_sheet[k][c["sheet"]].append((score, verdict == c["verdict"], decide(comps, bom, tau, shapes)[0], c["verdict"]))
    sheets = sorted(per_sheet[0])
    pick_rows = lambda k, chosen: [r for s in chosen for r in per_sheet[k][s]]
    result = {"run1": summary(pick_rows(0, sheets)), "run2": summary(pick_rows(1, sheets))}
    rng, diffs = random.Random(0), defaultdict(list)
    for _ in range(2000):
        chosen = rng.choices(sheets, k=len(sheets))
        a, b = summary(pick_rows(0, chosen)), summary(pick_rows(1, chosen))
        for m in a:
            diffs[m].append(b[m] - a[m])
    q = lambda xs: [sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1]]
    result["run2_minus_run1"] = {m: {"diff": result["run2"][m] - result["run1"][m], "ci95": q(v)} for m, v in diffs.items()}
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
