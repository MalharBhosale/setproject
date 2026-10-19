"""Selective-prediction summaries for the decision rules: AURC, a paired rule A vs rule B comparison, and review workload.

Usage: python src/selective_metrics.py <extracted_dir> <part> [tau]      tau defaults to the frozen 0.98
Each case gets a score: the highest tau (from decide.TAUS) at which the rule still decides it automatically
(cases with no findings are always decided: score +inf; cases sent to REVIEW at every tau: score -inf, and when forced
they take the plain verifier's verdict). Ranking cases by score gives an exact risk-coverage curve; AURC is the mean
risk over all coverage levels (lower is better). Bootstrap resamples whole sheets (2,000 times, seed 0).
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decide import TAUS, decide, reliable, tag_shapes  # noqa: E402


def case_score(comps, bom, shapes):
    """(score, verdict when decided, plain-verifier verdict) for one case under one rule."""
    plain = decide(comps, bom, 0.0, None)[0]
    score, verdict = -math.inf, plain
    for tau in TAUS:
        v = decide(comps, bom, tau, shapes)[0]
        if v == "REVIEW":
            break
        score, verdict = tau, v
    if verdict == "PASS":
        score = math.inf
    return score, verdict


def aurc(items):
    """items: [(score, correct)]. Mean selective risk over coverage k/n, k = 1..n, with tied scores entering together."""
    items = sorted(items, key=lambda x: -x[0])
    n, risks, wrong, k = len(items), [], 0, 0
    while k < n:
        j = k
        while j < n and items[j][0] == items[k][0]:
            j += 1
        group_wrong = sum(not c for _, c in items[k:j])
        for m in range(1, j - k + 1):  # inside a tie, errors are spread evenly
            risks.append((wrong + group_wrong * m / (j - k)) / (k + m))
        wrong += group_wrong
        k = j
    return sum(risks) / n


def main():
    ext, part = Path(sys.argv[1]), sys.argv[2]
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.98
    shapes = tag_shapes(json.loads(Path("docs/split.json").read_text(encoding="utf-8"))["train"])
    cases = [c for c in json.loads(Path("docs/benchmark_manifest.json").read_text(encoding="utf-8"))["cases"] if c["part"] == part]
    rules = {"A_confidence": None, "B_confidence_format": shapes}
    per_sheet = defaultdict(lambda: defaultdict(list))  # rule -> sheet -> [(score, correct, auto_at_tau, verdict_at_tau, truth)]
    workload = []
    for c in cases:
        comps = json.loads((ext / f"{c['sheet']}.json").read_text(encoding="utf-8"))
        bom = json.loads(Path(f"data/benchmark/boms/{c['case']}.json").read_text(encoding="utf-8"))
        for name, sh in rules.items():
            score, verdict = case_score(comps, bom, sh)
            at_tau, findings = decide(comps, bom, tau, sh)
            per_sheet[name][c["sheet"]].append((score, verdict == c["verdict"], at_tau, c["verdict"]))
            if name == "B_confidence_format" and at_tau == "REVIEW":
                workload.append((sum(not reliable(x, tau, sh) for x in comps) / len(comps), len(findings)))
        fmt = decide(comps, bom, 0.0, shapes)[0]  # format-only = rule B with no confidence threshold
        per_sheet["format_only"][c["sheet"]].append((0.0, fmt == c["verdict"], fmt, c["verdict"]))
    sheets = sorted(per_sheet["A_confidence"])

    def summary(name, chosen):
        rows = [r for s in chosen for r in per_sheet[name][s]]
        auto = [r for r in rows if r[2] != "REVIEW"]
        out = {"coverage_at_tau": len(auto) / len(rows), "risk_at_tau": sum(r[2] != r[3] for r in auto) / max(1, len(auto)),
               "auto_pass_of_true_pass": sum(r[2] == "PASS" and r[3] == "PASS" for r in rows) / max(1, sum(r[3] == "PASS" for r in rows))}
        if name != "format_only":
            out["aurc"] = aurc([(r[0], r[1]) for r in rows])
        return out

    result = {name: summary(name, sheets) for name in ("A_confidence", "B_confidence_format", "format_only")}
    rng = random.Random(0)
    diffs_aurc, diffs_risk = [], []
    for _ in range(2000):
        pick = rng.choices(sheets, k=len(sheets))
        a, b = summary("A_confidence", pick), summary("B_confidence_format", pick)
        diffs_aurc.append(a["aurc"] - b["aurc"])
        diffs_risk.append(a["risk_at_tau"] - b["risk_at_tau"])
    q = lambda xs: [sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1]]
    result["paired_A_minus_B"] = {"aurc_diff": result["A_confidence"]["aurc"] - result["B_confidence_format"]["aurc"], "aurc_diff_ci95": q(diffs_aurc),
                                  "risk_diff_at_tau": result["A_confidence"]["risk_at_tau"] - result["B_confidence_format"]["risk_at_tau"],
                                  "risk_diff_ci95": q(diffs_risk)}
    if workload:
        share = sorted(w[0] for w in workload)
        result["review_workload_rule_B"] = {"review_cases": len(workload), "median_share_of_components_untrusted": share[len(share) // 2],
                                            "median_findings_to_check": sorted(w[1] for w in workload)[len(workload) // 2]}
    out = Path("data/runs/decide_v1") / f"selective_{part}_{ext.parent.name}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
