"""95% sheet-level bootstrap intervals for coverage and risk at the frozen tau (rule B) and the no-abstention baseline.

Usage: python src/bootstrap_ci.py <part> <tau> [pipeline_dir]   e.g.  python src/bootstrap_ci.py test 0.98   (default data/runs/pipeline_v1)
Resamples whole sheets (2,000 times, seed 0) because the four cases on a sheet share one extraction.
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "src")
from decide import decide, tag_shapes

part, tau = sys.argv[1], float(sys.argv[2])
run = sys.argv[3] if len(sys.argv) > 3 else "data/runs/pipeline_v1"
cases = [c for c in json.load(open("docs/benchmark_manifest.json"))["cases"] if c["part"] == part]
shapes = tag_shapes(json.load(open("docs/split.json"))["train"])
by_sheet = defaultdict(list)
for c in cases:
    comps = json.load(open(f"{run}/extracted_{part}/{c['sheet']}.json", encoding="utf-8"))
    bom = json.load(open(f"data/benchmark/boms/{c['case']}.json", encoding="utf-8"))
    for name, t, sh in (("ruleB", tau, shapes), ("baseline", 0.0, None)):
        by_sheet[(name, c["sheet"])].append((c["verdict"], decide(comps, bom, t, sh)[0]))
sheets = sorted({c["sheet"] for c in cases})


def stats(results):
    auto = [(t, v) for t, v in results if v != "REVIEW"]
    return len(auto) / len(results), (sum(t != v for t, v in auto) / len(auto) if auto else 0.0)


rng = random.Random(0)
for name in ("baseline", "ruleB"):
    point = stats([r for s in sheets for r in by_sheet[(name, s)]])
    boots = sorted(stats([r for s in rng.choices(sheets, k=len(sheets)) for r in by_sheet[(name, s)]]) for _ in range(2000))
    cov = sorted(b[0] for b in boots); risk = sorted(b[1] for b in boots)
    q = lambda xs: (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1])
    print(f"{part} {name}: coverage {point[0]:.3f} [{q(cov)[0]:.3f}, {q(cov)[1]:.3f}]  risk {point[1]:.3f} [{q(risk)[0]:.3f}, {q(risk)[1]:.3f}]")
