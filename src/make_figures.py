"""Paper figure: risk-coverage curves for both decision rules on validation and test, from src/decide.py sweep files.

Usage: python src/make_figures.py      writes docs/paper/figures/risk_coverage.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SWEEPS, OUT = Path("data/runs/decide_v1"), Path("docs/paper/figures")
FROZEN_TAU, TARGET = 0.98, 0.05


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    styles = {("val", "confidence_only"): ("#9aa5b1", "--", "Rule A, validation"), ("test", "confidence_only"): ("#5b6b7c", "-", "Rule A, test"),
              ("val", "confidence_format"): ("#7fb3c9", "--", "Rule B, validation"), ("test", "confidence_format"): ("#1c7293", "-", "Rule B, test")}
    for (part, rule), (color, ls, label) in styles.items():
        rows = json.loads((SWEEPS / f"sweep_{part}_{rule}.json").read_text(encoding="utf-8"))["rows"]
        ax.plot([r["coverage"] for r in rows], [r["risk"] for r in rows], ls=ls, color=color, marker="o", ms=2.5, lw=1.4, label=label)
        if rule == "confidence_format":
            r = next(r for r in rows if abs(r["tau"] - FROZEN_TAU) < 1e-9)
            ax.plot(r["coverage"], r["risk"], marker="*", ms=12, color=color, zorder=5)
            ax.annotate(f"tau {FROZEN_TAU}: {r['coverage']:.0%} covered, {r['risk']:.1%} wrong", (r["coverage"], r["risk"]),
                        textcoords="offset points", xytext=(8, -12 if part == "val" else 6), fontsize=8, color=color)
    ax.axhline(TARGET, color="#b83227", lw=0.8, ls=":")
    ax.text(0.16, TARGET + 0.004, "5% target", fontsize=8, color="#b83227")
    ax.plot(1.0, 0.25, marker="s", ms=6, color="black")
    ax.annotate("no abstention: 25% wrong", (1.0, 0.25), textcoords="offset points", xytext=(-120, 6), fontsize=8)
    ax.set_xlabel("Coverage: share of cases decided automatically")
    ax.set_ylabel("Risk: share of automatic decisions that are wrong")
    ax.set_xlim(0.1, 1.04)
    ax.set_ylim(-0.01, 0.27)
    ax.legend(fontsize=8, loc="center left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=200)
    print("wrote", OUT / "risk_coverage.png")


if __name__ == "__main__":
    main()
