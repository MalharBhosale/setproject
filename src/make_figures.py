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


def pipeline():
    """Block diagram of the method: two inputs, extraction, verifier, decision, three outcomes."""
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3)
    box = dict(boxstyle="round,pad=0.35", fc="#eef4f8", ec="#1c7293", lw=1.1)
    blocks = {"img": (0.6, 2.2, "Raster P&ID\n(sheet image)"), "bom": (0.6, 0.8, "Bill of materials\n(independent list)"),
              "det": (2.35, 2.2, "Tiled YOLO11n\nsymbol detection"), "ocr": (4.05, 2.2, "Text region rules\n+ OCR reading"),
              "ver": (4.05, 0.8, "Deterministic verifier\n(4 matching passes)"), "dec": (6.0, 1.5, "Confidence-aware decision\ntrusted = det >= 0.5 and\nOCR conf >= tau and\ntag format seen in training")}
    for x, y, text in blocks.values():
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, bbox=box)
    arrow = dict(arrowstyle="-|>", color="#33475b", lw=1.1)
    for a, b in [((1.4, 2.2), (1.6, 2.2)), ((3.1, 2.2), (3.25, 2.2)), ((4.05, 1.85), (4.05, 1.2)), ((1.45, 0.8), (3.0, 0.8)),
                 ((5.1, 0.8), (4.95, 1.2)), ((4.85, 2.2), (4.95, 1.85))]:
        ax.annotate("", b, a, arrowprops=arrow)
    ax.text(4.12, 1.52, "components", fontsize=7.5, color="#33475b", va="center")
    ax.text(5.15, 0.68, "findings", fontsize=7.5, color="#33475b")
    ax.text(5.0, 2.3, "readings, confidences", fontsize=7.5, color="#33475b")
    for y, label, color in [(2.4, "FAIL: a supported discrepancy", "#b83227"), (1.5, "REVIEW: findings, none supported", "#c58b00"),
                            (0.6, "PASS: no findings", "#2e7d32")]:
        ax.annotate("", (7.45, y), (7.05, 1.5), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1))
        ax.text(7.5, y, label, fontsize=8.5, color=color, va="center", ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=200)
    print("wrote", OUT / "pipeline.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
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
