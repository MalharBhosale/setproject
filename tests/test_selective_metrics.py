"""Run with: python tests/test_selective_metrics.py   (also collectable by pytest)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from selective_metrics import aurc  # noqa: E402
from compare_runs import summary  # noqa: E402


def test_aurc_perfect_ranking_beats_reversed_ranking():
    good = [(0.9, True), (0.8, True), (0.1, False)]            # the error is ranked last
    bad = [(0.9, False), (0.8, True), (0.1, True)]             # the error is ranked first
    assert abs(aurc(good) - (0 + 0 + 1 / 3) / 3) < 1e-12
    assert abs(aurc(bad) - (1 + 1 / 2 + 1 / 3) / 3) < 1e-12
    assert aurc(good) < aurc(bad)


def test_aurc_spreads_errors_evenly_inside_ties():
    tied = [(math.inf, True), (0.5, False), (0.5, True)]
    # coverage 1/3: risk 0; then the tie enters with 0.5 error per item: 0.5/2, then 1/3
    assert abs(aurc(tied) - (0 + 0.25 + 1 / 3) / 3) < 1e-12


def test_compare_summary_counts_only_automatic_decisions_for_risk():
    rows = [(0.9, True, "FAIL", "FAIL"), (0.5, False, "FAIL", "PASS"), (-1.0, True, "REVIEW", "FAIL"), (2.0, True, "PASS", "PASS")]
    s = summary(rows)
    assert s["coverage"] == 0.75 and abs(s["risk"] - 1 / 3) < 1e-12 and 0 < s["aurc"] < 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
