"""Run with: python tests/test_detect_sheets.py   (also collectable by pytest). Does not need Ultralytics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detect_sheets import merge, score_sheet, to_sheet  # noqa: E402

W, H = 7168, 4562


def test_boxes_cut_by_inner_tile_edges_are_dropped_but_sheet_edges_are_kept():
    rows = [(5, 0.9, 0, 100, 40, 140), (5, 0.9, 500, 100, 540, 140), (5, 0.9, 990, 100, 1024, 140)]
    kept = to_sheet(rows, 768, 0, W, H)                     # interior tile: left and right edges are inner
    assert [p["box"] for p in kept] == [[1268, 100, 1308, 140]]
    kept = to_sheet(rows, 0, 0, W, H)                       # first tile: its left edge is the sheet edge
    assert [p["box"][0] for p in kept] == [0, 500]
    assert kept[0]["cls"] == 5 and kept[0]["score"] == 0.9


def test_merge_keeps_best_box_per_class_and_distinct_classes():
    a = {"cls": 1, "score": 0.9, "box": [0, 0, 50, 50]}
    b = {"cls": 1, "score": 0.6, "box": [2, 2, 52, 52]}      # same symbol seen from a neighbouring tile
    c = {"cls": 2, "score": 0.5, "box": [2, 2, 52, 52]}      # different class: not suppressed here
    assert merge([b, a, c]) == [a, c]


def test_scoring_counts_tp_fp_fn_and_wrong_class():
    truth = [{"cls": 1, "box": [0, 0, 50, 50]}, {"cls": 2, "box": [100, 0, 150, 50]}, {"cls": 3, "box": [200, 0, 250, 50]}]
    preds = [{"cls": 1, "score": 0.9, "box": [1, 1, 51, 51]},      # TP
             {"cls": 4, "score": 0.8, "box": [100, 0, 150, 50]},   # FP sitting on a class-2 symbol
             {"cls": 3, "score": 0.1, "box": [200, 0, 250, 50]}]   # below threshold: ignored, so class 3 is a FN
    tally, wrong = score_sheet(preds, truth, thr=0.25)
    assert tally[(1, "tp")] == 1 and tally[(4, "fp")] == 1 and wrong[4] == 1
    assert tally[(2, "fn")] == 1 and tally[(3, "fn")] == 1 and sum(tally.values()) == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
