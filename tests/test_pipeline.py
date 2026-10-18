"""Run with: python tests/test_pipeline.py   (also collectable by pytest). Does not need EasyOCR."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import assign, choose_text, dedupe, gap, join_boxes, tag_accuracy  # noqa: E402

SYM = [100, 100, 150, 150]


def test_gap():
    assert gap(SYM, [160, 100, 200, 120]) == 10
    assert gap(SYM, [120, 120, 130, 130]) == 0


def test_outside_class_takes_nearest_text_within_margin():
    texts = [([160, 100, 230, 120], "AB-12345", 0.9),   # 10 px away
             ([100, 70, 170, 90], "CD-99999", 0.8),     # 10 px above, tie broken by order: first nearest wins
             ([400, 100, 470, 120], "EF-11111", 0.99)]  # beyond the 80 px margin
    tag, conf = choose_text(4, SYM, texts)
    assert tag == "AB-12345" and conf == 0.9
    assert choose_text(4, SYM, [texts[2]]) == ("", 0.0)


def test_inside_class_joins_words_top_to_bottom_and_ignores_outside_text():
    texts = [([110, 128, 140, 145], "472", 0.7), ([110, 105, 140, 122], "SDL", 0.95), ([160, 100, 230, 120], "AB-1", 0.9)]
    assert choose_text(28, SYM, texts) == ("SDL 472", 0.7)


def test_assign_gives_each_text_to_one_owner():
    symbols = [(4, [100, 100, 150, 150]), (19, [100, 200, 150, 250]), (28, [300, 100, 350, 150]), (21, [160, 160, 170, 170])]
    texts = [([100, 155, 140, 165], "AB-1", 0.9),     # 5 px below the valve, 35 px above the reducer: the valve owns it
             ([155, 215, 175, 235], '6"', 0.8),       # 5 px right of the reducer
             ([310, 110, 340, 125], "SDL", 0.9),      # inside the bubble
             ([600, 600, 650, 620], "FAR", 0.9)]      # nobody's
    owned = assign(symbols, texts)
    assert {i: [t[1] for t in ts] for i, ts in owned.items()} == {0: ["AB-1"], 1: ['6"'], 2: ["SDL"]}


def test_dedupe_keeps_most_confident_copy():
    a, b = ([0, 0, 50, 20], "AB-1", 0.6), ([1, 0, 51, 20], "AB-12", 0.9)
    assert dedupe([a, b, ([200, 0, 250, 20], "X", 0.5)]) == [b, ([200, 0, 250, 20], "X", 0.5)]


def test_join_boxes_merges_split_tags_but_not_separate_texts():
    split = [[100, 100, 140, 120], [143, 101, 190, 120]]               # 3 px gap on one line, height 20
    assert join_boxes(split) == [[100, 100, 190, 120]]
    apart = [[100, 100, 140, 120], [170, 100, 210, 120]]               # 30 px gap: separate texts
    assert join_boxes(apart) == apart
    stacked = [[100, 100, 140, 120], [100, 124, 140, 144]]             # bubble lines one above the other: not joined
    assert join_boxes(stacked) == stacked
    vertical = [[100, 100, 120, 160], [101, 163, 120, 220]]            # split vertical tag
    assert join_boxes(vertical) == [[100, 100, 120, 220]]


def test_untagged_class_needs_no_text():
    assert choose_text(21, SYM, []) == ("", 1.0)


def test_tag_accuracy_counts_exact_wrong_and_undetected():
    sheet = {"symbols": [{"id": "s1", "cls": 4, "box": SYM}, {"id": "s2", "cls": 28, "box": [300, 300, 350, 350]},
                         {"id": "s3", "cls": 21, "box": [500, 0, 520, 20]}, {"id": "s4", "cls": 4, "box": [600, 0, 650, 50]}],
             "words": [{"id": "w1", "text": "AB-12345"}, {"id": "w2", "text": "SDL"}, {"id": "w3", "text": "472"},
                       {"id": "w4", "text": ""}, {"id": "w5", "text": "GH-1"}],
             "links": {"s1": {"words": ["w1"]}, "s2": {"words": ["w2", "w3"]}, "s3": {"words": ["w4"]}, "s4": {"words": ["w5"]}}}
    comps = [{"cls": 4, "tag": "AB-12345", "box": [101, 101, 151, 151]}, {"cls": 28, "tag": "SDL 412", "box": [300, 300, 350, 350]}]
    assert tag_accuracy(comps, sheet) == {"exact": 1, "wrong": 1, "symbol not detected": 1}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
