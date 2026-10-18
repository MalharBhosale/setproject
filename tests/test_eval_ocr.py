"""Run with: python tests/test_eval_ocr.py   (also collectable by pytest). Does not need any OCR engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_ocr import best_reading, edit_distance, summarise, tag_words  # noqa: E402


def test_edit_distance():
    assert edit_distance("KL-81809", "KL-81809") == 0
    assert edit_distance("KL-81809", "KL-8I809") == 1
    assert edit_distance("SDL 472", "SDL472") == 1
    assert edit_distance("", "AB") == 2


def test_best_reading_prefers_confidence_and_handles_nothing():
    assert best_reading([("8ı-7X", 0.3), ("KL-81809", 0.9)]) == ("KL-81809", 0.9)
    assert best_reading([]) == ("", 0.0)


def test_tag_words_only_identifier_classes_in_order():
    sheet = {"symbols": [{"id": "s1", "cls": 28}, {"id": "s2", "cls": 19}],
             "words": [{"id": "w1", "text": "SDL"}, {"id": "w2", "text": "472"}, {"id": "w3", "text": '6"'}],
             "links": {"s1": {"words": ["w1", "w2"]}, "s2": {"words": ["w3"]}}}
    assert [(s, k, w["text"]) for s, k, w in tag_words(sheet)] == [("s1", 0, "SDL"), ("s1", 1, "472")]


def test_summary_scores_tags_as_wholes():
    rows = [{"sheet": "1", "symbol": "s1", "k": 0, "truth": "SDL", "pred": "SDL", "rotated": 0},
            {"sheet": "1", "symbol": "s1", "k": 1, "truth": "472", "pred": "472 ", "rotated": 0},
            {"sheet": "1", "symbol": "s2", "k": 0, "truth": "KL-81809", "pred": "kl-81809", "rotated": 90}]
    s = summarise(rows, seconds=0.3)
    assert s["tags"] == 2 and s["tag_exact"] == 0.5 and s["tag_ignoring_case_and_spaces"] == 1.0
    assert s["rotated_words"] == 1 and s["rotated_word_exact"] == 0.0 and abs(s["ms_per_word"] - 100) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
