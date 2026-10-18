"""Run with: python tests/test_attribute_errors.py   (also collectable by pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attribute_errors import classify  # noqa: E402


def test_classify():
    others = ["AB-55555", '5"-EK-2648']
    assert classify("KL-80483", "KL-80483", others) == "exact"
    assert classify("", "KL-80483", others) == "no text"
    assert classify("AB-55555", "KL-80483", others) == "association"      # a neighbour's tag
    assert classify('5"-EK-2648', "KL-80483", others) == "association"    # a line label
    assert classify("1L-80483", "KL-80483", others) == "misread"          # own text, one character wrong
    assert classify("KL-", "KL-80483", others) == "misread"               # truncated own text


if __name__ == "__main__":
    test_classify()
    print("ok test_classify")
