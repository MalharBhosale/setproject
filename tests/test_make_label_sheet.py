"""Run with: python tests/test_make_label_sheet.py   (also collectable by pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_label_sheet import context_box, page  # noqa: E402


def test_context_box_stays_inside_the_image():
    assert context_box("instrumentation", (2, 2, 20, 20), 100, 100) == (0, 0, 24, 24)
    x1, y1, x2, y2 = context_box("valve", (90, 90, 99, 99), 100, 100)
    assert x1 < 90 and y1 < 90 and (x2, y2) == (100, 100)


def test_page_escapes_suggestions_and_keeps_node_ids():
    row = {"plan": "3", "node": "inlet/outlet7", "label": "inlet/outlet", "suggestion": 'PT "1"<b>', "crop": "3_inlet-outlet7.png"}
    out = page([row])
    assert 'data-key="3_inlet/outlet7"' in out and "&quot;1&quot;&lt;b&gt;" in out and '<b>"' not in out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
