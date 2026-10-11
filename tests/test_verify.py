"""Run with: python tests/test_verify.py   (also collectable by pytest)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verify import verify  # noqa: E402

DRAWING = [
    {"cls": 1, "tag": "AB-10001"}, {"cls": 1, "tag": "AB-10002"}, {"cls": 28, "tag": "SDL 100"}, {"cls": 28, "tag": "SDL 100"},
    {"cls": 19, "tag": '6"'}, {"cls": 19, "tag": '6"'}, {"cls": 21, "tag": ""},
]


def bom(tagged=None, counted=None):
    return {
        "tagged": [{"cls": 1, "tag": "AB-10001"}, {"cls": 1, "tag": "AB-10002"}, {"cls": 28, "tag": "SDL 100"},
                   {"cls": 28, "tag": "SDL 100"}] if tagged is None else tagged,
        "counted": [{"cls": 19, "text": '6"', "quantity": 2}, {"cls": 21, "text": "", "quantity": 1}] if counted is None else counted,
    }


def kinds(findings):
    return sorted(f["type"] for f in findings)


def test_matching_drawing_and_bom_gives_no_findings():
    assert verify(DRAWING, bom()) == []  # includes a repeated tag and untagged items


def test_each_discrepancy_type_is_reported_once():
    base = bom()["tagged"]
    assert kinds(verify(DRAWING, bom(base + [{"cls": 1, "tag": "AB-77777"}]))) == ["MISSING"]
    assert kinds(verify(DRAWING, bom(base[1:]))) == ["EXTRA"]
    assert kinds(verify(DRAWING, bom([{"cls": 1, "tag": "AB-10091"}] + base[1:]))) == ["TAG"]
    assert kinds(verify(DRAWING, bom([{"cls": 4, "tag": "AB-10001"}] + base[1:]))) == ["TYPE"]
    f = verify(DRAWING, bom(counted=[{"cls": 19, "text": '6"', "quantity": 3}, {"cls": 21, "text": "", "quantity": 1}]))
    assert kinds(f) == ["QUANTITY"] and f[0]["drawing"]["quantity"] == 2 and f[0]["bom"]["quantity"] == 3


def test_tag_rule_boundaries():
    base = bom()["tagged"]
    far = verify(DRAWING, bom([{"cls": 1, "tag": "AB-19991"}] + base[1:]))        # 3 characters differ
    assert kinds(far) == ["EXTRA", "MISSING"]
    other_class = verify(DRAWING, bom([{"cls": 4, "tag": "AB-10091"}] + base[1:]))  # near tag but different class
    assert kinds(other_class) == ["EXTRA", "MISSING"]
    f = verify(DRAWING, bom([{"cls": 1, "tag": "AB-10003"}] + base[1:]))           # near both drawing tags: closest unmatched wins
    assert kinds(f) == ["TAG"] and f[0]["drawing"]["tag"] == "AB-10001"


def test_counted_item_absent_from_bom_is_a_quantity_finding():
    f = verify(DRAWING, bom(counted=[{"cls": 19, "text": '6"', "quantity": 2}]))
    assert kinds(f) == ["QUANTITY"] and f[0]["bom"]["quantity"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
