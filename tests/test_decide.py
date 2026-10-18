"""Run with: python tests/test_decide.py   (also collectable by pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decide import choose, decide, reliable, shape  # noqa: E402


def comp(cls, tag, conf, det=0.9):
    return {"cls": cls, "tag": tag, "ocr_conf": conf, "det_score": det, "box": [0, 0, 1, 1]}


BOM = {"tagged": [{"cls": 1, "tag": "AB-12345"}, {"cls": 1, "tag": "AB-22222"}], "counted": [{"cls": 19, "text": '6"', "quantity": 1}]}


def test_clean_read_passes():
    comps = [comp(1, "AB-12345", 0.9), comp(1, "AB-22222", 0.9), comp(19, '6"', 0.9)]
    assert decide(comps, BOM, 0.5)[0] == "PASS"


def test_low_confidence_misread_goes_to_review_not_fail():
    comps = [comp(1, "AB-12345", 0.9), comp(1, "A8-27772", 0.3), comp(19, '6"', 0.9)]  # misread, low confidence
    verdict, findings = decide(comps, BOM, 0.5)
    assert verdict == "REVIEW" and {f["type"] for f in findings} == {"EXTRA", "MISSING"}
    assert decide(comps, BOM, 0.0)[0] == "FAIL"  # tau 0 is the plain verifier


def test_confident_extra_component_fails():
    comps = [comp(1, "AB-12345", 0.9), comp(1, "AB-22222", 0.9), comp(1, "AB-99999", 0.95), comp(19, '6"', 0.9)]
    verdict, findings = decide(comps, BOM, 0.5)
    assert verdict == "FAIL" and [f["supported"] for f in findings] == [True]


def test_quantity_unsupported_when_its_class_has_a_shaky_component():
    comps = [comp(1, "AB-12345", 0.9), comp(1, "AB-22222", 0.9), comp(19, '8"', 0.2)]  # size misread
    assert decide(comps, BOM, 0.5)[0] == "REVIEW"


def test_shape_hides_letters_and_flags_implausible_readings():
    assert shape("KL-81809") == shape("AB-12345") == "AA-99999"     # prefix letters are not revealed
    assert shape('2"x9"') == '9"x9"' and shape("INS ( 43°C )") == "AAA ( 99°A )"
    shapes = {1: {"AA-99999"}}
    assert reliable(comp(1, "KL-80483", 0.95), 0.5, shapes)
    assert not reliable(comp(1, "KL-80-", 0.99), 0.5, shapes)      # truncated: confident but malformed
    assert not reliable(comp(1, "1J-74522", 0.92), 0.5, shapes)    # I read as 1
    assert reliable(comp(21, "", 0.0), 0.5, shapes)                 # untagged class needs no text


def test_choose_takes_highest_coverage_within_risk():
    rows = [{"tau": 0.0, "coverage": 1.0, "risk": 0.25}, {"tau": 0.5, "coverage": 0.6, "risk": 0.04},
            {"tau": 0.7, "coverage": 0.4, "risk": 0.0}]
    assert choose(rows)["tau"] == 0.5
    assert choose([{"tau": 0.0, "coverage": 1.0, "risk": 0.3}]) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
