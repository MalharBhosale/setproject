"""Run with: python tests/test_convert_annotations.py   (also collectable by pytest)"""
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convert_annotations import bom, check, convert_sheet, safe_load  # noqa: E402


def _write_sheet(folder):
    """A two-symbol sheet reproducing the quirks seen in the real data."""
    obj = lambda rows: np.array(rows, dtype=object)
    np.save(folder / "7_symbols.npy", obj([["symbol_1", [10, 10, 40, 30], "21"], ["symbol_2", [100, 10, 160, 40], 4]]))
    np.save(folder / "7_words.npy", obj([["word_1", [100, 8, 130, 0], "GH", 0], ["word_2", [132, 8, 160, 0], "019", 90]]))
    np.save(folder / "7_lines.npy", obj([["line_1", [0, 20, 200, 20], '2"-XC-8202', "solid"]]))
    np.save(folder / "7_linker.npy", obj([["symbol_1", ["", "line_1"]], ["symbol_2", ["word_1,word_2", "line_1"]]]))
    np.save(folder / "7_KeyValue.npy", np.array([["DRAWING NAME", "SAMPLE.JPG"]]))
    np.save(folder / "7_Table.npy", np.array([["ISSUE", "DATE"]]))
    np.save(folder / "7_lines2.npy", np.array([[1, 2, 3, 4, 1]]))


def test_convert_normalises_quirks_and_builds_bom():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        _write_sheet(folder)
        sheet = convert_sheet(folder, 7)
    assert [s["cls"] for s in sheet["symbols"]] == [21, 4]          # '21' string becomes int
    assert sheet["words"][0]["box"] == [100, 0, 130, 8]              # y order fixed
    assert sheet["words"][1]["rotation"] == 90
    assert check(sheet) == []
    assert bom(sheet) == [
        {"symbol": "symbol_1", "cls": 21, "tag": "", "line": "line_1"},  # untagged symbol keeps an empty tag
        {"symbol": "symbol_2", "cls": 4, "tag": "GH 019", "line": "line_1"},
    ]


def test_check_reports_dangling_reference():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        _write_sheet(folder)
        sheet = convert_sheet(folder, 7)
    sheet["links"]["symbol_2"]["words"] = ["word_99"]
    assert any("linked word missing" in p for p in check(sheet))


class _Evil:
    def __reduce__(self):
        return (os.getcwd, ())  # stands in for any non-NumPy callable


def test_safe_load_refuses_foreign_pickle():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evil.npy"
        np.save(path, np.array([_Evil()], dtype=object))
        try:
            safe_load(path)
        except pickle.UnpicklingError as e:
            assert "refusing" in str(e)
        else:
            raise AssertionError("foreign pickle was loaded")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
