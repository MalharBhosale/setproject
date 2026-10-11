"""Run with: python tests/test_make_split.py   (also collectable by pytest)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_split import make_split  # noqa: E402

IDS = [str(i) for i in range(500)]


def test_split_is_disjoint_complete_and_sized():
    s = make_split(IDS)
    assert {k: len(v) for k, v in s.items()} == {"train": 350, "val": 75, "test": 75}
    everything = s["train"] + s["val"] + s["test"]
    assert len(set(everything)) == 500 and set(everything) == set(IDS)


def test_split_is_deterministic_and_order_independent():
    assert make_split(IDS) == make_split(reversed(IDS))
    assert make_split(IDS, seed=1) != make_split(IDS, seed=2)


def test_wrong_sheet_count_is_rejected():
    for bad in (IDS[:-1], IDS[:-1] + ["0"]):
        try:
            make_split(bad)
        except ValueError:
            continue
        raise AssertionError("bad sheet list accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
