"""Run with: python tests/test_train_detector.py   (also collectable by pytest). Does not need Ultralytics."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_detector import find_root, small_subset, write_yaml  # noqa: E402


def _tree(base):
    root = base / "pid-tiles" / "tiles"  # Kaggle nests uploads: /kaggle/input/<slug>/<folder>
    for part in ("train", "val"):
        for sub in ("images", "labels"):
            (root / part / sub).mkdir(parents=True)
        for k in range(3):
            (root / part / "images" / f"{k}_0_0.jpg").write_bytes(b"jpg")
            (root / part / "labels" / f"{k}_0_0.txt").write_text("" if k == 0 else "4 0.5 0.5 0.1 0.1\n")
    return root


def test_find_root_locates_nested_tiles_and_rejects_missing_val():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(Path(tmp))
        assert find_root(tmp) == root
        (root / "val" / "images").rename(root / "val" / "gone")
        try:
            find_root(tmp)
        except FileNotFoundError:
            return
        raise AssertionError("missing val/images accepted")


def test_yaml_lists_32_zero_based_classes():
    with tempfile.TemporaryDirectory() as tmp:
        text = write_yaml(Path(tmp), Path(tmp) / "out").read_text()
    assert "train: train/images" in text and "val: val/images" in text
    assert "  0: c01" in text and "  31: c32" in text and "  32:" not in text


def test_subset_copies_only_labelled_tiles_with_their_images():
    with tempfile.TemporaryDirectory() as tmp:
        sub = small_subset(_tree(Path(tmp)), Path(tmp) / "small", n_train=5, n_val=1)
        train = sorted(p.stem for p in (sub / "train" / "labels").glob("*.txt"))
        assert train == ["1_0_0", "2_0_0"]  # the empty tile 0 is skipped
        assert sorted(p.stem for p in (sub / "train" / "images").glob("*.jpg")) == train
        assert len(list((sub / "val" / "labels").glob("*.txt"))) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
