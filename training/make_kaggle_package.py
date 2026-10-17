"""Bundle the train/val tiles and the training script into one zip for upload as a PRIVATE Kaggle dataset.

Usage: python training/make_kaggle_package.py
Writes data/kaggle/pid_tiles.zip and prints its SHA-256 for the experiment record.
"""
import hashlib
import zipfile
from pathlib import Path

TILES, OUT = Path("data/tiles"), Path("data/kaggle/pid_tiles.zip")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for part in ("train", "val") for p in (TILES / part).rglob("*") if p.is_file())
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:  # tiles are JPEGs already; compressing again gains nothing
        for p in files:
            z.write(p, Path("tiles") / p.relative_to(TILES))
        z.write(Path(__file__).with_name("train_detector.py"), "train_detector.py")
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
        bad = z.testzip()
    assert bad is None and len(names) == len(files) + 1, (bad, len(names), len(files))
    h = hashlib.sha256()
    with open(OUT, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB at a time: the zip is ~2 GB and RAM is 8 GB
            h.update(chunk)
    digest = h.hexdigest()
    print(f"{OUT}: {len(names)} files, {OUT.stat().st_size / 2**30:.2f} GB, sha256 {digest}")


if __name__ == "__main__":
    main()
