"""Bundle the word-crop dataset (src/train_reader.py build) and the trainer for a PRIVATE Kaggle dataset, and write the notebook.

Usage: python training/make_reader_package.py
Writes data/kaggle/reader_package.zip and training/kaggle_train_reader.ipynb; prints the zip's SHA-256.
"""
import hashlib
import json
import zipfile
from pathlib import Path

FILES = ["train_crops.npy", "val_crops.npy", "train_texts.json", "val_texts.json", "charset.json"]
ZIP, NB = Path("data/kaggle/reader_package.zip"), Path("training/kaggle_train_reader.ipynb")


def notebook():
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}
    cells = [
        md("""
# Text reader training (CRNN + CTC) on Dataset-P&ID word crops

**Settings:** Accelerator = GPU (T4 or P100). Internet is not needed. *Add data*: your **private** dataset made from `reader_package.zip`.

Trains for 12 epochs on 78,783 training-sheet word crops and keeps the epoch with the best exact-word accuracy on 16,967 validation-sheet crops. No test-sheet data is included.

**Outputs** in `/kaggle/working`: `crnn.pt` (weights), `train_log.json` (per-epoch loss and validation accuracy), `environment.json`. Download all three.
"""),
        code("""
import glob, json, os, platform, sys, torch
script = glob.glob('/kaggle/input/**/train_reader.py', recursive=True)
assert len(script) == 1, script
data_dir = os.path.dirname(glob.glob('/kaggle/input/**/train_crops.npy', recursive=True)[0])
sys.path.insert(0, os.path.dirname(script[0]))
import train_reader
env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'torch': torch.__version__,
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'data_dir': data_dir}
json.dump(env, open('/kaggle/working/environment.json', 'w'), indent=1)
print(env)
"""),
        code("""
train_reader.train(data_dir, '/kaggle/working', epochs=12, batch=64)
"""),
        code("""
print(json.load(open('/kaggle/working/train_log.json'))[-3:])
print('Download crnn.pt, train_log.json and environment.json from the Output panel.')
"""),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main():
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(Path("data/reader") / f, f"reader/{f}")
        z.write("src/train_reader.py", "reader/train_reader.py")
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None and len(z.namelist()) == len(FILES) + 1
    h = hashlib.sha256()
    with open(ZIP, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    notebook()
    print(f"{ZIP}: {ZIP.stat().st_size / 2**20:.0f} MB, sha256 {h.hexdigest()}; notebook {NB}")


if __name__ == "__main__":
    main()
