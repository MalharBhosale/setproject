"""Train the symbol detector on Dataset-P&ID tiles with Ultralytics YOLO (AGPL-3.0).

Kaggle (GPU):  python train_detector.py --data-root /kaggle/input --out /kaggle/working
Local check:   python training/train_detector.py --data-root data/tiles --out <dir> --smoke

The data root must contain train/images, train/labels, val/images, val/labels (as written by src/make_tiles.py),
at any depth. Test sheets are never part of the tiles, so they cannot leak into training.
"""
import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

SEED = 0
# Dataset class N is YOLO id N-1. Names are placeholders: the dataset's class names are not verified.
NAMES = {i: f"c{i + 1:02d}" for i in range(32)}


def find_root(root):
    """The shallowest directory under root that holds train/images and val/images."""
    hits = sorted((p.parent.parent for p in Path(root).rglob("images") if p.parent.name == "train"), key=lambda p: len(p.parts))
    for d in hits:
        if (d / "val" / "images").is_dir():
            return d
    raise FileNotFoundError(f"no train/images with a matching val/images under {root}")


def write_yaml(root, out):
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"path: {Path(root).resolve().as_posix()}", "train: train/images", "val: val/images", "names:"]
    lines += [f"  {i}: {n}" for i, n in NAMES.items()]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "data.yaml"


def small_subset(root, out, n_train=16, n_val=8):
    """Copy a few labelled tiles so a CPU run finishes in minutes."""
    for part, n in (("train", n_train), ("val", n_val)):
        labelled = [p for p in sorted((root / part / "labels").glob("*.txt")) if p.stat().st_size][:n]
        for sub in ("images", "labels"):
            (out / part / sub).mkdir(parents=True, exist_ok=True)
        for lab in labelled:
            shutil.copy(lab, out / part / "labels" / lab.name)
            shutil.copy(root / part / "images" / f"{lab.stem}.jpg", out / part / "images" / f"{lab.stem}.jpg")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1024)  # tiles are 1024 px: no downscaling
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hours", type=float, default=10.0, help="stop training after this long (Kaggle sessions end at 12 h)")
    ap.add_argument("--smoke", action="store_true", help="16 train and 8 val tiles, 1 epoch, CPU")
    a = ap.parse_args(argv)

    import torch
    import ultralytics
    from ultralytics import YOLO

    out = Path(a.out)
    root = find_root(a.data_root)
    if a.smoke:
        root = small_subset(root, out / "smoke_data")
        a.epochs, a.imgsz, a.batch, a.hours = 1, 640, 4, None
    data = write_yaml(root, out)
    device = 0 if torch.cuda.is_available() and not a.smoke else "cpu"
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "ultralytics": ultralytics.__version__, "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "data_root": str(root), "args": vars(a), "seed": SEED}
    (out / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    print(json.dumps(env, indent=1))

    model = YOLO(a.model)
    model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, time=a.hours, seed=SEED, deterministic=True,
                patience=15, device=device, project=str(out / "runs"), name="detector", exist_ok=True, plots=True,
                workers=0 if a.smoke else 4)
    metrics = model.val(data=str(data), imgsz=a.imgsz, batch=a.batch, split="val", device=device,
                        project=str(out / "runs"), name="val", exist_ok=True)
    result = {"summary": {k: float(v) for k, v in metrics.results_dict.items()},
              "map50_95_per_class": {NAMES[i]: float(v) for i, v in enumerate(metrics.box.maps)},
              "note": "Tile-level metrics on validation tiles. Overlapping tiles count some symbols twice; "
                      "sheet-level evaluation after merging tiles is a separate step."}
    (out / "val_metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
