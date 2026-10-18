"""Run the tile detector over whole sheets and score it at sheet level.

Usage: python src/detect_sheets.py <weights.pt> <out_dir> [part]      part defaults to "val"
Writes <out_dir>/predictions_<part>/<sheet>.json (all boxes with score >= MIN_SCORE, sheet coordinates)
and <out_dir>/sheet_eval_<part>.json. Refuses the test part unless --allow-test is given.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tiles import OVERLAP, TILE, tile_origins  # noqa: E402

MIN_SCORE = 0.01   # kept on disk so later calibration can use low-confidence boxes
EVAL_SCORE = 0.25  # threshold for the precision / recall reported here
EDGE = 2           # px: a box this close to an inner tile edge is cut off; the symbol is whole in a neighbour


def to_sheet(boxes, x0, y0, width, height, tile=TILE):
    """Shift tile boxes (cls, score, x1, y1, x2, y2) to sheet coordinates, dropping boxes cut by an inner tile edge."""
    out = []
    for cls, score, x1, y1, x2, y2 in boxes:
        cut = (x1 <= EDGE and x0 > 0) or (y1 <= EDGE and y0 > 0) or \
              (x2 >= tile - EDGE and x0 + tile < width) or (y2 >= tile - EDGE and y0 + tile < height)
        if not cut:
            out.append({"cls": int(cls), "score": float(score), "box": [x1 + x0, y1 + y0, x2 + x0, y2 + y0]})
    return out


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def merge(preds, thr=0.5):
    """Class-wise greedy non-maximum suppression across tiles."""
    kept = []
    for p in sorted(preds, key=lambda p: -p["score"]):
        if all(k["cls"] != p["cls"] or iou(k["box"], p["box"]) < thr for k in kept):
            kept.append(p)
    return kept


def score_sheet(preds, truth, thr=EVAL_SCORE):
    """Per-class TP / FP / FN at IoU >= 0.5, plus how many false positives sit on a symbol of another class."""
    preds = sorted((p for p in preds if p["score"] >= thr), key=lambda p: -p["score"])
    used, tally, wrong_class = set(), Counter(), Counter()
    for p in preds:
        best = max(((iou(p["box"], t["box"]), i) for i, t in enumerate(truth) if i not in used and t["cls"] == p["cls"]), default=(0, None))
        if best[0] >= 0.5:
            used.add(best[1])
            tally[(p["cls"], "tp")] += 1
        else:
            tally[(p["cls"], "fp")] += 1
            if any(iou(p["box"], t["box"]) >= 0.5 for i, t in enumerate(truth) if t["cls"] != p["cls"]):
                wrong_class[p["cls"]] += 1
    for i, t in enumerate(truth):
        if i not in used:
            tally[(t["cls"], "fn")] += 1
    return tally, wrong_class


def predict_sheet(model, image_path, batch=8):
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    image = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    origins = [(x0, y0) for y0 in tile_origins(h) for x0 in tile_origins(w)]
    preds = []
    for k in range(0, len(origins), batch):
        chunk = origins[k:k + batch]
        results = model.predict([image[y0:y0 + TILE, x0:x0 + TILE] for x0, y0 in chunk], imgsz=TILE, conf=MIN_SCORE, verbose=False)
        for (x0, y0), r in zip(chunk, results):
            b = r.boxes
            rows = [(c + 1, s, *xyxy) for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]  # back to dataset class numbers
            preds += to_sheet(rows, x0, y0, w, h)
    return merge(preds)


def main():
    args = [a for a in sys.argv[1:] if a != "--allow-test"]
    weights, out = args[0], Path(args[1])
    part = args[2] if len(args) > 2 else "val"
    if part == "test" and "--allow-test" not in sys.argv:
        sys.exit("refusing to touch the test sheets: thresholds must be frozen first (pass --allow-test only for the final run)")
    from ultralytics import YOLO
    model = YOLO(weights)
    sheets = json.loads(Path("docs/split.json").read_text(encoding="utf-8"))[part]
    pred_dir = out / f"predictions_{part}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    total, wrong = Counter(), Counter()
    for n, sid in enumerate(sheets, 1):
        preds = predict_sheet(model, f"data/images/{sid}.jpg")
        (pred_dir / f"{sid}.json").write_text(json.dumps(preds), encoding="utf-8")
        truth = json.loads(Path(f"data/annotations/{sid}.json").read_text(encoding="utf-8"))["symbols"]
        t, wc = score_sheet(preds, truth)
        total.update(t)
        wrong.update(wc)
        if n % 15 == 0:
            print(f"  {n}/{len(sheets)} sheets", flush=True)
    per_class = {}
    for c in range(1, 33):
        tp, fp, fn = total[(c, "tp")], total[(c, "fp")], total[(c, "fn")]
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "fp_on_other_class": wrong[c]}
    tp, fp, fn = (sum(v[k] for v in per_class.values()) for k in ("tp", "fp", "fn"))
    summary = {"part": part, "sheets": len(sheets), "score_threshold": EVAL_SCORE, "iou": 0.5, "tp": tp, "fp": fp, "fn": fn,
               "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "fp_on_other_class": sum(wrong.values())}
    (out / f"sheet_eval_{part}.json").write_text(json.dumps({"summary": summary, "per_class": per_class}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    worst = sorted(per_class.items(), key=lambda kv: kv[1]["precision"] + kv[1]["recall"])[:5]
    print("lowest classes:", [(c, round(v["precision"], 3), round(v["recall"], 3), v["fp_on_other_class"]) for c, v in worst])


if __name__ == "__main__":
    main()
