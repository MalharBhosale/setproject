"""Qualitative real-drawing check: run the synthetic-trained detector on the 12 real OPEN100 P&IDs (PID2Graph).

Usage: python src/eval_open100.py <weights.pt> <out_dir>
Our 32 classes are mapped to PID2Graph's coarse labels (mapping derived in docs/dataset_card.md). Labels with no
counterpart in our classes (tank, pump, inlet/outlet) cannot be detected and are counted separately.
Runs at native size and upscaled SCALE x (symbols are about 4x smaller than in Dataset-P&ID: median 22 vs 88 px).
"""
import glob
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_sheets import EVAL_SCORE, iou, predict_sheet  # noqa: E402
from pid2graph import load_graph  # noqa: E402

COARSE = {**{c: "valve" for c in [*range(1, 8), *range(9, 17), 25]}, **{c: "general" for c in [8, *range(17, 24)]},
          24: "arrow", **{c: "instrumentation" for c in range(26, 33)}}
IGNORED = {"crossing", "connector", "background"}
SCALE = 4  # shortcut: chosen from the plans' annotated symbol sizes; a deployed system would need the drawing scale


def load_truth(path):
    return [(lab, box) for lab, box in load_graph(path)[0].values() if lab not in IGNORED]


def score(preds, truth):
    """Greedy one-to-one matching at IoU >= 0.5; 'same' requires the mapped coarse label to agree."""
    preds = sorted((p for p in preds if p["score"] >= EVAL_SCORE), key=lambda p: -p["score"])
    t, used_any, used_same = Counter(), set(), set()
    for p in preds:
        lab = COARSE[p["cls"]]
        any_m = max(((iou(p["box"], b), i) for i, (l, b) in enumerate(truth) if i not in used_any), default=(0, None))
        if any_m[0] >= 0.5:
            used_any.add(any_m[1])
        same = max(((iou(p["box"], b), i) for i, (l, b) in enumerate(truth) if i not in used_same and l == lab), default=(0, None))
        if same[0] >= 0.5:
            used_same.add(same[1])
            t[(lab, "tp")] += 1
        else:
            t[(lab, "fp")] += 1
    for i, (l, _) in enumerate(truth):
        t[(l, "found_any")] += i in used_any
        t[(l, "fn")] += i not in used_same and l in set(COARSE.values())
        t[(l, "total")] += 1
    return t


def main():
    from PIL import Image
    from ultralytics import YOLO
    Image.MAX_IMAGE_PIXELS = None
    model, out = YOLO(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for scale in (1, SCALE):
        tally = Counter()
        for g in sorted(glob.glob("data/pid2graph/open100/*.graphml"), key=lambda p: int(Path(p).stem)):
            img = Image.open(Path(g).with_suffix(".png")).convert("RGB")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "plan.png"
                img.resize((img.width * scale, img.height * scale), Image.LANCZOS).save(path)
                preds = predict_sheet(model, path)
            for p in preds:
                p["box"] = [v / scale for v in p["box"]]
            (out / f"open100_{Path(g).stem}_x{scale}.json").write_text(json.dumps(preds), encoding="utf-8")
            tally.update(score(preds, load_truth(g)))
        labels = sorted({k[0] for k in tally})
        per = {l: {"total": tally[(l, "total")], "detected_any_label": tally[(l, "found_any")],
                   "tp": tally[(l, "tp")], "fp": tally[(l, "fp")], "fn": tally[(l, "fn")],
                   "precision": tally[(l, "tp")] / max(1, tally[(l, "tp")] + tally[(l, "fp")]),
                   "recall": tally[(l, "tp")] / max(1, tally[(l, "total")])} for l in labels}
        results[f"x{scale}"] = per
        print(f"=== scale x{scale} ===")
        for l, v in per.items():
            print(f"  {l:16} total {v['total']:4}  located (any label) {v['detected_any_label']:4}  right label {v['tp']:4}  "
                  f"false {v['fp']:4}  precision {v['precision']:.2f}  recall {v['recall']:.2f}")
    (out / "open100_eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
