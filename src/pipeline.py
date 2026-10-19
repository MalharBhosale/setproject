"""End-to-end extraction (detector boxes + EasyOCR tags) and verification against the benchmark BOMs. No abstention yet.

Usage: python src/pipeline.py <predictions_dir> <out_dir> [part]      part defaults to "val"
<predictions_dir> holds <sheet>.json from src/detect_sheets.py. Writes <out_dir>/extracted_<part>/<sheet>.json,
<out_dir>/tags_<part>.json (tag accuracy against ground truth) and <out_dir>/verify_<part>.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_sheets import EVAL_SCORE, iou  # noqa: E402
from make_benchmark import TYPES  # noqa: E402
from verify import score_cases  # noqa: E402

# Measured on training sheets: classes 26-32 carry their text inside the symbol; 21 and 24 carry none;
# every other class has one word outside, at most 78 px from the symbol box.
INSIDE = frozenset(range(26, 33))
UNTAGGED = frozenset({21, 24})
MARGIN = 80
ALLOW = ' -0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"()x°'  # characters the dataset's texts use (tuned: +6.4 points tag exact)


def gap(a, b):
    """Chebyshev gap in px between two boxes; 0 if they touch or overlap."""
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3], 0)


def choose_text(cls, sym, texts):
    """texts: [(box, text, conf)] in sheet coordinates. Returns (tag, confidence) for one symbol."""
    if cls in UNTAGGED:
        return "", 1.0
    inside = lambda b: sym[0] <= (b[0] + b[2]) / 2 <= sym[2] and sym[1] <= (b[1] + b[3]) / 2 <= sym[3]
    if cls in INSIDE:
        picked = sorted((t for t in texts if inside(t[0])), key=lambda t: (t[0][1] + t[0][3]) / 2)  # top to bottom
    else:
        outside = [t for t in texts if not inside(t[0]) and gap(sym, t[0]) <= MARGIN]
        picked = [min(outside, key=lambda t: gap(sym, t[0]))] if outside else []
    if not picked:
        return "", 0.0
    return " ".join(t[1].strip() for t in picked), min(t[2] for t in picked)


INSET = 0.12  # fraction of a bubble's width trimmed on each side so its outline is not read as text
READ_PAD = 4  # margin around a found text box when reading it (tuned in Step 4.7)
JOIN = 0.6    # boxes on one line closer than this many text heights are one piece of text (the detector splits tags at hyphens)
# Vertical text in this dataset always reads after a clockwise turn (495 of 495 readable tall words on 12 training sheets).
VERTICAL_TURN = 3  # np.rot90 k: 3 = clockwise


def dedupe(texts, thr=0.5):
    """Regions overlap, so the same text is found more than once; keep the most confident copy."""
    kept = []
    for t in sorted(texts, key=lambda t: -t[2]):
        if all(iou(k[0], t[0]) < thr for k in kept):
            kept.append(t)
    return kept


def assign(symbols, texts):
    """Give each text to one symbol: the bubble it sits inside, else the nearest outside-text symbol within MARGIN.

    symbols: [(cls, box)]. Returns {symbol index: [texts]}, so no symbol can take a neighbour's tag.
    """
    owned = {}
    for t in texts:
        cx, cy = (t[0][0] + t[0][2]) / 2, (t[0][1] + t[0][3]) / 2
        bubble = [i for i, (c, b) in enumerate(symbols) if c in INSIDE and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
        if bubble:
            owned.setdefault(bubble[0], []).append(t)
            continue
        near = [(gap(b, t[0]), i) for i, (c, b) in enumerate(symbols) if c not in INSIDE | UNTAGGED and gap(b, t[0]) <= MARGIN]
        if near:
            owned.setdefault(min(near)[1], []).append(t)
    return owned


def join_boxes(boxes):
    """Merge [x1, y1, x2, y2] boxes that continue each other along one line of text (horizontal or vertical)."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ha, hb = a[3] - a[1], b[3] - b[1]
                wa, wb = a[2] - a[0], b[2] - b[0]
                tall = ha > wa and hb > wb
                if tall:
                    overlap, thick, along = min(a[2], b[2]) - max(a[0], b[0]), min(wa, wb), max(b[1] - a[3], a[1] - b[3])
                else:
                    overlap, thick, along = min(a[3], b[3]) - max(a[1], b[1]), min(ha, hb), max(b[0] - a[2], a[0] - b[2])
                if overlap >= 0.5 * thick and along <= JOIN * thick:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def read_region(reader, page, rx1, ry1, rx2, ry2, read_fn=None):
    """Find text in a region with EasyOCR's detector, then read each piece: with read_fn(crops) -> [(text, conf)] if given
    (the domain-trained reader, src/train_reader.py), otherwise with EasyOCR and the tuned settings (v1)."""
    import numpy as np
    h, w = page.shape
    rx1, ry1, rx2, ry2 = max(0, rx1), max(0, ry1), min(w, rx2), min(h, ry2)
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return []
    horizontal, free = reader.detect(np.ascontiguousarray(page[ry1:ry2, rx1:rx2]))
    boxes = [[b[0], b[2], b[1], b[3]] for b in horizontal[0]]  # detector gives x_min, x_max, y_min, y_max
    boxes += [[min(p[0] for p in q), min(p[1] for p in q), max(p[0] for p in q), max(p[1] for p in q)] for q in free[0]]
    found = []
    for x1, y1, x2, y2 in join_boxes(boxes):
        X1, Y1, X2, Y2 = int(x1) + rx1, int(y1) + ry1, int(x2) + rx1, int(y2) + ry1
        crop = page[max(0, Y1 - READ_PAD):Y2 + READ_PAD, max(0, X1 - READ_PAD):X2 + READ_PAD]
        if crop.size:
            k = VERTICAL_TURN if (Y2 - Y1) > (X2 - X1) else 0
            found.append(([X1, Y1, X2, Y2], np.ascontiguousarray(np.rot90(crop, k))))
    if read_fn is not None:
        return [(box, text, conf) for (box, _), (text, conf) in zip(found, read_fn([c for _, c in found])) if text]
    out = []
    for box, crop in found:
        res = reader.recognize(crop, detail=1, allowlist=ALLOW)
        if res:
            out.append((box, " ".join(r[1] for r in res), float(min(r[2] for r in res))))
    return out


def extract_sheet(reader, page, preds, thr=EVAL_SCORE, read_fn=None):
    kept = [p for p in preds if p["score"] >= thr]
    texts = []
    for p in kept:
        if p["cls"] in UNTAGGED:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in p["box"])
        if p["cls"] in INSIDE:
            dx, dy = int((x2 - x1) * INSET), int((y2 - y1) * INSET)
            texts += read_region(reader, page, x1 + dx, y1 + dy, x2 - dx, y2 - dy, read_fn)
        else:
            texts += read_region(reader, page, x1 - MARGIN, y1 - MARGIN, x2 + MARGIN, y2 + MARGIN, read_fn)
    owned = assign([(p["cls"], p["box"]) for p in kept], dedupe(texts))
    comps = []
    for i, p in enumerate(kept):
        tag, conf = choose_text(p["cls"], p["box"], owned.get(i, []))
        comps.append({"cls": p["cls"], "tag": tag, "box": p["box"], "det_score": p["score"], "ocr_conf": conf})
    return comps

def tag_accuracy(comps, sheet):
    """Match extracted components to true symbols (same class, IoU >= 0.5) and compare tags."""
    words = {w["id"]: w["text"] for w in sheet["words"]}
    tally, used = Counter(), set()
    for s in sheet["symbols"]:
        truth = " ".join(words[i] for i in sheet["links"][s["id"]]["words"])
        if s["cls"] in UNTAGGED:
            continue
        m = max(((iou(c["box"], s["box"]), i) for i, c in enumerate(comps) if i not in used and c["cls"] == s["cls"]), default=(0, None))
        if m[0] < 0.5:
            tally["symbol not detected"] += 1
            continue
        used.add(m[1])
        tally["exact" if comps[m[1]]["tag"] == truth else "wrong"] += 1
    return tally


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    read_fn = None
    if "--reader=crnn" in sys.argv:  # v2: domain-trained reader instead of EasyOCR recognition
        import train_reader
        net, chars = train_reader.load()
        read_fn = lambda crops: train_reader.read(net, chars, crops)
    preds_dir, out = Path(args[0]), Path(args[1])
    part = args[2] if len(args) > 2 else "val"
    if part == "test" and "--allow-test" not in sys.argv:
        sys.exit("refusing to touch the test sheets: freeze the configuration first (then pass --allow-test once)")
    import easyocr
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    sheets = json.loads(Path("docs/split.json").read_text(encoding="utf-8"))[part]
    ext_dir = out / f"extracted_{part}"
    ext_dir.mkdir(parents=True, exist_ok=True)
    comps_by_sheet, tags = {}, Counter()
    for n, sid in enumerate(sheets, 1):
        f = ext_dir / f"{sid}.json"
        if f.exists():  # resumable: extraction is the slow part
            comps = json.loads(f.read_text(encoding="utf-8"))
        else:
            page = np.asarray(Image.open(f"data/images/{sid}.jpg").convert("L"))
            comps = extract_sheet(reader, page, json.loads((preds_dir / f"{sid}.json").read_text(encoding="utf-8")), read_fn=read_fn)
            f.write_text(json.dumps(comps, ensure_ascii=False), encoding="utf-8")
        comps_by_sheet[sid] = comps
        tags.update(tag_accuracy(comps, json.loads(Path(f"data/annotations/{sid}.json").read_text(encoding="utf-8"))))
        if n % 15 == 0:
            print(f"  {n}/{len(sheets)} sheets", flush=True)
    total = sum(tags.values())
    tag_summary = {k: v for k, v in tags.items()} | {"exact_share": tags["exact"] / total}
    (out / f"tags_{part}.json").write_text(json.dumps(tag_summary, indent=1), encoding="utf-8")
    print("tags:", tag_summary)

    cases = [c for c in json.loads(Path("docs/benchmark_manifest.json").read_text(encoding="utf-8"))["cases"] if c["part"] == part]
    score, verdicts, wrong = score_cases(cases, lambda sid: comps_by_sheet[sid])
    v = verdicts[part]
    result = {"cases": len(cases),
              "verdicts": {f"true {a} -> reported {b}": n for (a, b), n in sorted(v.items())},
              "false_fail_rate_on_matching_pairs": v[("PASS", "FAIL")] / max(1, v[("PASS", "PASS")] + v[("PASS", "FAIL")]),
              "findings": {t: dict(score[t]) for t in TYPES},
              "false_findings_by_type": dict(Counter(k for _, extra, _ in wrong for k in (s.split("'")[1] for s in extra)))}
    (out / f"verify_{part}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
