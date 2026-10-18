"""Compare OCR engines on reading the tags of identifier-class symbols (validation sheets only).

Usage: python src/eval_ocr.py <out_dir> [easyocr|rapidocr ...]
Each ground-truth word box is cropped from the sheet and recognised on its own, so this measures reading, not
finding, the text. Tall boxes (vertical text) are tried rotated both ways; the more confident reading is kept.
Writes <out_dir>/ocr_<engine>.json (every word) and prints a summary.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_benchmark import IDENTIFIER_CLASSES  # noqa: E402

# px of margin around each word box, tuned per engine on the first 5 validation sheets (which are then not reported):
# EasyOCR peaked at 4 (tried 0, 2, 4, 12, 24, 40), RapidOCR at 24 (tried 4, 12, 24, 40).
PAD = {"easyocr": 4, "rapidocr": 24}
TUNING_SHEETS = 5


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_reading(readings):
    """readings: [(text, confidence)] for the candidate orientations; keep the most confident."""
    return max(readings, key=lambda r: r[1]) if readings else ("", 0.0)


def tag_words(sheet):
    """(symbol id, word index within tag, word) for every word linked to an identifier-class symbol."""
    words = {w["id"]: w for w in sheet["words"]}
    for s in sheet["symbols"]:
        if s["cls"] in IDENTIFIER_CLASSES:
            for k, wid in enumerate(sheet["links"][s["id"]]["words"]):
                yield s["id"], k, words[wid]


def make_engine(name):
    if name == "easyocr":
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        def read(img):
            out = reader.recognize(img, detail=1)
            return (" ".join(t for _, t, _ in out), min((c for _, _, c in out), default=0.0)) if out else ("", 0.0)
        return read
    if name == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()

        def read(img):
            res, _ = engine(img, use_det=False, use_cls=False, use_rec=True)
            return (res[0][1], float(res[0][2])) if res else ("", 0.0)  # each result is [box, text, score]
        return read
    raise ValueError(name)


def evaluate(name, sheets, out):
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    read = make_engine(name)
    rows, seconds = [], 0.0
    for sid in sheets:
        sheet = json.loads(Path(f"data/annotations/{sid}.json").read_text(encoding="utf-8"))
        page = np.asarray(Image.open(f"data/images/{sid}.jpg").convert("L"))
        for sym, k, w in tag_words(sheet):
            x1, y1, x2, y2 = w["box"]
            p = PAD[name]
            crop = page[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p]
            tall = (y2 - y1) > (x2 - x1)
            t0 = time.perf_counter()
            text, conf = best_reading([read(np.ascontiguousarray(np.rot90(crop, r))) for r in ((1, 3) if tall else (0,))])
            seconds += time.perf_counter() - t0
            rows.append({"sheet": sid, "symbol": sym, "k": k, "truth": w["text"], "pred": text, "conf": conf, "rotated": w["rotation"], "tall": tall})
    (out / f"ocr_{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows, seconds


def summarise(rows, seconds):
    tags = {}
    for r in sorted(rows, key=lambda r: r["k"]):
        t = tags.setdefault((r["sheet"], r["symbol"]), [[], []])
        t[0].append(r["truth"])
        t[1].append(r["pred"].strip())
    exact = sum(" ".join(t) == " ".join(p) for t, p in tags.values())
    loose = sum(" ".join(t).replace(" ", "").upper() == " ".join(p).replace(" ", "").upper() for t, p in tags.values())
    chars = sum(len(r["truth"]) for r in rows)
    errors = sum(edit_distance(r["pred"].strip(), r["truth"]) for r in rows)
    rot = [r for r in rows if r["rotated"]]
    return {"words": len(rows), "tags": len(tags), "tag_exact": exact / len(tags), "tag_ignoring_case_and_spaces": loose / len(tags),
            "word_exact": sum(r["pred"].strip() == r["truth"] for r in rows) / len(rows), "char_error_rate": errors / chars,
            "rotated_words": len(rot), "rotated_word_exact": sum(r["pred"].strip() == r["truth"] for r in rot) / max(1, len(rot)),
            "ms_per_word": 1000 * seconds / len(rows)}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    sheets = json.loads(Path("docs/split.json").read_text(encoding="utf-8"))["val"][TUNING_SHEETS:]
    summary = {}
    for name in sys.argv[2:] or ["rapidocr", "easyocr"]:
        summary[name] = summarise(*evaluate(name, sheets, out))
        print(name, json.dumps(summary[name], indent=1), flush=True)
    (out / "ocr_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
