"""Domain-trained text reader: a small CRNN (conv + BiLSTM + CTC) trained on word crops from the training sheets.

Usage:
    python src/train_reader.py build          # word crops -> data/reader/{train,val}.npy (+ labels)
    python src/train_reader.py build-detected <part> <n_sheets>   # v2b: crops cut by the pipeline's text finding
    python src/train_reader.py train [data_dir] [out_dir]   # best epoch on val crops -> <out_dir>/crnn.pt (GPU if available)
    python src/train_reader.py eval [val|val_det] [weights]   # word-exact accuracy on a crop set

Off-the-shelf EasyOCR read 80.8% of tags exactly from perfect word boxes; Gadekar et al. (2026) report 0.995 after
fine-tuning TrOCR on this dataset's word crops. This reader trains from scratch on CPU instead.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

H, W = 32, 256          # crop size fed to the network (grey, height-normalised, right-padded with white)
PAD = 4                 # px of margin around a word box (as in the pipeline)
OUT = Path("data/reader")
SEED = 0


def charset(texts):
    return sorted({c for t in texts for c in t})


def encode(text, chars):
    index = {c: i + 1 for i, c in enumerate(chars)}  # 0 is the CTC blank
    return [index[c] for c in text]


def greedy_decode(best_path, chars):
    """Collapse repeats, drop blanks (CTC greedy decoding). best_path: per-timestep argmax ids."""
    out, prev = [], 0
    for k in best_path:
        if k != prev and k != 0:
            out.append(chars[k - 1])
        prev = k
    return "".join(out)


def to_input(crop, rng=None):
    """Grey crop (uint8) -> HxW uint8, aspect kept, white-padded. With rng: random margin trim / extra jitter."""
    from PIL import Image
    if rng is not None:  # imitate imperfect text boxes: trim or extend up to 3 px per side
        h, w = crop.shape
        t, b, l, r = (rng.randint(0, 3) for _ in range(4))
        crop = crop[min(t, h // 4):h - min(b, h // 4), min(l, w // 8):w - min(r, w // 8)]
    h, w = crop.shape
    nw = max(1, min(W, round(w * H / h)))
    img = np.asarray(Image.fromarray(crop).resize((nw, H), Image.BILINEAR))
    out = np.full((H, W), 255, np.uint8)
    out[:, :nw] = img
    return out


def word_crops(sheet_ids):
    """Yield (crop, text) for every non-empty word box; tall boxes turned clockwise (this dataset's convention)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    for sid in sheet_ids:
        sheet = json.loads(Path(f"data/annotations/{sid}.json").read_text(encoding="utf-8"))
        page = np.asarray(Image.open(f"data/images/{sid}.jpg").convert("L"))
        for w in sheet["words"]:
            x1, y1, x2, y2 = w["box"]
            if not w["text"].strip() or x2 - x1 < 3 or y2 - y1 < 3:
                continue
            crop = page[max(0, y1 - PAD):y2 + PAD, max(0, x1 - PAD):x2 + PAD]
            if y2 - y1 > x2 - x1:
                crop = np.rot90(crop, 3)
            yield np.ascontiguousarray(crop), w["text"]


def build():
    split = json.loads(Path("docs/split.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val"):
        crops, texts = [], []
        for crop, text in word_crops(split[part]):
            if round(crop.shape[1] * H / crop.shape[0]) <= W:  # longer than the canvas: very rare, skipped
                crops.append(crop)
                texts.append(text)
        np.save(OUT / f"{part}_crops.npy", np.array(crops, dtype=object), allow_pickle=True)
        (OUT / f"{part}_texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        print(part, len(texts), "words")
    chars = charset(json.loads((OUT / "train_texts.json").read_text(encoding="utf-8")))
    (OUT / "charset.json").write_text(json.dumps(chars, ensure_ascii=False), encoding="utf-8")
    print("charset", len(chars), "".join(chars))


def label_box(box, words):
    """True text for a crop cut by the pipeline: the words it covers by at least half their area, in reading order.
    None if it covers no word, or several words of vertical text (rare; order would be ambiguous)."""
    x1, y1, x2, y2 = box
    hit = []
    for w in words:
        a, b, c, d = w["box"]
        inter = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
        if w["text"].strip() and inter >= 0.5 * max(1, (c - a) * (d - b)):
            hit.append(w)
    tall = (y2 - y1) > (x2 - x1)
    if not hit or (tall and len(hit) > 1):
        return None
    return " ".join(w["text"] for w in sorted(hit, key=lambda w: (w["box"][1] // 10, w["box"][0])))


def detected_crops(sheet_ids):
    """Yield (crop, text) for text found by the pipeline's own text finding (pipeline.read_region, frozen settings),
    in the regions of the true symbols, so training crops match what the reader meets at inference."""
    import easyocr
    from PIL import Image
    import pipeline
    Image.MAX_IMAGE_PIXELS = None
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    for n, sid in enumerate(sheet_ids, 1):
        sheet = json.loads(Path(f"data/annotations/{sid}.json").read_text(encoding="utf-8"))
        page = np.asarray(Image.open(f"data/images/{sid}.jpg").convert("L"))
        seen = set()
        for s in sheet["symbols"]:
            if s["cls"] in pipeline.UNTAGGED:
                continue
            x1, y1, x2, y2 = s["box"]
            if s["cls"] in pipeline.INSIDE:
                dx, dy = int((x2 - x1) * pipeline.INSET), int((y2 - y1) * pipeline.INSET)
                region = (x1 + dx, y1 + dy, x2 - dx, y2 - dy)
            else:
                region = (x1 - pipeline.MARGIN, y1 - pipeline.MARGIN, x2 + pipeline.MARGIN, y2 + pipeline.MARGIN)
            crops = []
            keep = lambda cs: crops.extend(cs) or [("x", 1.0)] * len(cs)  # capture the exact crops the reader would get
            for (box, _, _), crop in zip(pipeline.read_region(reader, page, *region, read_fn=keep), crops):
                text = label_box(box, sheet["words"])
                if text is not None and tuple(box) not in seen:
                    seen.add(tuple(box))
                    yield crop, text
        print(f"  {n}/{len(sheet_ids)} sheets", flush=True)


def build_detected(part, n_sheets):
    """Pipeline-cut crops from the first n sheets of a split -> data/reader/<part>_det_{crops.npy,texts.json}."""
    chars = set(json.loads((OUT / "charset.json").read_text(encoding="utf-8")))
    sheets = json.loads(Path("docs/split.json").read_text(encoding="utf-8"))[part][:int(n_sheets)]
    crops, texts = [], []
    for crop, text in detected_crops(sheets):
        if set(text) <= chars and min(crop.shape) > 2 and round(crop.shape[1] * H / crop.shape[0]) <= W:
            crops.append(crop)
            texts.append(text)
    np.save(OUT / f"{part}_det_crops.npy", np.array(crops, dtype=object), allow_pickle=True)
    (OUT / f"{part}_det_texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(part, len(texts), "pipeline crops from", len(sheets), "sheets")


def model(n_classes):
    import torch.nn as nn

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            def block(i, o, pool):
                return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)] + ([nn.MaxPool2d(pool)] if pool else [])
            self.cnn = nn.Sequential(*block(1, 32, 2), *block(32, 64, 2), *block(64, 128, None), *block(128, 128, (2, 1)),
                                     *block(128, 256, None), *block(256, 256, (2, 1)), *block(256, 256, (2, 1)))  # -> 256 x 1 x W/4
            self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(256, n_classes)

        def forward(self, x):                           # x: B x 1 x H x W, values 0..1
            f = self.cnn(x).squeeze(2).permute(0, 2, 1)  # B x T x C
            return self.fc(self.rnn(f)[0])              # B x T x classes

    return CRNN()


def load(path=OUT / "crnn.pt"):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = model(len(ck["chars"]) + 1)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["chars"]


def read(net, chars, crops):
    """Read a list of grey crops. Returns [(text, confidence)], confidence = product of the chosen characters' probabilities."""
    import torch
    if not crops:
        return []
    x = torch.from_numpy(np.stack([to_input(c) for c in crops])[:, None].astype(np.float32) / 255.0)
    with torch.no_grad():
        prob = net(x.to(next(net.parameters()).device)).softmax(-1).cpu()
    best_p, best = prob.max(-1)
    out = []
    for path, p in zip(best.tolist(), best_p):
        text = greedy_decode(path, chars)
        keep = [i for i, k in enumerate(path) if k != 0 and (i == 0 or k != path[i - 1])]
        out.append((text, float(p[keep].prod()) if keep else 0.0))
    return out


def train(data=OUT, out=OUT, epochs=12, batch=64):
    import torch
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, "| torch", torch.__version__, flush=True)
    chars = json.loads((data / "charset.json").read_text(encoding="utf-8"))
    load_set = lambda name: (list(np.load(data / f"{name}_crops.npy", allow_pickle=True)),  # our own files, not untrusted input
                             json.loads((data / f"{name}_texts.json").read_text(encoding="utf-8")))
    tr_c, tr_t = load_set("train")
    if (data / "train_det_crops.npy").exists():  # v2b: add crops cut by the pipeline's own text finding
        c, t = load_set("train_det")
        tr_c, tr_t = tr_c + c, tr_t + t
    va_name = "val_det" if (data / "val_det_crops.npy").exists() else "val"  # select on the crops met at inference
    va_c, va_t = load_set(va_name)
    print("train", len(tr_t), "| select on", va_name, len(va_t), flush=True)
    net = model(len(chars) + 1).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(tr_t) + batch - 1) // batch))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    best, log = -1.0, []
    for ep in range(1, epochs + 1):
        net.train()
        order = list(range(len(tr_t)))
        rng.shuffle(order)
        total = 0.0
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            x = torch.from_numpy(np.stack([to_input(tr_c[i], rng) for i in idx])[:, None].astype(np.float32) / 255.0).to(device)
            targets = [encode(tr_t[i], chars) for i in idx]
            logp = net(x).log_softmax(-1).permute(1, 0, 2)  # T x B x C for CTC
            loss = ctc(logp, torch.tensor([c for t in targets for c in t], device=device),
                       torch.full((len(idx),), logp.shape[0], dtype=torch.long, device=device),
                       torch.tensor([len(t) for t in targets], device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        net.eval()
        acc = sum(t == p for t, (p, _) in zip(va_t, (r for k in range(0, len(va_t), 256) for r in read(net, chars, list(va_c[k:k + 256]))))) / len(va_t)
        log.append({"epoch": ep, "train_loss": total / len(tr_t), "val_word_exact": acc, "val_set": va_name})
        print(f"epoch {ep}: loss {total / len(tr_t):.4f}  val word exact {acc:.4f}", flush=True)
        if acc > best:
            best = acc
            torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "chars": chars, "epoch": ep, "val_word_exact": acc, "val_set": va_name},
                       out / "crnn.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("best val word exact", best)


def evaluate(name="val", path=OUT / "crnn.pt"):
    net, chars = load(path)
    va_c = np.load(OUT / f"{name}_crops.npy", allow_pickle=True)
    va_t = json.loads((OUT / f"{name}_texts.json").read_text(encoding="utf-8"))
    preds = [r for k in range(0, len(va_t), 256) for r in read(net, chars, list(va_c[k:k + 256]))]
    right = [c for t, (p, c) in zip(va_t, preds) if p == t]
    wrong = [(t, p, round(c, 3)) for t, (p, c) in zip(va_t, preds) if p != t]
    print(f"val words {len(va_t)}  exact {len(right) / len(va_t):.4f}  wrong {len(wrong)}")
    print("mean confidence right", round(sum(right) / max(1, len(right)), 3), "wrong", round(sum(c for _, _, c in wrong) / max(1, len(wrong)), 3))
    print("examples:", wrong[:12])


if __name__ == "__main__":
    {"build": build, "build-detected": lambda: build_detected(*sys.argv[2:4]), "train": lambda: train(*sys.argv[2:4]),
     "eval": lambda: evaluate(*sys.argv[2:4])}[sys.argv[1]]()
