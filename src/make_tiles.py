"""Extract Dataset-P&ID images and cut sheets into overlapping tiles with YOLO labels for detector training.

Usage:
    python src/make_tiles.py extract            # data/image_2.tar.gz -> data/images/<sheet>.jpg
    python src/make_tiles.py tile [sheet ...]   # default: every train and val sheet in docs/split.json

Tiles go to data/tiles/<part>/images and labels to data/tiles/<part>/labels. Test sheets are never tiled here:
they are only seen at inference time.
"""
import json
import sys
import tarfile
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # sheets are about 7168 x 4562 px
TILE = 1024
OVERLAP = 256  # must exceed the largest symbol side so every symbol lies whole inside at least one tile
ARCHIVE, IMAGES, ANNOTATIONS = Path("data/image_2.tar.gz"), Path("data/images"), Path("data/annotations")
TILES, SPLIT = Path("data/tiles"), Path("docs/split.json")


def extract():
    IMAGES.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for m in tar:
            name = Path(m.name).name  # flatten: never trust paths inside an archive
            if m.isfile() and name.endswith(".jpg") and name[:-4].isdigit():
                (IMAGES / name).write_bytes(tar.extractfile(m).read())
                n += 1
    print("extracted", n, "images to", IMAGES)


def tile_origins(size, tile=TILE, overlap=OVERLAP):
    """Start offsets along one axis: fixed stride, with a final tile flush against the far edge."""
    if size <= tile:
        return [0]
    last = size - tile
    return list(range(0, last, tile - overlap)) + [last]


def labels_for_tile(symbols, x0, y0, tile=TILE):
    """YOLO rows (class, cx, cy, w, h; normalised) for symbols whose centre is inside the tile, clipped to it."""
    rows = []
    for s in symbols:
        x1, y1, x2, y2 = s["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not (x0 <= cx < x0 + tile and y0 <= cy < y0 + tile):
            continue
        a, b, c, d = max(x1, x0) - x0, max(y1, y0) - y0, min(x2, x0 + tile) - x0, min(y2, y0 + tile) - y0
        rows.append((s["cls"] - 1, (a + c) / 2 / tile, (b + d) / 2 / tile, (c - a) / tile, (d - b) / tile))
    return rows


def tile_sheet(sheet_id, part):
    sheet = json.loads((ANNOTATIONS / f"{sheet_id}.json").read_text(encoding="utf-8"))
    image = Image.open(IMAGES / f"{sheet_id}.jpg").convert("RGB")
    for sub in ("images", "labels"):
        (TILES / part / sub).mkdir(parents=True, exist_ok=True)
    n = empty = boxes = 0
    for y0 in tile_origins(image.height):
        for x0 in tile_origins(image.width):
            rows = labels_for_tile(sheet["symbols"], x0, y0)
            name = f"{sheet_id}_{x0}_{y0}"
            image.crop((x0, y0, x0 + TILE, y0 + TILE)).save(TILES / part / "images" / f"{name}.jpg", quality=95)
            (TILES / part / "labels" / f"{name}.txt").write_text(
                "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows), encoding="ascii")
            n, empty, boxes = n + 1, empty + (not rows), boxes + len(rows)
    return n, empty, boxes


def main():
    if sys.argv[1:2] == ["extract"]:
        return extract()
    if sys.argv[1:2] != ["tile"]:
        sys.exit(__doc__)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    part_of = {s: p for p in ("train", "val") for s in split[p]}
    wanted = sys.argv[2:] or sorted(part_of, key=int)
    if bad := [s for s in wanted if s not in part_of]:
        sys.exit(f"refusing to tile sheets outside train/val (test sheets must stay unseen): {bad}")
    totals = [0, 0, 0]
    for sid in wanted:
        totals = [a + b for a, b in zip(totals, tile_sheet(sid, part_of[sid]))]
    print(f"sheets={len(wanted)} tiles={totals[0]} empty={totals[1]} labelled boxes={totals[2]}")


if __name__ == "__main__":
    main()
