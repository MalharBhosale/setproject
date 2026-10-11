"""Write the fixed train / validation / test split of Dataset-P&ID sheets to docs/split.json.

Usage: python src/make_split.py
The split is by whole sheet and must be made BEFORE any derived data (tiles, seeded discrepancies)
so that nothing derived from a test sheet can reach training.
"""
import json
import random
from pathlib import Path

SEED = 20261011
SIZES = {"train": 350, "val": 75, "test": 75}
ANNOTATIONS = Path("data/annotations")
OUT = Path("docs/split.json")


def make_split(sheet_ids, seed=SEED, sizes=SIZES):
    ids = sorted(sheet_ids, key=int)
    if len(set(ids)) != len(ids) or len(ids) != sum(sizes.values()):
        raise ValueError(f"expected {sum(sizes.values())} unique sheets, got {len(ids)} ({len(set(ids))} unique)")
    random.Random(seed).shuffle(ids)
    split, start = {}, 0
    for name, n in sizes.items():
        split[name] = sorted(ids[start:start + n], key=int)
        start += n
    return split


def main():
    split = make_split(p.stem for p in ANNOTATIONS.glob("*.json"))
    OUT.write_text(json.dumps({"seed": SEED, "unit": "sheet", **split}, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in split.items()}, "->", OUT)


if __name__ == "__main__":
    main()
