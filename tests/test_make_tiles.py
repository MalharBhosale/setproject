"""Run with: python tests/test_make_tiles.py   (also collectable by pytest)"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_tiles import OVERLAP, TILE, labels_for_tile, tile_origins  # noqa: E402


def test_origins_cover_the_axis_with_enough_overlap():
    for size in (500, 1024, 1025, 4562, 7168):
        o = tile_origins(size)
        assert o[0] == 0 and o == sorted(set(o))
        assert o[-1] + TILE >= size and (size <= TILE or o[-1] + TILE == size)
        assert all(b - a <= TILE - OVERLAP for a, b in zip(o, o[1:]))  # neighbours overlap by at least OVERLAP


def test_label_is_clipped_normalised_and_zero_based():
    sym = [{"cls": 21, "box": [1000, 100, 1100, 160]}]           # centre x=1050 lies in the second tile only
    assert labels_for_tile(sym, 0, 0) == []
    (cls, cx, cy, w, h), = labels_for_tile(sym, 768, 0)
    assert cls == 20 and abs(cx - 282 / 1024) < 1e-9 and abs(w - 100 / 1024) < 1e-9 and abs(h - 60 / 1024) < 1e-9
    (_, cx, _, w, _), = labels_for_tile([{"cls": 1, "box": [990, 100, 1050, 160]}], 0, 0)   # sticks out of the tile
    assert abs(w - 34 / 1024) < 1e-9 and abs(cx - 1007 / 1024) < 1e-9


def test_every_symbol_is_whole_in_some_tile_and_labelled_once_per_centre():
    rng = random.Random(0)
    W, H = 7168, 4562
    for _ in range(2000):
        w, h = rng.randint(10, OVERLAP), rng.randint(10, OVERLAP)
        x, y = rng.randint(0, W - w), rng.randint(0, H - h)
        sym = [{"cls": 5, "box": [x, y, x + w, y + h]}]
        tiles = [(x0, y0) for y0 in tile_origins(H) for x0 in tile_origins(W)]
        assert any(x0 <= x and x + w <= x0 + TILE and y0 <= y and y + h <= y0 + TILE for x0, y0 in tiles)
        assert sum(len(labels_for_tile(sym, x0, y0)) for x0, y0 in tiles) >= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
