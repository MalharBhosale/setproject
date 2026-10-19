"""Run with: python tests/test_train_reader.py   (also collectable by pytest). Needs numpy, Pillow and torch; no data."""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_reader import H, W, charset, encode, greedy_decode, model, read, to_input  # noqa: E402

CHARS = charset(["AB-12345", '6"', "SDL 472"])


def test_encode_decode_round_trip_with_ctc_collapsing():
    ids = encode("AB-1122", CHARS)
    path = [0, ids[0], ids[0], 0, ids[1], ids[2], ids[3], ids[3], 0, ids[4], 0, ids[5], ids[6], 0, ids[6]]
    assert greedy_decode(path, CHARS) == "AB-1122"   # a blank separates genuine double letters
    assert greedy_decode([0, 0, 0], CHARS) == ""


def test_to_input_keeps_aspect_and_pads_white():
    crop = np.zeros((16, 40), np.uint8)               # black 16 x 40 -> 32 x 80, rest white
    out = to_input(crop)
    assert out.shape == (H, W) and out[:, :80].max() == 0 and out[:, 80:].min() == 255
    jittered = to_input(np.zeros((30, 100), np.uint8), random.Random(0))
    assert jittered.shape == (H, W)


def test_untrained_model_reads_without_crashing():
    import torch
    torch.manual_seed(0)
    net = model(len(CHARS) + 1).eval()
    out = read(net, CHARS, [np.full((20, 60), 255, np.uint8), np.zeros((20, 120), np.uint8)])
    assert len(out) == 2 and all(isinstance(t, str) and 0.0 <= c <= 1.0 for t, c in out)


def test_label_box_takes_covered_words_in_reading_order():
    from train_reader import label_box
    words = [{"box": [0, 0, 30, 10], "text": "SDL"}, {"box": [0, 14, 30, 24], "text": "472"}, {"box": [100, 0, 140, 10], "text": "RV-1"}]
    assert label_box([-2, -2, 32, 26], words) == "SDL 472"          # bubble: two lines, top first
    assert label_box([0, 0, 18, 10], words) == "SDL"                # covers 60% of the word: still labelled
    assert label_box([0, 0, 10, 10], words) is None                 # a third of a word: not labelled
    assert label_box([50, 50, 60, 90], words) is None               # no text at all


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
