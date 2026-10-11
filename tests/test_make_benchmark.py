"""Run with: python tests/test_make_benchmark.py   (also collectable by pytest)"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_benchmark import IDENTIFIER_CLASSES, cases_for_sheet, clean_bom, seed_discrepancies  # noqa: E402


def _sheet():
    """12 tagged valves (classes 1 and 28, one repeated tag) plus counted items."""
    tagged = [(1, f"AB-{10000 + i}") for i in range(6)] + [(28, f"SDL {100 + i}") for i in range(5)] + [(28, "SDL 100")]
    counted = [(19, '6"')] * 2 + [(21, "")] * 3 + [(30, "STA")]
    symbols, words, links = [], [], {}
    for k, (cls, text) in enumerate(tagged + counted):
        sid, wid = f"symbol_{k}", f"word_{k}"
        symbols.append({"id": sid, "box": [k * 10, 0, k * 10 + 8, 8], "cls": cls})
        words.append({"id": wid, "box": [0, 0, 1, 1], "text": text, "rotation": 0})
        links[sid] = {"words": [wid], "line": "line_1"}
    return {"sheet": "7", "symbols": symbols, "words": words, "links": links}


def test_clean_bom_groups_rows_as_specified():
    bom, evidence = clean_bom(_sheet())
    assert len(bom["tagged"]) == 12 and len(evidence) == 12
    assert all(r["cls"] in IDENTIFIER_CLASSES for r in bom["tagged"])
    assert bom["counted"] == [{"cls": 19, "text": '6"', "quantity": 2}, {"cls": 21, "text": "", "quantity": 3},
                              {"cls": 30, "text": "STA", "quantity": 1}]
    assert evidence[0]["box"] == [0, 0, 8, 8]  # evidence row i points at the symbol behind tagged row i


def _multiset(rows):
    return Counter((r["cls"], r["tag"]) for r in rows)


def test_each_edit_changes_the_bom_exactly_as_recorded():
    bom, evidence = clean_bom(_sheet())
    original = _multiset(bom["tagged"])
    seen = Counter()
    for seed in range(300):
        seeded, found = seed_discrepancies(bom, evidence, 3, random.Random(seed))
        assert len(found) == 3 and bom == clean_bom(_sheet())[0]  # clean BOM is never mutated
        expected = Counter(original)
        for d in found:
            seen[d["type"]] += 1
            if d["type"] == "QUANTITY":
                assert d["after"]["quantity"] != d["before"]["quantity"] and d["after"]["quantity"] >= 0
                continue
            if d["before"]:
                assert d["before"]["tag"] != "SDL 100" and "box" in d  # repeated tag never edited; evidence given
                expected[(d["before"]["cls"], d["before"]["tag"])] -= 1
            if d["after"]:
                expected[(d["after"]["cls"], d["after"]["tag"])] += 1
            if d["type"] in ("MISSING", "TAG"):
                assert (d["after"]["cls"], d["after"]["tag"]) not in original  # new tags never collide
            if d["type"] == "TAG":
                a, b = d["before"]["tag"], d["after"]["tag"]
                assert d["before"]["cls"] == d["after"]["cls"] and len(a) == len(b)
                assert 1 <= sum(x != y for x, y in zip(a, b)) <= 2 and a[:3] == b[:3]  # prefix and format kept
            if d["type"] == "TYPE":
                assert d["before"]["tag"] == d["after"]["tag"] and d["before"]["cls"] != d["after"]["cls"]
        assert _multiset(seeded["tagged"]) == +expected
    assert set(seen) == {"MISSING", "EXTRA", "TAG", "TYPE", "QUANTITY"}


def test_cases_are_deterministic_and_sized():
    a, b = list(cases_for_sheet(_sheet())), list(cases_for_sheet(_sheet()))
    assert a == b
    assert [(c[0], len(c[2])) for c in a] == [("7_clean", 0), ("7_d1", 1), ("7_d3", 3), ("7_d5", 5)]
    assert list(cases_for_sheet(_sheet(), seed=1)) != a


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
