"""Run with: python tests/test_pid2graph.py   (also collectable by pytest). No network needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pid2graph import load_graph, matched_share, vote  # noqa: E402

GRAPHML = """<?xml version="1.0"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
 <key id="d0" for="node" attr.name="label" attr.type="string"/>
 <key id="d1" for="node" attr.name="xmin" attr.type="long"/><key id="d2" for="node" attr.name="ymin" attr.type="long"/>
 <key id="d3" for="node" attr.name="xmax" attr.type="long"/><key id="d4" for="node" attr.name="ymax" attr.type="long"/>
 <key id="d5" for="edge" attr.name="edge_label" attr.type="string"/>
 <graph edgedefault="undirected">
  <node id="0"><data key="d0">valve</data><data key="d1">10</data><data key="d2">20</data><data key="d3">40</data><data key="d4">50</data></node>
  <node id="1"><data key="d0">connector</data><data key="d1">60</data><data key="d2">30</data><data key="d3">62</data><data key="d4">32</data></node>
  <edge source="0" target="1"><data key="d5">solid</data></edge>
 </graph>
</graphml>"""


def test_load_graph_reads_labels_boxes_and_edges():
    nodes, edges = load_graph(GRAPHML)
    assert nodes == {"0": ("valve", [10.0, 20.0, 40.0, 50.0]), "1": ("connector", [60.0, 30.0, 62.0, 32.0])}
    assert edges == [("0", "1", "solid")]


def test_vote_picks_the_sheet_sharing_most_boxes():
    index = {(10, 20, 40, 50): {"7"}, (0, 0, 5, 5): {"7", "9"}, (1, 1, 3, 3): {"9"}}
    assert vote([[10, 20, 40, 50], [0.2, 0, 5, 5]], index) == ("7", 2, 1)
    assert vote([[99, 99, 100, 100]], index)[0] is None


def test_matched_share():
    assert matched_share([[0, 0, 10, 10], [20, 0, 30, 10]], [[1, 0, 11, 10]]) == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
