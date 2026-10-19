"""PID2Graph (Stürmer et al. 2025, Zenodo 14803338, CC BY-SA 4.0): fetch only the parts we use, and align graphs to our sheets.

Usage:
    python src/pid2graph.py fetch    # range-reads the 8.9 GB zip: our 500 sheets' graphs + the 12 real OPEN100 plans (~21 MB)
    python src/pid2graph.py map      # writes docs/pid2graph_graph_to_sheet.json (PID2Graph graph number -> our sheet number)

PID2Graph numbers Dataset-P&ID's sheets differently from the original dataset, so graphs must go through the map.
"""
import io
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_sheets import iou  # noqa: E402

URL = "https://zenodo.org/records/14803338/files/PID2Graph.zip?download=1"
OUT = Path("data/pid2graph")
MAP = Path("docs/pid2graph_graph_to_sheet.json")
G = "{http://graphml.graphdrawing.org/xmlns}"
JUNCTIONS = {"crossing", "connector", "background"}


class HttpFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP range requests, so zipfile reads only what it needs."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else n
        if n == 0 or self.pos >= self.size:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{min(self.pos + n, self.size) - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.status != 206:  # a server ignoring Range would silently stream the whole 8.9 GB
                raise IOError(f"server ignored the Range request (status {r.status})")
            data = r.read()
        self.pos += len(data)
        return data


def fetch():
    z = zipfile.ZipFile(HttpFile(URL))
    pick = {"dataset_pid": [i for i in z.infolist() if i.filename.startswith("PID2Graph/Complete/Dataset PID/") and i.filename.endswith(".graphml")],
            "open100": [i for i in z.infolist() if i.filename.startswith("PID2Graph/Complete/PID2Graph OPEN100/") and not i.is_dir()]}
    for sub, members in pick.items():
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for i in members:
            name = PurePosixPath(i.filename).name  # flatten: never trust archive paths
            (OUT / sub / name).write_bytes(z.read(i))  # zipfile checks each member's CRC
        print(f"{sub}: {len(members)} files")


def load_graph(source):
    """Nodes {id: (label, [xmin, ymin, xmax, ymax])} and edges [(source, target, edge_label)] from a GraphML path or text."""
    root = ET.fromstring(source) if isinstance(source, str) and source.lstrip().startswith("<") else ET.parse(source).getroot()
    keys = {k.get("id"): k.get("attr.name") for k in root.iter(G + "key")}
    nodes = {}
    for n in root.iter(G + "node"):
        v = {keys[d.get("key")]: d.text for d in n.iter(G + "data")}
        nodes[n.get("id")] = (v["label"], [float(v[k]) for k in ("xmin", "ymin", "xmax", "ymax")])
    edges = [(e.get("source"), e.get("target"), next((d.text for d in e.iter(G + "data")), None)) for e in root.iter(G + "edge")]
    return nodes, edges


def vote(symbol_boxes, index):
    """index: {box tuple: set of our sheet ids}. Returns (best sheet, its votes, runner-up votes)."""
    votes = Counter(s for b in symbol_boxes for s in index.get(tuple(int(round(c)) for c in b), ()))
    top = votes.most_common(2) + [(None, 0), (None, 0)]
    return top[0][0], top[0][1], top[1][1]


def matched_share(ours, theirs):
    """Share of our symbol boxes that overlap one of their boxes at IoU >= 0.5 (one-to-one, greedy)."""
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(ours) for j, b in enumerate(theirs)), reverse=True)
    used_a, used_b = set(), set()
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return len(used_a) / max(1, len(ours))


def map_graphs():
    index, ours = {}, {}
    for f in Path("data/annotations").glob("*.json"):
        s = json.loads(f.read_text(encoding="utf-8"))
        ours[s["sheet"]] = [y["box"] for y in s["symbols"]]
        for b in ours[s["sheet"]]:
            index.setdefault(tuple(b), set()).add(s["sheet"])
    mapping, worst = {}, 1.0
    for g in sorted((OUT / "dataset_pid").glob("*.graphml"), key=lambda p: int(p.stem)):
        nodes, _ = load_graph(g)
        boxes = [b for lab, b in nodes.values() if lab not in JUNCTIONS]
        best, n1, n2 = vote(boxes, index)
        share = matched_share(ours[best], boxes) if best else 0.0
        if best is None or share < 0.9 or n2 > 0.2 * n1:
            raise SystemExit(f"graph {g.stem}: no confident match (best {best}, votes {n1} vs {n2}, overlap {share:.2f})")
        mapping[g.stem], worst = best, min(worst, share)
    if len(set(mapping.values())) != len(mapping):
        raise SystemExit("two graphs mapped to the same sheet")
    MAP.write_text(json.dumps(mapping, indent=0), encoding="utf-8")
    print(f"mapped {len(mapping)} graphs one-to-one; worst symbol overlap {worst:.3f}; identity matches {sum(k == v for k, v in mapping.items())}")


if __name__ == "__main__":
    {"fetch": fetch, "map": map_graphs}[sys.argv[1]]()
