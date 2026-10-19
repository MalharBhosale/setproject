"""Build a local labelling page for tags on the 12 real OPEN100 plans (PID2Graph), with OCR suggestions pre-filled.

Usage: python src/make_label_sheet.py          -> data/open100_labels/index.html (+ crops/)
Open index.html in a browser, correct each suggested tag (clear it if the symbol has no tag), then press
"Export CSV" and save the file as data/open100_labels/labels.csv. Progress is kept in the browser between sessions.
The drawings' licence is unconfirmed: everything stays in data/ (git-ignored) and is for evaluation only.
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid2graph import load_graph  # noqa: E402

OUT = Path("data/open100_labels")
LABELLED = {"instrumentation", "valve", "general", "tank", "pump", "inlet/outlet"}
UP = 3  # crops are enlarged 3x: real symbols are about 22 px across
ALLOW = " -0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/\"()."


def context_box(label, box, w, h):
    """Region shown and read for a symbol: the bubble itself for instruments, the symbol plus surroundings otherwise."""
    x1, y1, x2, y2 = box
    m = 4 if label == "instrumentation" else 2.5 * max(x2 - x1, y2 - y1, 12)
    return max(0, int(x1 - m)), max(0, int(y1 - m)), min(w, int(x2 + m)), min(h, int(y2 + m))


def main():
    import easyocr
    import numpy as np
    from PIL import Image, ImageDraw
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    for g in sorted(Path("data/pid2graph/open100").glob("*.graphml"), key=lambda p: int(p.stem)):
        img = Image.open(g.with_suffix(".png")).convert("L")
        nodes, _ = load_graph(g)
        for nid, (label, box) in sorted(nodes.items(), key=lambda kv: (kv[1][1][1], kv[1][1][0])):
            if label not in LABELLED:
                continue
            cx1, cy1, cx2, cy2 = context_box(label, box, img.width, img.height)
            crop = img.crop((cx1, cy1, cx2, cy2)).resize(((cx2 - cx1) * UP, (cy2 - cy1) * UP), Image.LANCZOS)
            res = reader.readtext(np.asarray(crop), allowlist=ALLOW, rotation_info=[90, 270])
            res.sort(key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))  # top to bottom, then left to right
            suggestion = " ".join(r[1].strip() for r in res if r[2] > 0.3).strip()
            shown = crop.convert("RGB")
            ImageDraw.Draw(shown).rectangle([(box[0] - cx1) * UP, (box[1] - cy1) * UP, (box[2] - cx1) * UP, (box[3] - cy1) * UP],
                                            outline=(220, 0, 0), width=2)
            name = f"{g.stem}_{nid}.png".replace("/", "-")  # node ids such as "inlet/outlet79" contain a slash
            shown.save(OUT / "crops" / name)
            rows.append({"plan": g.stem, "node": nid, "label": label, "box": [round(v, 1) for v in box], "suggestion": suggestion, "crop": name})
        print(f"plan {g.stem}: {sum(r['plan'] == g.stem for r in rows)} symbols", flush=True)
    (OUT / "items.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    (OUT / "index.html").write_text(page(rows), encoding="utf-8")
    print(f"{len(rows)} symbols -> {OUT / 'index.html'}")


def page(rows):
    items = "\n".join(
        f'<div class="item" data-key="{r["plan"]}_{r["node"]}"><img src="crops/{r["crop"]}" alt="plan {r["plan"]} node {r["node"]}" loading="lazy">'
        f'<div class="meta"><b>plan {r["plan"]}</b> &middot; {html.escape(r["label"])} &middot; node {r["node"]}'
        f'<label>Tag (empty if none) <input type="text" value="{html.escape(r["suggestion"], quote=True)}" data-suggestion="{html.escape(r["suggestion"], quote=True)}"></label>'
        f'<label class="chk"><input type="checkbox" class="unsure"> unsure / unreadable</label></div></div>'
        for r in rows)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OPEN100 tag labels</title>
<style>
body{{font-family:Calibri,Arial,sans-serif;margin:16px;background:#fff;color:#1b2a3a}}
header{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccd;z-index:1}}
.item{{display:flex;gap:16px;align-items:center;padding:10px 0;border-bottom:1px solid #eee}}
.item img{{max-width:420px;max-height:260px;border:1px solid #ccd}}
.meta{{display:flex;flex-direction:column;gap:6px;min-width:260px}}
input[type=text]{{font-size:18px;padding:4px 6px;width:260px}} .chk{{font-size:14px}}
button{{font-size:16px;padding:6px 14px;margin-right:8px}}
</style></head><body>
<header><b>OPEN100 tag labelling</b> &mdash; correct each tag exactly as printed (clear it if the symbol has no tag).
Progress saves in this browser. <span id="count"></span><br>
<button id="export">Export CSV</button><button id="reset">Restore suggestions</button></header>
{items}
<script>
const KEY='open100-labels-v1';
let saved={{}}; try{{saved=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
const items=[...document.querySelectorAll('.item')];
function store(){{const s={{}};items.forEach(d=>{{s[d.dataset.key]={{tag:d.querySelector('input[type=text]').value,unsure:d.querySelector('.unsure').checked}}}});
  try{{localStorage.setItem(KEY,JSON.stringify(s))}}catch(e){{}}
  document.getElementById('count').textContent=items.length+' symbols'}}
items.forEach(d=>{{const v=saved[d.dataset.key]; if(v){{d.querySelector('input[type=text]').value=v.tag; d.querySelector('.unsure').checked=v.unsure}}
  d.querySelectorAll('input').forEach(i=>i.addEventListener('change',store))}});
store();
document.getElementById('export').onclick=()=>{{
  const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
  const lines=['plan,node,tag,unsure,suggestion'].concat(items.map(d=>{{const k=d.dataset.key,p=k.slice(0,k.indexOf('_')),n=k.slice(k.indexOf('_')+1);const t=d.querySelector('input[type=text]');
    return [p,n,q(t.value.trim()),d.querySelector('.unsure').checked?1:0,q(t.dataset.suggestion)].join(',')}}));
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{{type:'text/csv'}})); a.download='labels.csv'; a.click()}};
document.getElementById('reset').onclick=()=>{{if(confirm('Discard your edits and restore the OCR suggestions?')){{
  items.forEach(d=>{{const t=d.querySelector('input[type=text]'); t.value=t.dataset.suggestion; d.querySelector('.unsure').checked=false}}); store()}}}};
</script></body></html>"""


if __name__ == "__main__":
    main()
