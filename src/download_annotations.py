"""Download the per-sheet .npy annotations of Dataset-P&ID (Paliwal et al. 2021) into data/raw/<sheet>/.

Usage: python src/download_annotations.py
Safe to re-run: sheets that already have all 7 valid files are skipped.
"""
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_FOLDER_ID = "1gMm_YKBZtXB3qUKUpI-LF1HE_MgzwfeR"
OUT = Path("data/raw")
FILES_PER_SHEET = 7
NPY_MAGIC = b"\x93NUMPY"
ENTRY = re.compile(r'href="(https://drive\.google\.com/[^"]+)"[^>]*>.*?flip-entry-title">([^<]+)<')


def fetch(url):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)  # back off: Drive throttles bursts


def list_folder(folder_id):
    html = fetch(f"https://drive.google.com/embeddedfolderview?id={folder_id}#list").decode("utf-8", "replace")
    return [(name, url) for url, name in ENTRY.findall(html)]


def valid_files(folder):
    return [f for f in folder.glob("*.npy") if f.read_bytes()[:6] == NPY_MAGIC]


def download_sheet(entry):
    name, url = entry
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    if len(valid_files(dest)) == FILES_PER_SHEET:
        return name, []
    errors = []
    try:
        files = list_folder(url.rstrip("/").split("/")[-1])
    except Exception as e:
        return name, [f"listing failed: {e}"]
    for fname, furl in files:
        target = dest / fname
        if target.exists() and target.read_bytes()[:6] == NPY_MAGIC:
            continue
        file_id = re.search(r"/d/([^/]+)/", furl)
        if not fname.endswith(".npy") or not file_id:
            errors.append(f"unexpected entry {fname}")
            continue
        try:
            data = fetch(f"https://drive.google.com/uc?export=download&id={file_id.group(1)}")
        except Exception as e:
            errors.append(f"{fname}: {e}")
            continue
        if data[:6] != NPY_MAGIC:  # Drive returns an HTML page when it throttles
            errors.append(f"{fname}: not an npy file ({len(data)} bytes)")
            continue
        target.write_bytes(data)
    if len(valid_files(dest)) != FILES_PER_SHEET:
        errors.append(f"has {len(valid_files(dest))} valid files, expected {FILES_PER_SHEET}")
    return name, errors


def main():
    sheets = [(n, u) for n, u in list_folder(ROOT_FOLDER_ID) if n.isdigit()]
    print(f"sheets listed: {len(sheets)}", flush=True)
    failed = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (name, errors) in enumerate(pool.map(download_sheet, sheets), 1):
            if errors:
                failed[name] = errors
            if i % 50 == 0:
                print(f"  {i}/{len(sheets)} done, {len(failed)} with problems", flush=True)
    total = sum(len(valid_files(d)) for d in OUT.iterdir() if d.is_dir())
    print(f"valid files on disk: {total} (expected {len(sheets) * FILES_PER_SHEET})")
    for name, errors in sorted(failed.items(), key=lambda x: int(x[0])):
        print(f"  sheet {name}: {errors}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
