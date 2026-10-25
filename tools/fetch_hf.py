"""Download the files of a Hugging Face model repository whose paths match the given
globs into DEST, several at a time, resuming partial files. Uses HF_TOKEN for gated
repositories. Run from anywhere:

    python3 tools/fetch_hf.py REPO DEST [--include GLOB ...] [--jobs N]
"""

import argparse
import fnmatch
import json
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

API = "https://huggingface.co/api/models"
BASE = "https://huggingface.co"
CHUNK = 1 << 22


@dataclass
class Entry:
    path: str
    size: int


def request(url: str, start: int = 0) -> urllib.request.Request:
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if start > 0:
        req.add_header("Range", f"bytes={start}-")
    return req


def listing(repo: str) -> list[Entry]:
    with urllib.request.urlopen(request(f"{API}/{repo}/tree/main?recursive=1")) as r:
        items: list[dict[str, object]] = json.load(r)
    out: list[Entry] = []
    for it in items:
        if it.get("type") == "file":
            size = it.get("size")
            out.append(Entry(str(it["path"]), size if isinstance(size, int) else 0))
    return out


def fetch(repo: str, dest: Path, e: Entry) -> str:
    out = dest / e.path
    part = out.with_name(out.name + ".part")
    if out.exists() and out.stat().st_size == e.size:
        return f"{e.path}: present"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        start = part.stat().st_size if part.exists() else 0
        with urllib.request.urlopen(request(f"{BASE}/{repo}/resolve/main/{e.path}", start)) as r, part.open("ab") as f:
            shutil.copyfileobj(r, f, CHUNK)
        part.rename(out)
        return f"{e.path}: {e.size / 1e9:.2f} GB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--include", nargs="+", default=["*"])
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    files = [e for e in listing(args.repo) if any(fnmatch.fnmatch(e.path, g) for g in args.include)]
    if not files:
        sys.exit(f"{args.repo}: nothing matches {' '.join(args.include)}")
    else:
        print(f"{args.repo}: {len(files)} files, {sum(e.size for e in files) / 1e9:.1f} GB", flush=True)
        with ThreadPoolExecutor(args.jobs) as pool:
            for line in pool.map(lambda e: fetch(args.repo, args.dest, e), files):
                print(line, flush=True)


if __name__ == "__main__":
    main()
