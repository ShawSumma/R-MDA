"""Download a mixture of text datasets and write it as the training corpus.

Each source is a Hugging Face dataset of parquet files with a text column. For each
source, files are taken in listing order until its text reaches the requested
gigabytes; they are kept in data/<source>/ (reruns reuse them, interrupted downloads
resume). Every document is written to data/corpus.txt followed by a blank line,
source after source; training samples random windows, so each source's share of the
bytes is its share of the mixture. Run from the project root, in the venv that has
pyarrow:

    .venv/bin/python3 tools/fetch_corpus.py fineweb-edu=4 dclm=3 finemath=1

The token-id cache (data/corpus.txt.ids) is rebuilt on the next training run.
"""

import argparse
import fnmatch
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pyarrow.parquet as pq

API = "https://huggingface.co/api/datasets"
BASE = "https://huggingface.co/datasets"
DATA = Path("data")
OUT = DATA / "corpus.txt"


@dataclass
class Source:
    repo: str
    dir: str
    glob: str
    column: str


SOURCES = {
    "fineweb-edu": Source("HuggingFaceFW/fineweb-edu", "sample/100BT", "*.parquet", "text"),
    "dclm": Source("mlfoundations/dclm-baseline-1.0-parquet", "filtered", "*.parquet", "text"),
    "finemath": Source("HuggingFaceTB/finemath", "finemath-4plus", "*.parquet", "text"),
}


def next_page(link: str | None) -> str | None:
    for part in (link or "").split(","):
        if 'rel="next"' in part:
            return part.split(";")[0].strip().strip("<>")
    return None


def listing(src: Source) -> list[str]:
    url: str | None = f"{API}/{src.repo}/tree/main/{src.dir}?recursive=1"
    paths: list[str] = []
    while url is not None:
        with urllib.request.urlopen(url) as r:
            items: list[dict[str, object]] = json.load(r)
            url = next_page(r.headers.get("Link"))
        paths += [str(it["path"]) for it in items if it.get("type") == "file"]
    return sorted(p for p in paths if fnmatch.fnmatch(p, f"{src.dir}/{src.glob}"))


def download(src: Source, path: str, dst: Path) -> Path:
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(f"{BASE}/{src.repo}/resolve/main/{path}", headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req) as r:
        mode = "ab" if r.status == 206 else "wb"
        print(f"{dst}: downloading", flush=True)
        with part.open(mode) as f:
            shutil.copyfileobj(r, f, 1 << 24)
    part.rename(dst)
    return dst


def fetch(name: str, src: Source, path: str) -> Path:
    dst = DATA / name / path.replace("/", "_")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    else:
        return download(src, path, dst)


def write_source(f: BinaryIO, name: str, src: Source, budget: float) -> int:
    written = 0
    for path in listing(src):
        if written >= budget:
            break
        for batch in pq.ParquetFile(fetch(name, src, path)).iter_batches(columns=[src.column]):
            for text in batch.column(0).to_pylist():
                raw = str(text).strip().encode() + b"\n\n"
                f.write(raw)
                written += len(raw)
    print(f"{name}: {written / 1e9:.2f} GB", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mix", nargs="+", metavar="SOURCE=GB", help=f"sources: {' '.join(SOURCES)}")
    args = ap.parse_args()
    parts = [m.split("=", 1) for m in args.mix]
    unknown = [name for name, _ in parts if name not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown sources: {' '.join(unknown)}")
    else:
        DATA.mkdir(exist_ok=True)
        tmp = OUT.with_name(OUT.name + ".tmp")
        with tmp.open("wb") as f:
            total = sum(write_source(f, name, SOURCES[name], float(gb) * 1e9) for name, gb in parts)
        tmp.replace(OUT)
        print(f"{OUT}: {total / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
