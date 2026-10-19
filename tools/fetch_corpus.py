"""Download FineWeb-Edu shards and write their text as the training corpus.

FineWeb-Edu is Common Crawl web text filtered by an educational-quality classifier;
its sample-10BT subset is 14 parquet shards of ~0.7B GPT-2 tokens each. The shards
are kept in data/fineweb-edu/ (a rerun or a larger --shards reuses them; an
interrupted download resumes), and their documents are written to data/corpus.txt
separated by blank lines. Run from the project root, in the venv that has pyarrow:

    .venv/bin/python3 tools/fetch_corpus.py [--shards N]

then retrain the vocab (build/train_tokenizer) before training.
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
N_SHARDS = 14
SHARD_DIR = Path("data/fineweb-edu")
OUT = Path("data/corpus.txt")


def download(name: str, dst: Path) -> Path:
    part = dst.with_name(name + ".part")
    have = 0
    if part.exists():
        have = part.stat().st_size
    req = urllib.request.Request(f"{BASE}/{name}", headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req) as r:
        mode = "wb"
        start = 0
        if r.status == 206:
            mode = "ab"
            start = have
        print(f"{name}: downloading from byte {start}", flush=True)
        with part.open(mode) as f:
            shutil.copyfileobj(r, f, 1 << 24)
    part.rename(dst)
    return dst


def fetch(name: str) -> Path:
    dst = SHARD_DIR / name
    if dst.exists():
        return dst
    else:
        return download(name, dst)


def write_corpus(shards: list[Path]) -> None:
    tmp = OUT.with_name(OUT.name + ".tmp")
    docs = 0
    with tmp.open("wb") as f:
        for path in shards:
            for batch in pq.ParquetFile(path).iter_batches(columns=["text"]):
                for text in batch.column(0).to_pylist():
                    f.write(str(text).strip().encode())
                    f.write(b"\n\n")
                    docs += 1
    tmp.replace(OUT)
    print(f"{OUT}: {docs} documents, {OUT.stat().st_size / 1e9:.2f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=1, choices=range(1, N_SHARDS + 1), metavar=f"1..{N_SHARDS}")
    args = ap.parse_args()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    write_corpus([fetch(f"{i:03d}_00000.parquet") for i in range(args.shards)])


if __name__ == "__main__":
    main()
