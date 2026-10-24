"""Compare the C tokenizer with Hugging Face's on the start of the corpus: tokenizes the
first BYTES of data/corpus.txt (cut where the C side may cut) with the `tokenizers`
package and checks the ids against data/corpus.txt.ids, which build/train or build/sample
writes. Needs `pip install tokenizers` in the interpreter that runs it.

    python3 tools/check_tokenizer.py [--bytes N]
"""

import argparse
import struct
from array import array
from pathlib import Path

from tokenizers import Tokenizer

CORPUS = Path("data/corpus.txt")
IDS = Path("data/corpus.txt.ids")
SPEC = Path("models/tokenizer.json")
HDR = struct.Struct("<QQQQ")


def safe_cut(raw: bytes, at: int) -> int:
    i = at
    while i < len(raw) and (raw[i] != 0x0A or not 0x21 <= raw[i - 1] <= 0x7E):
        i += 1
    return i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bytes", type=int, default=20_000_000)
    args = ap.parse_args()
    with CORPUS.open("rb") as f:
        raw = f.read(args.bytes + 65536)
    text = raw[: safe_cut(raw, args.bytes)].decode()
    hf = Tokenizer.from_file(str(SPEC)).encode(text, add_special_tokens=False).ids
    with IDS.open("rb") as f:
        f.read(HDR.size)
        mine = array("I")
        mine.frombytes(f.read(4 * len(hf)))
    n = min(len(hf), len(mine))
    diff = next((i for i in range(n) if hf[i] != mine[i]), n)
    if diff == n and len(mine) == len(hf):
        print(f"{len(hf)} tokens from {len(text)} chars: identical")
    else:
        print(f"first difference at token {diff} of {len(hf)}: hf {hf[diff:diff + 8]} mine {list(mine[diff:diff + 8])}")


if __name__ == "__main__":
    main()
