"""Convert RWKV's World vocabulary (rwkv_vocab_v20230424.txt: one "id literal length"
line per token, the literal a Python str or bytes) to the tokenizer file the C side
reads: the count, every token's bytes in id order, then zero merges, which makes the
tokenizer encode by greedy longest match. Id 0 is RWKV's end-of-text token, which text
never encodes to; it gets a placeholder entry. Run from the project root:

    python3 tools/world_vocab.py [SRC [DST]]
"""

import argparse
import ast
from pathlib import Path

END = b"<|endoftext|>"


def escape(raw: bytes) -> bytes:
    return raw.replace(b"\\", b"\\\\").replace(b"\n", b"\\n")


def parse(line: str) -> tuple[int, bytes]:
    head, rest = line.split(" ", 1)
    lit, length = rest.rsplit(" ", 1)
    value = ast.literal_eval(lit)
    raw = value.encode() if isinstance(value, str) else value
    if not isinstance(raw, bytes) or len(raw) != int(length):
        raise SystemExit(f"bad vocab line: {line!r}")
    else:
        return int(head), raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, nargs="?", default=Path("models/rwkv_vocab_v20230424.txt"))
    ap.add_argument("dst", type=Path, nargs="?", default=Path("models/world.txt"))
    args = ap.parse_args()
    tokens = dict(parse(l) for l in args.src.read_text(encoding="utf-8").splitlines() if l)
    count = max(tokens) + 1
    missing = [i for i in range(1, count) if i not in tokens]
    if missing:
        raise SystemExit(f"{args.src}: ids missing, first {missing[0]}")
    else:
        with args.dst.open("wb") as f:
            f.write(f"{count}\n".encode())
            for i in range(count):
                raw = tokens.get(i, END)
                f.write(f"{len(raw)} ".encode() + escape(raw) + b"\n")
            f.write(b"0\n")
        print(f"{args.dst}: {count} tokens")


if __name__ == "__main__":
    main()
