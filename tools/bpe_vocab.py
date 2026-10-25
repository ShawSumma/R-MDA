"""Convert a Hugging Face byte-level BPE tokenizer.json into the tokenizer file
src/tokenizer reads: the vocab as raw bytes in id order (added tokens included), then
the merges in rank order as "left right merged" ids. Refuses model options the C
encoder does not implement.

    python3 tools/bpe_vocab.py models/qwen38fn/tokenizer.json models/qwen.txt
"""

import argparse
import json
from pathlib import Path


def byte_decoder() -> dict[str, int]:
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    chars = list(keep)
    extra = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(keep, chars)}


def escape(raw: bytes) -> bytes:
    return raw.replace(b"\\", b"\\\\").replace(b"\n", b"\\n")


def merge_pair(m: str | list[str]) -> tuple[str, str]:
    match m:
        case str():
            a, b = m.split(" ")
            return a, b
        case [a, b]:
            return a, b
        case _:
            raise SystemExit(f"bad merge {m!r}")


def check_model(model: dict[str, object]) -> None:
    unsupported = {
        "type": model.get("type") != "BPE",
        "ignore_merges": bool(model.get("ignore_merges")),
        "byte_fallback": bool(model.get("byte_fallback")),
        "dropout": model.get("dropout") is not None,
        "continuing_subword_prefix": bool(model.get("continuing_subword_prefix")),
        "end_of_word_suffix": bool(model.get("end_of_word_suffix")),
    }
    bad = [k for k, v in unsupported.items() if v]
    if bad:
        raise SystemExit(f"unsupported BPE options: {' '.join(bad)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    model = spec["model"]
    check_model(model)
    dec = byte_decoder()
    vocab: dict[str, int] = model["vocab"]
    added: list[dict[str, str | int]] = spec["added_tokens"]
    count = max([*vocab.values(), *(int(a["id"]) for a in added)]) + 1
    entries: list[bytes] = [b""] * count
    for tok, i in vocab.items():
        entries[i] = bytes(dec[c] for c in tok)
    for a in added:
        entries[int(a["id"])] = str(a["content"]).encode()
    empty = [i for i, e in enumerate(entries) if not e]
    if empty:
        raise SystemExit(f"ids without a token: {empty[:8]}")
    merges = [merge_pair(m) for m in model["merges"]]
    with args.out.open("wb") as f:
        f.write(f"{count}\n".encode())
        for raw in entries:
            f.write(f"{len(raw)} ".encode() + escape(raw) + b"\n")
        f.write(f"{len(merges)}\n".encode())
        for a, b in merges:
            f.write(f"{vocab[a]} {vocab[b]} {vocab[a + b]}\n".encode())
    print(f"{args.out}: {count} tokens, {len(merges)} merges")


if __name__ == "__main__":
    main()
