"""Download SmolLM2-135M's tokenizer and weights into models/ (tokenizer.json,
smollm2.safetensors; kept for reruns) and write models/tokenizer.txt: the byte-level BPE
vocab as raw bytes in id order, then the merges in rank order as "left right merged" ids.
build/import turns the weights into a checkpoint. Run from the project root:

    python3 tools/fetch_smollm.py
"""

import json
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main"
SPEC = Path("models/tokenizer.json")
WEIGHTS = Path("models/smollm2.safetensors")
OUT = Path("models/tokenizer.txt")


def fetch(name: str, dst: Path) -> None:
    if not dst.exists():
        part = dst.with_name(dst.name + ".part")
        urllib.request.urlretrieve(f"{BASE}/{name}", part)
        part.rename(dst)


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


def write_tokenizer() -> None:
    spec = json.loads(SPEC.read_text())
    model = spec["model"]
    dec = byte_decoder()
    vocab: dict[str, int] = model["vocab"]
    count = max(vocab.values()) + 1
    entries: list[bytes] = [b""] * count
    for tok, i in vocab.items():
        entries[i] = bytes(dec[c] for c in tok)
    for added in spec["added_tokens"]:
        entries[added["id"]] = added["content"].encode()
    merges = [merge_pair(m) for m in model["merges"]]
    with OUT.open("wb") as f:
        f.write(f"{count}\n".encode())
        for raw in entries:
            f.write(f"{len(raw)} ".encode() + escape(raw) + b"\n")
        f.write(f"{len(merges)}\n".encode())
        for a, b in merges:
            f.write(f"{vocab[a]} {vocab[b]} {vocab[a + b]}\n".encode())
    print(f"{OUT}: {count} tokens, {len(merges)} merges")


def main() -> None:
    fetch("tokenizer.json", SPEC)
    fetch("model.safetensors", WEIGHTS)
    write_tokenizer()


if __name__ == "__main__":
    main()
