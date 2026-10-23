import argparse
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path

OLD_MAGIC = 0x38504B4341444D52
NEW_MAGIC = 0x39504B4341444D52
ALIGN = 128
HDR = struct.Struct("<Q8IIxxxxQQ")


@dataclass
class Cfg:
    d: int
    n_layer: int
    n_head: int
    seqlen: int
    batch: int
    ffn_hidden: int
    vocab: int
    block: int


def padded(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def sizes(cfg: Cfg, ada: int) -> list[int]:
    d = cfg.d
    hid = cfg.ffn_hidden
    out = [cfg.vocab * d, d, d]
    for _ in range(cfg.n_layer):
        out += [d * d] * 6 + [d, d, d, d * hid, d * hid, hid * d, ada]
    return out


def fold(old: array, d: int, power: float) -> array:
    half = d // 2
    cols = 4 * d
    out = array("f", [0.0]) * cols
    for r in range(half, d):
        base = r * cols
        for c in range(cols):
            out[c] += old[base + c]
    scale = float(half) ** power
    for c in range(cols):
        out[c] *= scale
    return out


def convert(old: array, cfg: Cfg, power: float) -> array:
    d = cfg.d
    old_sizes = sizes(cfg, d * 4 * d)
    new = array("f")
    at = 0
    for i, n in enumerate(old_sizes):
        block = old[at:at + n]
        if i >= 3 and (i - 3) % 13 == 12:
            block = fold(block, d, power)
        new.extend(block)
        new.extend(array("f", [0.0]) * (padded(len(block)) - len(block)))
        at += padded(n)
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description="fold the [d, 4d] ada matrix of a checkpoint into its [4d] vector")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    raw = args.src.read_bytes()
    fields = HDR.unpack_from(raw, 0)
    magic = fields[0]
    cfg = Cfg(*fields[1:9])
    step = fields[9]
    vocab_hash = fields[10]
    n_all = fields[11]
    if magic != OLD_MAGIC:
        raise SystemExit(f"{args.src}: not an unfolded checkpoint")
    arenas = []
    for i in range(3):
        a = array("f")
        a.frombytes(raw[HDR.size + i * 4 * n_all:HDR.size + (i + 1) * 4 * n_all])
        arenas.append(a)
    folded = [convert(arenas[0], cfg, -0.5), convert(arenas[1], cfg, -0.5), convert(arenas[2], cfg, 0.0)]
    new_all = len(folded[0])
    with args.dst.open("wb") as f:
        f.write(HDR.pack(NEW_MAGIC, cfg.d, cfg.n_layer, cfg.n_head, cfg.seqlen, cfg.batch, cfg.ffn_hidden, cfg.vocab, cfg.block, step, vocab_hash, new_all))
        for a in folded:
            f.write(a.tobytes())
    print(f"{args.dst}: step {step}, {n_all} -> {new_all} floats")


if __name__ == "__main__":
    main()
