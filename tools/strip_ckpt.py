"""Copy a checkpoint without its optimizer state, a third of the size. sample, generate
and probe_acts load the copy; train refuses to resume from it.

    python3 tools/strip_ckpt.py SRC DST
"""

import argparse
import struct
from pathlib import Path

HDR = struct.Struct("<Q12IIxxxxQQ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    with args.src.open("rb") as f:
        hdr = f.read(HDR.size)
        n_all = HDR.unpack(hdr)[15]
        weights = f.read(4 * n_all)
    if len(weights) != 4 * n_all:
        raise SystemExit(f"{args.src}: truncated")
    else:
        args.dst.write_bytes(hdr + weights)
        print(f"{args.dst}: {n_all} weights, {args.dst.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
