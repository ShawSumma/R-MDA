"""Print every tensor in a safetensors file: name, dtype, shape. Names can be
filtered by a substring.

    python3 tools/safet_header.py FILE [SUBSTRING]
"""

import argparse
import json
import struct
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("part", nargs="?", default="")
    args = ap.parse_args()
    with args.file.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header: dict[str, dict[str, str | list[int]]] = json.loads(f.read(n))
    for name, info in sorted(header.items()):
        if name != "__metadata__" and args.part in name:
            print(name, info["dtype"], info["shape"])


if __name__ == "__main__":
    main()
