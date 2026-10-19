"""Summarize build/train logs: the mean per-step loss and bits/byte over the last logged
steps of each run. Runs from the same seed draw the same batches, so these means compare
them on identical data (a paired comparison, much less noisy than the ema)."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

LINE = re.compile(r"step\s+(\d+)\s+lr \S+\s+loss (\S+) ema \S+\s+bits/byte (\S+) ema \S+.*?(\S+) s/step")


@dataclass
class Row:
    step: int
    loss: float
    bpb: float
    sec: float


def rows(path: Path) -> list[Row]:
    out: list[Row] = []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if m is not None:
            out.append(Row(int(m[1]), float(m[2]), float(m[3]), float(m[4])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--last", type=int, default=2000, help="steps to average over")
    args = ap.parse_args()
    for path in args.logs:
        rs = rows(path)
        if not rs:
            print(f"{path}: no step lines")
        else:
            end = rs[-1].step
            tail = [r for r in rs if r.step > end - args.last]
            loss = sum(r.loss for r in tail) / len(tail)
            bpb = sum(r.bpb for r in tail) / len(tail)
            print(f"{path}: step {end}, last {args.last} steps: loss {loss:.4f}, bits/byte {bpb:.4f}, {rs[-1].sec:.3f} s/step")


if __name__ == "__main__":
    main()
