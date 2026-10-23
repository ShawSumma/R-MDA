"""Train one run per learning rate, all with the same other arguments, then compare
them with logstat. Runs at the same step draw the same batches, so the comparison is
paired. Each run writes models/NAME-LR.bin and its log models/NAME-LR.log; an existing
checkpoint is refused, since train would resume it. Run from the project root:

    .venv/bin/python3 tools/sweep.py NAME STOP ACCUM OPT MODE BATCH --lr 1e-4 3e-4 1e-3
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train(ckpt: Path, lr: str, stop: int, rest: list[str]) -> Path:
    log = ckpt.with_suffix(".log")
    print(f"{ckpt}: lr {lr}", flush=True)
    with log.open("w") as f:
        subprocess.run(["build/train", lr, str(ckpt), str(stop), *rest], stdout=f, check=True)
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("stop", type=int)
    ap.add_argument("accum")
    ap.add_argument("opt")
    ap.add_argument("mode")
    ap.add_argument("batch")
    ap.add_argument("--lr", nargs="+", required=True)
    ap.add_argument("--last", type=int, default=500, help="steps logstat averages over")
    args = ap.parse_args()
    ckpts = [Path(f"models/{args.name}-{lr}.bin") for lr in args.lr]
    taken = [c for c in ckpts if c.exists()]
    if taken:
        sys.exit(f"already exist: {' '.join(map(str, taken))}")
    else:
        rest = [args.accum, args.opt, args.mode, args.batch]
        logs = [train(c, lr, args.stop, rest) for c, lr in zip(ckpts, args.lr)]
        subprocess.run([sys.executable, "tools/logstat.py", *map(str, logs), "--last", str(args.last)], check=True)


if __name__ == "__main__":
    main()
