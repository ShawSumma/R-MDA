"""Train several variants with the same lr, steps, accumulation, optimizer and batch, then
evaluate each with build/sample and print one table. A variant is NAME MODE INIT TEACHER
KD: INIT is a checkpoint to start from (copied, since train resumes its checkpoint),
TEACHER the teacher weights for distillation and KD the teacher's share of the target;
"-" leaves INIT or TEACHER out. Runs at the same
step draw the same batches, so the comparison is paired. Each variant writes
models/GROUP-NAME.bin, .log (train) and .eval (sample), and the table also goes to
models/GROUP.table; an existing checkpoint is refused. Run from the project root:

    .venv/bin/python3 tools/compare.py GROUP LR STOP ACCUM OPT BATCH \\
        --run distill block16 models/smollm2.bin models/smollm2.safetensors 1 \\
        --run scratch block16 - - 1
"""

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

NELBO = re.compile(r"held-out NELBO (\S+) bits/byte")
AR = re.compile(r"next-token NLL (\S+) bits/byte")
FILL = re.compile(r"masked-fill accuracy .* = (\S+)%")
SPEED = re.compile(r"(\S+) s/step")


@dataclass
class Variant:
    name: str
    mode: str
    init: str
    teacher: str
    kd: str


@dataclass
class Result:
    name: str
    nelbo: str
    ar: str
    fill: str
    sec: str


def last_match(pat: re.Pattern[str], text: str) -> str:
    found = pat.findall(text)
    if found:
        return found[-1]
    else:
        return "-"


def run(v: Variant, group: str, common: list[str]) -> Result:
    lr, stop, accum, opt, batch = common
    ckpt = Path(f"models/{group}-{v.name}.bin")
    if v.init != "-":
        shutil.copyfile(v.init, ckpt)
    teacher = [] if v.teacher == "-" else [v.teacher, v.kd]
    print(f"{ckpt}: {v.mode}, init {v.init}, teacher {v.teacher} at {v.kd}", flush=True)
    log = ckpt.with_suffix(".log")
    with log.open("w") as f:
        subprocess.run(["build/train", lr, str(ckpt), stop, accum, opt, v.mode, batch, *teacher], stdout=f, stderr=subprocess.STDOUT, check=True)
    ev = ckpt.with_suffix(".eval")
    with ev.open("w") as f:
        subprocess.run(["build/sample", str(ckpt), v.mode], stdout=f, check=True)
    text = ev.read_text(errors="replace")
    return Result(v.name, last_match(NELBO, text), last_match(AR, text), last_match(FILL, text), last_match(SPEED, log.read_text(errors="replace")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group")
    ap.add_argument("lr")
    ap.add_argument("stop")
    ap.add_argument("accum")
    ap.add_argument("opt")
    ap.add_argument("batch")
    ap.add_argument("--run", nargs=5, action="append", required=True, metavar=("NAME", "MODE", "INIT", "TEACHER", "KD"))
    args = ap.parse_args()
    variants = [Variant(*r) for r in args.run]
    taken = [p for v in variants if (p := Path(f"models/{args.group}-{v.name}.bin")).exists()]
    if taken:
        sys.exit(f"already exist: {' '.join(map(str, taken))}")
    else:
        common = [args.lr, args.stop, args.accum, args.opt, args.batch]
        results = [run(v, args.group, common) for v in variants]
        lines = [f"{'run':<16} {'NELBO bpb':>10} {'AR bpb':>8} {'fill %':>7} {'s/step':>7}"]
        lines += [f"{r.name:<16} {r.nelbo:>10} {r.ar:>8} {r.fill:>7} {r.sec:>7}" for r in results]
        table = "\n".join(lines) + "\n"
        Path(f"models/{args.group}.table").write_text(table)
        print(table, end="")


if __name__ == "__main__":
    main()
