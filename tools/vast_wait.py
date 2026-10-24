"""Wait until a vast.ai instance is running, then print its ssh address. Exits non-zero
if the instance stops or fails instead, or after --timeout seconds. Run from the
project root, with the vastai CLI in the venv and its API key set:

    .venv/bin/python3 tools/vast_wait.py INSTANCE_ID
"""

import argparse
import json
import subprocess
import sys
import time
from typing import TypedDict

VASTAI = ".venv/bin/vastai"
FAILED = {"exited", "offline", "unknown"}


class Instance(TypedDict):
    actual_status: str | None
    status_msg: str | None
    ssh_host: str
    ssh_port: int
    public_ipaddr: str
    ports: dict[str, list[dict[str, str]]] | None


def show(iid: str) -> Instance:
    out = subprocess.run([VASTAI, "show", "instance", iid, "--raw"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def direct_port(inst: Instance) -> str | None:
    ports = inst["ports"] or {}
    mapped = ports.get("22/tcp")
    if mapped:
        return mapped[0]["HostPort"]
    else:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    t0 = time.monotonic()
    last = ""
    while time.monotonic() - t0 < args.timeout:
        inst = show(args.id)
        state = inst["actual_status"] or "pending"
        msg = (inst["status_msg"] or "").strip()
        if f"{state} {msg}" != last:
            print(f"{time.monotonic() - t0:6.0f}s {state}: {msg}", flush=True)
            last = f"{state} {msg}"
        match state:
            case "running":
                print(f"proxy:  ssh -p {inst['ssh_port']} root@{inst['ssh_host']}")
                port = direct_port(inst)
                if port is not None:
                    print(f"direct: ssh -p {port} root@{inst['public_ipaddr'].strip()}")
                return
            case s if s in FAILED or msg.startswith("Error"):
                sys.exit(f"instance {args.id} is {s}: {msg}")
            case _:
                time.sleep(15)
    sys.exit(f"instance {args.id} not running after {args.timeout:.0f} s")


if __name__ == "__main__":
    main()
