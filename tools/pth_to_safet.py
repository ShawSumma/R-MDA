"""Convert a PyTorch .pth state dict (a zip of a pickle plus raw storages) to a
safetensors file without torch: a restricted unpickler maps torch's tensor rebuild
calls to (storage, offset, shape, stride) records, and each tensor's bytes are copied
from its storage. Only contiguous tensors are supported. Prints every tensor's name,
dtype and shape. Run from anywhere:

    python3 tools/pth_to_safet.py SRC.pth DST.safetensors
"""

import argparse
import json
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

DTYPES = {
    "BFloat16Storage": ("BF16", 2),
    "HalfStorage": ("F16", 2),
    "FloatStorage": ("F32", 4),
}


@dataclass
class Storage:
    key: str
    dtype: str
    size: int


@dataclass
class Tensor:
    storage: Storage
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass
class StorageType:
    name: str


def rebuild(storage: Storage, offset: int, shape: tuple[int, ...], stride: tuple[int, ...], *_: object) -> Tensor:
    return Tensor(storage, offset, tuple(shape), tuple(stride))


class Loader(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        match (module, name):
            case ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild
            case ("collections", "OrderedDict"):
                return OrderedDict
            case ("torch", n) if n in DTYPES:
                return StorageType(n)
            case _:
                raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

    def persistent_load(self, pid: tuple[object, ...]) -> Storage:
        _, kind, key, _, _ = pid
        if not isinstance(kind, StorageType) or not isinstance(key, str):
            raise pickle.UnpicklingError(f"unsupported storage {pid!r}")
        else:
            dtype, size = DTYPES[kind.name]
            return Storage(key, dtype, size)


def contiguous(t: Tensor) -> bool:
    expect = 1
    for dim, st in zip(reversed(t.shape), reversed(t.stride)):
        if dim > 1 and st != expect:
            return False
        expect *= dim
    return True


def numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def copy_tensor(z: zipfile.ZipFile, root: str, t: Tensor, out: BinaryIO) -> None:
    with z.open(f"{root}/data/{t.storage.key}") as f:
        f.seek(t.offset * t.storage.size)
        out.write(f.read(numel(t.shape) * t.storage.size))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    with zipfile.ZipFile(args.src) as z:
        pkl = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        root = pkl.rsplit("/", 1)[0]
        with z.open(pkl) as f:
            state: dict[str, Tensor] = Loader(f).load()
        bad = [k for k, t in state.items() if not contiguous(t)]
        if bad:
            raise SystemExit(f"non-contiguous tensors: {' '.join(bad)}")
        else:
            header: dict[str, object] = {}
            off = 0
            for k, t in state.items():
                n = numel(t.shape) * t.storage.size
                header[k] = {"dtype": t.storage.dtype, "shape": list(t.shape), "data_offsets": [off, off + n]}
                off += n
                print(f"{k} {t.storage.dtype} {list(t.shape)}")
            raw = json.dumps(header, separators=(",", ":")).encode()
            raw += b" " * (-len(raw) % 8)
            with args.dst.open("wb") as out:
                out.write(len(raw).to_bytes(8, "little"))
                out.write(raw)
                for t in state.values():
                    copy_tensor(z, root, t, out)
            print(f"{args.dst}: {len(state)} tensors, {off / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
