"""Compare build/check_import's logits (models/import_check.bin: token ids and our fp32
logits) with the official RWKV-7 forward, written here from BlinkDL's
rwkv_v7_demo.py and run in fp32 on the original .pth. Prints the max logit error,
argmax agreement and the mean KL from the reference to ours. Needs torch.

    python3 tools/check_rwkv.py MODEL.pth [models/import_check.bin]
"""

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def shifted(x: Tensor) -> Tensor:
    return torch.cat([torch.zeros_like(x[:1]), x[:-1]]) - x


def time_mix(p: dict[str, Tensor], i: int, x: Tensor, v_first: Tensor | None, heads: int) -> tuple[Tensor, Tensor]:
    b = f"blocks.{i}.att."
    t, c = x.shape
    n = c // heads
    xx = shifted(x)
    xr, xw, xk, xv, xa, xg = (x + xx * p[b + m].view(c) for m in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))
    r = xr @ p[b + "receptance.weight"].T
    w = -F.softplus(-(p[b + "w0"].view(c) + torch.tanh(xw @ p[b + "w1"]) @ p[b + "w2"])) - 0.5
    k = xk @ p[b + "key.weight"].T
    v = xv @ p[b + "value.weight"].T
    if v_first is None:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(p[b + "v0"].view(c) + (xv @ p[b + "v1"]) @ p[b + "v2"])
    a = torch.sigmoid(p[b + "a0"].view(c) + (xa @ p[b + "a1"]) @ p[b + "a2"])
    g = torch.sigmoid(xg @ p[b + "g1"]) @ p[b + "g2"]
    kk = F.normalize((k * p[b + "k_k"].view(c)).view(t, heads, n), dim=-1, p=2.0).view(t, c)
    k = k * (1 + (a - 1) * p[b + "k_a"].view(c))
    decay = torch.exp(-torch.exp(w))
    state = torch.zeros(heads, n, n, dtype=x.dtype)
    out = torch.empty_like(x)
    for s in range(t):
        vk = v[s].view(heads, n, 1) @ k[s].view(heads, 1, n)
        ab = (-kk[s]).view(heads, n, 1) @ (kk[s] * a[s]).view(heads, 1, n)
        state = state * decay[s].view(heads, 1, n) + state @ ab + vk
        out[s] = (state @ r[s].view(heads, n, 1)).view(c)
    y = F.group_norm(out, heads, p[b + "ln_x.weight"], p[b + "ln_x.bias"], eps=64e-5)
    bonus = (r.view(t, heads, n) * k.view(t, heads, n) * p[b + "r_k"]).sum(-1, keepdim=True) * v.view(t, heads, n)
    y = y + bonus.view(t, c)
    return (y * g) @ p[b + "output.weight"].T, v_first


def channel_mix(p: dict[str, Tensor], i: int, x: Tensor) -> Tensor:
    b = f"blocks.{i}.ffn."
    k = x + shifted(x) * p[b + "x_k"].view(-1)
    k = torch.relu(k @ p[b + "key.weight"].T) ** 2
    return k @ p[b + "value.weight"].T


def layer_norm(p: dict[str, Tensor], name: str, x: Tensor) -> Tensor:
    return F.layer_norm(x, x.shape[-1:], p[name + ".weight"], p[name + ".bias"])


def forward(p: dict[str, Tensor], ids: Tensor) -> Tensor:
    c = p["emb.weight"].shape[1]
    heads = c // 64
    layers = 1 + max(int(k.split(".")[1]) for k in p if k.startswith("blocks."))
    x = layer_norm(p, "blocks.0.ln0", p["emb.weight"][ids])
    v_first: Tensor | None = None
    for i in range(layers):
        y, v_first = time_mix(p, i, layer_norm(p, f"blocks.{i}.ln1", x), v_first, heads)
        x = x + y
        x = x + channel_mix(p, i, layer_norm(p, f"blocks.{i}.ln2", x))
    return layer_norm(p, "ln_out", x) @ p["head.weight"].T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("ours", type=Path, nargs="?", default=Path("models/import_check.bin"))
    args = ap.parse_args()
    raw = args.ours.read_bytes()
    n, vocab = struct.unpack_from("<II", raw, 0)
    ids = torch.tensor(struct.unpack_from(f"<{n}I", raw, 8), dtype=torch.long)
    mine = torch.frombuffer(bytearray(raw[8 + 4 * n :]), dtype=torch.float32).view(n, vocab)
    sd = torch.load(args.model, map_location="cpu", weights_only=True)
    p = {k: t.float() for k, t in sd.items()}
    with torch.no_grad():
        ref = forward(p, ids)
    err = (ref - mine).abs().max().item()
    agree = (ref.argmax(-1) == mine.argmax(-1)).float().mean().item()
    kl = F.kl_div(mine.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean").item()
    print(f"{n} tokens: max |logit error| {err:.4f}, argmax agreement {100 * agree:.1f}%, KL(ref || ours) {kl:.6f} nats/token")


if __name__ == "__main__":
    main()
