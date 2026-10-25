"""Compare build/check_qwen's logits (models/qwen_check.bin: token ids and our logits)
with Hugging Face's Qwen4-Exp text model, run one decoder layer at a time in fp32 on the
CPU from the safetensors shards (the n-gram embedding is looked up row by row instead of
loaded). Also checks our token ids against Hugging Face tokenizers on the same text.
Prints the max logit error, argmax agreement, KL from the reference to ours, and both
models' loss on the text. Needs torch, transformers with qwen4_exp, and tokenizers.

    python3 tools/check_qwen.py [models/qwen38fn] [models/qwen_check.bin] [data/corpus.txt]
"""

import argparse
import json
import math
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
)

Tensor = torch.Tensor
PREFIX = "model.language_model."


class Shards:
    def __init__(self, root: Path) -> None:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.root = root
        self.where: dict[str, str] = index["weight_map"]

    def get(self, name: str) -> Tensor:
        with safe_open(self.root / self.where[name], framework="pt") as f:
            return f.get_tensor(name)

    def rows(self, name: str, idx: Tensor) -> Tensor:
        with safe_open(self.root / self.where[name], framework="pt") as f:
            sl = f.get_slice(name)
            return torch.stack([sl[int(i) : int(i) + 1][0] for i in idx.flatten()]).view(*idx.shape, -1)

    def prefixed(self, prefix: str) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for k in self.where:
            if k.startswith(prefix) and ".ngram_embedding." not in k:
                t = self.get(k)
                out[k[len(prefix) :]] = t.float() if t.is_floating_point() else t
        return out


class NGramLookup(torch.nn.Module):
    def __init__(self, shards: Shards, layer: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.empty(0), persistent=False)
        self.shards = shards
        self.base = f"{PREFIX}layers.{layer}.ple.ple_embedding.ngram_embedding.shard_"
        self.per = shards.get(self.base + "0.weight").shape[0]

    def forward(self, ids: Tensor) -> Tensor:
        flat = ids.flatten()
        out = torch.stack([self.shards.rows(f"{self.base}{int(i) // self.per}.weight", torch.tensor([int(i) % self.per]))[0] for i in flat])
        return out.view(*ids.shape, -1).float()


def build_layer(cfg: Qwen4ExpTextConfig, shards: Shards, layer: int) -> Qwen4ExpTextDecoderLayer:
    with torch.device("meta"):
        mod = Qwen4ExpTextDecoderLayer(cfg, layer)
    if mod.ple is not None:
        mod.ple.ple_embedding.ngram_embedding = NGramLookup(shards, layer)
    mod.load_state_dict(shards.prefixed(f"{PREFIX}layers.{layer}."), strict=True, assign=True)
    return mod.eval()


def reference(cfg: Qwen4ExpTextConfig, shards: Shards, ids: Tensor) -> Tensor:
    n = ids.shape[0]
    emb = shards.get(PREFIX + "embed_tokens.weight")[ids].float()
    h = emb.unsqueeze(0).repeat(1, 1, cfg.hc_count)
    rotary = Qwen4ExpTextRotaryEmbedding(cfg)
    pos = torch.arange(n).view(1, 1, n).expand(3, 1, n)
    pe = rotary(h, pos)
    mask = torch.full((n, n), torch.finfo(torch.float32).min).triu(1).view(1, 1, n, n)
    batch = ids.view(1, n)
    if n > cfg.indexer_budget:
        raise SystemExit(f"{n} tokens: past the indexer budget {cfg.indexer_budget}, the reference would need its sparse mask")
    for layer in range(cfg.num_hidden_layers):
        mod = build_layer(cfg, shards, layer)
        if mod.layer_type != "linear_attention":
            mod.self_attn.indexer.forward = lambda *a: torch.zeros_like(mask)
        with torch.no_grad():
            h = mod(h, position_embeddings=pe, attention_mask=mask, conv_mask=None, past_key_values=None, ple_input_ids=batch)
        print(f"layer {layer}: |h| {h.abs().max().item():.2f}", flush=True)
    with torch.device("meta"):
        mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    mixer.load_state_dict(shards.prefixed(PREFIX + "hyper_connection_mixer."), strict=True, assign=True)
    with torch.no_grad():
        x = mixer(h)[0]
        return x @ shards.get("lm_head.weight").float().T


def loss(logits: Tensor, ids: Tensor) -> float:
    return F.cross_entropy(logits[:-1], ids[1:]).item() / math.log(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, nargs="?", default=Path("models/qwen38fn"))
    ap.add_argument("ours", type=Path, nargs="?", default=Path("models/qwen_check.bin"))
    ap.add_argument("text", type=Path, nargs="?", default=Path("data/corpus.txt"))
    args = ap.parse_args()
    raw = args.ours.read_bytes()
    n, vocab = struct.unpack_from("<II", raw, 0)
    ids = torch.tensor(struct.unpack_from(f"<{n}I", raw, 8), dtype=torch.long)
    mine = torch.frombuffer(bytearray(raw[8 + 4 * n :]), dtype=torch.float32).view(n, vocab)
    text = args.text.read_bytes()[:65536].decode("utf-8", errors="ignore")
    hf = Tokenizer.from_file(str(args.model / "tokenizer.json")).encode(text, add_special_tokens=False).ids[:n]
    same = sum(int(a == b) for a, b in zip(hf, ids.tolist()))
    print(f"tokenizer: {same}/{n} ids equal Hugging Face's")
    cfg = Qwen4ExpTextConfig(**json.loads((args.model / "config.json").read_text())["text_config"])
    cfg._attn_implementation = "eager"
    torch.set_num_threads(32)
    ref = reference(cfg, Shards(args.model), ids)
    err = (ref - mine).abs().max().item()
    agree = (ref.argmax(-1) == mine.argmax(-1)).float().mean().item()
    kl = F.kl_div(mine.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean").item()
    print(f"{n} tokens: max |logit error| {err:.4f}, argmax agreement {100 * agree:.1f}%, KL(ref || ours) {kl:.6f} nats/token")
    print(f"loss: reference {loss(ref, ids):.4f} bits/token, ours {loss(mine, ids):.4f} bits/token")


if __name__ == "__main__":
    main()
