"""Compare build/check_teacher's logits (models/teacher_check.bin) with SmolLM2-135M run by
Hugging Face transformers on the same token ids: max logit error, argmax agreement and
the mean KL from the reference to ours. Needs `pip install torch transformers`.

    python3 tools/check_teacher.py
"""

import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

SRC = Path("models/teacher_check.bin")
MODEL = "HuggingFaceTB/SmolLM2-135M"


def main() -> None:
    raw = SRC.read_bytes()
    n, vocab = struct.unpack_from("<II", raw, 0)
    ids = torch.tensor(struct.unpack_from(f"<{n}I", raw, 8), dtype=torch.long)
    mine = torch.frombuffer(bytearray(raw[8 + 4 * n :]), dtype=torch.float32).view(n, vocab)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    with torch.no_grad():
        ref = model(ids.unsqueeze(0)).logits[0]
    err = (ref - mine).abs().max().item()
    agree = (ref.argmax(-1) == mine.argmax(-1)).float().mean().item()
    kl = torch.nn.functional.kl_div(mine.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean").item()
    print(f"{n} tokens: max |logit error| {err:.4f}, argmax agreement {100 * agree:.1f}%, KL(ref || ours) {kl:.5f} nats/token")


if __name__ == "__main__":
    main()
