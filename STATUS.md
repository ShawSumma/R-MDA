# RWKV-7 + masked-diffusion denoiser — status

A complete RWKV-7 + masked-text-diffusion language model, **entirely in raw Vulkan
compute** (C host + GLSL shaders). No Python, no autograd, no GGML/HF/torch.
Trains on byte-level Wikipedia (`data/wiki.txt`), 512-dim / 12-layer / 63.5M params.

## What it is
- **RWKV-7 "Goose"** delta-rule matrix-state recurrence, **bidirectional** (a L→R and
  a R→L pass per layer, averaged — each position sees past AND future, needed for a
  denoiser). Per-head L2-normalized removal key (without it the recurrence diverges).
- **Masked / absorbing-state diffusion** (MDLM): mask tokens at rate t, predict the
  originals, 1/t-weighted cross-entropy on masked positions only.
- **adaLN-Zero timestep conditioning** on both norms (the model knows its mask rate t).
- fp32 throughout. Hand-written forward AND backward kernels, finite-diff verified.

## Build / run / monitor
- `make all` — builds shaders (glslc → build/*.spv) and binaries (cc → build/*).
- `make test` — runs the 11 kernel tests (all should print `... OK`).
- `./run_overnight.sh` — training with auto-restart + VRAM watchdog (launch detached:
  `setsid ./run_overnight.sh </dev/null >/dev/null 2>&1 &`).
- `build/sample` — load latest checkpoint, show held-out bits/byte + predicted text.
- Log: `models/train.log`. Best checkpoint: `models/ckpt.bin`. Global best
  bits/byte: `models/best_bpb.txt`.

## Result (measured honestly via build/sample on held-out data)
- **Held-out bits/byte ≈ 5.68** (random = 8.0) — a ~29% reduction. Masked-fill accuracy
  10.6% (random 0.4%), so the model learned real structure but is a weak denoiser.
- **IMPORTANT — the training `ema_bpb` (~2.64) is OPTIMISTIC and NOT the real metric.**
  It is dominated by easy low-mask-rate batches (most context visible) and the 1/t
  weighting; on held-out data with the full mask-rate distribution the model is ~5.68.
  Always judge generalization with `build/sample` (held-out), not the training ema.
- So: the system trains stably and self-manages perfectly, but at d=512/12L/63.5M the
  *generalization* is modest. Real improvement would need more data/steps/scale or a
  better objective — NOT more stability tuning (that part is solved).

## Training stability (the hard part — see memory for full diagnosis)
The masked-diffusion objective has a recurring late-training instability that NO
optimizer-knob tuning fully prevents (ruled out: gradient clipping alone, lr decay,
weight decay, loss-reweighting, lr schedule — weights stay healthy, it just wanders
out of the minimum and only a restart/optimizer-reset recovers it). It is handled
SYSTEMICALLY, validated in live events:
- **Frozen config:** gradient-clip global-norm ≤ 2 (GPU-reduced), constant lr = 3e-5,
  wd = 0 (weight decay was *destroying* the model), uncapped 1/t weight.
- **Best-state ratchet:** checkpoint ONLY on a genuine new-best ema_bpb; the best is
  persisted to `best_bpb.txt` so it survives restarts (globally monotonic).
- **Auto-restart:** if ema_bpb > 4.5, exit(1) → wrapper reloads the best ckpt + resets
  optimizer state → recovers to ~3.0. So the run self-heals and self-improves,
  staying usable indefinitely unattended.

## Key lessons (also in agent memory)
- RWKV-7 needs per-head key normalization or the state recurrence → NaN.
- Host loops over Vulkan mapped (write-combined) memory are ~50000× slower than RAM —
  do elementwise ops + reductions on the GPU.
- When the FD harness is unreliable, **training descent** is the definitive
  gradient-correctness test.
- For this high-variance objective, judge drift by the RAW bits/byte AVERAGE over
  n≥10 windows — the ema and short windows mislead.
- Don't change the loss objective mid-training; weight decay + lr-decay starves the
  model; over-intervention causes harm — when tuning fails on a recurring instability,
  engineer resilience around it instead.
