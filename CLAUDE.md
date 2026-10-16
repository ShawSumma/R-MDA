# r-mda

RWKV-7 blocks trained as a masked-diffusion text denoiser, in raw Vulkan compute (C host
+ GLSL). ARCHITECTURE.md is the design and optimization log.

## Code structure

- `makefile` -- `shader/*.comp` -> `build/*.spv` (glslc), each `src/*.c` -> `build/<name>`.
  `make test` runs every `build/test_*`. Binaries load `build/*.spv`, `data/wiki.txt` and
  `models/` by relative path, so run them from the project root.
- `src/vkc.h` -- Vulkan harness: device, host-visible buffers, kernel loading, dispatch.
  `vkc_begin`/`vkc_flush` record many dispatches into one submit; outside a batch every
  `vkc_run` is its own submit + wait.
- `src/tokenizer.h` -- Unigram subword tokenizer runtime (vocab load, Viterbi encode).
- `src/train_tokenizer.c` -- trains `models/vocab.txt` from `data/wiki.txt`.
- `src/data.h` -- tokenizes the corpus, builds masked-diffusion batches (ids, target, mask,
  1/t weight).
- `src/model.h` -- config, parameter registry (`param_t`: w, g, AdamW moments, fp16
  mirror), kernel handles, config validation, `model_init`.
- `src/fwd.h` -- scratch buffers, per-layer saved activations (`act_t`), dispatch helpers,
  and the one forward (`forward_rec`) shared by training and inference; `model_forward`
  wraps it for eval/generation.
- `src/bwd.h` -- matmul backward helpers.
- `src/step.h` -- `train_step`: forward + backward recorded as one GPU batch.
- `src/train.c` -- training loop, grad clipping, AdamW, checkpoints (`models/ckpt.bin`).
- `src/sample.c` -- held-out bits/byte and masked-fill accuracy from the checkpoint.
- `src/generate.c` -- iterative-unmasking generation; `build/generate <steps>`.
- `src/bench.c` -- forward-pass timing.
- `src/probe_coopmat.c` -- lists the device's cooperative-matrix configs.
- `src/test_matmul_f16.c` -- fp16 coopmat matmuls vs the scalar fp32 matmul.
- `src/test_grad.c` -- finite-difference check of every param's gradient (tiny model,
  fp32 reference mode `ref32`) and train_step vs model_forward loss equality.
- `shader/` -- one compute kernel per file; the table in ARCHITECTURE.md section 3 says
  what each does.
- `notes/` -- chunked-WKV derivation and a reference/WIP kernel (not built).
