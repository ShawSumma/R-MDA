# r-mda

RWKV-7 blocks trained as a masked-diffusion text denoiser, in raw Vulkan compute (C host
+ GLSL).

## Rules

- Run `build/bench` before and after a performance change; its per-kernel GPU profile
  says where the time goes.
- Any buffer the host reads back is `mda_vkc_alloc_cached`; bulk host reads of
  device-local buffers go through a cached staging buffer and `mda_vkc_copy`.
- A change to what the parameters mean (their order, or a formula that reads them)
  bumps `MDA_CKPT_MAGIC` in src/ckpt.h; the header only catches config and layout-size
  changes.
- After changing the corpus or the tokenizer trainer, rerun `build/train_tokenizer`;
  checkpoints trained on the old vocab are refused.
- If train stops with "overflowed", run `build/probe_acts` on `models/ckpt-overflow.bin`
  before changing anything.
- Python tools run from `.venv` (pyarrow is installed there).

## Style

- C17 (`-std=gnu17`), no C23 features: `bool` comes from `<stdbool.h>`, no typed enums,
  no `{}` initializers.
- Every name carries the namespace `mda_` plus its file's name: `mda_fwd_rec`,
  `mda_fwd_act_t`, `MDA_MODEL_MAX_PARAMS`; a program's own functions use its file name
  too (`mda_train_lr`).
- Headers hold, in this order: includes, defines, typedefs, enums, structs, prototypes.
  `.c` files hold only includes and functions. Order declarations by dependency and use,
  not alphabetically.
- An enum is a `typedef uint32_t x_t;` in the typedefs plus an anonymous `enum { ... }`
  of its constants after them; never a typedef merged with the enum.
- Struct members are laid out to avoid padding, with bools last.
- No comments (C, GLSL, makefile, Python). No `static`, `inline`, function bodies in
  headers, `exit()`, ternaries, `++`/`--`, out parameters, `qsort`, `rand`/`srand`
  (pseudo-randomness comes from `src/hash`), or plain `int` (except `main`).
- No side effects in a condition, including calls that write or print. No reassigned
  status flags. No guard clause that returns and falls through to more code: when one
  branch returns, the rest goes in `else`; split long functions into stages instead of
  nesting.
- Use `switch` to dispatch on a value. Each call and signature sits on one line, or one
  argument per line with the closing `);` on its own line.
- printf padding only in tables with a header row; no decorative output.
- The makefile uses `!=`, `${VAR}` and `${VAR:%.c=%.o}`, not GNU functions or `:=`;
  header dependencies come from the compiler (`-MMD -MP`).

## Code structure

- `makefile` -- `shader/*.comp` -> `build/*.spv` (glslc); the modules in `MODS` ->
  `build/obj/*.o` -> `build/libmda.a`; each program in `PROGS` (`src/<name>.c`) ->
  `build/<name>` linked against it; `build/**/*.d` hold the header dependencies. `make
  test` runs every `build/test_*`. Binaries load
  `build/*.spv`, `data/corpus.txt` and `models/` by relative path, so run them from the
  project root.
- Errors return to `main` as `false`/NULL/`ok` results: a Vulkan or config failure sets
  the sticky `mda_vkc_t.fail` (every vkc call is then a no-op).
- `tools/fetch_corpus.py` -- downloads FineWeb-Edu shards into `data/fineweb-edu/` and
  writes their text as `data/corpus.txt`; run it with `.venv/bin/python3` (pyarrow).
- `tools/logstat.py` -- mean loss and bits/byte over the tail of `build/train` logs.
- `src/vkc` -- Vulkan harness: device selection + feature checks, device-local
  (`mda_vkc_alloc`) vs host-cached (`mda_vkc_alloc_cached`) buffers, views, copy/fill,
  kernel loading, dispatch. `mda_vkc_begin`/`mda_vkc_flush` record a transient batch
  into one submit, `mda_vkc_record`/`mda_vkc_end`/`mda_vkc_submit` a persistent command;
  outside a batch every `mda_vkc_run` is its own submit + wait. Also the per-kernel GPU
  timestamp profiler.
- `src/pc.h` -- push-constant layouts of every shader, `MDA_PC_NO_SLOT`, the vecops ops.
- `src/hash` -- pure hash of (stream, key, index) to integers, uniforms and normals.
- `src/sort` -- heapsort of (key, index) pairs, ties broken by index.
- `src/util` -- clock, path suffixing, argv defaults, ok/FAIL, min/max.
- `src/tokenizer` -- Unigram subword tokenizer runtime (vocab read, Viterbi encode,
  vocab hash) and the corpus and vocab paths.
- `src/vocab` -- Unigram vocab training (seeding, EM, pruning).
- `src/data` -- loads the vocab and tokenizes the corpus on every core into
  `data/corpus.txt.ids` (reused while vocab and corpus are unchanged; last 1% held out
  as the eval split); `mda_data_batch` fills a batch (ids, target, mask, 1/t weight) as
  a function of its key (the training step), so a run is reproducible and a resume
  never replays batches.
- `src/model` -- config (`mda_model_cfg`, checked by `mda_model_init`), parameter
  registry (`mda_model_param_t`: views of w, g, AdamW moments and fp16 mirror into
  per-role arenas, plus the transposed fp16 mirror), kernel handles.
- `src/fwd` -- the GPU optimizer state block (`mda_fwd_opt_t`), scratch buffers (with the
  host batch view of the inputs), per-layer saved activations (`mda_fwd_act_t`), dispatch
  helpers, mirror refresh, and the one forward (`mda_fwd_rec`) shared by training and
  inference; `mda_fwd_run` wraps it for eval/generation.
- `src/bwd` -- matmul backward helpers (coopmat with transposed operands; scalar in
  ref32).
- `src/step` -- `mda_step_trainer_t`: forward + backward and optimizer (grad norm, clip,
  dynamic loss scale with overflow skip, AdamW, mirrors) recorded once as the persistent
  step command; `mda_step_train` runs it, `mda_step_grad` runs forward + backward alone.
- `src/ckpt` -- checkpoint format (header with config + vocab hash, then the arenas),
  atomic save, validated load.
- `src/train.c` (+ `train.h`) -- training loop: warmup-stable-decay lr, logging,
  periodic checkpoints, resume; `build/train [lr [checkpoint [stop]]]` (default
  `models/ckpt.bin`; a stop step adds the decay).
- `src/sample.c` (+ `sample.h`) -- held-out bits/byte and masked-fill accuracy from a
  checkpoint; `build/sample [checkpoint]`.
- `src/generate.c` -- iterative-unmasking generation;
  `build/generate [steps [checkpoint [temp]]]`.
- `src/bench.c` (+ `bench.h`) -- forward / forward+backward / training-step timing on
  synthetic tokens, plus the per-kernel GPU profile of a step.
- `src/train_tokenizer.c` -- trains `models/vocab.txt` from `data/corpus.txt`.
- `src/probe_coopmat.c` -- lists the device's cooperative-matrix configs.
- `src/probe_memory.c` -- lists memory heaps/types with GPU and host bandwidth per type.
- `src/probe_acts.c` -- for a checkpoint whose training went non-finite: finds a batch
  whose fp16 forward overflows, checks the fp32 forward, prints per-layer max
  activations.
- `src/test_matmul_f16.c` -- fp16 coopmat matmuls vs the scalar fp32 matmul, plus the
  scaled fp16 path on an operand far past fp16's range.
- `src/test_grad.c` -- train vs inference loss, fp16 vs fp32 backward,
  finite-difference check of every param's gradient (tiny model, fp32 reference mode
  `ref32`), and the GPU optimizer step and overflow skip vs the host.
- `src/test_tokenizer.c` (+ `test_tokenizer.h`) -- vocab file round trip (NUL, newline,
  backslash) and encode.
- `src/test_ckpt.c` -- checkpoint write/read round trip through memory.
- `shader/` -- one compute kernel per file. `muon.comp` is built but not yet used by
  host code.
