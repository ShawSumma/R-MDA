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
- Compare training changes with `tools/logstat.py` on runs from the same seed: batches
  are a hash of the step, so the runs are paired.

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
  as the eval split); `mda_data_batch` fills a batch (ids, target, mask, loss weight,
  mask rate t) as a function of its key (the training step), so a run is reproducible
  and a resume never replays batches. Schedules: `BELL` (diffusion training: t from a
  truncated N(0.5, 0.1), unweighted loss), `ELBO` (evaluation: t uniform on (0, 1], 1/t
  weight, so the loss is the NELBO), `AR` (next-token targets, nothing masked).
- `src/model` -- config (`mda_model_cfg`, checked by `mda_model_init`), parameter
  registry (`mda_model_param_t`: views of w, g, AdamW moments and fp16 mirror into
  per-role arenas, plus the transposed fp16 mirror), kernel handles.
- `src/fwd` -- the GPU optimizer state block (`mda_fwd_opt_t`), scratch buffers (with the
  host batch view of the inputs), per-layer saved activations (`mda_fwd_act_t`), dispatch
  helpers, mirror refresh, and the one forward (`mda_fwd_rec`: embed, mask-rate adaLN
  conditioning, blocks, head tied to the token embedding) shared by training and
  inference; `mda_fwd_run` wraps it for eval/generation. With `mda_model_t.causal` set
  at record time only the left-to-right wkv pass runs, making the same network an AR
  model. A config with `block` (`mda_model_cfg_block`, mode names `block` / `block<n>`)
  is block diffusion jointly with AR: the batch's first half holds the second half's
  windows unmasked, run causally with next-token targets (the AR loss); each noisy
  block's left-to-right wkv pass starts from the clean rows' state at the block start
  (`bstate`, its gradient flows back through `dbstate`), its right-to-left pass and
  token shift stay inside the block. Checkpoints only record the parameter shapes, so
  the same weights load as AR, diffusion or block diffusion. With `cache` set
  (`mda_fwd_cache_t`: per-layer clean wkv state and token-shift carry, double
  buffered) the clean rows start from the cached state and write their end state to
  the other buffer, which `mda_fwd_cache_commit` makes current; a block-length model
  then continues a sequence one block at a time, exactly as the full-length model
  would. With `stage` set, `mda_fwd_run` also writes the temperature softmax of every
  row there in the same submit.
- `src/bwd` -- matmul backward helpers (coopmat with transposed operands; scalar in
  ref32).
- `src/step` -- `mda_step_trainer_t`: three persistent commands, forward + backward
  writing the grads (first micro-batch) or adding to them (the rest), and the optimizer
  (grad norm, clip, dynamic loss scale with overflow skip, Muon's Newton-Schulz update on
  the hidden matrices with AdamW for the rest or AdamW alone, mirrors);
  `mda_step_begin` + `mda_step_micro` run one update of `accum` micro-batches,
  `mda_step_train` a one-micro-batch update, `mda_step_grad` forward + backward alone.
- `src/ckpt` -- checkpoint format (header with config + vocab hash, then the arenas),
  atomic save, validated load.
- `src/train.c` (+ `train.h`) -- training loop: warmup-stable-decay lr, logging,
  periodic checkpoints, resume;
  `build/train [lr [checkpoint [stop [accum [adamw|muon [diffusion|block<n>|ar]]]]]]`
  (defaults 3e-4, `models/ckpt.bin`, no stop, 8 micro-batches per update, muon,
  diffusion; a stop update adds the decay over the last fifth).
- `src/sample.c` (+ `sample.h`) -- held-out NELBO bits/byte (the comparable number),
  masked-token bits/byte, masked-fill accuracy and, in block mode, the clean rows' AR
  NLL, on fixed eval batches; `build/sample [checkpoint [diffusion|block<n>|ar]]`.
- `src/generate.c` (+ `generate.h`) -- generation over several rounds, with forwards
  per token, repeated trigrams, unigram entropy and, given an AR judge checkpoint,
  generative perplexity; `build/generate [steps [checkpoint [temp [mode [judge
  [policy]]]]]]`. Modes: `diffusion` (the full-length model over the whole sequence),
  `block<n>` (an n-token model with the clean-state cache, block by block: each block
  is decoded, then one more forward advances the cache over it), `ar` (block mode with
  `spec0`, exact cached left-to-right sampling). Policies: `random` / `ltr` (reveal
  order, fixed steps per block or per sequence), `remdm<eta>` (random order, and after
  every step but the last each decoded token is re-masked with probability eta, ReMDM),
  `spec<k>` (block mode:
  self-speculative decoding, the noisy rows draft up to k tokens and the clean rows' AR
  distribution accepts or resamples them exactly).
- `src/bench.c` (+ `bench.h`) -- forward / forward+backward / training-step timing on
  synthetic tokens, plus the per-kernel GPU profile of a step, then decode-forward
  timing and profile of the full-length block model vs the block-length cached one;
  `build/bench [adamw|muon]`.
- `src/train_tokenizer.c` -- trains `models/vocab.txt` from `data/corpus.txt`.
- `src/probe_coopmat.c` -- lists the device's cooperative-matrix configs.
- `src/probe_memory.c` -- lists memory heaps/types with GPU and host bandwidth per type.
- `src/probe_acts.c` -- for a checkpoint whose training went non-finite: finds a batch
  whose fp16 forward overflows, checks the fp32 forward, prints per-layer max
  activations.
- `src/test_matmul_f16.c` -- fp16 coopmat matmuls vs the scalar fp32 matmul, plus the
  scaled fp16 path on an operand far past fp16's range.
- `src/test_grad.c` -- train vs inference loss, fp16 vs fp32 backward,
  finite-difference check of every param's gradient along sign(g) (tiny model, fp32
  reference mode `ref32`, in diffusion, causal and block mode), gradient accumulation vs
  summed grads, the GPU AdamW step, Muon updates vs a host Newton-Schulz, the overflow
  skip, one-block block mode equal to plain diffusion, block causality (no leak
  from later blocks, injection from earlier ones), and a block-length model with the
  cache equal to the full-length block model.
- `src/test_tokenizer.c` (+ `test_tokenizer.h`) -- vocab file round trip (NUL, newline,
  backslash) and encode.
- `src/test_ckpt.c` -- checkpoint write/read round trip through memory.
- `shader/` -- one compute kernel per file.
