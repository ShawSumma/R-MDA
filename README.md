# RWKV Masked Diffusion Architecture

Masked-diffusion language models with a bidirectional RWKV-7 backbone, trained and sampled on one GPU. Written in C with Vulkan compute shaders, with no deep-learning framework.

## Overview

A masked-diffusion language model generates text by starting from a fully masked sequence and repeatedly predicting and revealing tokens, in any order. It is trained by masking a random fraction of real text and learning to recover it. Because the model must see context on both sides of each token, these models are usually bidirectional Transformers.

This project uses RWKV-7's linear recurrence instead of attention. Each layer runs the recurrence left to right and right to left and averages the two results, which makes it bidirectional at linear cost. The recurrence also makes block decoding cheap: when text is generated 16 tokens at a time, everything before the current block is held in a fixed-size state rather than in a cache that grows with the text.

In block mode, the model is trained as an autoregressive model and a block-diffusion model at once. Each training sequence appears twice in the batch. One copy is unmasked and read left to right with a next-token loss. The other is masked and denoised one block at a time, starting each block from the unmasked copy's state at that point. The generator uses this for self-speculative decoding: the diffusion side drafts several tokens in one forward pass, and the autoregressive side verifies them in the same pass. Block mode can also split the layers into an encoder that reads only the unmasked copy and a decoder, the top layers, that the masked copy runs through, which makes training and decoding cheaper.

The model has SmolLM2-135M's shape and tokenizer: width 576, 30 layers, 9 heads of 64, a SwiGLU feed-forward layer of width 1536, a 49,152-token vocabulary, tied input and output embeddings, and about 159M parameters with the RWKV-7 additions. Training can start from SmolLM2's weights, converted the way RADLADS converts a Transformer to RWKV, and distil SmolLM2's next-token predictions into the autoregressive side.

## Requirements

- Linux or another POSIX system.
- A C compiler, `make`, the Vulkan headers and loader, and `glslc`.
- Python 3 with `pyarrow`, to download the corpus.
- A GPU with Vulkan 1.3, `VK_KHR_cooperative_matrix` with 16 by 16 by 16 fp16 tiles and fp32 accumulation, `VK_EXT_shader_atomic_float`, fp16 storage and arithmetic, a subgroup size of 32 in compute shaders, and subgroup arithmetic and clustered operations.

It has been tested on an NVIDIA RTX 6000 Ada and, before the RWKV-7 kernels, on an AMD Radeon 8060S (Ryzen AI Max) with Mesa's RADV driver. NVIDIA RTX 20-series and newer, and AMD RDNA3 and newer with RADV from Mesa 23.3 on, have the cooperative-matrix support. AMD RDNA2 and older have none. If a device is unsuitable, the programs print the reason at startup. `build/probe_coopmat` and `build/probe_memory` show what a device supports. Training at the default batch allocates about 20 GB of GPU memory; `build/bench` prints the amount for other settings.

## Building

```sh
make
make test
```

`make` compiles the shaders to `build/*.spv` and the programs to `build/`. `make test` also runs the four test programs. Run every program from the repository root, because they load shaders and data by relative path.

## Getting started

```sh
python3 -m venv .venv
.venv/bin/pip install pyarrow
.venv/bin/python3 tools/fetch_corpus.py --shards 1
mkdir -p models
python3 tools/fetch_smollm.py
build/import models/smollm2.safetensors models/ckpt.bin
build/train 3e-4 models/ckpt.bin 20000 2 muon block16d10 8 models/smollm2.safetensors
build/sample models/ckpt.bin block16d10
build/generate 8 models/ckpt.bin 0.8 block16d10 "" conf
```

1. Create a Python environment with `pyarrow` in `.venv`.
2. Download one FineWeb-Edu shard and write its text to `data/corpus.txt`.
3. Create `models/`, which holds the tokenizer and checkpoints.
4. Download SmolLM2-135M's tokenizer and weights, and write the tokenizer in `models/tokenizer.txt`.
5. Convert SmolLM2's weights into a step-0 checkpoint.
6. Train it for 20,000 steps in block mode with 16-token blocks and a 10-layer decoder, distilling SmolLM2. The first run tokenizes the corpus and caches the token ids in `data/corpus.txt.ids`.
7. Evaluate on held-out text.
8. Decode block by block with 8 denoising steps per block, revealing the most confident tokens first.

Other ways to train and decode:

```sh
build/train 3e-4 models/scratch.bin 20000 2 muon block16 8
build/train 3e-4 models/diff.bin 20000 2 muon diffusion 8
build/generate 64 models/diff.bin 0.8 diffusion
build/generate 0 models/ckpt.bin 0.8 block16d10 "" spec8
build/generate 64 models/diff.bin 0.8 diffusion models/ckpt.bin
```

1. Train a block model from scratch, without a teacher and with every layer seeing the masked copy.
2. Train a plain diffusion model over whole 256-token sequences.
3. Sample it with 64 denoising steps at temperature 0.8.
4. Decode the block model with speculative decoding and 8-token drafts.
5. Score the diffusion model's samples with the block model's autoregressive side.

## Programs

All arguments are positional. To set one, give all the ones before it. A checkpoint does not record whether it was trained as `diffusion`, `block` or `ar`, so every program loads it in whatever mode it is given. A mode that differs from the training mode gives poor results but no error.

The modes are `diffusion`, `block` (16-token blocks), `blockN` (`N`-token blocks; `N` must divide 256 and be a multiple of 4), `blockNdD` (the same with only the top `D` layers seeing the masked copy; `D` must be below 30), and `ar`.

### build/train

```
build/train [lr] [ckpt] [stop] [accum] [opt] [mode] [batch] [teacher] [kd]
```

| Argument | Default | Meaning |
|---|---|---|
| `lr` | `3e-4` | Peak learning rate. |
| `ckpt` | `models/ckpt.bin` | Checkpoint file. If it exists, training resumes from it with the arguments given now, so a new run needs a new path, or a checkpoint from `build/import` to start from SmolLM2. |
| `stop` | `0` | Last step. `0` trains until stopped. |
| `accum` | `2` | Micro-batches per step. |
| `opt` | `muon` | `muon`, or `adamw` to use Adam for everything. |
| `mode` | `diffusion` | One of the modes above. |
| `batch` | `8` | Sequences per micro-batch. Block mode doubles the rows. |
| `teacher` | none | SmolLM2's safetensors file. Its next-token distribution becomes the target of the autoregressive rows: the unmasked copy in block mode, every row in `ar` mode. |
| `kd` | `1` | With a teacher, the teacher's share of those rows' target; the rest is the real next token. |

At startup, `train` prints the model size and how much GPU memory it allocated. Every 5 steps, it prints the loss, bits per byte, gradient norm, clip and skip counts, and seconds per step. The loss and bits per byte are always against the real text, also when distilling. It saves a checkpoint every 2 minutes and at the last step. A run can be stopped at any time and resumed on any machine, since batches depend only on the step number.

If 100 steps in a row overflow fp16, `train` saves the weights to `models/ckpt-overflow.bin` and stops. `build/probe_acts` shows which layer overflowed. Skipped steps leave the weights unchanged, so the regular checkpoint is still usable. Resuming it with a lower learning rate is the usual next step.

### build/import

```
build/import [weights] [out]
```

`import` writes a step-0 checkpoint from SmolLM2's weights, `models/smollm2.safetensors` and `models/smollm2.bin` by default. The embeddings, norms and feed-forward layers are copied. Attention's query, key, value and output projections become the recurrence's receptance, key, value and output projections, with the 3 key-value heads repeated over the 9 query heads, and the token shift starts at the current token. The recurrence's own parameters start from their usual initialization. The imported model is a poor language model until it has trained for a while.

### build/sample

```
build/sample [ckpt] [mode]
```

`sample` evaluates 512 held-out batches in the given mode, which should be the training mode. It reports:

- The negative ELBO in bits per byte, which is an upper bound on the negative log-likelihood.
- Masked-token bits per byte, and accuracy at low, medium and high mask rates.
- In block mode, the unmasked stream's next-token loss.

It then prints one example with the model's fills.

### build/generate

```
build/generate [steps] [ckpt] [temp] [mode] [judge] [policy]
```

| Argument | Default | Meaning |
|---|---|---|
| `steps` | `16` | Denoising passes, over the whole sequence in `diffusion` mode or per block in block mode. `ar` and `specK` ignore it, but it must still be given to reach the later arguments. |
| `ckpt` | `models/ckpt.bin` | Checkpoint to sample from. |
| `temp` | `0.8` | Sampling temperature. `0` is greedy. |
| `mode` | `diffusion` | One of the modes above. |
| `judge` | none | Checkpoint of an `ar` model, or of a block model (its unmasked side is one), used to score the samples. `""` means none, to reach `policy` without a judge. |
| `policy` | `random` | `random`, `ltr`, `conf`, `remdmE`, or `specK`. |

The policies:

- `random` reveals masked positions in random order.
- `ltr` reveals them left to right.
- `conf` reveals the positions whose sampled token had the highest probability first.
- `remdmE`, for example `remdm0.05`, reveals in random order and also re-masks each revealed token with probability `E` after every step except the last. This lets early mistakes be corrected.
- `specK` uses speculative decoding with drafts of `K` tokens, at most 16. It needs block mode and ignores `steps`.

In `ar` mode, `generate` samples one token per forward pass and ignores the policy.

`generate` produces 2 sequences of 256 tokens (4 in a split block mode, so that the unmasked rows fill the matmul tiles), 8 times, each starting from a newline. It prints the first round, then:

- Forward passes per sequence.
- Tokens per forward pass.
- Tokens per second.
- The rate of repeated trigrams.
- Unigram entropy.
- The draft acceptance rate.
- With a judge, the generative perplexity.

### Other programs

- `build/bench [opt] [batch] [mode]` prints how much GPU memory training with those settings allocates. It times each matmul shape, the forward pass, the backward pass and a step of one micro-batch, prints GPU time per kernel, and compares cached and uncached block decoding. The defaults, `muon 8 diffusion`, are `train`'s. It needs no data.
- `build/check_teacher [weights]` runs the SmolLM2 teacher on the start of the corpus and writes its logits to `models/teacher_check.bin` for `tools/check_teacher.py`, then prints the teacher's held-out next-token bits per byte, to compare with `sample`'s unmasked-stream loss.
- `build/probe_acts [ckpt] [batches]` finds the first batch whose fp16 loss is not finite and prints the largest activation in each layer. The defaults are `models/ckpt-overflow.bin 500`.
- `build/probe_coopmat` lists the cooperative-matrix shapes the driver supports.
- `build/probe_memory` lists the memory types and measures their bandwidth.

## Tools

- `tools/fetch_corpus.py [--shards N]` downloads 1 to 14 FineWeb-Edu shards to `data/fineweb-edu/` and writes their text to `data/corpus.txt`. Interrupted downloads resume. The token-id cache is rebuilt on the next run.
- `tools/fetch_smollm.py` downloads SmolLM2-135M's `tokenizer.json` and `model.safetensors` to `models/` and writes the tokenizer in the format the C side reads, `models/tokenizer.txt`.
- `tools/logstat.py LOG... [--last N] [--end STEP]` averages the logged loss and bits per byte over the last `N` steps of each training log, 2000 by default. Runs draw the same batches at the same step, so this compares runs on identical data. To get a log, redirect the output of `build/train` to a file.
- `tools/sweep.py NAME STOP ACCUM OPT MODE BATCH --lr LR...` trains one run per learning rate with the other settings shared, writing `models/NAME-LR.bin` and its log `models/NAME-LR.log`, then compares the runs with `logstat.py`.
- `tools/compare.py GROUP LR STOP ACCUM OPT BATCH --run NAME MODE INIT TEACHER KD ...` trains one run per variant, each with its own mode, starting checkpoint, teacher (`-` for none) and teacher share, evaluates each with `build/sample`, and prints a table of the results, also written to `models/GROUP.table`.
- `tools/strip_ckpt.py SRC DST` copies a checkpoint without its optimizer state, a third of the size. `sample` and `generate` load the copy; `train` cannot resume from it.
- `tools/check_tokenizer.py [--bytes N]` checks the token-id cache against Hugging Face's `tokenizers` on the start of the corpus. It needs that package.
- `tools/check_teacher.py` checks `build/check_teacher`'s logits against SmolLM2 run by Hugging Face `transformers`. It needs `torch` and `transformers`.
- `tools/gen_unicode.py` regenerates `src/unicode.c`, the letter and number tables the pre-tokenizer uses.
- `tools/vast_wait.py ID` waits for a vast.ai instance to start and prints its ssh address. It needs the `vastai` command-line tool in `.venv`.

## How it works

### Model

Each layer applies, in order:

1. RMSNorm and a learned per-channel scale and shift.
2. Token shift, a learned mix of each position with the previous one.
3. Three 576-by-576 projections for the key `k`, value `v` and receptance `r`, and four low-rank MLPs for the decay `w`, the write strength `a`, the value-residual gate and the output gate `g`.
4. The recurrence below, once in each direction, with the two outputs averaged.
5. Per-head RMS normalization, plus a bonus term, times the output gate, then an output projection and a residual add.
6. RMSNorm and a learned per-channel scale and shift.
7. A SwiGLU feed-forward layer and a residual add.

The output is an RMSNorm followed by a product with the token embedding matrix. The mask token has its own embedding and is not in the output vocabulary. There is no timestep input: the model infers the noise level from how many tokens are masked.

The recurrence keeps a 64-by-64 state `S` per head, from keys to values. Before the scan, each head's inputs become:

- the removal key `kk = normalize(k * xi)`,
- the write strength `a = sigmoid(...)` and the replacement key `kt = k * (1 + (a - 1) * alpha)`,
- the decay `w = exp(-0.6065 * sigmoid(...))`, between about 0.55 and 1,
- the value `v = v0 + gate * (v' - v0)`, where `v0` is the first layer's value (the value residual).

Then, for each token:

```
sk[j]   = sum_i S[i][j] * kk[i]
S[i][j] = w[i] * S[i][j] - a[i] * kk[i] * sk[j] + kt[i] * v[j]
out[j]  = sum_i r[i] * S[i][j]
```

`sk` is what the state currently returns for the key. Each step removes a fraction `a` of it along that key, writes the new value along the replacement key, and lets old content fade through `w`. The bonus adds `(sum_i r[i] * rho[i] * kt[i]) * v` to each head's output, so the current token is read directly as well as through the state. `xi`, `alpha` and `rho` are learned per channel. This is RWKV-7's time mix.

In a split block mode, the bottom layers process only the unmasked copy. The masked copy's embeddings enter at the first decoder layer, which also supplies their `v0`, and in every decoder layer the masked rows' left-to-right scan starts each block from that same layer's unmasked state.

### Tokenizer

The tokenizer is SmolLM2's byte-level BPE with 49,152 tokens, reimplemented in C: every Unicode number is split into single characters, then GPT-2's pre-tokenizer splits words, numbers, whitespace and punctuation, then merges apply in rank order. On 50 MB of the corpus, the ids are identical to Hugging Face's.

### Training

Each token of a 256-token window is masked with probability `t`, and the loss is the cross-entropy at the masked positions. Training draws `t` from a normal distribution with mean 0.5 and standard deviation 0.1. Evaluation draws `t` uniformly and weights the loss by `1/t`, which makes it an upper bound on the negative log-likelihood. Losses are reported in bits per byte of text, so models with different tokenizers or objectives can be compared. The last 1% of the corpus is held out.

Block mode doubles the batch, so each window appears once unmasked and once masked. The unmasked copy is read left to right with a next-token loss. The masked copy is processed one block at a time: its left-to-right scan starts each block from the unmasked copy's state at that point, and its right-to-left scan starts each block from zero. A masked token therefore sees all earlier text and its own block, but nothing after its block.

With a teacher, SmolLM2 runs on the unmasked copy in the same command buffer, and the unmasked rows' target becomes its next-token distribution instead of the real next token.

Matmul inputs are fp16 and everything else is fp32. Each activation and gradient is scaled by a power of two that brings its largest absolute value between 16384 and 32768, and the matmul undoes the scale. Gradients use a dynamic loss scale: a step that overflows is skipped and the scale is halved, and the scale doubles after 2000 good steps. Gradients are clipped to norm 1. All of this is decided on the GPU, and each training step is a prerecorded command buffer, so the CPU never waits on the GPU in the middle of a step.

By default, the 7 full weight matrices in each layer are trained with Muon, and everything else with Adam, without weight decay. The learning rate warms up over 50 steps and then stays constant. If a last step is given, it falls linearly to zero over the final 20% of steps. Batches are derived from the step number, so a resumed run sees the same data as an uninterrupted one.

### Generation

Plain diffusion starts from a masked 256-token sequence. Each step samples a token at every position, then reveals enough of the still-masked positions to finish in the remaining steps.

Block decoding generates 16 tokens at a time. Each block takes `steps` passes over 16 tokens, plus one more pass that stores the state after the finished block. That state stands in for all earlier text, so the cost per block does not grow with the length of the text. In a split mode, the decoding passes run the masked rows through the decoder layers only.

Speculative decoding does two jobs in one forward pass. The unmasked rows hold the accepted tokens followed by the current drafts, and their next-token predictions accept or reject each draft. The masked rows hold the accepted tokens followed by masks, and their predictions become the next drafts. With standard speculative-sampling acceptance, the output follows the autoregressive distribution exactly.

## Tests

`make test` runs four programs:

- `test_grad` checks every parameter's gradient against finite differences, fp16 gradients against fp32, and both optimizers against host implementations, in diffusion, autoregressive, block and split block mode. It also checks overflow handling, that a block cannot see later blocks, and that cached block decoding gives the same losses as the full sequence.
- `test_matmul_f16` checks the fp16 matmul kernels against an fp32 kernel.
- `test_ckpt` checks that a checkpoint saves and loads unchanged, and that a weights-only copy loads but cannot be resumed.
- `test_tokenizer` checks that the tokenizer file saves and loads unchanged, and the pre-tokenizer and merge order on a toy vocabulary.

## Layout

- `src/` holds the C sources.
- `shader/` holds the GLSL compute kernels.
- `tools/` holds the Python scripts.
- `build/` holds the programs, objects and compiled shaders.
- `data/` holds the corpus, its token-id cache and the downloaded shards.
- `models/` holds the tokenizer, SmolLM2's weights and checkpoints.

The library modules in `src/`:

- `vkc.c`: Vulkan device, buffers, pipelines and command recording.
- `model.c`: configuration, parameters and kernel loading.
- `fwd.c`: forward pass, fp16 scaling and the decoding cache.
- `bwd.c`: matmul backward pass.
- `step.c`: backward pass, optimizers and the recorded training step.
- `teacher.c`: the SmolLM2 teacher's forward pass.
- `safet.c`: safetensors reading.
- `data.c`: corpus loading and batch construction.
- `tokenizer.c`: tokenizer file and encoding.
- `unicode.c`: generated Unicode tables.
- `ckpt.c`: checkpoints.
- `hash.c`: deterministic random numbers.
- `sort.c`: sorting.
- `util.c`: small helpers.

Each program has its own source file in `src/`, named after the program.

## Settings

Settings are constants in the sources. Changing the model shape or the tokenizer makes existing checkpoints unloadable.

| Setting | Location |
|---|---|
| Model shape | `mda_model_cfg` in `src/model.c` |
| Warmup, clip norm, checkpoint interval | `src/train.h` |
| Optimizer settings | `src/step.h` |
| Masking schedule | `src/data.h` |
