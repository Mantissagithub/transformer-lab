# Transformer Research Framework

A plug-and-play PyTorch transformer where attention, FFN, normalization, positional encoding, residual style, optimizer, scheduler, loss, and dataset are all swappable via Hydra config groups. Add a new component by writing one class with a `@register` decorator and one YAML — no edits to the trainer or model builder.

## Why this exists

Originally a hand-rolled transformer for meeting summarization with three connection variants (residual / HyperConnection / MHC), toggled via a Python dict. Swapping anything else meant editing several files. This rewrite makes the base composable: an experiment is one YAML, every component is replaceable, and the trainer is component-agnostic.

## Architecture

```
configs/
  config.yaml                # top-level Hydra entry with `defaults:` list
  model/                     # base model dimensions
  attention/                 # mha, gqa, mqa, sliding_window
  feedforward/               # relu_ffn, swiglu, geglu
  normalization/             # layernorm, rmsnorm
  positional/                # sinusoidal, rope, alibi
  connection/                # residual, hyperconnection, mhc
  optimizer/                 # adamw, muon_adamw, lion, adafactor
  scheduler/                 # none, cosine_warmup, linear_warmup
  loss/                      # cross_entropy
  data/                      # meetingbank
  logging/                   # tensorboard, neptune
  experiment/                # composable experiment recipes

src/
  registry.py                # name -> class lookup for every component kind
  components/                # all swappable parts; @register decorators run at import
  model/                     # generic blocks, encoder, decoder, transformer, builder
  training/                  # trainer, logging, checkpoint
  cli/train.py               # @hydra.main entrypoint
```

The keystone is the **Connection abstraction** (`src/components/connections/base.py`). Every connection (residual / HC / MHC) implements `init_state(x)`, `apply(state, sublayer_fn)`, `to_output(state)`. Blocks are state-shape-agnostic — the same block code runs whether the state is `(b, s, d)` (residual) or `(b, s, n, d)` (hyper variants).

## Install

```bash
pip install -e .
# optional
pip install -e .[neptune,dev]
```

## Run

The base config trains a residual / AdamW transformer on MeetingBank:

```bash
python -m src.cli.train
```

Compose an experiment that's already wired up (reproduces the previous best run):

```bash
python -m src.cli.train +experiment=meeting_summarization_mhc
```

Override anything inline — Hydra picks the right group:

```bash
python -m src.cli.train \
  attention=gqa \
  feedforward=swiglu \
  normalization=rmsnorm \
  positional=rope \
  connection=mhc \
  optimizer=lion \
  scheduler=cosine_warmup \
  training.lr=3e-4
```

Run a fast sanity pass on CPU:

```bash
python -m src.cli.train training.num_epochs=1 data.limit=64
```

## Adding a new component

Adding a new attention mechanism (the recipe is identical for FFN, norm, positional, connection, optimizer, scheduler, loss, dataset):

1. Write the class in `src/components/attention/my_attn.py`. Inherit from `AttentionBase`. Decorate with `@ATTENTION.register("my_attn")`.
2. Make sure the package's `__init__.py` imports your module so the decorator runs at import time.
3. Add `configs/attention/my_attn.yaml` with at least `name: my_attn` plus any kwargs.
4. Use it: `python -m src.cli.train attention=my_attn`.

That's the whole loop. No trainer / builder / block edits.

## Logging

TensorBoard by default — runs land in `runs/<experiment_name>/`. To use Neptune, set `logging=neptune` and provide `NEPTUNE_API_TOKEN` via environment:

```bash
NEPTUNE_API_TOKEN=… python -m src.cli.train logging=neptune
```

The previous implementation had a hardcoded API token. That has been removed; tokens must come from the environment.

## Tests

```bash
pytest tests/ -q
```

The suite covers (a) every registered component instantiating from its YAML defaults, (b) `init_state -> apply -> to_output` round-trips for every connection, (c) encoder/decoder block output shapes for every connection, and (d) a full builder smoke matrix that mixes attentions × FFNs × norms × connections to confirm arbitrary swaps compose without code edits.

## What's shipping in this version

- **Attention**: `mha`, `gqa`, `mqa`, `sliding_window`
- **FFN**: `relu_ffn`, `swiglu`, `geglu`
- **Norm**: `layernorm` (fixed `sqrt(std)` bug from the original — was `(x - mean) / (sqrt(std) + eps)`, now correctly `(x - mean) / (std + eps)`), `rmsnorm`
- **Positional**: `sinusoidal` (fixed `100000` → `10000` constant from the original paper), `rope`, `alibi`
- **Connection**: `residual`, `hyperconnection`, `mhc`
- **Optimizer**: `adamw`, `muon_adamw` (2D params → Muon if available else AdamW; rest → AdamW), `lion`, `adafactor`
- **Scheduler**: `none`, `cosine_warmup`, `linear_warmup`
- **Loss**: `cross_entropy` (label smoothing + ignore_index from cfg)
- **Dataset**: `meetingbank`

## Out of scope (deliberately)

MoE, distributed training, beam-search inference. The base is structured so they can be added the same way as everything else — write a class, register it, drop a YAML.

## References

1. Vaswani et al. (2017). ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
2. HyperConnections — https://arxiv.org/pdf/2409.19606
3. Manifold Constrained HyperConnections — https://www.arxiv.org/pdf/2512.24880
