# Transformer Research Framework

A plug-and-play PyTorch framework for transformer research. Every component — attention, FFN, norm, positional, connection, projection, optimizer, scheduler, loss, dataset — is a Hydra config group. Supports both encoder-decoder summarization and decoder-only causal-LM pretraining at the ~500M-param scale (DDP/FSDP, bf16, gradient accumulation, `torch.compile`, KV-cache generation).

## Why this exists

Originally a hand-rolled transformer for meeting summarization with three connection variants (residual / HyperConnection / MHC), toggled via a Python dict. Swapping anything else meant editing several files. This rewrite makes the base composable: an experiment is one YAML, every component is replaceable, and the trainer is component-agnostic. It has since grown a decoder-only path so the same registry/Hydra surface drives foundation-style pretraining.

## Two paths

The framework supports two model kinds, selected by `model.kind`:

1. **Encoder-decoder** (`kind: encoder_decoder`, default) — summarization on MeetingBank / Multi-News. Compatible with all attention/FFN/norm/positional variants and the connection abstraction (residual, HyperConnection, MHC, sandwich-norm).
2. **Decoder-only causal LM** (`kind: causal_lm`) — streaming FineWeb-Edu, packed sequences, KV-cache `.generate()`. Designed for ~500M-param pretraining on a single node or sharded across many GPUs.

## Architecture

```
configs/
  config.yaml                         # top-level Hydra entry with `defaults:` list
  model/                              # transformer_base, transformer_modern, transformer_500m
  attention/                          # mha, gqa, gqa_rope, mqa, sliding_window, sliding_gqa, gemma3_hybrid
  feedforward/                        # relu_ffn, swiglu, geglu
  normalization/                      # layernorm, rmsnorm
  positional/                         # sinusoidal, rope, alibi, rope_nope_hybrid
  connection/                         # residual, residual_sandwich, hyperconnection, mhc
  projection/                         # linear, tied
  optimizer/                          # adamw, muon_adamw, lion, adafactor
  scheduler/                          # none, cosine_warmup, linear_warmup
  loss/                               # cross_entropy
  data/                               # meetingbank, multi_news, fineweb_edu
  logging/                            # tensorboard, neptune, wandb
  experiment/                         # meeting_summarization_{residual,mhc,modern},
                                      # multinews_modern, pretrain_500m

src/
  registry.py                         # name -> class lookup for every component kind
  components/                         # all swappable parts; @register decorators run at import
    projection/                       # linear, tied (weight-tied input/output embeddings)
  model/
    blocks.py                         # generic encoder/decoder + CausalBlock (with forward_with_cache)
    builder.py                        # build_transformer + build_causal_lm
    causal_lm.py                      # CausalLM with KV-cache .generate()
    encoder.py / decoder.py / transformer.py
  training/
    trainer.py                        # bf16/fp16 autocast, grad accum, max_steps, compile, TUI
    distributed.py                    # DDP + FSDP wrap (transformer_auto_wrap_policy on CausalBlock)
    tui.py                            # Rich live header + dual progress bars + log panel
    hf_push.py                        # upload last checkpoint + .hydra config to HF Hub
    hf_credentials.py                 # interactive HF_TOKEN / HF_REPO_ID pre-flight, persists to .env
    logging.py / checkpoint.py
  cli/train.py                        # @hydra.main entrypoint
```

The keystone is the **Connection abstraction** (`src/components/connections/base.py`). Every connection (residual / sandwich / HC / MHC) implements `init_state(x)`, `apply(state, sublayer_fn)`, `to_output(state)`. Blocks are state-shape-agnostic — the same block code runs whether the state is `(b, s, d)` (residual) or `(b, s, n, d)` (hyper variants).

## Install

```bash
pip install -e .
# optional
pip install -e .[neptune,wandb,dev]
```

`.env` is picked up automatically (via `python-dotenv`) for `HF_TOKEN`, `HF_REPO_ID`, `NEPTUNE_API_TOKEN`, and `WANDB_API_KEY`. Copy `.env.example` to `.env` and fill in.

## Run

The base config trains a residual / AdamW transformer on MeetingBank:

```bash
python -m src.cli.train
```

Reproduce the previous best run (MHC):

```bash
python -m src.cli.train +experiment=meeting_summarization_mhc
```

Modern stack on MeetingBank — Gemma3 hybrid attention (5× sliding + 1× full per layer pattern), RoPE/NoPE hybrid, RMSNorm, SwiGLU, sandwich-norm, tied embeddings:

```bash
python -m src.cli.train +experiment=meeting_summarization_modern
```

Modern stack on Multi-News (shared BPE tokenizer, Muon-AdamW):

```bash
python -m src.cli.train +experiment=multinews_modern
```

Decoder-only ~500M pretraining on FineWeb-Edu (single GPU, bf16, grad accum × 16):

```bash
python -m src.cli.train +experiment=pretrain_500m
```

Same, sharded across 8 GPUs with FSDP:

```bash
torchrun --nproc_per_node=8 -m src.cli.train \
  +experiment=pretrain_500m \
  training.distributed.enabled=true \
  training.distributed.strategy=fsdp \
  training.distributed.fsdp.mixed_precision=bf16
```

Override anything inline — Hydra picks the right group:

```bash
python -m src.cli.train \
  attention=gqa_rope \
  feedforward=swiglu \
  normalization=rmsnorm \
  positional=rope \
  connection=residual_sandwich \
  optimizer=lion \
  scheduler=cosine_warmup \
  training.precision=bf16 \
  training.gradient_accumulation_steps=16 \
  training.max_steps=10000 \
  training.lr=3e-4
```

Quick CPU sanity pass:

```bash
python -m src.cli.train training.num_epochs=1 data.limit=64
```

## Adding a new component

Adding a new attention mechanism (the recipe is identical for FFN, norm, positional, connection, projection, optimizer, scheduler, loss, dataset):

1. Write the class in `src/components/attention/my_attn.py`. Inherit from `AttentionBase`. Decorate with `@ATTENTION.register("my_attn")`.
2. Make sure the package's `__init__.py` imports your module so the decorator runs at import time.
3. Add `configs/attention/my_attn.yaml` with at least `name: my_attn` plus any kwargs.
4. Use it: `python -m src.cli.train attention=my_attn`.

That's the whole loop. No trainer / builder / block edits.

## Training UX

**TUI.** `training.tui=true` (default) renders a Rich live view: a header with params / layers / d_model / attention type, dual progress bars (current epoch + total steps), and a rolling log panel with events every step. Set `training.tui=false` for plain log piping (CI, multi-GPU stderr).

**Precision / compile.** `training.precision=bf16|fp16|fp32`. fp16 wires `torch.amp.GradScaler`; bf16 uses autocast with no scaler. `training.compile=true` enables `torch.compile`; pass a dict for compiler options.

**Gradient accumulation + max_steps.** `training.gradient_accumulation_steps` accumulates micro-batches before stepping; `training.max_steps` overrides `num_epochs × len(loader)` and is the right knob for streaming datasets where the loader length is unknown.

**Distributed.** `training.distributed.enabled=true`, with `strategy: ddp | fsdp`. FSDP wraps `CausalBlock` via `transformer_auto_wrap_policy`; sharding ∈ `full_shard | shard_grad_op | hybrid_shard`; mixed precision ∈ `bf16 | fp16`. Launch via `torchrun`; the trainer reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from the environment.

**Logging.** TensorBoard by default — runs land in `runs/<experiment_name>/`. At startup, `ensure_logging_backend()` prompts for which tracker to use: (1) tensorboard only, (2) wandb, (3) neptune. Picking wandb/neptune asks for the API key (offering to persist it to `.env`) and the project. Power users can skip the prompt by passing `logging=wandb` or `logging=neptune` on the Hydra command line — the API key is still validated, but the picker is bypassed:

```bash
NEPTUNE_API_TOKEN=… python -m src.cli.train logging=neptune
WANDB_API_KEY=…       python -m src.cli.train logging=wandb
```

The previous implementation had a hardcoded API token. That has been removed; tokens must come from the environment or the startup prompt.

**HF Hub push.** `training.hf.push=true` uploads the last checkpoint and `.hydra/config.yaml` snapshot to `training.hf.repo_id` (or the `HF_REPO_ID` env var) at the end of training. On startup, `ensure_hf_credentials()` validates `HF_TOKEN` via `whoami`, prompts for any missing values, and offers to persist them to `.env` so subsequent runs are non-interactive. Copy `.env.example` to `.env` for fully unattended runs.

## Generation (causal LM)

```python
out = model.generate(
    prompt_ids,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    eos_id=tokenizer.eos_id,
)
```

Runs prefill once, then steps token-by-token using per-layer `(k, v)` tuples. (The `KVCache` LRU helper in `src/components/attention/kv_cache.py` is a memoization helper, not the generation primitive — it is intentionally not used here.)

## Tests

```bash
pytest tests/ -q
```

The suite covers (a) every registered component instantiating from its YAML defaults, (b) `init_state -> apply -> to_output` round-trips for every connection, (c) encoder/decoder block output shapes for every connection, (d) a builder smoke matrix that mixes attentions × FFNs × norms × connections to confirm arbitrary swaps compose without code edits, plus the causal-LM forward/generate path (`test_causal_lm.py`) and a RoPE smoke test (`test_rope.py`).

## What's shipping in this version

- **Model kinds**: `encoder_decoder`, `causal_lm`
- **Attention**: `mha`, `gqa`, `gqa_rope`, `mqa`, `sliding_window`, `sliding_gqa`, `gemma3_hybrid` (per-layer pattern: 5× sliding + 1× full)
- **FFN**: `relu_ffn`, `swiglu`, `geglu`
- **Norm**: `layernorm` (fixed `sqrt(std)` bug from the original — was `(x - mean) / (sqrt(std) + eps)`, now correctly `(x - mean) / (std + eps)`), `rmsnorm`
- **Positional**: `sinusoidal` (fixed `100000` → `10000` constant from the original paper), `rope`, `alibi`, plus `skip_every_n` for RoPE/NoPE hybrids (drop RoPE every Nth layer)
- **Connection**: `residual`, `residual_sandwich` (pre + post norm), `hyperconnection`, `mhc`
- **Projection**: `linear`, `tied` (weight-tied input/output embeddings)
- **Optimizer**: `adamw`, `muon_adamw` (2D params → Muon if available else AdamW; rest → AdamW), `lion`, `adafactor`
- **Scheduler**: `none`, `cosine_warmup`, `linear_warmup`
- **Loss**: `cross_entropy` (label smoothing + `ignore_index` from cfg)
- **Dataset**: `meetingbank`, `multi_news` (shared BPE tokenizer), `fineweb_edu` (streaming, packed)
- **Training**: bf16/fp16 autocast, `torch.compile`, gradient accumulation, `max_steps`, DDP, FSDP, Rich TUI, HF Hub push

## Out of scope (deliberately)

MoE, beam-search inference, full RLHF / DPO, evaluation harness (lm-eval-harness integration is left to the user). The base is structured so each can be added the same way as everything else — write a class, register it, drop a YAML.

## References

1. Vaswani et al. (2017). ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
2. HyperConnections — https://arxiv.org/pdf/2409.19606
3. Manifold Constrained HyperConnections — https://www.arxiv.org/pdf/2512.24880
4. Gemma 3 (hybrid local/global attention) — https://arxiv.org/abs/2503.19786
5. GQA — Ainslie et al. (2023), https://arxiv.org/abs/2305.13245
6. RoPE — Su et al. (2021), https://arxiv.org/abs/2104.09864
7. NoPE — Kazemnejad et al. (2023), https://arxiv.org/abs/2305.19466
8. FineWeb-Edu — https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
9. Muon optimizer (Jordan) — https://kellerjordan.github.io/posts/muon/

## License

Apache-2.0. See [`LICENSE`](LICENSE).
