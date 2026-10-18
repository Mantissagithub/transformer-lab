# Transformer

A modular PyTorch transformer for research. Every component, including attention, FFN, normalization, positional encoding, residual connection, optimizer, and dataset, is registered by name and selected from a YAML. Swap anything without touching the trainer.

Supports both encoder-decoder summarization (MeetingBank, Multi-News) and decoder-only causal-LM pretraining at the ~500M-parameter scale (FineWeb-Edu, FSDP, bf16, KV-cache generation).

## Why this exists

Started as a hand-rolled transformer for meeting summarization. Trying any new attention or connection variant meant editing several files. This rewrite makes the base composable: experiments are one YAML each, components are decoupled, and the trainer doesn't know what attention you picked.

## Quick start

```bash
pip install -e .

# default: residual transformer on MeetingBank
python -m src.cli.train

# ~500M decoder-only pretrain on FineWeb-Edu
python -m src.cli.train +experiment=pretrain_500m

# mix and match anything inline
python -m src.cli.train attention=gqa_rope feedforward=swiglu normalization=rmsnorm
```

## Adding a new component

Same four steps for every kind (attention, FFN, norm, optimizer, dataset, ...):

1. Write the class. Inherit from the base, decorate with `@<KIND>.register("name")`.

   ```python
   # src/components/attention/my_attn.py
   @ATTENTION.register("my_attn")
   class MyAttention(AttentionBase):
       ...
   ```

2. Import the module in the package's `__init__.py` so the decorator runs at import time.
3. Add `configs/attention/my_attn.yaml` with `name: my_attn` and any kwargs.
4. Use it: `python -m src.cli.train attention=my_attn`.

No trainer or builder edits needed.

## What's available

| Group | Choices |
|-------|---------|
| Attention | mha, gqa, gqa_rope, mqa, sliding_window, sliding_gqa, gemma3_hybrid, csa, hca, mla, msa, kda. See [`docs/attention.md`](docs/attention.md) for the variant notes and diagrams. |
| FFN | relu, swiglu, geglu. See [`docs/feedforward.md`](docs/feedforward.md). |
| Normalization | layernorm, rmsnorm. See [`docs/normalization.md`](docs/normalization.md). |
| Positional | sinusoidal, rope, alibi, rope/nope hybrid. See [`docs/positional.md`](docs/positional.md). |
| Connection | residual, residual_sandwich, hyperconnection, mhc. See [`docs/connections.md`](docs/connections.md). |
| Optimizer | adamw, muon_adamw, lion, adafactor, ademamix, mars_adamw. See [`docs/optimizers.md`](docs/optimizers.md). |
| Scheduler | cosine_warmup, linear_warmup, inverse_sqrt_warmup, polynomial_warmup, wsd, none. See [`docs/schedulers.md`](docs/schedulers.md). |
| Dataset | meetingbank, multi_news, fineweb_edu, c4, wikitext, wikipedia. See [`docs/datasets.md`](docs/datasets.md). |

Plus: bf16/fp16 autocast, gradient accumulation, `torch.compile`, DDP/FSDP, HF Hub push, KV-cache `.generate()` across every attention variant, Rich TUI.

Evaluation metrics include loss, perplexity, token/top-5 accuracy, ROUGE, BLEU, throughput, latency, and peak memory. What they measure and how they're computed are documented in [`docs/metrics.md`](docs/metrics.md).

## Experiment logbook

### 2026-07-29: Kimi Delta Attention

Added Kimi Delta Attention with FLA's chunkwise KDA kernel on CUDA, while keeping the exact recurrence as the CPU reference. The 20-epoch MeetingBank run finished all 12,920 optimizer updates in about one hour on the local RTX 4060 Laptop GPU. It used bf16, physical batch size 1, and eight-step gradient accumulation for an effective batch size of 8.

#### training run

![KDA training loss](docs/assets/kda_training_loss.svg)

This is the unsmoothed `train/loss` series from the completed run. The final logged training loss was 1.3462.

#### checkpoint selection

The final checkpoint was not the best one, so I evaluated every checkpoint that the run actually saved on the full MeetingBank validation split:

| saved epoch | optimizer step | validation loss |
|---:|---:|---:|
| 0 | 646 | 3.3420 |
| **4** | **3,230** | **2.5381** |
| 8 | 5,814 | 2.5980 |
| 12 | 8,398 | 2.7164 |
| 16 | 10,982 | 2.8413 |
| 19 | 12,920 | 2.9069 |

Epoch 4 is the checkpoint on the Hub. Later epochs still raise token accuracy a little, but validation cross-entropy gets worse, so uploading epoch 19 just because it was last would hide the overfitting.

Future runs do this selection while training: validation runs halfway through every epoch and again at epoch end, and `*_best.pt` is overwritten only when `current_val_loss < best_val_loss`. That keeps one useful checkpoint on disk instead of another 354 MiB file every few epochs.

#### benchmark

The published checkpoint was then evaluated through the same [`benchmark_hf_collection.py`](scripts/benchmark_hf_collection.py) path used for the other attention models: core metrics over the complete 861-batch validation loader and generation metrics over 128 greedy summaries.

<table>
  <tr>
    <td align="center"><strong>evaluation quality</strong><br><img src="benchmarks/attention/assets/eval_quality_causal.png" alt="KDA causal evaluation quality benchmark"></td>
    <td align="center"><strong>generation quality</strong><br><img src="benchmarks/attention/assets/generation_quality_causal.png" alt="KDA causal generation quality benchmark"></td>
  </tr>
  <tr>
    <td align="center"><strong>throughput</strong><br><img src="benchmarks/attention/assets/throughput.png" alt="attention benchmark throughput"></td>
    <td align="center"><strong>quality vs. efficiency</strong><br><img src="benchmarks/attention/assets/tradeoff.png" alt="attention benchmark quality versus efficiency"></td>
  </tr>
</table>

| metric | KDA epoch 4 |
|---|---:|
| validation loss | 2.5381 |
| perplexity | 12.6559 |
| token accuracy | 0.5497 |
| top-5 accuracy | 0.7400 |
| ROUGE-1 | 0.2556 |
| ROUGE-2 | 0.0853 |
| ROUGE-L | 0.2055 |
| BLEU | 7.90 |
| evaluation throughput | 4,789 tok/s |
| generation throughput | 92.86 tok/s |
| average forward latency | 13.17 ms |
| peak CUDA memory | 189.5 MB |
| parameters | 30,896,560 |
| checkpoint size | 370.9 MB |

The raw checkpoint ranking and standalone outputs live in [`benchmarks/kda`](benchmarks/kda). KDA is also merged into the combined loss, quality, throughput, and tradeoff charts in [`benchmarks/attention`](benchmarks/attention). The best checkpoint, exact 12,920-point loss CSV/SVG, config, tokenizer, architecture image, and metric-bearing model card are published at [`Pradheep1647/meeting_summarization_kda-meetingbank-bs8-e20-bf16-4`](https://huggingface.co/Pradheep1647/meeting_summarization_kda-meetingbank-bs8-e20-bf16-4) and included in the [`transformer-lab` collection](https://huggingface.co/collections/Pradheep1647/transformer-lab-6a07fe3185f5728e217997e0).

## MSA paper-alignment: old vs new

The first cut of MSA (minimax sparse attention) was close to the paper but not exact. [`fix(attention): align msa with paper`](../../commit/e1a3421) corrected it. To check the fix was actually worth it I trained both versions on the same footing: MeetingBank causal summarization, 20 epochs, batch 8, fp32, lr 1e-4, and identical seed/data. I then evaluated the final checkpoints on the validation split (core metrics over 100 batches, ROUGE/BLEU over 16 generated summaries).

![MSA old vs paper-aligned training loss](assets/msa_paper_compare_loss.png)

![MSA old vs paper-aligned eval metrics](assets/msa_paper_compare_eval.png)

| metric | old (`2d018af`) | new (`e1a3421`) | Δ new−old |
|--------|----------------:|----------------:|----------:|
| eval loss | 2.484 | 2.571 | +0.087 |
| perplexity | 11.99 | 13.08 | +1.09 |
| token acc | 0.518 | 0.502 | −0.017 |
| top-5 acc | 0.744 | 0.732 | −0.012 |
| ROUGE-1 | 0.084 | **0.177** | +0.093 |
| ROUGE-2 | 0.050 | **0.102** | +0.052 |
| ROUGE-L | 0.083 | **0.173** | +0.090 |
| BLEU | 2.20 | **6.77** | +4.58 |
| eval tok/s | 7186 | 7278 | +92 |

The new implementation is the one from the official MiniMax tech report; the old one was my own approximation of it. Two things changed in `msa.py`, and both are about dropping my shortcuts in favour of what the report actually specifies:

1. **how the block selector gets trained.** my old version kept hard top-k for the forward pass but added the index branch's block log-probs straight onto the sparse attention logits (`block_score_bias`), purely so `w_iq`/`w_ik` would get gradient from the LM loss. it works, but it contaminates the attention the model actually uses. every value ends up weighted by a blend of real query-key affinity and a coarse block-level score. the report doesn't do that. the new version detaches the index branch entirely, keeps the forward logits pure `q·k`, and trains the selector with a separate KL loss (`kl_alignment_loss`) that matches the index distribution to the full attention's. the selector learns to *predict* which blocks real attention wants instead of leaking into it.

2. **the local block is now mandatory.** old top-k just took the k highest-scoring blocks, so the block containing the query itself could get dropped when the index scores were noisy. the new one reserves a slot for the local block and fills the rest with the best non-local blocks, exactly as in the report. for a decoder the most recent tokens are the ones you can least afford to miss.

with that, the numbers make sense. under teacher forcing the old score-bias acts like a mild prior: the gold token is handed over at every step, nothing goes off the rails, and the extra bias even nudges perplexity slightly lower. that is why old looks a hair better on eval loss / ppl / token accuracy. but that regime never stresses the selector. the moment you generate free-running, the two fixes pay off: clean attention, a selector trained to mimic it, and guaranteed local context mean errors stop compounding. that's the ~2× ROUGE and ~3× BLEU jump (ROUGE-L 0.083 → 0.173, BLEU 2.2 → 6.8). throughput is identical, so it's a pure correctness win. the teacher-forced numbers that "favour" the old impl are an artifact of the crutch, not a real edge. judge attention on generation, not perplexity.

### MLA baseline

Trained MLA on MeetingBank using the existing causal summarization setup: 20 epochs, batch size 8, fp32, lr 1e-4, and seed 42 in about 2 hours on Apple M5 Pro (MPS).

#### training run

![MLA training loss](benchmarks/mla/assets/loss_curves.png)

The final logged training loss was 2.47865 and the final validation loss was 3.26368.

#### checkpoint selection

Validation ran halfway through each epoch and at epoch end. The trainer kept the checkpoint with the lowest validation loss:

| checkpoint | epoch index | optimizer step | validation loss |
|---|---:|---:|---:|
| `meeting_causal_mla_best.pt` | 12 | 8,411 | **3.08166** |

Best checkpoint was used for evaluation.

#### benchmark

Evaluated the best MLA checkpoint on Apple M5 Pro / MPS in fp32 with weight absorption disabled. Core metrics use the full validation split with batch size 8. Generation metrics use 128 greedy summaries, processed one at a time. Kept the results separate from the cross-model attention benchmark because they use different hardware.

<table>
  <tr>
    <td align="center"><strong>evaluation quality</strong><br><img src="benchmarks/mla/assets/eval_quality_causal.png" alt="MLA evaluation quality"></td>
    <td align="center"><strong>generation quality</strong><br><img src="benchmarks/mla/assets/generation_quality_causal.png" alt="MLA generation quality"></td>
  </tr>
  <tr>
    <td align="center"><strong>throughput</strong><br><img src="benchmarks/mla/assets/throughput.png" alt="MLA throughput"></td>
    <td align="center"><strong>quality vs. efficiency</strong><br><img src="benchmarks/mla/assets/tradeoff.png" alt="MLA quality versus efficiency comparison"></td>
  </tr>
</table>

| metric | MLA |
|---|---:|
| validation loss | 3.08165996 |
| perplexity | 21.7946 |
| token accuracy | 0.4969 |
| ROUGE-L | 0.1474 |
| BLEU | 3.9311 |
| evaluation throughput, tok/s | 3,822.60 |
| generation throughput, tok/s | 298.87 |

#### weight absorption inference comparison

[`scripts/benchmark_mla.py`](scripts/benchmark_mla.py) compares absorption off/on using one checkpoint and identical synthetic token sequences. It reports prefill time, decode time per token, and logit agreement in [standard_vs_absorbed.json](benchmarks/mla/standard_vs_absorbed.json).

![Standard vs. absorbed MLA](benchmarks/mla/assets/standard_vs_absorbed.png)

## Tests

```bash
pytest tests/ -q
```

## License

MIT. See [`LICENSE`](LICENSE).
