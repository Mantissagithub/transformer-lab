# Transformer Lab — Attention Benchmark

Attention variants trained on MeetingBank, grouped into the two model families they were trained in — **encoder–decoder transformers** and **decoder-only (causal-LM)** models. Because the families optimise different objectives, the quality metrics are benchmarked **separately per family**; only the cross-cutting views (training loss, throughput, and the speed/quality overview) place them on a shared axis.

**Setup** — dataset `meetingbank` / `validation` · generation samples `128` · precision `fp32` by checkpoint · [model collection](https://huggingface.co/collections/Pradheep1647/transformer-lab-6a07fe3185f5728e217997e0).

## Throughput

![Throughput](assets/throughput.png)

Evaluation and autoregressive-generation speed in tokens per second — the metric that is directly comparable across both families.

## Quality vs. efficiency

> Cross-family overview only. ROUGE-L against generation speed for all variants (circles = decoder-only, squares = encoder–decoder). Quality is *not* comparable across families — see each family's own section below for the head-to-head numbers.

![Quality vs. efficiency](assets/tradeoff.png)

## Encoder–decoder transformers

<details>
<summary>Raw metrics — Encoder–decoder transformers</summary>

| Attention Variant | Repo | Status | Loss | PPL | Tok Acc | ROUGE-L | BLEU | Tok/s | Gen tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

</details>

## Decoder-only (causal-LM)

> **Why this family scores much lower on generation.** Encoder–decoder models get cross-attention that aligns the decoder directly to the source transcript, making the copy-heavy summarization task far easier; decoder-only models must learn that purely in-context. These models also compress or sparsify the KV cache, discarding source detail that copying needs, and they show the classic teacher-forcing gap — msa reaches the lowest perplexity of any model yet near-zero BLEU, because low next-token loss does not survive free-running generation. Compare these variants against each other, not against the encoder–decoder family above.

**Highlights**

- Lowest perplexity — **mla** (21.8).
- Best generation overlap (ROUGE-L) — **mla** (0.147).
- Fastest generation — **mla** (299 tok/s).

### Evaluation quality

![Evaluation quality](assets/eval_quality_causal.png)

Teacher-forced perplexity (lower is better) and next-token accuracy on the held-out validation split.

### Generation quality

![Generation quality](assets/generation_quality_causal.png)

Summary-generation overlap against reference summaries (ROUGE-L and BLEU).

<details>
<summary>Raw metrics — Decoder-only (causal-LM)</summary>

| Attention Variant | Repo | Status | Loss | PPL | Tok Acc | ROUGE-L | BLEU | Tok/s | Gen tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mla (multi-head latent attention) | [`junuxyz/mla-meetingbank-bs8-e20-fp32-12`](https://huggingface.co/junuxyz/mla-meetingbank-bs8-e20-fp32-12) | ok | 3.082 | 21.79 | 0.4969 | 0.1474 | 3.931 | 3823 | 298.9 |

</details>

## Models

**Decoder-only**

- <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="16" alt="🤗"/> **mla** [`junuxyz/mla-meetingbank-bs8-e20-fp32-12`](https://huggingface.co/junuxyz/mla-meetingbank-bs8-e20-fp32-12) (ok)
