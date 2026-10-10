# Attention variants

This page keeps the attention family in one place. Each variant has its own architecture diagram, paper tag, local implementation/config links, and a short note on when to use it.

The diagrams use a black-canvas explainer style: token tensors, projection boxes, per-head lanes, attention heatmaps, and a bottom formula strip. They are exported as image assets from HTML-style layouts.

## `mha`

![MHA architecture](assets/mha.png)

Multi-head attention is the baseline from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). It projects the input into separate query, key, and value tensors, computes dense token-to-token attention, and applies the attention pattern to the values.

- Implementation: [`mha.py`](../src/components/attention/mha.py)
- Config: [`mha.yaml`](../configs/attention/mha.yaml)
- Use when: you want the clean reference path before testing cache or context-efficiency changes.

## `mqa`

![MQA architecture](assets/mqa.png)

Multi-query attention follows [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). Query heads stay separate, but all heads share one key stream and one value stream, reducing decode-time KV bandwidth.

- Implementation: [`mqa.py`](../src/components/attention/mqa.py)
- Config: [`mqa.yaml`](../configs/attention/mqa.yaml)
- Use when: KV-cache size and memory bandwidth matter more than per-head KV capacity.

## `gqa`

![GQA architecture](assets/gqa.png)

Grouped-query attention follows [GQA](https://arxiv.org/abs/2305.13245). It keeps multiple KV heads, but fewer than the number of query heads, so it sits between MHA and MQA.

- Implementation: [`gqa.py`](../src/components/attention/gqa.py)
- Config: [`gqa.yaml`](../configs/attention/gqa.yaml)
- Use when: you want most of MHA's modeling shape with a smaller KV cache.

## `gqa_rope`

![GQA with RoPE architecture](assets/gqa_rope.png)

`gqa_rope` combines [GQA](https://arxiv.org/abs/2305.13245) with rotary position embeddings from [RoFormer](https://arxiv.org/abs/2104.09864). In this repo, RoPE is applied inside attention to Q and K before grouped-KV attention runs.

- Implementation: [`gqa_rope.py`](../src/components/attention/gqa_rope.py)
- Config: [`gqa_rope.yaml`](../configs/attention/gqa_rope.yaml)
- Use when: you want grouped KV heads plus rotary positions.

## `sliding_window`

![Sliding-window attention architecture](assets/sliding_window.png)

Sliding-window attention uses the local-window idea associated with [Longformer](https://arxiv.org/abs/2004.05150). It keeps dense heads, but the attention mask only allows nearby tokens through.

- Implementation: [`sliding_window.py`](../src/components/attention/sliding_window.py)
- Config: [`sliding_window.yaml`](../configs/attention/sliding_window.yaml)
- Use when: you want a local-context baseline without changing the head layout.

## `sliding_gqa`

![Sliding GQA architecture](assets/sliding_gqa.png)

`sliding_gqa` combines local-window attention with grouped KV heads. It has both the reduced attention span of sliding-window attention and the smaller KV layout of GQA.

- Implementation: [`sliding_gqa.py`](../src/components/attention/sliding_gqa.py)
- Config: [`sliding_gqa.yaml`](../configs/attention/sliding_gqa.yaml)
- Use when: you want local attention and grouped cache savings together.

## `gemma3_hybrid`

![Gemma 3 hybrid attention architecture](assets/gemma3_hybrid.png)

The Gemma-style hybrid pattern is tagged to the [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786). It composes existing repo variants: several local `sliding_gqa` layers followed by a global `gqa_rope` layer.

- Implementation: composed in [`builder.py`](../src/model/builder.py)
- Config: [`gemma3_hybrid.yaml`](../configs/attention/gemma3_hybrid.yaml)
- Use when: you want an interleaved local/global pattern without adding a new kernel.

## `mla`

![MLA architecture](assets/mla.png)

Multi-head latent attention is tagged to [DeepSeek-V2](https://arxiv.org/abs/2405.04434). It caches a compact latent KV state plus a decoupled RoPE key, then reconstructs per-head K/V during attention.

- Implementation: [`mla.py`](../src/components/attention/mla.py)
- Config: [`mla.yaml`](../configs/attention/mla.yaml)
- Use when: KV-cache memory is the main target. The current implementation saves cache memory, but does not yet reduce decode-time projection FLOPs.

## `csa`

![CSA architecture](assets/csa.png)

Compressed sparse attention is tagged to the [DeepSeek-V4 docs](https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4). It compresses KV blocks, scores compressed entries with a sparse indexer, gathers top-k memory, keeps a local branch, and projects the result back through grouped output projection.

- Implementation: [`csa.py`](../src/components/attention/csa.py)
- Config: [`csa.yaml`](../configs/attention/csa.yaml)
- Use when: you want a long-context experiment that keeps local detail while sparsifying compressed long-range memory.

## `hca`

![HCA architecture](assets/hca.png)

Heavily compressed attention is also tagged to the [DeepSeek-V4 docs](https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4). It is the simpler compressed sibling of CSA: single-stream block compression, no sparse indexer, no local branch, then shared-KV attention over compressed entries.

- Implementation: [`hca.py`](../src/components/attention/hca.py)
- Config: [`hca.yaml`](../configs/attention/hca.yaml)
- Use when: you want a simpler compressed-memory attention path with heavier KV reduction.

## `msa`

![MSA architecture](assets/msa.png)

MiniMax sparse attention follows [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392). It is a GQA-based block-sparse self-attention: a cheap index branch runs one index query per GQA group against a single shared index key head, block-max-pools token scores into per-block scores, always keeps the query's local block, and uses the remaining budget for top-k block selection. The main branch then runs exact GQA attention only over tokens gathered from those selected blocks, so each query attends to at most `top_k * block_size` keys instead of the whole sequence.

The implementation keeps the paper's separation between routing and attention: index scores choose blocks, but they are not added as a bias to the main attention logits. Because hard top-k routing is not differentiable, the module exposes `kl_alignment_loss(...)` to train the index branch against the group-averaged main-branch attention distribution on the selected token support, with detached teacher and hidden-state inputs.

- Implementation: [`msa.py`](../src/components/attention/msa.py)
- Config: [`msa.yaml`](../configs/attention/msa.yaml)
- Use when: you want long-context attention that keeps GQA's KV savings and adds learned block sparsity, picking which KV blocks each query reads instead of reading them all. For training, add the returned KL alignment term to the language-modeling loss.

## `kda`

<img src="assets/kimi-linear-architecture.jpg" alt="Official Kimi Linear architecture with the KDA block expanded" width="720">

*Official Kimi Linear architecture figure from [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear/blob/master/figures/arch.png), © 2025 Moonshot AI, [MIT licensed](assets/kimi-linear-architecture.LICENSE). The lower-right panel is the KDA layer implemented here. The left side is Moonshot's hybrid stack; this repo's current experiment uses KDA in every attention layer instead of mixing KDA and MLA.*

Kimi Delta Attention comes from [Kimi Linear](https://arxiv.org/abs/2510.26692), with the two changes used by [Kimi K3](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf): lower-bounded channel decay and a full-rank output gate.

The input first splits into q, k, and v projections. Each branch runs through a causal depthwise convolution and SiLU; q and k are then L2-normalized. A separate low-rank branch produces one decay value for every key channel, while beta decides how strongly the current key-value pair should rewrite memory. The state update is the useful bit: decay the old matrix, read what it currently predicts for k, then write only the prediction error. Each head therefore carries one fixed `head_dim × head_dim` state instead of a token-by-token KV cache.

K3 bounds the log-decay with `g = -5 sigmoid(exp(A) z)` and uses `alpha = exp(g)`. This keeps every step's log-decay inside `(-5, 0)`. After reading the updated state with q, the implementation applies head-wise RMSNorm, a full-rank sigmoid gate from the original input, and the output projection.

- Implementation: [`kda.py`](../src/components/attention/kda.py)
- Config: [`kda.yaml`](../configs/attention/kda.yaml)
- Training config: [`meeting_summarization_kda.yaml`](../configs/experiment/meeting_summarization_kda.yaml)
- Use when: you want to experiment with K3-style finite-state attention and constant-size decode memory. Run it as a causal LM with `positional=rope`; RoPE stays identity here, so KDA owns position and recency as intended.

## Model compatibility

The encoder-decoder path needs attention modules that support cross-attention and bidirectional encoder attention. In this repo, `mha`, `mqa`, `gqa`, `gqa_rope`, `sliding_window`, `sliding_gqa`, and `gemma3_hybrid` satisfy that path.

`csa`, `hca`, `mla`, `msa`, and `kda` are self-attention-only variants and should run through the causal-LM path. The builder enforces this split so an experiment does not silently train with the wrong topology.
