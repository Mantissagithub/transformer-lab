from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.masks import causal_mask

from .blocks import CausalBlock

# The KVCache LRU class in src/components/attention/kv_cache.py is a memoization
# helper, not the right primitive for per-layer past_kv during generation. We use
# plain (k, v) tuples instead, one per layer.

PastKVs = List[Tuple[torch.Tensor, torch.Tensor]]


class CausalLM(nn.Module):
    def __init__(
        self,
        embed: nn.Module,
        pos: nn.Module,
        layers: nn.ModuleList,
        norm: nn.Module,
        projection: nn.Module,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.pos = pos
        self.layers = layers
        self.norm = norm
        self.projection = projection

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kvs: Optional[PastKVs] = None,
        return_kvs: bool = False,
    ) -> torch.Tensor:
        b, sq = input_ids.shape
        past_len = 0 if past_kvs is None else past_kvs[0][0].shape[-2]
        x = self.pos(self.embed(input_ids))

        if past_kvs is None:
            mask = causal_mask(sq).to(x.device)
        else:
            total = past_len + sq
            full = causal_mask(total).to(x.device)
            mask = full[:, past_len:total, :total]

        if past_kvs is None and not return_kvs:
            for layer in self.layers:
                x = layer(x, mask)
            x = self.norm(x)
            return self.projection(x)

        new_kvs: PastKVs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = layer.forward_with_cache(x, mask, past_kv)
            new_kvs.append(new_kv)
        x = self.norm(x)
        logits = self.projection(x)
        if return_kvs:
            return logits, new_kvs
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        ids = prompt_ids
        logits, past_kvs = self.forward(ids, past_kvs=None, return_kvs=True)
        out_ids = [ids]
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            if temperature == 0.0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / max(temperature, 1e-8)
                if top_k is not None:
                    v, _ = torch.topk(next_logits, k=top_k, dim=-1)
                    next_logits = next_logits.masked_fill(
                        next_logits < v[:, -1:], float("-inf")
                    )
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            out_ids.append(next_id)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
            logits, past_kvs = self.forward(next_id, past_kvs=past_kvs, return_kvs=True)
        if was_training:
            self.train()
        return torch.cat(out_ids, dim=-1)
