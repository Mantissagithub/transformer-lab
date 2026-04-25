import time
from collections import OrderedDict

import torch
import torch.nn as nn


class KVCache(nn.Module):
    """LRU-bounded key/value cache. Kept here so generation work has a home."""

    def __init__(self, max_size: int) -> None:
        super().__init__()
        self.max_size = max_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache: "OrderedDict[object, tuple]" = OrderedDict()

    def get_by_key(self, key):
        if key not in self.cache:
            return None
        value, _ = self.cache[key]
        self.cache[key] = (value, int(time.time() * 1000))
        return value

    def put(self, key, value) -> None:
        ts = int(time.time() * 1000)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = (value, ts)
            return
        self.cache[key] = (value, ts)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def to_gpu(self) -> None:
        if self.device != "cuda":
            return
        gpu_cache: "OrderedDict[object, tuple]" = OrderedDict()
        for k, (v, ts) in self.cache.items():
            v = v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
            gpu_cache[k] = (v, ts)
        self.cache = gpu_cache
