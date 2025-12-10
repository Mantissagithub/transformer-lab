# this is the file i'm gonna code up the kv cache with lru eviction strategy
# when i think in first principles it's just a set of values like [{keys, values, timestamp}] where timestamp is the last used time
# also we can store that in the constatn mem or the hbm as obviously pytorch uses the gpu

import time
import torch
# import sys
import torch.nn as nn
from collections import OrderedDict

class KVCache(nn.Module):
  def __init__(self, max_size):
    super().__init__()
    self.max_size = max_size
    # self.cache = [] # [key, value, timestamp]
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.cache = OrderedDict() # (key) -> (value, timestamp)

  def get_by_key(self, key):
    if key not in self.cache:
      print(f"key: {key} not found!!")
      return None

    value, _ = self.cache[key]
    current_time_ms = int(time.time() * 1000)
    self.cache[key] = (value, current_time_ms)
    print(f"found key: {key} with value: {value}")
    return value

  def put(self, key, value):
    current_time_ms = int(time.time()*1000)
    if key in self.cache:
      self.cache.move_to_end(key)
      self.cache[key] = (value, current_time_ms)
      print(f"key: {key} already found, updated!!")
      return

    self.cache[key] = (value, current_time_ms)

    if len(self.cache) > self.max_size:
      lru_cache = next(iter(self.cache))
      self.cache.pop(lru_cache)
      print(f"lru evicted: {lru_cache}")

    print(f"successfully added key: {key}, value: {value} with timestamp_ms: {current_time_ms}")

  def to_gpu(self):
    if self.device == 'cuda':
      # i can use torch.tensor over here
      gpu_cache = OrderedDict()
      for k, (v, ts) in self.cache.items():
        if not isinstance(v, torch.Tensor):
          v = torch.tensor(v, device=self.device)
        else:
          v = v.to(self.device)
        gpu_cache[k] = (v, ts)
      self.cache = gpu_cache
      print("moved to gpu")
    else:
      print("gpu cannot be found!!")


