# this is the file i'm gonna code up the kv cache with lru eviction strategy
# when i think in first principles it's just a set of values like {keys[i], values[i], timestamp[i]} where timestamp is the last used time
# also we can store that in the constatn mem or the hbm as obviously pytorch uses the gpu

import time
import torch
import sys

class KVCache:
  def __init__(self, max_size):
    self.max_size = max_size
    self.cache = [] # [key, value, timestamp]

  def put_into_cuda_const_mem(self):
    if not torch.cuda.is_available():
      print("either pytorch is not able to access the gpu, or there is no gpu!!")
      return

    self.register_buffer('kv_cache', self.cache)

  def sort_lru(self):
    self.cache.sort(key=self.cache[2]).reverse()

  def get_by_key(self, key):
    for x in self.cache:
      if key == x[0]:
        print(f"found the key: {key}, and the value is {x[1]}")
        return x[1]
    print(f"key: {key} not found!!")
    sys.exit(0)

  def put(self, key, value):
    current_time_ms = int(time.time()*1000)
    self.cache.append([key, value, current_time_ms])
    if len(self.cache) > self.max_size:
      self.sort_lru()
      self.cache.pop()
      print("the least recently used element is popped!!")
    print(f"added key; {key}, value: {value} at timestamp in ms: {current_time_ms}")

