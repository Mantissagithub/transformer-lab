import torch 
import torch.nn as nn
import math

class ProjectionLayer(nn.Module):
    def __init__(self, d_model_size : int, vocab_size : int):
        super().__init__()
        self.d_model_size = d_model_size
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model_size, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)