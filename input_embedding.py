import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model_size : int, vocab_size : int):
        super().__init__()
        self.d_model_size = d_model_size
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, d_model_size)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model_size)
    

    