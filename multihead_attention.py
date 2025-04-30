import torch
import torch.nn as nn
import math

class MultiHeadAttentionNetwork(nn.Module):
    def __init__(self, d_model_size: int, h : int, dropout : float):
        super().__init__()
        self.d_model_size = d_model_size
        self.h = h
        assert d_model_size%h == 0, "d_model_size is not divisible by h"

        self.dk = d_model_size//h
        self.wq = nn.Linear(d_model_size, d_model_size)
        self.wk = nn.Linear(d_model_size, d_model_size)
        self.wv = nn.Linear(d_model_size, d_model_size)

        self.wo = nn.Linear(d_model_size, d_model_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores
    
    def forward(self, k, q, v, mask):
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dk).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dk).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionNetwork.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.dk)
        
        return self.wo(x)