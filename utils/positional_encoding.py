import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model_size : int, seq_len : int, dropout : float) -> None:
        super().__init__()
        self.d_model_size = d_model_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(seq_len, d_model_size)

        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_size, 2).float() * (-math.log(100000.0)/ d_model_size))


        positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        positional_encoding[:, 1::2] = torch.cos(positions * div_term)

        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('pe', positional_encoding)


    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)    
        return self.dropout(x)
