import torch
import torch.nn as nn
import math

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model_size : int, d_ff : int, dropout : float):
        super().__init__()
        self.d_model = d_model_size  # Add for consistency with hyper connection access
        self.d_model_size = d_model_size
        self.linear_layer_1 = nn.Linear(d_model_size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer_2 = nn.Linear(d_ff, d_model_size)

    def forward(self, x):
        return self.linear_layer_2(self.dropout(torch.relu(self.linear_layer_1(x))))
