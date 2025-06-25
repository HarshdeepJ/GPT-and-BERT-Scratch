import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_prob: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.network = nn.Sequential(
            self.linear_1,
            self.activation,
            self.dropout,
            self.linear_2,
        )
    
    def forward(self, x):
        x = self.network(x)
        return x