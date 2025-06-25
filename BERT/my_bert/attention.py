import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q: torch.Tensor, k:torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    d_k = k.size(-1)
    attention_score = torch.matmul(q, k.transpose(-2, -1))
    scaled_attention_scores = attention_score / math.sqrt(d_k)
    if mask is not None:
        scaled_attention_scores = scaled_attention_scores.masked_fill(mask == 0, -1e9)
    attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k:torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = q.size(0)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_output, _ = scaled_dot_product_attention(q, k, v, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attention_output)
        return output