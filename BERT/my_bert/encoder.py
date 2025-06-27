from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads)
        self.ffn = PositionwiseFeedForward(d_model = d_model, d_ff = d_ff, dropout_prob = dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
    
    def forward(self, x, mask = None):
        residual = x
        x = self.norm1(x)
        attention_output = self.attention(q = x, k = x, v = x, mask = mask)
        x = self.dropout1(attention_output)
        x = x + residual

        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = self.dropout2(ffn_output)
        x = x + residual

        return x