import numpy as np
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout_prob):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = PositionalEncoding(d_model= embed_size, max_len = max_len, dropout_prob = dropout_prob)
        self.segment_embeddings = nn.Embedding(2, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, segment_ids):
        device = input_ids.device
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids, device = device)

        token_embs = self.token_embeddings(input_ids)
        segment_embs = self.segment_embeddings(segment_ids)
        base_embeddings = token_embs + segment_embs

        final_embeddings = self.position_embeddings(base_embeddings)


        normalized_embeddings = self.norm(final_embeddings)
        final_output = self.dropout(normalized_embeddings)

        return final_output