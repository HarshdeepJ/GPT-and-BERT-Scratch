import torch.nn as nn
from .embedding import BERTEmbedding
from .encoder import TransformerEncoderBlock

class BertModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_len, dropout_prob: float = 0.01, num_layers: int = 12):
<<<<<<< HEAD
        super().__init__()
=======
>>>>>>> db5ae0b3460336e48ab6d2b028b1fac6107b503f
        self.embedding = BERTEmbedding(vocab_size = vocab_size, embed_size = d_model, max_len = max_len, dropout_prob = dropout_prob)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderBlock(
                d_model = d_model,
                num_heads = num_heads,
                d_ff = d_ff,
                dropout_prob = dropout_prob,
            ) for _ in range(num_layers)
            ]
        )
    
    def forward(self, input_ids, segment_ids, attention_mask =  None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        x = self.embedding(input_ids = input_ids, segment_ids = segment_ids)

        for layer in self.encoder_layers:
            x = layer(x, mask = attention_mask)
        
        return x