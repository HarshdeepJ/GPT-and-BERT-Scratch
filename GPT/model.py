import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias, eps = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
    
    def forward(self, x):   
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, bias):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias = bias)
        
        self.c_proj = nn.Linear(n_embd, n_embd, bias = bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.register_buffer('bias_mask', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    
    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim = 2)
        
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

        self.ffn_1 = nn.Linear(n_embd, 4*n_embd, bias = bias)
        self.ffn_2 = nn.Linear(4*n_embd, n_embd, bias = bias)

        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.activation(self.ffn_1(x))
        x = self.dropout_layer(self.ffn_2(x))
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, bias):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias = bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embd, bias = bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, bias):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, block_size, dropout, bias) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, bias = bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        elif isinstance(module, (nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, target: torch.LongTensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B,T = idx.size()
        assert T <= self.block_size, 'Positional Embeddings are only defined up to block_size'
        
        tok_emb = self.transformer.wte(idx)
        
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos)
        
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if target is not None:
            logits = logits.view(-1, self.vocab_size)
            loss = F.cross_entropy(logits, target.view(-1), ignore_index=-1)
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature: float = 1.0, top_k: int = None):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx
                if idx.size(1)>self.block_size:
                    idx_cond = idx[:, -self.block_size:]
                
                logits, _ =  self(idx_cond)
                logits = logits[:, -1, :]
                logits = logits/temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('-Inf')
                
                prob_dist = F.softmax(logits, dim = -1)
                idx_next = torch.multinomial(prob_dist, num_samples=1)
                idx = torch.cat((idx, idx_next), dim = 1)
            
            return idx