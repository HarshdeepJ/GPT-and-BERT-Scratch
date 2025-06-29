import os
import json

class CharacterTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
    
    def build_vocab(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f'Vocabulary built. Found {self.vocab_size} unique characters.')
    
    def encode(self, s: str) -> list[int]:
        if not self.stoi:
            raise RuntimeError('Vocabulary not built. Call build_vocab() first.')
        return [self.stoi[c] for c in s]
    
    def decode(self, l: list[int]) -> str:
        if not self.itos:
            raise RuntimeError('Vocabulary not built. Call build_vocab() first.')
        s = ''
        for i in l:
            s+=self.itos[l]
        return s

    def save_vocab(self, path: str):
        vocab_data = {
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self.vocab_size,
        }

        with open(path, 'w', encoding = 'utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)
        print(f'Vocabulary saved to {path}')
    
    def load_vocab(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Vocabulary file not found at {path}')
        
        with open(path, 'r', encoding = 'utf-8') as f:
            vocab_data = json.load(f)
        
        self.stoi = vocab_data['stoi']
        self.itos = {int(k): v for k, v in vocab_data['itos'].items()}

        print(f'Vocabulary loaded from {path}. Vocab size: {self.vocab_size}')