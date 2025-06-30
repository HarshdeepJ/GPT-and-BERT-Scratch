import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from .tokenizer import CharacterTokenizer

class TextDataset(Dataset):
    def __init__(self, text_data: str, tokenizer: CharacterTokenizer, block_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        print('Encoding text data...')
        self.data = torch.tensor(tokenizer.encode(text_data), dtype = torch.long)
        print(f'Text data encoded. Total tokens: {len(self.data)}')
        if len(self.data) < self.block_size + 1:
            raise ValueError(
                f'Dataset too small for given block_size.'
                f'Total tokens: {len(self.data)}, required: {self.block_size + 1}'
            )
    
    def __len__(self) -> int:
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y