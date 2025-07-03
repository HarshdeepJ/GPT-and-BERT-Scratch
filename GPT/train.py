import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from torch.utils.tensorboard import SummaryWriter

from config import *
from tokenizer import CharacterTokenizer
from dataset import TextDataset
from model import GPT

os.makedirs(out_dir, exist_ok=True)
logs_dir = os.path.join('logs', time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(logs_dir, exist_ok=True)

writer = SummaryWriter(log_dir=logs_dir)
print(f'Logging to TensorBoard directory: {logs_dir}')
print(f'Checkpoints will be saved to: {out_dir}')

with open(dataset_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharacterTokenizer()
tokenizer.build_vocab(text)

tokenizer_vocab_path = os.path.join(out_dir, 'tokenizer_vocab.json')
tokenizer.save_vocab(tokenizer_vocab_path)
vocab_size = tokenizer.vocab_size

n = int(0.9 * len(text))
train_data = text[:n]
val_data = text[:n]

train_dataset = TextDataset(train_data, tokenizer, block_size)
val_dataset = TextDataset(val_data, tokenizer, block_size)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False, num_workers=0, pin_memory=True)

train_iter = iter(train_loader)
val_iter = iter(val_loader)

model = GPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout, bias).to(device)
optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay, betas = (beta1, beta2))

@torch.no_grad()
def estimate_loss(model, data_loader, eval_iters, current_iter, writer):
    out = {}
    model.eval()
    losses = []

    eval_loader_iter = iter(data_loader)

    for k in range(eval_iters):
        try:
            xb, yb = next(eval_loader_iter)
        except StopIteration:
            eval_loader_iter = iter(data_loader)
            xb, yb = next(eval_loader_iter)
        
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss

global_step = 0
for iter_num in tqdm(range(max_iters), desc = 'Training'):
    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)
    
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
    optimizer.step()

    global_step += 1

    if iter_num % log_interval == 0 or iter_num == max_iters - 1:
        writer.add_scalar('Loss/train', loss.item(), global_step)
        tqdm.write(f'Step {iter_num}: Train Loss: {loss.item(): .4f}')

    if iter_num % eval_interval == 0 or iter_num == max_iters -1:
        val_loss = estimate_loss(model, val_loader, eval_iters, iter_num, writer)
        writer.add_scalar('Loss/val', val_loss, global_step)
        tqdm.write(f'Step {iter_num}: Train Loss {loss.item(): .4f}, Val Loss {val_loss:.4f}')

    if iter_num % checkpoint_interval == 0 or iter_num == max_iters -1:
        checkpoint_path = os.path.join(out_dir, f'ckpt_{iter_num:05d}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': iter_num,
            'train_loss': loss.item(),
        }, checkpoint_path)
        tqdm.write(f'Checkpoint saved to {checkpoint_path}')

print('\n Training complete!')
writer.close()
print('TensorBoard writer closed.')