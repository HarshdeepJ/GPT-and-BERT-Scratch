import torch
import os

dataset_path = 'data/input.txt'
block_size = 256
n_layer = 1
n_head = 6
n_embd = 384
dropout = 0.2
bias = False
batch_size = 64
learning_rate = 3e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_norm_clip = 1.0
eval_interval = 500
eval_iters = 200
log_interval = 10
out_dir = 'checkpoints'
checkpoint_interval = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generate_max_new_tokens = 500
generate_temperature = 0.8
generate_top_k = None
generate_checkpoint_path = os.path.join(out_dir, 'ckpt_04999.pth')