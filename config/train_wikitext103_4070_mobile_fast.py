"""
Fast training config for WikiText-103 on smaller GPUs (e.g. 8GB).

Usage:
  python data/wikitext103/prepare.py
  python train.py config/train_wikitext103_4070_mobile_fast.py
"""

out_dir = "out-diffusion-wikitext103-4070-fast"
eval_interval = 1000
log_interval = 10
eval_iters = 50
always_save_checkpoint = True

wandb_log = False
wandb_project = "diffusiongpt"
wandb_run_name = "wikitext103-4070-fast"

dataset = "wikitext103"

gradient_accumulation_steps = 8
batch_size = 8
block_size = 256

diffusion_steps = 500
mask_schedule = "cosine"
denoise_loss = "reveal"
ensure_min_revealed_tokens = 16
loss_reduction = "token"
loss_weighting = "snr"
t_sampling = "pow"
t_sampling_power = 2.0

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 3000
lr_decay_iters = 100000
min_lr = 6e-5

compile = False
