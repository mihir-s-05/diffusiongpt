"""
Config for training on WikiText-103 with ~8GB GPUs.

Usage:
  python data/wikitext103/prepare.py
  python train.py config/train_wikitext103_4070_mobile.py
"""

out_dir = "out-diffusion-wikitext103-4070-mobile"
eval_interval = 2000
log_interval = 10
eval_iters = 50
always_save_checkpoint = True

wandb_log = False
wandb_project = "diffusiongpt"
wandb_run_name = "wikitext103-4070-mobile"

dataset = "wikitext103"

gradient_accumulation_steps = 16
batch_size = 4
block_size = 256

diffusion_steps = 200
mask_schedule = "cosine"
denoise_loss = "reveal"
ensure_min_revealed_tokens = 16
loss_reduction = "token"
loss_weighting = "none"
t_sampling = "uniform"
t_sampling_power = 1.0

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 200000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 200000
min_lr = 6e-5

compile = False
