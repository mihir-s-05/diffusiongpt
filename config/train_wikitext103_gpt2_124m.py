"""
Train a GPT-2 (124M-sized) DiffusionGPT on WikiText-103 (GPT-2 BPE).

Usage:
  python data/wikitext103/prepare.py
  python train.py config/train_wikitext103_gpt2_124m.py
"""

out_dir = "out-diffusion-wikitext103-gpt2-124m"
eval_interval = 2000
log_interval = 10
eval_iters = 100
always_save_checkpoint = True

wandb_log = False
wandb_project = "diffusiongpt"
wandb_run_name = "wikitext103-gpt2-124m"

dataset = "wikitext103"

gradient_accumulation_steps = 8
batch_size = 16
block_size = 1024

diffusion_steps = 500
mask_schedule = "cosine"
denoise_loss = "reveal"
ensure_min_revealed_tokens = 32
loss_reduction = "token"
loss_weighting = "snr"
t_sampling = "pow"
t_sampling_power = 2.0

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 10000
lr_decay_iters = 600000
min_lr = 6e-5

