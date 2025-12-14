# Train a GPT-2 (124M-sized) DiffusionGPT on OpenWebText.
#
# Prepare data first (requires `pip install datasets tqdm`):
#   python diffusionGPT/data/openwebtext/prepare.py
#
# Then train (single node, multi-GPU example):
#   torchrun --standalone --nproc_per_node=8 diffusionGPT/train.py diffusionGPT/config/train_openwebtext_gpt2_124m.py

out_dir = "out-diffusion-openwebtext-gpt2-124m"
eval_interval = 2000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True

wandb_log = False
wandb_project = "diffusiongpt"
wandb_run_name = "owt-gpt2-124m"

dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

diffusion_steps = 200
mask_schedule = "cosine"

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
