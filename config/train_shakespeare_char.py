# Train a small DiffusionGPT on tiny Shakespeare (character-level).
#
# First prepare the dataset:
#   python diffusionGPT/data/shakespeare_char/prepare.py
#
# Then train:
#   python diffusionGPT/train.py diffusionGPT/config/train_shakespeare_char.py

out_dir = "out-diffusion-shakespeare-char"
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = "diffusiongpt"
wandb_run_name = "shakespeare-char"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

diffusion_steps = 200
mask_schedule = "linear"

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# On CPU:
# device = "cpu"
# compile = False
