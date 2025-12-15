# Train a small DiffusionGPT on tiny Shakespeare (character-level).
#
# First prepare the dataset:
#   python data/shakespeare_char/prepare.py
#
# Then train:
#   python train.py config/train_shakespeare_char.py

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
mask_schedule = "cosine"
exact_masked_tokens = False
denoise_loss = "masked"
ensure_min_revealed_tokens = 16
loss_reduction = "token"
loss_weighting = "none"
t_sampling = "uniform"
t_sampling_power = 1.0

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0

learning_rate = 6e-4
weight_decay = 1e-2
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5
beta2 = 0.95
warmup_iters = 200

# On CPU:
# device = "cpu"
# compile = False
