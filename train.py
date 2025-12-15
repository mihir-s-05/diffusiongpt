"""
Train DiffusionGPT (masked-token diffusion) on .bin datasets.

This is intentionally close in spirit to nanoGPT/train.py: a single-file
training loop with minimal indirection.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import DiffusionGPTConfig, DiffusionGPT
from diffusion import MaskSchedule, q_sample_mask, q_reveal_mask


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _has_working_triton() -> bool:
    try:
        import triton
        return True
    except ImportError:
        return False


def _resolve_dataset_dir(dataset_name: str, data_dir_override: str) -> str:
    """Resolve the directory containing train.bin/val.bin (and optionally meta.pkl)."""

    if data_dir_override:
        return os.path.abspath(data_dir_override)

    preferred = os.path.abspath(os.path.join(THIS_DIR, "data", dataset_name))
    if os.path.isdir(preferred):
        return preferred

    legacy = os.path.abspath(os.path.join(THIS_DIR, "..", "nanoGPT", "data", dataset_name))
    if os.path.isdir(legacy):
        return legacy

    raise FileNotFoundError(
        f"Could not find dataset directory for dataset='{dataset_name}'.\n"
        f"Tried:\n- {preferred}\n- {legacy}\n\n"
        "Either prepare data under diffusionGPT/data/<dataset>/ or pass --data_dir=..."
    )

# -----------------------------------------------------------------------------
# defaults (override via config file and/or CLI: `--key=value`)
# I/O
out_dir = "out-diffusion"
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = "scratch"

# wandb logging
wandb_log = False
wandb_project = "diffusiongpt"
wandb_run_name = "run"

# data
dataset = "shakespeare_char"
data_dir = ""
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# diffusion
diffusion_steps = 200
mask_schedule = "linear"  # linear|cosine|pow
mask_schedule_power = 1.0
exact_masked_tokens = False 
denoise_loss = "reveal"  # masked|reveal
loss_reduction = "token"  # token|example
# Optional weighting across timesteps to avoid collapsing to the unigram solution.
# - inverse_mask_ratio balances examples with different numbers of masked tokens.
# - 1_minus_mask_ratio emphasizes low-noise (more-context) examples.
# - snr combines both (stronger bias to low-noise).
loss_weighting = "none"  # none|1_minus_mask_ratio|inverse_mask_ratio|snr
ensure_min_masked_tokens = 8  # 0 disables; helps avoid degenerate batches at small t
ensure_min_revealed_tokens = 16  # only used when denoise_loss='reveal'
# Timestep sampling distribution. Biasing toward small t often helps the model
# learn to use context before mastering very-high-noise denoising.
t_sampling = "pow"  # uniform|pow
t_sampling_power = 2.0

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# adamw optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# DDP settings
backend = "nccl" if torch.cuda.is_available() and torch.distributed.is_nccl_available() else "gloo"

# system
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else ("float16" if torch.cuda.is_available() else "float32")
)
# torch.compile is great, but on Windows CUDA it commonly fails due to Triton.
# Users can still override this via config/CLI.
compile = torch.cuda.is_available() and os.name != "nt"
# -----------------------------------------------------------------------------

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open(os.path.join(THIS_DIR, "configurator.py"), encoding="utf-8").read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
        "device is set to CUDA, but torch.cuda.is_available() is False. "
        "Install a CUDA-enabled PyTorch build (or pass --device=cpu)."
    )

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
if device_type == "cpu" and not torch.cuda.is_available():
    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build is None:
        print(
            "NOTE: CUDA is not available; this looks like a CPU-only PyTorch build "
            f"(torch.__version__={torch.__version__}). Install a CUDA-enabled torch to use GPU."
        )
    else:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        extra = f", CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}" if cuda_visible_devices is not None else ""
        print(
            "NOTE: CUDA is not available (torch.version.cuda="
            f"{cuda_build!r}{extra}). If you expected GPU, check NVIDIA drivers/CUDA runtime."
        )
print(f"using device: {device} ({device_type}), dtype: {dtype}")

# -----------------------------------------------------------------------------
# data loader (expects train.bin/val.bin, optionally meta.pkl)
dataset_dir = _resolve_dataset_dir(dataset, data_dir)
config["dataset_dir"] = dataset_dir
print(f"using dataset dir: {dataset_dir}")


def get_batch(split: str) -> torch.Tensor:
    # recreate np.memmap every batch to avoid a memory leak
    filename = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(dataset_dir, filename), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x0 = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x0 = x0.pin_memory().to(device, non_blocking=True)
    else:
        x0 = x0.to(device)
    return x0


# -----------------------------------------------------------------------------
# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# vocab_size discovery (char datasets have meta.pkl; otherwise default to GPT-2 BPE)
meta_path = os.path.join(dataset_dir, "meta.pkl")
base_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    base_vocab_size = int(meta["vocab_size"])
    print(f"found vocab_size = {base_vocab_size} (inside {meta_path})")
else:
    base_vocab_size = 50257
    print("no meta.pkl found, defaulting vocab_size to 50257 (GPT-2 BPE)")

schedule = MaskSchedule(kind=mask_schedule, power=mask_schedule_power)

# timestep sampling
def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    if t_sampling == "uniform":
        return torch.randint(1, diffusion_steps + 1, (batch_size,), device=device)
    if t_sampling == "pow":
        if float(t_sampling_power) <= 0:
            raise ValueError("t_sampling_power must be > 0 for t_sampling='pow'")
        u = torch.rand((batch_size,), device=device)
        t = (u.pow(float(t_sampling_power)) * float(diffusion_steps)).to(dtype=torch.long) + 1
        return t.clamp_(1, diffusion_steps)
    raise ValueError(f"Unknown t_sampling: {t_sampling}")


def loss_weights(t: torch.Tensor) -> torch.Tensor:
    """
    Per-example weights w(t) used by the training objective.

    We clamp very small mask ratios to 1/T to avoid extreme weights with cosine schedules.
    """

    mask_ratio = schedule.mask_ratio(t, diffusion_steps=diffusion_steps)  # (B,)
    min_ratio = 1.0 / float(diffusion_steps)
    ratio = mask_ratio.clamp_min(min_ratio)

    if loss_weighting == "none":
        return torch.ones_like(mask_ratio, dtype=torch.float32, device=t.device)
    if loss_weighting == "1_minus_mask_ratio":
        return (1.0 - mask_ratio).clamp_min(min_ratio).to(dtype=torch.float32)
    if loss_weighting == "inverse_mask_ratio":
        return (1.0 / ratio).to(dtype=torch.float32)
    if loss_weighting == "snr":
        return ((1.0 - mask_ratio).clamp_min(min_ratio) / ratio).to(dtype=torch.float32)
    raise ValueError(f"Unknown loss_weighting: {loss_weighting}")


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute diffusion denoising loss on masked positions (targets != -1).

    Supports:
    - loss_reduction='token': average across masked tokens (optionally weighted by w(t))
    - loss_reduction='example': average per example (optionally weighted by w(t))
    """

    per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
        reduction="none",
    ).view_as(targets)
    mask = targets.ne(-1)

    if loss_reduction == "token":
        if loss_weighting == "none":
            denom = mask.sum().clamp_min(1)
            return (per_token * mask).sum() / denom
        w = loss_weights(t).to(dtype=per_token.dtype)[:, None]  # (B, 1)
        denom = (mask * w).sum().clamp_min(1)
        return (per_token * mask * w).sum() / denom

    if loss_reduction == "example":
        denom = mask.sum(dim=1).clamp_min(1)
        per_example = (per_token * mask).sum(dim=1) / denom
        if loss_weighting == "none":
            return per_example.mean()
        w = loss_weights(t).to(dtype=per_example.dtype)
        return (per_example * w).sum() / w.sum().clamp_min(1e-8)

    raise ValueError(f"Unknown loss_reduction: {loss_reduction}")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=base_vocab_size,
    dropout=dropout,
    diffusion_steps=diffusion_steps,
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = DiffusionGPT(DiffusionGPTConfig(**model_args))
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "dropout", "diffusion_steps"]:
        model_args[k] = checkpoint_model_args[k]
    model = DiffusionGPT(DiffusionGPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"Unknown init_from: {init_from}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size

model.to(device)

scaler = torch.amp.GradScaler(device="cuda", enabled=(dtype == "float16" and device_type == "cuda"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"WARNING: could not load GradScaler state, continuing without it: {e}")
checkpoint = None

if compile and device_type == "cuda" and not _has_working_triton():
    print(
        "compile=True but Triton is not available/working on this system; "
        "falling back to eager mode. (Tip: pass --compile=False to silence.)"
    )
    compile = False

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x0 = get_batch(split)
            t = sample_timesteps(x0.size(0), x0.device)
            xt, targets_x0, masked_at_t = q_sample_mask(
                x0,
                t,
                mask_token_id=raw_model.mask_token_id,
                diffusion_steps=diffusion_steps,
                schedule=schedule,
                min_masked_tokens=ensure_min_masked_tokens,
                exact_masked_tokens=exact_masked_tokens,
            )
            if denoise_loss == "masked":
                targets = targets_x0
            elif denoise_loss == "reveal":
                reveal = q_reveal_mask(
                    masked_at_t,
                    t,
                    diffusion_steps=diffusion_steps,
                    schedule=schedule,
                    min_revealed_tokens=ensure_min_revealed_tokens,
                )
                targets = x0.clone()
                targets[~reveal] = -1
            else:
                raise ValueError(f"Unknown denoise_loss: {denoise_loss}")
            with ctx:
                logits, _ = model(xt, t, targets=None)
                loss = compute_loss(logits, targets, t)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# training loop
raw_model = model.module if ddp else model
x0 = get_batch("train")
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    # lightweight stats for debugging the noise distribution
    t_mean = 0.0
    mask_ratio_mean = 0.0
    masked_tokens_mean = 0.0
    predicted_tokens_mean = 0.0

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

        # sample diffusion timestep per example
        t = sample_timesteps(x0.size(0), x0.device)
        xt, targets_x0, masked_at_t = q_sample_mask(
            x0,
            t,
            mask_token_id=raw_model.mask_token_id,
            diffusion_steps=diffusion_steps,
            schedule=schedule,
            min_masked_tokens=ensure_min_masked_tokens,
            exact_masked_tokens=exact_masked_tokens,
        )
        if denoise_loss == "masked":
            targets = targets_x0
        elif denoise_loss == "reveal":
            reveal = q_reveal_mask(
                masked_at_t,
                t,
                diffusion_steps=diffusion_steps,
                schedule=schedule,
                min_revealed_tokens=ensure_min_revealed_tokens,
            )
            targets = x0.clone()
            targets[~reveal] = -1
        else:
            raise ValueError(f"Unknown denoise_loss: {denoise_loss}")
        t_mean += float(t.float().mean().item())
        mask_ratio_mean += float(schedule.mask_ratio(t, diffusion_steps=diffusion_steps).mean().item())
        masked_tokens_mean += float(masked_at_t.sum(dim=1).float().mean().item())
        predicted_tokens_mean += float(targets.ne(-1).sum(dim=1).float().mean().item())

        with ctx:
            logits, _ = model(xt, t, targets=None)
            loss = compute_loss(logits, targets, t)
            loss = loss / gradient_accumulation_steps

        x0 = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        denom = float(gradient_accumulation_steps)
        t_mean_print = t_mean / denom
        mask_ratio_print = mask_ratio_mean / denom
        masked_tokens_print = masked_tokens_mean / denom
        predicted_tokens_print = predicted_tokens_mean / denom
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}% "
            f"(t_mean {t_mean_print:.1f}, mask_ratio {mask_ratio_print:.3f}, "
            f"masked {masked_tokens_print:.1f}, pred {predicted_tokens_print:.1f})"
        )

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
