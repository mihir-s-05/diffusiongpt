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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import DiffusionGPTConfig, DiffusionGPT
from diffusion import MaskSchedule, q_sample_mask


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _has_working_triton() -> bool:
    """
    torch.compile(..., backend="inductor") typically requires Triton on CUDA.
    On Windows, Triton is often unavailable, so prefer an eager fallback.
    """
    try:
        import triton  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _resolve_dataset_dir(dataset_name: str, data_dir_override: str) -> str:
    """
    Resolve the directory containing train.bin/val.bin (and optionally meta.pkl).

    By default this looks under diffusionGPT/data/<dataset>/, making diffusionGPT
    self-contained. For backward compatibility, it also falls back to the legacy
    nanoGPT/data/<dataset>/ location if present.
    """

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
init_from = "scratch"  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = "diffusiongpt"
wandb_run_name = "run"

# data
dataset = "shakespeare_char"  # expects diffusionGPT/data/<dataset>/{train,val}.bin by default
data_dir = ""  # optional: override the dataset directory path
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# diffusion
diffusion_steps = 200
mask_schedule = "linear"  # linear|cosine|pow
mask_schedule_power = 1.0

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# adamw optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
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
        except Exception as e:
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
    # If inductor hits an unsupported pattern or backend issue, prefer eager fallback.
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
    unoptimized_model = model
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
            t = torch.randint(1, diffusion_steps + 1, (x0.size(0),), device=x0.device)
            xt, targets, _ = q_sample_mask(
                x0,
                t,
                mask_token_id=raw_model.mask_token_id,
                diffusion_steps=diffusion_steps,
                schedule=schedule,
            )
            with ctx:
                _, loss = model(xt, t, targets=targets)
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

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

        # sample diffusion timestep per example
        t = torch.randint(1, diffusion_steps + 1, (x0.size(0),), device=x0.device)
        xt, targets, _ = q_sample_mask(
            x0,
            t,
            mask_token_id=raw_model.mask_token_id,
            diffusion_steps=diffusion_steps,
            schedule=schedule,
        )

        with ctx:
            _, loss = model(xt, t, targets=targets)
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
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
