"""
Sample from a trained DiffusionGPT checkpoint using MaskGIT-style iterative decoding.
"""

import os
import sys
import math
import time
import pickle
from contextlib import nullcontext

import torch

from model import DiffusionGPTConfig, DiffusionGPT
from diffusion import MaskSchedule


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_dataset_dir(dataset_name: str, data_dir_override: str) -> str:
    """Resolve the directory containing meta.pkl for encoding/decoding."""

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
init_from = "resume"
out_dir = "out-diffusion"
data_dir = ""

start = "\n"
num_samples = 5
max_new_tokens = 256

temperature = 1.0
top_k = 200
sample = True  # if False, uses argmax (deterministic)

animate = False
animate_delay = 0.05
animate_step = 1
mask_char = "_"

mask_schedule = "auto"  # auto|linear|cosine|pow
mask_schedule_power = 1.0
diffusion_steps = 0

seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else ("float16" if torch.cuda.is_available() else "float32")
)
compile = False

exec(open(os.path.join(THIS_DIR, "configurator.py"), encoding="utf-8").read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if "cuda" in device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "device is set to CUDA, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build (or pass --device=cpu)."
        )
    torch.cuda.manual_seed(seed)
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
# load model
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model_args = checkpoint["model_args"]
gptconf = DiffusionGPTConfig(**model_args)
model = DiffusionGPT(gptconf)

state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

ckpt_cfg = checkpoint.get("config", {})
if mask_schedule == "auto":
    mask_schedule = ckpt_cfg.get("mask_schedule", "linear")
    mask_schedule_power = float(ckpt_cfg.get("mask_schedule_power", 1.0))

T = diffusion_steps if diffusion_steps > 0 else model.config.diffusion_steps
schedule = MaskSchedule(kind=mask_schedule, power=mask_schedule_power)

# -----------------------------------------------------------------------------
# encoding/decoding (char-level meta.pkl if available; otherwise GPT-2 BPE)
encode = None
decode = None

dataset = None
if "config" in checkpoint and "dataset" in checkpoint["config"]:
    dataset = checkpoint["config"]["dataset"]

dataset_dir = None
if data_dir:
    dataset_dir = os.path.abspath(data_dir)
elif "config" in checkpoint and "dataset_dir" in checkpoint["config"]:
    dataset_dir = checkpoint["config"]["dataset_dir"]
elif "config" in checkpoint and "data_dir" in checkpoint["config"] and checkpoint["config"]["data_dir"]:
    dataset_dir = _resolve_dataset_dir(dataset or "", checkpoint["config"]["data_dir"])
elif dataset is not None:
    dataset_dir = _resolve_dataset_dir(dataset, "")

meta_path = os.path.join(dataset_dir, "meta.pkl") if dataset_dir else None
if meta_path and os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    if model.config.vocab_size < 50000:
        raise FileNotFoundError(
            "meta.pkl was not found, but this checkpoint's vocab_size looks non-GPT2.\n"
            "If this is a character-level dataset, run `python diffusionGPT/data/<dataset>/prepare.py` "
            "and ensure meta.pkl is alongside train.bin/val.bin, or pass --data_dir=..."
        )
    try:
        import tiktoken
    except ImportError:
        raise SystemExit(
            "tiktoken is required to sample GPT-2 BPE checkpoints when meta.pkl is not available.\n"
            "Install with: pip install tiktoken"
        ) from None
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
prompt_ids = encode(start)


def sample_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-position categorical sampling from logits.
    Returns (tokens, confidence) with shape (B, S).
    """

    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), logits.size(-1))
        v, ix = torch.topk(logits, k=k, dim=-1)
        probs = torch.softmax(v, dim=-1)
        flat_probs = probs.view(-1, k)
        sampled = torch.multinomial(flat_probs, num_samples=1).view(logits.size(0), logits.size(1))
        tokens = ix.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        logZ = torch.logsumexp(logits, dim=-1)
        chosen = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        confidence = torch.exp(chosen - logZ)
        return tokens, confidence

    probs = torch.softmax(logits, dim=-1)
    flat_probs = probs.view(-1, probs.size(-1))
    tokens = torch.multinomial(flat_probs, num_samples=1).view(logits.size(0), logits.size(1))
    logZ = torch.logsumexp(logits, dim=-1)
    chosen = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    confidence = torch.exp(chosen - logZ)
    return tokens, confidence


def decode_with_mask(ids: list[int], mask_token_id: int, decoder, mask_display: str = "_") -> str:
    """Decode token ids, replacing mask tokens with a visible character."""
    result = []
    for i in ids:
        if i == mask_token_id:
            result.append(mask_display)
        else:
            result.append(decoder([i]))
    return "".join(result)


def clear_line():
    """Clear the current line in terminal."""
    sys.stdout.write("\033[2K\033[G")
    sys.stdout.flush()


def move_cursor_up(n: int):
    """Move cursor up n lines."""
    if n > 0:
        sys.stdout.write(f"\033[{n}A")
        sys.stdout.flush()


@torch.no_grad()
def generate_one(prompt: list[int], decoder=None, show_animation: bool = False) -> list[int]:
    prompt = prompt[-model.config.block_size :]
    prompt_len = len(prompt)
    seq_len = min(model.config.block_size, prompt_len + int(max_new_tokens))

    x = torch.full((1, seq_len), model.mask_token_id, dtype=torch.long, device=device)
    if prompt_len:
        x[0, :prompt_len] = torch.tensor(prompt, dtype=torch.long, device=device)

    if seq_len == prompt_len:
        return x[0].tolist()

    prompt_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    prompt_mask[:, :prompt_len] = True

    prev_lines = 0

    for t in range(T, 0, -1):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        with ctx:
            logits, _ = model(x, t_tensor)

        masked = (x == model.mask_token_id) & (~prompt_mask)
        if sample:
            sampled_tokens, _ = sample_from_logits(logits)
        else:
            scaled_logits = logits / temperature if temperature != 1.0 else logits
            sampled_tokens = torch.argmax(scaled_logits, dim=-1)

        tokens = torch.where(masked, sampled_tokens, x)

        if prompt_len:
            tokens[0, :prompt_len] = torch.tensor(prompt, dtype=torch.long, device=device)

        scaled_logits = logits / temperature if temperature != 1.0 else logits
        logZ = torch.logsumexp(scaled_logits, dim=-1)
        chosen = scaled_logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        confidence = torch.exp(chosen - logZ)
        confidence[prompt_mask] = float("inf")

        n_gen = seq_len - prompt_len
        if n_gen <= 0:
            x = tokens
            continue

        if t == 1:
            x = tokens
            if show_animation and decoder is not None:
                n_masked = 0
                mask_ratio_pct = 0.0
                display_text = decode_with_mask(x[0].tolist(), model.mask_token_id, decoder, mask_char)
                move_cursor_up(prev_lines)
                lines = display_text.split('\n')
                header = f"[t={t:3d}/{T}] {mask_ratio_pct:5.1f}% masked"
                print(header)
                for line in lines:
                    clear_line()
                    print(line)
                prev_lines = len(lines) + 1
                time.sleep(animate_delay)
            continue

        next_ratio = float(schedule.mask_ratio(torch.tensor(t - 1), diffusion_steps=T).item())
        n_mask_next = int(math.floor(next_ratio * n_gen))
        if n_mask_next <= 0:
            x = tokens
            continue

        scores = confidence.clone()
        scores[prompt_mask] = float("inf")
        mask_idx = torch.topk(-scores, k=n_mask_next, dim=1).indices

        x = tokens.clone()
        x.scatter_(1, mask_idx, model.mask_token_id)
        if prompt_len:
            x[0, :prompt_len] = torch.tensor(prompt, dtype=torch.long, device=device)

        if show_animation and decoder is not None and t % animate_step == 0:
            n_masked = (x == model.mask_token_id).sum().item()
            mask_ratio_pct = 100.0 * n_masked / seq_len
            display_text = decode_with_mask(x[0].tolist(), model.mask_token_id, decoder, mask_char)

            move_cursor_up(prev_lines)

            lines = display_text.split('\n')
            header = f"[t={t:3d}/{T}] {mask_ratio_pct:5.1f}% masked"
            print(header)
            for line in lines:
                clear_line()
                print(line)
            prev_lines = len(lines) + 1

            time.sleep(animate_delay)

    return x[0].tolist()


with torch.no_grad():
    for i in range(num_samples):
        if animate:
            print(f"\n=== Sample {i+1}/{num_samples} ===")
            print()
        ids = generate_one(prompt_ids, decoder=decode, show_animation=animate)
        if not animate:
            print(decode(ids))
        print("---------------")
