# diffusionGPT

`diffusionGPT/` is a diffusion-style, non-autoregressive text model built in the spirit of `nanoGPT/` (but self-contained).

Instead of next-token prediction, `DiffusionGPT` is trained as a denoiser: given a noisy sequence `x_t` (created by masking tokens) and a timestep `t`, it predicts the original tokens `x_0`. Sampling uses **MaskGIT-style iterative refinement**: start from all masks and repeatedly predict tokens, re-masking the least confident positions until no masks remain.

This is a discrete, masked-token diffusion variant (not a continuous Gaussian diffusion in embedding space).

## Quick start (Shakespeare char)

1) Prepare data:

```bash
python data/shakespeare_char/prepare.py
```

2) Train:

```bash
python train.py config/train_shakespeare_char.py
```

3) Sample:

```bash
python sample.py --out_dir=out-diffusion-shakespeare-char --start="\n" --max_new_tokens=256
```

## Command-line syntax

All scripts use `--key=value` syntax for arguments, **including booleans**:

```bash
# Correct:
python sample.py --animate=True --temperature=0.8

# Wrong (will error):
python sample.py --animate --temperature 0.8
```

## Sampling options

| Argument | Default | Description |
|----------|---------|-------------|
| `--out_dir` | `out-diffusion` | Checkpoint directory |
| `--start` | `"\n"` | Prompt text (or `FILE:path.txt`) |
| `--max_new_tokens` | `256` | Tokens to generate after prompt |
| `--num_samples` | `5` | Number of samples to generate |
| `--temperature` | `1.0` | Sampling temperature (lower = more deterministic) |
| `--top_k` | `200` | Top-k sampling (0 = disabled) |
| `--sample` | `True` | Use sampling; `False` for argmax |
| `--diffusion_steps` | `0` | Override sampling steps (0 = use checkpoint value) |

### Animation

Watch the diffusion denoising process in real-time:

```bash
python sample.py --out_dir=out-diffusion-shakespeare-char --start="To be" --animate=True --num_samples=1
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--animate` | `False` | Show denoising step-by-step |
| `--animate_delay` | `0.05` | Seconds between frames |
| `--animate_step` | `1` | Show every N steps (higher = faster) |
| `--mask_char` | `_` | Character displayed for masked tokens |

Example with faster animation:

```bash
python sample.py --out_dir=out-diffusion-shakespeare-char --start="ROMEO:" --max_new_tokens=128 --animate=True --animate_step=5 --animate_delay=0.1 --num_samples=1
```

## Notes

- The model uses **bidirectional attention** (not causal). It's meant for parallel denoising / refinement, not left-to-right likelihood evaluation.
- Prompts are supported by fixing the prompt tokens and only generating the remaining positions (similar to inpainting).
- Data loading expects `train.bin` / `val.bin` under `data/<dataset>/` by default (override with `--data_dir=...`).
- `train.py` and `sample.py` default to `cuda` when `torch.cuda.is_available()` is true.

## OpenWebText (GPT-2 BPE)

After running `python data/openwebtext/prepare.py` (requires `pip install datasets tqdm`), you can start a GPT-2-sized diffusion run with:

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_openwebtext_gpt2_124m.py
```
