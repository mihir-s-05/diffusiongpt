# diffusionGPT

`diffusionGPT/` is a diffusion-style, non-autoregressive text model built in the spirit of `nanoGPT/` (but self-contained).

Instead of next-token prediction, `DiffusionGPT` is trained as a denoiser: given a noisy sequence `x_t` (created by masking tokens) and a timestep `t`, it predicts the original tokens `x_0`. Sampling uses **MaskGIT-style iterative refinement**: start from all masks and repeatedly predict tokens, re-masking the least confident positions until no masks remain.

This is a discrete, masked-token diffusion variant (not a continuous Gaussian diffusion in embedding space).

## Quick start (Shakespeare char)

1) Prepare data:

```bash
python diffusionGPT/data/shakespeare_char/prepare.py
```

2) Train:

```bash
python diffusionGPT/train.py diffusionGPT/config/train_shakespeare_char.py
```

3) Sample:

```bash
python diffusionGPT/sample.py --out_dir=out-diffusion-shakespeare-char --start="\n" --max_new_tokens=256
```

## Notes

- The model uses **bidirectional attention** (not causal). It’s meant for parallel denoising / refinement, not left-to-right likelihood evaluation.
- Prompts are supported by fixing the prompt tokens and only generating the remaining positions (similar to inpainting).
- Data loading expects `train.bin` / `val.bin` under `diffusionGPT/data/<dataset>/` by default (override with `--data_dir=...`).
- `diffusionGPT/train.py` and `diffusionGPT/sample.py` default to `cuda` when `torch.cuda.is_available()` is true.

## OpenWebText (GPT-2 BPE)

After running `python diffusionGPT/data/openwebtext/prepare.py` (requires `pip install datasets tqdm`), you can start a GPT-2-sized diffusion run with:

```bash
torchrun --standalone --nproc_per_node=8 diffusionGPT/train.py diffusionGPT/config/train_openwebtext_gpt2_124m.py
```
