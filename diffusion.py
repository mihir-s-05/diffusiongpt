"""
Discrete (masked-token) diffusion utilities used by DiffusionGPT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskSchedule:
    """
    mask_ratio(t) in [0,1], where 0 = no corruption and 1 = fully masked.

    Supported schedules:
    - linear: mask_ratio = t/T
    - cosine: mask_ratio = sin^2(pi/2 * t/T)
    - pow: mask_ratio = (t/T)^power
    """

    kind: str = "linear"
    power: float = 1.0

    def mask_ratio(self, t: torch.Tensor, diffusion_steps: int) -> torch.Tensor:
        t = t.to(dtype=torch.float32)
        T = float(diffusion_steps)
        if self.kind == "linear":
            ratio = t / T
        elif self.kind == "cosine":
            ratio = torch.sin(0.5 * math.pi * (t / T)) ** 2
        elif self.kind == "pow":
            ratio = (t / T) ** float(self.power)
        else:
            raise ValueError(f"Unknown schedule kind: {self.kind}")
        return ratio.clamp_(0.0, 1.0)


def q_sample_mask(
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    mask_token_id: int,
    diffusion_steps: int,
    schedule: MaskSchedule,
    generator: torch.Generator | None = None,
    min_masked_tokens: int = 0,
    exact_masked_tokens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample xt ~ q(xt | x0, t) using masked-token corruption.

    Returns:
    - xt: corrupted tokens, shape (B, S)
    - targets: original tokens with ignore_index=-1 for unmasked positions, shape (B, S)
    - mask: boolean mask of corrupted positions, shape (B, S)
    """

    if t.dim() == 0:
        t = t.expand(x0.size(0))
    assert t.shape == (x0.size(0),)

    mask_ratio = schedule.mask_ratio(t, diffusion_steps=diffusion_steps)  # (B,)

    if exact_masked_tokens:
        # Mask exactly k_i ~= mask_ratio(t_i) * seq_len tokens per example.
        bsz, seq_len = x0.shape
        mask = torch.zeros((bsz, seq_len), device=x0.device, dtype=torch.bool)
        for i in range(bsz):
            k = int(round(float(mask_ratio[i].item()) * seq_len))
            k = max(int(min_masked_tokens), k)
            k = min(k, seq_len)
            if k <= 0:
                continue
            perm = torch.randperm(seq_len, device=x0.device, generator=generator)
            mask[i, perm[:k]] = True
    else:
        noise = torch.rand(x0.shape, device=x0.device, generator=generator)
        mask = noise < mask_ratio[:, None]
        if min_masked_tokens > 0:
            # Ensure each sequence has at least `min_masked_tokens` masked positions.
            # This avoids degenerate loss (e.g. all ignore_index) at very small mask ratios.
            masked_counts = mask.sum(dim=1)
            for i in range(x0.size(0)):
                need = int(min_masked_tokens) - int(masked_counts[i].item())
                if need <= 0:
                    continue
                # If everything is already masked, there's nothing to do.
                unmasked = (~mask[i]).nonzero(as_tuple=False).view(-1)
                if unmasked.numel() == 0:
                    continue
                need = min(need, unmasked.numel())
                perm = torch.randperm(unmasked.numel(), device=x0.device, generator=generator)
                chosen = unmasked[perm[:need]]
                mask[i, chosen] = True

    xt = x0.masked_fill(mask, mask_token_id)
    targets = x0.clone()
    targets[~mask] = -1
    return xt, targets, mask


def q_reveal_mask(
    masked_at_t: torch.Tensor,
    t: torch.Tensor,
    *,
    diffusion_steps: int,
    schedule: MaskSchedule,
    generator: torch.Generator | None = None,
    min_revealed_tokens: int = 0,
) -> torch.Tensor:
    """
    Sample a subset of masked positions to be "revealed" on the reverse step t -> (t-1).

    For an absorbing mask process where q(x_t | x0) masks each token with probability r_t,
    and r_t is given by `schedule.mask_ratio(t)`, the conditional probability that a
    token is *newly* masked at step t (i.e. was unmasked at t-1 but masked at t) given
    that it is masked at t is:

        p(reveal | masked_at_t) = (r_t - r_{t-1}) / r_t

    This function returns a boolean mask `reveal` with:
      reveal[i, j] == True  => position (i, j) should be predicted at this step
      reveal is always a subset of `masked_at_t`.
    """

    if t.dim() == 0:
        t = t.expand(masked_at_t.size(0))
    assert t.shape == (masked_at_t.size(0),)

    t = t.to(dtype=torch.long, device=masked_at_t.device)
    t_prev = (t - 1).clamp_min(0)

    r_t = schedule.mask_ratio(t, diffusion_steps=diffusion_steps)  # (B,)
    r_prev = schedule.mask_ratio(t_prev, diffusion_steps=diffusion_steps)  # (B,)

    r_t = r_t.clamp_min(1.0 / float(diffusion_steps))
    reveal_p = ((r_t - r_prev).clamp_min(0.0) / r_t).clamp_(0.0, 1.0)  # (B,)

    noise = torch.rand(masked_at_t.shape, device=masked_at_t.device, generator=generator)
    reveal = masked_at_t & (noise < reveal_p[:, None])

    if min_revealed_tokens > 0:
        revealed_counts = reveal.sum(dim=1)
        for i in range(masked_at_t.size(0)):
            need = int(min_revealed_tokens) - int(revealed_counts[i].item())
            if need <= 0:
                continue
            candidates = (masked_at_t[i] & (~reveal[i])).nonzero(as_tuple=False).view(-1)
            if candidates.numel() == 0:
                continue
            need = min(need, candidates.numel())
            perm = torch.randperm(candidates.numel(), device=masked_at_t.device, generator=generator)
            chosen = candidates[perm[:need]]
            reveal[i, chosen] = True

    return reveal
