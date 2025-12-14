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
    noise = torch.rand(x0.shape, device=x0.device, generator=generator)
    mask = noise < mask_ratio[:, None]

    xt = x0.masked_fill(mask, mask_token_id)
    targets = x0.clone()
    targets[~mask] = -1
    return xt, targets, mask

