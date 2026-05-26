"""Weight purification operator.

Applies a per-channel soft interpolation between W_poisoned and W_clean,
using the sync-derived α_c as the interpolation coefficient.

The operator is mathematically:

    W'_c = W_p_c - α_c * (W_p_c - W_clean_c)
         = (1 - α_c) * W_p_c + α_c * W_clean_c

so α_c = 1 means "fully roll back this channel to W_clean", α_c = 0
means "keep W_poisoned unchanged".  Per-channel α_c is derived from
the sync score σ_c via a softmax (concentrating mass on high-σ
channels) clamped by α_max.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor


def channel_alpha_from_sync(
    sigma: Tensor,
    *,
    alpha_max: float = 1.0,
    beta_softmax: float = 0.5,
    floor: float = 0.0,
) -> Tensor:
    """Convert sync degree σ to per-channel rollback coefficient α.

    σ → α uses a softmax-normalised concentration:
       p_c = softmax(σ_c / β)
       α_c = α_max * p_c * C       (so α sums to α_max * C, mean to α_max)
       α_c <- max(α_c, floor)

    The intuition:
      * If σ is uniform, softmax is uniform => α_c ≡ α_max for every
        channel, recovering Backbone-Soup at α=α_max.
      * If σ is concentrated on K cohort channels, softmax puts mass
        on those K, so α is heterogeneous: cohort channels get α
        much larger than α_max, non-cohort channels get α near 0.

    Args:
        sigma : (C,) non-negative sync degree
        alpha_max: scale parameter, the "average rollback strength"
        beta_softmax: softmax temperature; smaller => sharper
        floor: per-channel minimum α (set to a small positive number
               like 0.05 if you want ALL channels to drift slightly
               toward W_clean as a safety net)

    Returns:
        α : (C,) per-channel rollback coefficient, each in [floor, alpha_max * C / 1]
            (typically in [0, alpha_max * scale])
    """
    if sigma.dim() != 1:
        raise ValueError(f"sigma must be 1-D; got {sigma.shape}")
    C = sigma.shape[0]
    # Softmax over channels
    logits = sigma / max(beta_softmax, 1e-6)
    p = torch.softmax(logits, dim=0)
    # Scale so the mean is alpha_max
    alpha = alpha_max * p * C
    # Clamp to [floor, +inf)
    alpha = alpha.clamp(min=float(floor))
    return alpha


def purify_weights(
    W_poisoned: Tensor,
    W_clean: Tensor,
    alpha: Tensor,
    *,
    channel_axis: int = 0,
) -> Tensor:
    """Apply per-channel soft interpolation: W' = (1 - α) W_p + α W_c.

    Args:
        W_poisoned : (..., C, ...) poisoned tensor
        W_clean    : same shape as W_poisoned
        alpha      : (C,) per-channel rollback coefficient in [0, +∞)
                     (clamped at 1.0 internally for hard interpolation)
        channel_axis: which axis of W_poisoned is the channel dimension

    Returns:
        W_purified : same shape as W_poisoned
    """
    if W_poisoned.shape != W_clean.shape:
        raise ValueError(f"shape mismatch: {W_poisoned.shape} vs {W_clean.shape}")
    C_in_W = W_poisoned.shape[channel_axis]
    if alpha.shape != (C_in_W,):
        raise ValueError(f"alpha must be ({C_in_W},); got {alpha.shape}")
    # Clamp α to [0, 1] for the convex combination
    alpha_c = alpha.clamp(0.0, 1.0)
    # Reshape α for broadcasting: put channel axis at the right spot
    shape = [1] * W_poisoned.dim()
    shape[channel_axis] = C_in_W
    alpha_b = alpha_c.view(shape)
    return (1.0 - alpha_b) * W_poisoned + alpha_b * W_clean
