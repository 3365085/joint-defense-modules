"""Sync score computation for CCSync.

The mathematical core. No learning, all closed-form.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


def correlation_matrix(X: Tensor, *, standardise: bool = True) -> Tensor:
    """Pearson correlation matrix of columns.

    X : (N, C) — N samples, C channels.
    standardise: if True, z-score each column before the inner
                 product (canonical Pearson R).  If False, return
                 raw covariance.

    Returns : (C, C) symmetric matrix with 1.0 on the diagonal when
              standardise=True.
    """
    if X.dim() != 2:
        raise ValueError(f"X must be (N, C); got {X.shape}")
    Xc = X - X.mean(dim=0, keepdim=True)
    if standardise:
        std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        Xn = Xc / std
    else:
        Xn = Xc
    N = Xc.shape[0]
    # Use the same divisor as the std computation (N for unbiased=False)
    # so that diag(R) = 1.0 exactly when standardise=True.
    if standardise:
        return (Xn.T @ Xn) / max(1, N)
    return (Xn.T @ Xn) / max(1, N - 1)


def compute_excess_correlation(
    X_trig: Tensor,
    X_clean: Tensor,
    *,
    standardise: bool = True,
) -> Tensor:
    """Excess co-firing correlation: S = ρ_trig - ρ_clean.

    A backdoor channel cohort gains synchronised firing when the
    trigger pool is presented (S > 0 inside the cohort); clean
    channels' inter-correlations are unchanged or even slightly
    weaker (S near zero or negative).

    Args:
        X_trig : (N_trig, C) trigger pool channel activations
        X_clean: (N_clean, C) clean target-absent pool activations

    Returns:
        S : (C, C) excess-correlation matrix
    """
    rho_t = correlation_matrix(X_trig, standardise=standardise)
    rho_c = correlation_matrix(X_clean, standardise=standardise)
    return rho_t - rho_c


def sync_score(S: Tensor, *, tau: float = 0.10) -> Tensor:
    """Per-channel sync degree.

    σ_c = Σ_{j != c} max(0, S_{c,j}) * 𝟙[S_{c,j} > τ]

    Sums positive excess correlations above threshold τ.  A backdoor
    channel collects a strong signal because it acquires several
    high-S cohort partners; a clean channel barely accumulates
    anything.

    Args:
        S  : (C, C) excess correlation matrix
        tau: threshold below which excess correlations are ignored

    Returns:
        σ : (C,) per-channel sync degree (non-negative)
    """
    if S.dim() != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square; got {S.shape}")
    C = S.shape[0]
    # mask diagonal
    Smasked = S.clone()
    Smasked.fill_diagonal_(0.0)
    # threshold
    above = (Smasked > float(tau))
    return torch.where(above, Smasked, torch.zeros_like(Smasked)).sum(dim=1)


def cohort_topk(sigma: Tensor, k: int) -> List[int]:
    """Return indices of the top-k channels by sync degree."""
    k = min(k, sigma.shape[0])
    _, idx = sigma.topk(k)
    return [int(i) for i in idx.tolist()]
