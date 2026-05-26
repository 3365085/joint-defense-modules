"""CCSync — Channel Co-firing Sync for backdoor weight purification.

CCSync is a closed-form, training-free purification mechanism that
identifies the backdoor channel cohort by **co-firing synchronisation
under the trigger pool** and selectively rolls back the fine-tune
drift Δ = W_poisoned - W_clean only on those channels.

It is the operational instantiation of CTM's "synchronisation as
latent representation" abstraction, applied to weight-level OD
backdoor cleanup.  Unlike the dynamical CTM-style attempts (JES /
NeuroChrono / CSC v1-v3), CCSync abandons RNN evolution: the
synchronisation signal is computed DIRECTLY from the channel
covariance matrix on the trigger pool, with the clean-pool covariance
subtracted as baseline.  The resulting per-channel sync score σ is
used as a continuous channel-wise interpolation coefficient between
W_poisoned and W_clean.

Key formulas:

  ρ_trig  = Pearson correlation of channel activations on triggered samples
  ρ_clean = Pearson correlation of channel activations on clean target-absent samples
  S       = ρ_trig - ρ_clean
  σ_c     = Σ_{j != c} max(0, S_{c,j}) * 𝟙[S_{c,j} > τ]
  α_c     = α_max * softmax(σ / β)_c * C
  W'_c    = W_p_c - α_c * (W_p_c - W_clean_c)

There is exactly ONE primary mathematical object (the sync score σ),
ONE primary loss (none — closed-form), ONE deployment artefact (a
purified .pt file).  No new module is added to the detector at
inference.

This subpackage does not import cels_od, ccs, jes, oc3_*, autodetox,
hybrid_purify, weight_soup.  It depends only on torch.
"""
from .schema import CCSyncConfig, CCSyncReport
from .sync_score import (
    correlation_matrix,
    sync_score,
    cohort_topk,
    compute_excess_correlation,
)
from .purify import purify_weights, channel_alpha_from_sync

__all__ = [
    "CCSyncConfig",
    "CCSyncReport",
    "correlation_matrix",
    "compute_excess_correlation",
    "sync_score",
    "cohort_topk",
    "channel_alpha_from_sync",
    "purify_weights",
]
