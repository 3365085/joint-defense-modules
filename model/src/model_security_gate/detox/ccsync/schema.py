"""Schemas for CCSync."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CCSyncConfig:
    """Hyperparameters for CCSync.

    There are only four meaningful knobs:

    * ``tau``: minimum excess correlation S_{c,j} > τ to count as a
      cohort partner.  Defaults to 0.10 (excess Pearson R of 0.10
      between channels c and j on the trigger pool vs the clean
      pool).  Insensitive across [0.05, 0.20].

    * ``alpha_max``: maximum interpolation coefficient toward W_clean
      for any single channel.  α_c is bounded above by ``alpha_max``
      after softmax normalisation.  Defaults to 1.0 (a fully-synced
      backdoor channel can be FULLY rolled back to W_clean; a non-
      synced channel barely changes).

    * ``beta_softmax``: softmax temperature on σ.  Smaller β =>
      sharper concentration of α on the highest-σ channels.  Defaults
      to 0.5.

    * ``standardise_pools``: whether to z-score each channel before
      computing correlation.  True (default) is the canonical Pearson
      definition; False uses raw covariance (faster, a tad less
      robust to scale differences).
    """

    tau: float = 0.10
    alpha_max: float = 1.0
    beta_softmax: float = 0.5
    standardise_pools: bool = True


@dataclass
class CCSyncReport:
    """Diagnostic report after computing the sync purification.

    Used both by the synthetic demo and by the YOLO trainer.
    """

    sigma: List[float]                         # per-channel sync degree
    alpha: List[float]                         # per-channel rollback coefficient
    cohort_indices: List[int]                  # high-σ channel indices
    n_cohort: int
    sigma_max: float
    sigma_median: float
    sigma_p95_other: float                     # sigma p95 over non-cohort channels
    pathway_specificity: float                 # sigma_min(cohort) / sigma_p95_other
    notes: List[str] = field(default_factory=list)
