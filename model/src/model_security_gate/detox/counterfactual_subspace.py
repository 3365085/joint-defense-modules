from __future__ import annotations

"""Counterfactual feature primitives for multi-family backdoor detox.

These functions are small, testable building blocks for the original OTSN
(Orthogonal Trigger Subspace Neutralization) algorithm.  They can be called from
future YOLO hook jobs that export features, or directly inside a PyTorch training
loop when features are available.

Notation:
    clean_features:     features of the same image without trigger
    triggered_features: features of the image with trigger
    delta = triggered - clean

The main idea is to estimate the low-rank trigger-causal subspace from paired
deltas, then penalize target evidence that aligns with this subspace on
target-absent samples while preserving object-present clean evidence.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np


@dataclass(frozen=True)
class TriggerSubspace:
    basis: np.ndarray
    singular_values: np.ndarray
    explained_variance: float
    n_pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis.tolist(),
            "singular_values": self.singular_values.tolist(),
            "explained_variance": float(self.explained_variance),
            "n_pairs": int(self.n_pairs),
        }


def _flatten_features(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


def estimate_trigger_subspace(
    clean_features: np.ndarray,
    triggered_features: np.ndarray,
    *,
    rank: int = 4,
    center: bool = True,
) -> TriggerSubspace:
    """Estimate trigger-causal basis from paired feature deltas via SVD."""

    clean = _flatten_features(clean_features)
    trig = _flatten_features(triggered_features)
    if clean.shape != trig.shape:
        raise ValueError(f"feature shape mismatch: clean={clean.shape} triggered={trig.shape}")
    if clean.shape[0] == 0:
        raise ValueError("at least one paired feature is required")
    delta = trig - clean
    if center:
        delta = delta - delta.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(delta, full_matrices=False)
    k = max(1, min(int(rank), vh.shape[0]))
    denom = float(np.sum(s * s))
    explained = float(np.sum(s[:k] * s[:k]) / denom) if denom > 0 else 0.0
    return TriggerSubspace(basis=vh[:k].astype(np.float32), singular_values=s[:k].astype(np.float32), explained_variance=explained, n_pairs=int(clean.shape[0]))


def project_out_subspace(features: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the component of features that lies in ``basis``.

    ``basis`` is expected to be row-orthonormal as returned by SVD.  The function
    accepts any leading shape and flattens feature dimensions from axis 1 onward.
    """

    arr = _flatten_features(features)
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != arr.shape[1]:
        raise ValueError(f"basis shape {b.shape} is incompatible with features {arr.shape}")
    coeff = arr @ b.T
    recon = coeff @ b
    return (arr - recon).reshape(np.asarray(features).shape)


def subspace_alignment_score(features: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return per-sample squared fraction of energy explained by the trigger basis."""

    arr = _flatten_features(features)
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != arr.shape[1]:
        raise ValueError(f"basis shape {b.shape} is incompatible with features {arr.shape}")
    proj = (arr @ b.T) @ b
    denom = np.sum(arr * arr, axis=1) + 1e-12
    return np.sum(proj * proj, axis=1) / denom


def threshold_excess(values: np.ndarray, cap: float) -> np.ndarray:
    """Elementwise ``max(values - cap, 0)`` used for threshold-aware guards."""

    return np.maximum(np.asarray(values, dtype=np.float64) - float(cap), 0.0)


def target_absent_threshold_penalty(target_scores: np.ndarray, cap: float = 0.25, power: float = 2.0) -> float:
    """Penalty for target-absent false-positive scores above a deployment cap."""

    excess = threshold_excess(np.asarray(target_scores, dtype=np.float64), cap)
    if excess.size == 0:
        return 0.0
    return float(np.mean(np.power(excess, float(power))))


def preserve_floor_penalty(current_scores: np.ndarray, reference_scores: np.ndarray, margin: float = 0.0) -> float:
    """Penalty when current target-present evidence falls below a reference floor."""

    cur = np.asarray(current_scores, dtype=np.float64)
    ref = np.asarray(reference_scores, dtype=np.float64)
    if cur.shape != ref.shape:
        raise ValueError(f"score shape mismatch: current={cur.shape} reference={ref.shape}")
    gap = np.maximum((ref - float(margin)) - cur, 0.0)
    return float(np.mean(gap * gap)) if gap.size else 0.0


def otsn_penalty(
    features: np.ndarray,
    target_scores: np.ndarray,
    basis: np.ndarray,
    *,
    score_cap: float = 0.25,
    alignment_cap: float = 0.05,
) -> dict[str, float]:
    """Compute OTSN target-absent penalty terms.

    This is designed for target-absent samples.  It penalizes target scores above
    ``score_cap`` and feature alignment above ``alignment_cap``.  Training loops
    can combine the returned losses with normal task and distillation losses.
    """

    score_loss = target_absent_threshold_penalty(target_scores, score_cap)
    align = subspace_alignment_score(features, basis)
    align_loss = target_absent_threshold_penalty(align, alignment_cap)
    return {
        "target_score_loss": float(score_loss),
        "subspace_alignment_loss": float(align_loss),
        "mean_alignment": float(np.mean(align)) if align.size else 0.0,
        "max_alignment": float(np.max(align)) if align.size else 0.0,
    }


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def teacher_kl_np(student_logits: np.ndarray, teacher_logits: np.ndarray, temperature: float = 1.0) -> float:
    """Small numpy KL for smoke tests and hook-exported logits.

    For in-graph training, use the analogous torch implementation in the trainer.
    """

    t = max(float(temperature), 1e-6)
    ps = softmax_np(np.asarray(student_logits) / t)
    pt = softmax_np(np.asarray(teacher_logits) / t)
    kl = np.sum(pt * (np.log(pt + 1e-12) - np.log(ps + 1e-12)), axis=-1)
    return float(np.mean(kl) * t * t)


def diversity_score_for_variants(metadata_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score hard-suite diversity from variant metadata.

    The score is a simple average of normalized unique counts across common
    axes.  It is meant as an audit signal, not a statistical proof.
    """

    axes = ["trigger_x", "trigger_y", "trigger_scale", "brightness", "jpeg_quality", "variant"]
    n = len(metadata_rows)
    if n <= 0:
        return {"n": 0, "score": 0.0, "unique": {}}
    unique: dict[str, int] = {}
    parts: list[float] = []
    for axis in axes:
        vals = [str(row.get(axis, "")) for row in metadata_rows if row.get(axis, "") != ""]
        if not vals:
            continue
        u = len(set(vals))
        unique[axis] = u
        parts.append(min(1.0, u / max(2.0, math.sqrt(n))))
    score = float(sum(parts) / len(parts)) if parts else 0.0
    return {"n": n, "score": score, "unique": unique}
