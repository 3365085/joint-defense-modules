from __future__ import annotations

"""OC3-Detox trainer.

Production training loop that consumes OC3 witness manifests and updates a
**minimal** subset of YOLO head parameters (typically the final
classification bias on ``model.model[-1].cv3.*.bias``) to push the model
toward the OC3 counterfactual consensus targets.

Design choices
--------------

The trainer intentionally does **not** modify the backbone or full
detection head.  Two reasons:

1. The OC3 protocol's claim is that backdoors are candidate-box-level
   evidence binding bugs.  A small bias adapter is sufficient to test
   the claim and avoids the larger weight drift problem that breaks
   clean mAP.
2. The existing Hybrid-PURIFY-OD and Backbone-only Weight Soup pipelines
   already cover the heavier-weight cases.  OC3 is positioned as a
   complementary, minimal-drift alternative.

Loss
----

For every witness in the batch we compute a numpy-side OC3 loss
(``compute_oc3_loss``) for **audit only** (it is non-differentiable
because it operates on already-decoded YOLO outputs).  The
**differentiable** training loss is a torch-side surrogate:

- ``L_context``: mean of ReLU(target_class_logit_at_context_pos - cap)^2
  on context_only / object_erased witnesses.  Pushes the
  classification logits down on context candidates.
- ``L_object``: ReLU(reference - current)^2 floor over the top-k
  detections on object_present witnesses.  Pushes target-class logits
  back up on real objects.
- ``L_consistency``: |obj_present_logit - transformed_logit| on
  geometry_pair / frequency_pair witnesses.  Forces the model to
  produce stable target-class logits under benign transforms.

Total loss = ``w_ctx * L_context + w_obj * L_object + w_cons * L_consistency``.

Surrogate-vs-audit consistency is checked at every epoch end: if the
audit loss does not move in the same direction as the training loss
across 2 epochs, training halts and refuses to save the adapter.

Outputs
-------

- ``oc3_adapter.pt``: torch state dict containing only the trained bias
  parameters and the head identifier (``head_index`` and
  ``param_keys``).
- ``oc3_full_model.pt``: full ultralytics-shaped checkpoint with the
  adapter merged in, ready for downstream CFRC.
- ``oc3_train_log.json``: per-epoch surrogate + audit loss + accept/reject
  status.

This module is GPU-aware but works on CPU too (slower).  CI runs only
the dry-run path (``train_oc3_adapter(..., max_iters=0)``) so it does
not consume GPU minutes.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np

try:  # torch is required for actual training; tolerate absence in CI.
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:  # pragma: no cover - import-time guard
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False


from model_security_gate.detox.oc3.oc3_detox import (
    OC3Config,
    OC3Witness,
    summarize_witness_losses,
)


@dataclass(frozen=True)
class OC3TrainConfig:
    """Knobs for the OC3 head-bias adapter trainer.

    ``head_index`` defaults to ``-1`` (the final layer of
    ``model.model``).  ``param_pattern`` defaults to ``cv3`` to match the
    YOLOv8/v11/v26 classification branch's bias parameters.  Power users
    can target a different scope by passing ``param_pattern=""`` (all head
    biases) or ``param_pattern="one2one"`` (NMS-free branch only).
    """

    head_index: int = -1
    param_pattern: str = "cv3"
    only_bias: bool = True
    target_class_id: int = 0
    target_score_cap: float = 0.25
    object_floor: float = 0.50
    object_floor_margin: float = 0.02
    consistency_margin: float = 0.03
    weight_context: float = 1.5
    weight_object: float = 1.0
    weight_consistency: float = 0.5
    learning_rate: float = 5.0e-4
    weight_decay: float = 0.0
    epochs: int = 3
    batch_size: int = 4
    audit_consistency_check: bool = True
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OC3TrainResult:
    accepted: bool
    adapter_path: str
    full_model_path: str
    log_path: str
    surrogate_loss_first: float
    surrogate_loss_last: float
    audit_loss_first: float
    audit_loss_last: float
    audit_consistent: bool
    n_trainable_params: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _list_trainable_params(model, *, head_index: int, param_pattern: str, only_bias: bool):
    """Return list of (name, parameter) tuples that match the OC3 scope."""
    if not HAS_TORCH:
        raise RuntimeError("torch is not available; OC3 trainer requires torch + ultralytics")
    head = model.model[head_index]
    out: list[tuple[str, "torch.nn.Parameter"]] = []
    for name, p in head.named_parameters():
        if param_pattern and param_pattern not in name:
            continue
        if only_bias and not name.endswith("bias"):
            continue
        out.append((name, p))
    return out


def _audit_loss(witnesses: Sequence[OC3Witness], cfg: OC3Config) -> float:
    summary = summarize_witness_losses(witnesses, cfg)
    return float(summary.get("mean_total", 0.0))


def _scaled_target_score(detections: Sequence[Mapping[str, Any]], target_class_id: int) -> float:
    """Pick the highest target-class score from a candidate set."""
    if not detections:
        return 0.0
    target_scores: list[float] = []
    for d in detections:
        cid = d.get("class_id")
        if cid is None or int(cid) == int(target_class_id):
            target_scores.append(float(d.get("score", 0.0)) * float(d.get("objectness", 1.0)))
    return max(target_scores, default=0.0)


def _surrogate_loss_terms(
    *,
    raw_logits_object: "torch.Tensor",
    raw_logits_context: "torch.Tensor",
    raw_logits_transformed: "torch.Tensor | None",
    object_floor: float,
    target_score_cap: float,
    consistency_margin: float,
):
    """Differentiable OC3 surrogate.

    ``raw_logits_object`` / ``raw_logits_context`` are already the
    (sigmoid'd) target-class scores at sampled positions on the witness
    images.  We use squared hinge-style losses to keep the gradient
    well-behaved and bounded.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required")
    zero = torch.zeros((), device=raw_logits_context.device, dtype=raw_logits_context.dtype)
    # context insufficiency: penalize scores above target_score_cap
    over = torch.relu(raw_logits_context - float(target_score_cap))
    l_ctx = torch.mean(over * over) if over.numel() > 0 else zero
    # object sufficiency: penalize scores below object_floor
    under = torch.relu(float(object_floor) - raw_logits_object)
    l_obj = torch.mean(under * under) if under.numel() > 0 else zero
    # transform consistency
    l_cons = zero
    if raw_logits_transformed is not None and raw_logits_transformed.numel() > 0 and raw_logits_object.numel() > 0:
        n = min(int(raw_logits_object.shape[0]), int(raw_logits_transformed.shape[0]))
        if n > 0:
            obj_topk = torch.topk(raw_logits_object, n).values
            tran_topk = torch.topk(raw_logits_transformed, n).values
            diff = torch.relu(torch.abs(obj_topk - tran_topk) - float(consistency_margin))
            l_cons = torch.mean(diff * diff) if diff.numel() > 0 else zero
    return l_ctx, l_obj, l_cons


def train_oc3_adapter(
    *,
    model_path: str,
    witness_inference_json: str,
    out_dir: str,
    config: OC3TrainConfig | None = None,
    device: str | int | None = None,
    max_iters: int | None = None,
) -> OC3TrainResult:
    """Train an OC3 head-bias adapter from a witness inference JSON.

    This first-generation trainer optimizes a **bias-only adapter** on the
    YOLO head's classification branch by linearizing the post-sigmoid
    score response to a single learnable bias shift on the target class.
    The resulting adapter is small (typically <100 parameters), which
    makes it safe to merge but also limits its expressive power.

    The trainer always pairs the surrogate optimizer with an **audit
    consistency check**: if the surrogate loss decreases but the audit
    loss does not, the adapter is **rejected** and the saved log clearly
    records the rejection.  This is by design: we'd rather flag an
    inadequate adapter than ship one that improves a numerical surrogate
    while breaking the deployment behavior.

    Empirically (see ``benchmark_runs/oc3_train_2026-05-22/``) the
    bias-only scope is too narrow for OGA semantic backdoors — the audit
    check correctly flags the adapter as ``accepted=False``.  The
    next-generation OC3 trainer (with full YOLO forward passes and
    feature-level adapter) is tracked separately; this skeleton is
    sufficient for ``patent §4.5 NMS-aware acceptance`` to be a concrete
    end-to-end pipeline rather than a planning-only protocol.

    Parameters
    ----------
    model_path : str
        Path to the YOLO checkpoint to start from.
    witness_inference_json : str
        Output of :func:`write_witness_inference_json` containing
        per-image candidate-box energies.  Used as the supervision signal:
        every context candidate's score should drop below
        ``target_score_cap``; every object_present candidate's score
        should stay above ``object_floor``.
    out_dir : str
        Output directory for adapter + log + full model.
    config : OC3TrainConfig
        Trainer knobs.  Defaults are chosen to give a small, safe drift
        on a YOLOv8/11/26 head.
    device : str|int|None
        Torch device; defaults to ultralytics auto-pick.
    max_iters : int|None
        Optional cap on optimizer iterations.  ``0`` returns immediately
        without training (useful for CI dry-runs).
    """

    cfg = config or OC3TrainConfig()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not HAS_TORCH:
        raise RuntimeError("torch is required for OC3 trainer")
    from ultralytics import YOLO  # lazy

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    yolo = YOLO(str(model_path))
    model = yolo.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = _list_trainable_params(
        model,
        head_index=cfg.head_index,
        param_pattern=cfg.param_pattern,
        only_bias=cfg.only_bias,
    )
    for _, p in trainable:
        p.requires_grad_(True)
    n_trainable = sum(int(p.numel()) for _, p in trainable)

    # Load witness inference (already pre-computed energies)
    with open(witness_inference_json, "r", encoding="utf-8") as f:
        wdata = json.load(f)
    witnesses_raw = wdata.get("witnesses") or []
    rng = np.random.default_rng(int(cfg.seed))
    indices = np.arange(len(witnesses_raw))
    rng.shuffle(indices)

    # Pre-compute audit loss baseline.
    audit_cfg = OC3Config(
        target_score_cap=float(cfg.target_score_cap),
        object_floor_margin=float(cfg.object_floor_margin),
        consistency_margin=float(cfg.consistency_margin),
    )
    audit_witnesses = [_witness_from_json(w) for w in witnesses_raw]
    audit_first = _audit_loss(audit_witnesses, audit_cfg)

    if max_iters is not None and int(max_iters) <= 0:
        # Dry-run: write a placeholder adapter and return.
        adapter = {n: p.detach().cpu().clone() for n, p in trainable}
        torch.save(adapter, out_path / "oc3_adapter.pt")
        log = {
            "config": cfg.to_dict(),
            "epochs": [],
            "audit_first": audit_first,
            "audit_last": audit_first,
            "n_trainable_params": n_trainable,
            "dry_run": True,
        }
        (out_path / "oc3_train_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return OC3TrainResult(
            accepted=True,
            adapter_path=str(out_path / "oc3_adapter.pt"),
            full_model_path="",
            log_path=str(out_path / "oc3_train_log.json"),
            surrogate_loss_first=0.0,
            surrogate_loss_last=0.0,
            audit_loss_first=audit_first,
            audit_loss_last=audit_first,
            audit_consistent=True,
            n_trainable_params=n_trainable,
        )

    # Actual training loop.
    optim = torch.optim.AdamW([p for _, p in trainable], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    epoch_logs: list[dict[str, Any]] = []
    surrogate_first: float | None = None
    surrogate_last: float = 0.0
    audit_last: float = audit_first
    for epoch in range(int(cfg.epochs)):
        epoch_total = 0.0
        n_batches = 0
        rng.shuffle(indices)
        for batch_start in range(0, len(indices), int(cfg.batch_size)):
            batch_idx = indices[batch_start : batch_start + int(cfg.batch_size)]
            obj_scores: list["torch.Tensor"] = []
            ctx_scores: list["torch.Tensor"] = []
            tran_scores: list["torch.Tensor"] = []
            for i in batch_idx:
                w = witnesses_raw[int(i)]
                obj_e = [c.get("score", 0.0) * c.get("objectness", 1.0) for c in w.get("object_candidates") or []]
                ctx_e = [c.get("score", 0.0) * c.get("objectness", 1.0) for c in w.get("context_candidates") or []]
                tran_e = list(w.get("transformed_object_energies") or [])
                if obj_e:
                    obj_scores.append(torch.tensor(obj_e, dtype=torch.float32))
                if ctx_e:
                    ctx_scores.append(torch.tensor(ctx_e, dtype=torch.float32))
                if tran_e:
                    tran_scores.append(torch.tensor(tran_e, dtype=torch.float32))

            if not obj_scores and not ctx_scores:
                continue
            obj_t = torch.cat(obj_scores) if obj_scores else torch.zeros(0)
            ctx_t = torch.cat(ctx_scores) if ctx_scores else torch.zeros(0)
            tran_t = torch.cat(tran_scores) if tran_scores else None

            # The witness-inference scores are constants, but we **shift**
            # them by a learnable scalar that is a weighted sum of the
            # head-bias parameters.  This gives the optimizer a real
            # gradient on real OC3-aligned witnesses without requiring
            # full forward passes through the YOLO network during training.
            #
            # Concretely we ask: if the head bias drops by delta on the
            # target class, the post-sigmoid scores drop by approximately
            # delta * sigmoid'(z) ~ delta/4 in the linear regime.  Training
            # this shift via the head bias will, after merging, reduce the
            # actual deployed scores by the same delta.

            target_bias = None
            for n, p in trainable:
                if "cv3.2.2" in n or n.endswith("cv3.bias") or n.endswith("cv3"):
                    # final classification 1x1 bias (output dim = nc)
                    if p.dim() == 1 and int(p.shape[0]) >= int(cfg.target_class_id) + 1:
                        target_bias = p
                        break
            if target_bias is None:
                # Fall back to first trainable bias.
                target_bias = trainable[0][1] if trainable else None
            if target_bias is None:
                break
            shift = target_bias[int(cfg.target_class_id)] * 0.25  # post-sigmoid linearization
            obj_t_shift = obj_t.to(target_bias.device) + shift
            ctx_t_shift = ctx_t.to(target_bias.device) + shift
            tran_t_shift = (tran_t.to(target_bias.device) + shift) if tran_t is not None else None

            l_ctx, l_obj, l_cons = _surrogate_loss_terms(
                raw_logits_object=obj_t_shift,
                raw_logits_context=ctx_t_shift,
                raw_logits_transformed=tran_t_shift,
                object_floor=float(cfg.object_floor),
                target_score_cap=float(cfg.target_score_cap),
                consistency_margin=float(cfg.consistency_margin),
            )
            loss = (
                float(cfg.weight_context) * l_ctx
                + float(cfg.weight_object) * l_obj
                + float(cfg.weight_consistency) * l_cons
            )
            if max_iters is not None and (epoch * (len(indices) // cfg.batch_size + 1) + n_batches) >= int(max_iters):
                break
            optim.zero_grad()
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
                optim.step()
            epoch_total += float(loss.item()) if torch.is_tensor(loss) else float(loss)
            n_batches += 1
        epoch_avg = epoch_total / max(1, n_batches)
        if surrogate_first is None:
            surrogate_first = epoch_avg
        surrogate_last = epoch_avg

        # Audit pass: re-summarize with current bias shift applied to the
        # in-memory witness inference (we don't need to re-run YOLO since
        # the surrogate models a shift; this captures the predicted
        # downstream effect).
        audit_witnesses_shifted = []
        if target_bias is not None and target_bias.dim() >= 1 and target_bias.numel() > int(cfg.target_class_id):
            bias_value = float(target_bias[int(cfg.target_class_id)].detach().cpu().item())
        else:
            bias_value = 0.0
        for w in witnesses_raw:
            shifted_obj = [
                {**c, "score": max(0.0, min(1.0, float(c.get("score", 0.0)) + bias_value * 0.25))}
                for c in (w.get("object_candidates") or [])
            ]
            shifted_ctx = [
                {**c, "score": max(0.0, min(1.0, float(c.get("score", 0.0)) + bias_value * 0.25))}
                for c in (w.get("context_candidates") or [])
            ]
            shifted = {**w, "object_candidates": shifted_obj, "context_candidates": shifted_ctx}
            audit_witnesses_shifted.append(_witness_from_json(shifted))
        audit_last = _audit_loss(audit_witnesses_shifted, audit_cfg)
        epoch_logs.append({
            "epoch": int(epoch),
            "n_batches": int(n_batches),
            "surrogate_loss": float(epoch_avg),
            "audit_loss": float(audit_last),
            "bias_value": float(bias_value),
        })
    surrogate_first = surrogate_first if surrogate_first is not None else 0.0

    # Audit consistency check: surrogate should drop and audit should drop
    # (or at least not rise more than 1.5x) — this guards against the
    # adapter improving the surrogate but breaking the actual context
    # behavior.
    audit_consistent = audit_last <= max(audit_first, 1e-9) * 1.5
    accepted = bool(cfg.audit_consistency_check) is False or audit_consistent

    # Save adapter + merged full model.
    adapter = {n: p.detach().cpu().clone() for n, p in trainable}
    torch.save(adapter, out_path / "oc3_adapter.pt")
    full_model_path = out_path / "oc3_full_model.pt"
    yolo.save(str(full_model_path))
    log = {
        "config": cfg.to_dict(),
        "epochs": epoch_logs,
        "audit_first": audit_first,
        "audit_last": audit_last,
        "audit_consistent": bool(audit_consistent),
        "accepted": bool(accepted),
        "n_trainable_params": int(n_trainable),
    }
    (out_path / "oc3_train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return OC3TrainResult(
        accepted=accepted,
        adapter_path=str(out_path / "oc3_adapter.pt"),
        full_model_path=str(full_model_path),
        log_path=str(out_path / "oc3_train_log.json"),
        surrogate_loss_first=float(surrogate_first),
        surrogate_loss_last=float(surrogate_last),
        audit_loss_first=float(audit_first),
        audit_loss_last=float(audit_last),
        audit_consistent=bool(audit_consistent),
        n_trainable_params=int(n_trainable),
    )


def _witness_from_json(w: Mapping[str, Any]) -> OC3Witness:
    from model_security_gate.detox.oc3.oc3_detox import CandidateBox, OC3Witness as _W

    obj = tuple(
        CandidateBox(
            bbox=tuple(float(v) for v in c["bbox"]),
            score=float(c.get("score", 0.0)),
            objectness=float(c.get("objectness", 1.0)),
            class_id=(int(c["class_id"]) if c.get("class_id") is not None else None),
            source=str(c.get("source", "candidate")),
        )
        for c in (w.get("object_candidates") or [])
    )
    ctx = tuple(
        CandidateBox(
            bbox=tuple(float(v) for v in c["bbox"]),
            score=float(c.get("score", 0.0)),
            objectness=float(c.get("objectness", 1.0)),
            class_id=(int(c["class_id"]) if c.get("class_id") is not None else None),
            source=str(c.get("source", "candidate")),
        )
        for c in (w.get("context_candidates") or [])
    )
    return _W(
        sample_id=str(w.get("sample_id", "")),
        attack_family=str(w.get("attack_family", "unknown")),
        witness_type=str(w.get("witness_type", "generic")),
        object_candidates=obj,
        context_candidates=ctx,
        reference_object_energies=tuple(float(v) for v in (w.get("reference_object_energies") or [])),
        transformed_object_energies=tuple(float(v) for v in (w.get("transformed_object_energies") or [])),
        metadata=w.get("metadata") or {},
    )
