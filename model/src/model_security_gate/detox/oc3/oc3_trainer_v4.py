from __future__ import annotations

"""OC3-Detox trainer v4: trigger-paired self-consensus (no external teacher).

This is the canonical OC3 trainer that preserves the algorithm's original
three core claims:

1. **Candidate-box level reasoning** (not neuron, not subspace).
2. **No external teacher** (consensus is built between two views of the
   *same* trainable model).
3. **NMS-aware candidate energy** (max-cell context cap, in-bbox object
   floor for ODA, paired logit-map self-consistency for trigger-pair).

The key novelty of v4 over v3: instead of importing a clean teacher's
logit map (which would degrade OC3 to a NAD-style distillation defense),
v4 pairs each attack_eval image with its clean source image (same UUID,
no trigger).  Both views are run through the **same** trainable model
and the OC3 ``transform_consensus`` term forces their per-cell sigmoid
maps to agree (within margin) inside the GT bbox region.

This formulation:
- Treats the trigger application as a "transform" the OC3 protocol
  already covers (transform_paired witness).
- Makes the model *invariant* to the trigger across paired views.
- Works for both OGA (push trigger-side scores down to clean-side) and
  ODA (push trigger-side scores up to clean-side) with the same loss
  surface — which side moves depends on the optimization dynamics.

Loss terms used:
- ``L_context_max_cell``: on every ``context_only`` witness (target
  absent), the per-cell max sigmoid must be ≤ ``target_score_cap``.
- ``L_object_inbbox``: on every ``object_present`` witness, the in-bbox
  max sigmoid must be ≥ ``object_floor``.
- ``L_trigger_pair_consensus``: on every ``trigger_paired_*`` pair,
  the *triggered* model output and the *clean* model output should
  agree per-cell (mean L2 across all 3 FPN scales of the inference
  branch).  This is the no-teacher self-consistency loss.
- ``L_l2sp``: keep the head adapter close to its initial state.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import time

import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False


from model_security_gate.detox.oc3.oc3_trainer_v2 import (
    _bbox_to_grid,
    _hook_cv3,
    _list_trainable_params,
    _read_image_tensor,
)


@dataclass(frozen=True)
class OC3TrainV4Config:
    head_index: int = -1
    param_pattern: str = "auto"  # 'auto' | 'cv3' | 'one2one_cv3'
    only_bias: bool = False
    target_class_id: int = 0
    target_score_cap: float = 0.20
    object_floor: float = 0.30
    consistency_margin: float = 0.05
    weight_context: float = 1.0
    weight_object: float = 1.0
    weight_trigger_consensus: float = 5.0  # the no-teacher self-consensus regularizer
    weight_l2sp: float = 5e-3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 5
    batch_size: int = 4
    imgsz: int = 416
    audit_consistency_check: bool = True
    seed: int = 42
    grad_clip: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OC3TrainV4Result:
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
    n_witness_images_used: int
    n_pairs_used: int
    param_pattern: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_param_pattern(head) -> str:
    has_one2one = any(name.startswith("one2one_cv3") for name, _ in head.named_modules())
    return "one2one_cv3" if has_one2one else "cv3"


def _filter_inference_outputs(cls_outputs: list, param_pattern: str):
    """Pick only the inference-branch outputs from hook captures."""
    out = []
    for entry in cls_outputs:
        name, t = entry if isinstance(entry, tuple) else (None, entry)
        if param_pattern == "one2one_cv3":
            if name and "one2one_cv3" in name:
                out.append((name, t))
        else:
            if name and name.startswith("cv3.") and "one2one" not in name:
                out.append((name, t))
    if not out and cls_outputs:
        out = cls_outputs
    return out


def _per_image_oc3_terms(
    *,
    cls_outputs: list,
    target_class_id: int,
    witness_type: str,
    bboxes: list[tuple[float, float, float, float]],
    orig_hw: tuple[int, int],
    cfg: OC3TrainV4Config,
):
    """Compute (l_ctx, l_obj) for non-pair witnesses using max-cell semantics."""
    if not cls_outputs:
        return None, None
    first = cls_outputs[0][1] if isinstance(cls_outputs[0], tuple) else cls_outputs[0]
    zero = torch.zeros((), device=first.device, dtype=first.dtype)
    l_ctx = zero.clone()
    l_obj = zero.clone()

    is_target_absent = witness_type in ("context_only", "object_erased")
    is_object_present = witness_type in ("object_present", "object_transplant")

    n_terms = 0
    for entry in cls_outputs:
        _, cls_out = entry if isinstance(entry, tuple) else (None, entry)
        sig = torch.sigmoid(cls_out[0, int(target_class_id)])  # (H, W)
        gh, gw = sig.shape

        if is_target_absent:
            over = torch.relu(sig - float(cfg.target_score_cap))
            l_ctx = l_ctx + over.max() * over.max()
            n_terms += 1
        elif is_object_present and bboxes:
            mask_inside = torch.zeros((gh, gw), device=sig.device, dtype=sig.dtype)
            for bb in bboxes:
                gx1, gy1, gx2, gy2 = _bbox_to_grid(bb, orig_hw, gh, gw)
                mask_inside[gy1:gy2, gx1:gx2] = 1.0
            if mask_inside.sum() > 0:
                inside_max = (sig * mask_inside).max()
            else:
                inside_max = torch.zeros((), device=sig.device, dtype=sig.dtype)
            outside = sig * (1.0 - mask_inside)
            n_outside = (1.0 - mask_inside).sum().clamp(min=1.0)
            under = torch.relu(float(cfg.object_floor) - inside_max)
            over = torch.relu(outside - float(cfg.target_score_cap))
            l_obj = l_obj + under * under
            l_ctx = l_ctx + (over * over).sum() / n_outside
            n_terms += 1

    if n_terms > 0:
        l_ctx = l_ctx / n_terms
        l_obj = l_obj / n_terms
    return l_ctx, l_obj


def _trigger_consensus(
    clean_outputs: list,
    triggered_outputs: list,
    *,
    bboxes: list[tuple[float, float, float, float]],
    orig_hw: tuple[int, int],
    target_class_id: int,
    margin: float,
    asymmetric: bool = True,
):
    """Per-cell consensus loss between two views of the same model.

    L = max(|sig_clean − sig_triggered| − margin, 0)^2 averaged over the
    in-bbox region (when bboxes provided) or the whole image otherwise.

    When ``asymmetric=True`` (default), the **clean side is detached**
    from the autograd graph so the gradient only flows through the
    *triggered* side.  This forces the optimizer to move the trigger
    prediction toward the clean prediction, not the other way around.
    Without this, the optimizer can take the easy shortcut of pulling
    the clean prediction down to match the suppressed trigger
    prediction (catastrophic for ODA where clean is correct and trigger
    is wrong).

    Both views still share the same head parameters; the asymmetry only
    blocks gradient flow on the clean side, it does not change the
    forward behavior or the no-teacher property (the clean side is
    still the *current* trainable model, not an external teacher).
    """
    if not clean_outputs or not triggered_outputs:
        first = clean_outputs[0][1] if clean_outputs else triggered_outputs[0][1]
        return torch.zeros((), device=first.device, dtype=first.dtype)
    n_terms = 0
    total = None
    for c_entry, t_entry in zip(clean_outputs, triggered_outputs):
        _, c_out = c_entry if isinstance(c_entry, tuple) else (None, c_entry)
        _, t_out = t_entry if isinstance(t_entry, tuple) else (None, t_entry)
        if c_out.shape != t_out.shape:
            continue
        c_sig = torch.sigmoid(c_out[0, int(target_class_id)])
        t_sig = torch.sigmoid(t_out[0, int(target_class_id)])
        if asymmetric:
            # Detach clean side: gradient only flows through triggered side.
            c_sig = c_sig.detach()
        gh, gw = c_sig.shape

        if bboxes:
            mask = torch.zeros((gh, gw), device=t_sig.device, dtype=t_sig.dtype)
            for bb in bboxes:
                gx1, gy1, gx2, gy2 = _bbox_to_grid(bb, orig_hw, gh, gw)
                mask[gy1:gy2, gx1:gx2] = 1.0
            denom = mask.sum().clamp(min=1.0)
            diff = torch.abs(c_sig - t_sig) - float(margin)
            diff = torch.relu(diff) * mask
            term = (diff * diff).sum() / denom
        else:
            diff = torch.abs(c_sig - t_sig) - float(margin)
            diff = torch.relu(diff)
            term = (diff * diff).mean()

        total = term if total is None else total + term
        n_terms += 1

    if total is None:
        first = clean_outputs[0][1] if clean_outputs else triggered_outputs[0][1]
        return torch.zeros((), device=first.device, dtype=first.dtype)
    return total / max(1, n_terms)


def train_oc3_adapter_v4(
    *,
    model_path: str,
    witness_manifest_json: str,
    out_dir: str,
    config: OC3TrainV4Config | None = None,
    device: str | int | None = None,
    max_iters: int | None = None,
) -> OC3TrainV4Result:
    """Train OC3 v4 adapter with trigger-pair self-consensus (no teacher)."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required for v4 trainer")
    from ultralytics import YOLO  # lazy

    cfg = config or OC3TrainV4Config()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    yolo = YOLO(str(model_path))
    model = yolo.model
    model.eval()
    if device is None:
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        dev = f"cuda:{int(device)}"
    else:
        dev = str(device)
    model.to(dev)
    head = model.model[cfg.head_index]

    param_pattern = cfg.param_pattern
    if param_pattern == "auto":
        param_pattern = _detect_param_pattern(head)

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = _list_trainable_params(head, param_pattern=param_pattern, only_bias=cfg.only_bias)
    for _, p in trainable:
        p.requires_grad_(True)
    n_trainable = sum(int(p.numel()) for _, p in trainable)
    init_state: dict[str, "torch.Tensor"] = {n: p.detach().clone() for n, p in trainable}

    with open(witness_manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest.get("records") or []
    target_class_ids = [int(c) for c in (manifest.get("target_class_ids") or [int(cfg.target_class_id)])]
    target_id = int(target_class_ids[0])

    # Group records by base_image_id for pair lookup.
    by_base: dict[str, dict[str, dict]] = {}
    for r in records:
        by_base.setdefault(r["base_image_id"], {})[r["witness_type"]] = r

    # Build a flat work list:
    # - paired items: (clean_path, trigger_path, bboxes) for trigger consensus.
    # - erased_paired items: (original_triggered_path, erased_path) for
    #   physical-object backdoors where trigger_paired pair is a degenerate
    #   self-self copy. The erased counterpart is produced by the OC3
    #   ``object_erased`` witness builder (HSV+inpaint for color objects).
    # - solo items: (path, witness_type, bboxes) for context/object terms.
    paired_items: list[dict] = []
    solo_items: list[dict] = []
    for base_id, recs in by_base.items():
        clean = recs.get("trigger_paired_clean")
        trig = recs.get("trigger_paired_triggered")
        erased = recs.get("object_erased")

        if clean and trig:
            # Detect degenerate same-image pairs (physical-object backdoors
            # where the witness builder copied the same image twice). When
            # detected, replace the clean side with the object_erased witness
            # if available — same OC3 transform_consensus protocol, same
            # trainable model, same asymmetric detach. No external teacher.
            same_path = (clean.get("image_path") == trig.get("image_path"))
            if same_path and erased:
                paired_items.append({
                    "base_id": base_id,
                    "clean_path": erased["image_path"],
                    "trigger_path": trig["image_path"],
                    "pair_kind": "object_erased_paired",
                    "bboxes": [tuple(float(v) for v in b) for b in (clean.get("object_bboxes_xyxy") or trig.get("object_bboxes_xyxy") or [])],
                })
            else:
                paired_items.append({
                    "base_id": base_id,
                    "clean_path": clean["image_path"],
                    "trigger_path": trig["image_path"],
                    "pair_kind": "trigger_paired",
                    "bboxes": [tuple(float(v) for v in b) for b in (clean.get("object_bboxes_xyxy") or trig.get("object_bboxes_xyxy") or [])],
                })

        for wtype in ("context_only", "object_present"):
            r = recs.get(wtype)
            if r is None:
                continue
            solo_items.append({
                "base_id": base_id,
                "witness_type": wtype,
                "image_path": r["image_path"],
                "bboxes": [tuple(float(v) for v in b) for b in (r.get("object_bboxes_xyxy") or [])],
            })
        # object_erased is also useful as a solo context_only-style term:
        # the erased scene should NOT detect the target. Keep it as a solo
        # item EVEN when used in a pair, because the OC3 protocol applies
        # context_insufficiency on it independently of the paired consensus.
        if erased is not None:
            solo_items.append({
                "base_id": base_id,
                "witness_type": "object_erased",
                "image_path": erased["image_path"],
                "bboxes": [tuple(float(v) for v in b) for b in (erased.get("object_bboxes_xyxy") or [])],
            })

    rng = np.random.default_rng(int(cfg.seed))

    if max_iters is not None and int(max_iters) <= 0:
        adapter = {n: p.detach().cpu().clone() for n, p in trainable}
        torch.save(adapter, out_path / "oc3_adapter.pt")
        return OC3TrainV4Result(
            accepted=True,
            adapter_path=str(out_path / "oc3_adapter.pt"),
            full_model_path="",
            log_path="",
            surrogate_loss_first=0.0,
            surrogate_loss_last=0.0,
            audit_loss_first=0.0,
            audit_loss_last=0.0,
            audit_consistent=True,
            n_trainable_params=n_trainable,
            n_witness_images_used=0,
            n_pairs_used=0,
            param_pattern=param_pattern,
        )

    optim = torch.optim.AdamW([p for _, p in trainable], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    epoch_logs: list[dict[str, Any]] = []
    surrogate_first: float | None = None
    surrogate_last: float = 0.0
    audit_first: float | None = None
    audit_last: float = 0.0
    iter_done = 0
    t_start = time.time()

    # Combine into one shuffled list of "tasks" each epoch: pair tasks +
    # solo tasks.  Pair tasks consume 2 forwards but produce a pair
    # consensus loss + 2 OC3 single-image losses.
    all_tasks = (
        [{"kind": "pair", **p} for p in paired_items]
        + [{"kind": "solo", **s} for s in solo_items]
    )
    n_witness_images = sum(2 if t["kind"] == "pair" else 1 for t in all_tasks)

    for epoch in range(int(cfg.epochs)):
        order = np.arange(len(all_tasks))
        rng.shuffle(order)
        epoch_total = 0.0
        n_batches = 0
        for batch_start in range(0, len(order), int(cfg.batch_size)):
            batch_idx = order[batch_start : batch_start + int(cfg.batch_size)]
            optim.zero_grad()
            batch_loss = torch.zeros((), device=dev, dtype=torch.float32)
            for i in batch_idx:
                task = all_tasks[int(i)]
                if task["kind"] == "pair":
                    try:
                        c_t, c_hw = _read_image_tensor(task["clean_path"], cfg.imgsz, dev)
                        t_t, t_hw = _read_image_tensor(task["trigger_path"], cfg.imgsz, dev)
                    except FileNotFoundError:
                        continue
                    # Forward pass on clean
                    c_outputs, c_hooks = _hook_cv3(head)
                    _ = model(c_t)
                    for h in c_hooks:
                        h.remove()
                    c_filtered = _filter_inference_outputs(c_outputs, param_pattern)
                    # Forward pass on trigger
                    t_outputs, t_hooks = _hook_cv3(head)
                    _ = model(t_t)
                    for h in t_hooks:
                        h.remove()
                    t_filtered = _filter_inference_outputs(t_outputs, param_pattern)

                    # Consensus loss.
                    l_consensus = _trigger_consensus(
                        c_filtered, t_filtered,
                        bboxes=task["bboxes"],
                        orig_hw=c_hw,
                        target_class_id=target_id,
                        margin=cfg.consistency_margin,
                    )
                    # Per-image OC3 anchors: object_present on the clean
                    # side (so the model keeps detecting helmets on the
                    # paired clean source), context_only on triggered
                    # side iff bboxes empty (OGA case).
                    if task["bboxes"]:
                        l_ctx_c, l_obj_c = _per_image_oc3_terms(
                            cls_outputs=c_filtered,
                            target_class_id=target_id,
                            witness_type="object_present",
                            bboxes=task["bboxes"],
                            orig_hw=c_hw,
                            cfg=cfg,
                        )
                        anchor = (
                            (float(cfg.weight_context) * l_ctx_c if l_ctx_c is not None else 0.0)
                            + (float(cfg.weight_object) * l_obj_c if l_obj_c is not None else 0.0)
                        )
                    else:
                        l_ctx_t, _ = _per_image_oc3_terms(
                            cls_outputs=t_filtered,
                            target_class_id=target_id,
                            witness_type="context_only",
                            bboxes=[],
                            orig_hw=t_hw,
                            cfg=cfg,
                        )
                        anchor = float(cfg.weight_context) * l_ctx_t if l_ctx_t is not None else 0.0
                    pair_loss = float(cfg.weight_trigger_consensus) * l_consensus + anchor
                    batch_loss = batch_loss + pair_loss
                else:
                    try:
                        s_t, s_hw = _read_image_tensor(task["image_path"], cfg.imgsz, dev)
                    except FileNotFoundError:
                        continue
                    s_outputs, s_hooks = _hook_cv3(head)
                    _ = model(s_t)
                    for h in s_hooks:
                        h.remove()
                    s_filtered = _filter_inference_outputs(s_outputs, param_pattern)
                    l_ctx, l_obj = _per_image_oc3_terms(
                        cls_outputs=s_filtered,
                        target_class_id=target_id,
                        witness_type=task["witness_type"],
                        bboxes=task["bboxes"],
                        orig_hw=s_hw,
                        cfg=cfg,
                    )
                    if l_ctx is None:
                        continue
                    batch_loss = batch_loss + (
                        float(cfg.weight_context) * l_ctx
                        + float(cfg.weight_object) * l_obj
                    )

            # L2-SP regularization.
            if float(cfg.weight_l2sp) > 0:
                l2sp = torch.zeros((), device=dev, dtype=torch.float32)
                for n, p in trainable:
                    if n in init_state:
                        diff = p - init_state[n].to(p.device)
                        l2sp = l2sp + (diff * diff).sum()
                batch_loss = batch_loss + float(cfg.weight_l2sp) * l2sp
            if not batch_loss.requires_grad:
                continue
            batch_loss.backward()
            if float(cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], float(cfg.grad_clip))
            optim.step()
            epoch_total += float(batch_loss.item())
            n_batches += 1
            iter_done += 1
            if max_iters is not None and iter_done >= int(max_iters):
                break
        epoch_avg = epoch_total / max(1, n_batches)
        if surrogate_first is None:
            surrogate_first = epoch_avg
        surrogate_last = epoch_avg

        # Audit pass.
        with torch.no_grad():
            audit_total = 0.0
            n_audit = 0
            for task in all_tasks:
                if task["kind"] == "pair":
                    try:
                        c_t, c_hw = _read_image_tensor(task["clean_path"], cfg.imgsz, dev)
                        t_t, t_hw = _read_image_tensor(task["trigger_path"], cfg.imgsz, dev)
                    except FileNotFoundError:
                        continue
                    c_outputs, c_hooks = _hook_cv3(head)
                    _ = model(c_t)
                    for h in c_hooks:
                        h.remove()
                    c_filtered = _filter_inference_outputs(c_outputs, param_pattern)
                    t_outputs, t_hooks = _hook_cv3(head)
                    _ = model(t_t)
                    for h in t_hooks:
                        h.remove()
                    t_filtered = _filter_inference_outputs(t_outputs, param_pattern)
                    l_consensus = _trigger_consensus(
                        c_filtered, t_filtered,
                        bboxes=task["bboxes"],
                        orig_hw=c_hw,
                        target_class_id=target_id,
                        margin=cfg.consistency_margin,
                    )
                    audit_total += float(l_consensus.item())
                    n_audit += 1
                else:
                    try:
                        s_t, s_hw = _read_image_tensor(task["image_path"], cfg.imgsz, dev)
                    except FileNotFoundError:
                        continue
                    s_outputs, s_hooks = _hook_cv3(head)
                    _ = model(s_t)
                    for h in s_hooks:
                        h.remove()
                    s_filtered = _filter_inference_outputs(s_outputs, param_pattern)
                    l_ctx, l_obj = _per_image_oc3_terms(
                        cls_outputs=s_filtered,
                        target_class_id=target_id,
                        witness_type=task["witness_type"],
                        bboxes=task["bboxes"],
                        orig_hw=s_hw,
                        cfg=cfg,
                    )
                    if l_ctx is None:
                        continue
                    audit_total += float(l_ctx.item() + l_obj.item())
                    n_audit += 1
        audit_avg = audit_total / max(1, n_audit)
        if audit_first is None:
            audit_first = audit_avg
        audit_last = audit_avg
        epoch_logs.append({
            "epoch": int(epoch),
            "n_batches": int(n_batches),
            "surrogate_loss": float(epoch_avg),
            "audit_loss": float(audit_avg),
            "elapsed_s": float(time.time() - t_start),
        })
        if max_iters is not None and iter_done >= int(max_iters):
            break

    audit_first = audit_first if audit_first is not None else 0.0
    surrogate_first = surrogate_first if surrogate_first is not None else 0.0
    audit_consistent = audit_last <= max(audit_first, 1e-9) * 1.5
    accepted = bool(cfg.audit_consistency_check) is False or audit_consistent

    adapter = {n: p.detach().cpu().clone() for n, p in trainable}
    torch.save(adapter, out_path / "oc3_adapter.pt")
    full_model_path = out_path / "oc3_full_model.pt"
    if hasattr(yolo, "ckpt") and isinstance(yolo.ckpt, dict):
        for stale in ("ema", "updates", "optimizer", "scaler", "train_args", "train_metrics", "train_results"):
            yolo.ckpt.pop(stale, None)
    yolo.save(str(full_model_path))
    log = {
        "config": cfg.to_dict(),
        "epochs": epoch_logs,
        "audit_first": audit_first,
        "audit_last": audit_last,
        "audit_consistent": bool(audit_consistent),
        "accepted": bool(accepted),
        "n_trainable_params": int(n_trainable),
        "n_witness_images": int(n_witness_images),
        "n_pairs_used": int(len(paired_items)),
        "param_pattern": param_pattern,
    }
    (out_path / "oc3_train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    return OC3TrainV4Result(
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
        n_witness_images_used=int(n_witness_images),
        n_pairs_used=int(len(paired_items)),
        param_pattern=param_pattern,
    )
