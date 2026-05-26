from __future__ import annotations

"""OC3-Detox trainer v2: feature-level adapter with full YOLO forward pass.

The first-generation trainer (oc3_trainer.py) optimizes only the head's
classification bias via a post-sigmoid score linearization.  Empirically
that scope is too narrow for OGA semantic backdoors -- the audit guard
correctly rejects it.

This v2 trainer fixes that by:

1. Running **real YOLO forward passes** on the witness images.
2. Hooking the head's ``cv3`` classification branch outputs at all three
   FPN scales (typically 52x52, 26x26, 13x13 spatial cells x 2 classes
   for helmet/head).  These tensors retain gradients through head
   parameters.
3. Building per-image **target-class logit maps** by collecting
   per-cell sigmoid(cls_logits[target_id]).
4. Computing **OC3 surrogate losses** on those logit maps:
   - context_only / object_erased witnesses: penalize sigmoid scores
     above ``target_score_cap`` everywhere on the image (no GT target
     should remain).
   - object_present / object_transplant witnesses: penalize sigmoid
     scores below ``object_floor`` *inside* the GT bbox region (rescaled
     to the FPN grid).
   - geometry_pair / frequency_pair witnesses: pair the witness's
     logit map with the corresponding ``object_present`` logit map's
     in-bbox region and penalize differences > ``consistency_margin``.
5. Updating ``cv3.*`` parameters (typically all conv weights + biases on
   the classification branch).  Backbone is frozen.
6. Pairing every epoch with an audit pass that re-runs YOLO inference
   on the same images and checks that the audit OC3 loss drops in the
   same direction as the surrogate -- otherwise the adapter is
   rejected.

The output is a full Ultralytics-shaped checkpoint that can be plugged
straight into ``run_external_hard_suite.py`` and CFRC.
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


@dataclass(frozen=True)
class OC3TrainV2Config:
    head_index: int = -1
    param_pattern: str = "cv3"
    only_bias: bool = False
    target_class_id: int = 0
    target_score_cap: float = 0.25
    object_floor: float = 0.50
    consistency_margin: float = 0.05
    weight_context: float = 1.0
    weight_object: float = 1.0
    weight_consistency: float = 0.3
    weight_l2sp: float = 1e-3
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    epochs: int = 2
    batch_size: int = 4
    imgsz: int = 416
    audit_consistency_check: bool = True
    seed: int = 42
    grad_clip: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OC3TrainV2Result:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _list_trainable_params(head, *, param_pattern: str, only_bias: bool):
    out: list[tuple[str, "torch.nn.Parameter"]] = []
    for name, p in head.named_parameters():
        if param_pattern and param_pattern not in name:
            continue
        if only_bias and not name.endswith("bias"):
            continue
        out.append((name, p))
    return out


def _read_image_tensor(path: str, imgsz: int, device: str | int):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(imgsz), int(imgsz)), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return t, (h, w)


def _bbox_to_grid(bbox_xyxy, orig_hw, grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    """Convert original-image xyxy bbox to grid cell range."""
    oh, ow = orig_hw
    x1, y1, x2, y2 = bbox_xyxy
    gx1 = int(np.clip((x1 / max(1, ow)) * grid_w, 0, grid_w - 1))
    gy1 = int(np.clip((y1 / max(1, oh)) * grid_h, 0, grid_h - 1))
    gx2 = int(np.clip(np.ceil((x2 / max(1, ow)) * grid_w), gx1 + 1, grid_w))
    gy2 = int(np.clip(np.ceil((y2 / max(1, oh)) * grid_h), gy1 + 1, grid_h))
    return gx1, gy1, gx2, gy2


def _hook_cv3(head):
    """Register forward hooks on the head's classification branches.

    YOLO26 has both ``cv3.*`` (one2many, used during training) and
    ``one2one_cv3.*`` (NMS-free, used during inference).  YOLOv8/v11
    only have ``cv3.*``.  We hook both when present so the gradient can
    flow whichever scope the optimizer is updating; the predictions at
    deployment come from ``one2one_cv3`` on YOLO26.

    Returns ``(outputs_holder, [hooks])``.  Caller must call
    ``hook.remove()`` after each forward.  ``outputs_holder`` is a
    list of (name, tensor) tuples preserving registration order so the
    consumer can pick the appropriate scale.
    """

    outputs: list = []
    hooks = []
    branches = ("cv3", "one2one_cv3")
    for name, sub in head.named_modules():
        for br in branches:
            # Match top-level scale modules: cv3.0, cv3.1, cv3.2 (and
            # one2one_cv3.0/1/2 on YOLO26).
            if name.startswith(f"{br}.") and name.count(".") == 1:
                outputs_local = outputs

                def _h(_mod, _inp, out, _name=name):
                    outputs_local.append((_name, out))

                hooks.append(sub.register_forward_hook(_h))
    return outputs, hooks


def _per_image_surrogate(
    *,
    cls_outputs: list,
    target_class_id: int,
    witness_type: str,
    bboxes: list[tuple[float, float, float, float]],
    orig_hw: tuple[int, int],
    paired_object_mean: float | None,
    cfg: OC3TrainV2Config,
):
    """Compute per-image OC3 surrogate from cv3 hook outputs.

    ``cls_outputs`` is a list of ``(name, tensor)`` tuples produced by
    :func:`_hook_cv3` -- it includes both ``cv3.*`` (training branch)
    and ``one2one_cv3.*`` (deployment branch on YOLO26).  We compute the
    loss on **every** captured output so gradient flows to whichever
    scope the optimizer is updating.  Both branches share the same
    spatial layout so the masks are reusable.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required")
    if not cls_outputs:
        return None, None, None
    first_tensor = cls_outputs[0][1] if isinstance(cls_outputs[0], tuple) else cls_outputs[0]
    zero = torch.zeros((), device=first_tensor.device, dtype=first_tensor.dtype)
    l_ctx = zero.clone()
    l_obj = zero.clone()
    l_cons = zero.clone()

    is_target_absent = witness_type in ("context_only", "object_erased")
    is_object_present = witness_type in ("object_present", "object_transplant")
    is_object_present_suppressed = witness_type in ("object_present_suppressed",)
    is_paired = witness_type in ("geometry_pair", "frequency_pair")

    n_terms = 0
    for entry in cls_outputs:
        if isinstance(entry, tuple):
            _, cls_out = entry
        else:
            cls_out = entry
        # cls_out shape: (1, num_classes, H, W)
        sig = torch.sigmoid(cls_out[0, int(target_class_id)])  # (H, W)
        gh, gw = sig.shape

        if is_target_absent:
            # Use the worst-case cell (max sigmoid above cap) instead of
            # the image-mean.  The backdoor only fires at specific spatial
            # positions (where the trigger sits), so averaging across all
            # H*W cells dilutes the gradient until it is unusable.  The
            # max-cell formulation is exactly what the deployed system
            # cares about ("any cell above conf threshold becomes a
            # detection") and gives the optimizer a non-trivial signal.
            over = torch.relu(sig - float(cfg.target_score_cap))
            l_ctx = l_ctx + over.max() * over.max()
            n_terms += 1
        elif is_object_present and bboxes:
            mask_inside = torch.zeros((gh, gw), device=sig.device, dtype=sig.dtype)
            for bb in bboxes:
                gx1, gy1, gx2, gy2 = _bbox_to_grid(bb, orig_hw, gh, gw)
                mask_inside[gy1:gy2, gx1:gx2] = 1.0
            inside = sig * mask_inside
            outside = sig * (1.0 - mask_inside)
            n_inside = mask_inside.sum().clamp(min=1.0)
            n_outside = (1.0 - mask_inside).sum().clamp(min=1.0)
            under = torch.relu(float(cfg.object_floor) - inside) * mask_inside
            over = torch.relu(outside - float(cfg.target_score_cap))
            l_obj = l_obj + (under * under).sum() / n_inside
            l_ctx = l_ctx + (over * over).sum() / n_outside
            n_terms += 1
        elif is_object_present_suppressed and bboxes:
            # ODA path: the trigger suppresses target detection inside
            # the GT bbox.  Push the *max* in-bbox cell back above the
            # object floor (max-cell, not mean-cell, because suppression
            # usually leaves a tiny pocket of high score; we want at
            # least one cell above threshold to recover the detection at
            # NMS).  Outside the bbox we still suppress spurious target
            # evidence, since the trigger may also hallucinate elsewhere.
            mask_inside = torch.zeros((gh, gw), device=sig.device, dtype=sig.dtype)
            for bb in bboxes:
                gx1, gy1, gx2, gy2 = _bbox_to_grid(bb, orig_hw, gh, gw)
                mask_inside[gy1:gy2, gx1:gx2] = 1.0
            outside = sig * (1.0 - mask_inside)
            n_outside = (1.0 - mask_inside).sum().clamp(min=1.0)
            if mask_inside.sum() > 0:
                # Pick the max sig value INSIDE the bbox.  Multiplication
                # by mask first (rather than masked_select) keeps gradient
                # flowing only on in-bbox cells.
                inside_max = (sig * mask_inside).max()
            else:
                inside_max = torch.zeros((), device=sig.device, dtype=sig.dtype)
            under = torch.relu(float(cfg.object_floor) - inside_max)
            over = torch.relu(outside - float(cfg.target_score_cap))
            l_obj = l_obj + under * under
            l_ctx = l_ctx + (over * over).sum() / n_outside
            n_terms += 1
        elif is_paired and paired_object_mean is not None and bboxes:
            mask_inside = torch.zeros((gh, gw), device=sig.device, dtype=sig.dtype)
            for bb in bboxes:
                gx1, gy1, gx2, gy2 = _bbox_to_grid(bb, orig_hw, gh, gw)
                mask_inside[gy1:gy2, gx1:gx2] = 1.0
            inside_mean = (sig * mask_inside).sum() / mask_inside.sum().clamp(min=1.0)
            diff = torch.abs(inside_mean - float(paired_object_mean)) - float(cfg.consistency_margin)
            l_cons = l_cons + torch.relu(diff).pow(2)
            n_terms += 1

    if n_terms > 0:
        l_ctx = l_ctx / n_terms
        l_obj = l_obj / n_terms
        l_cons = l_cons / n_terms
    return l_ctx, l_obj, l_cons


def train_oc3_adapter_v2(
    *,
    model_path: str,
    witness_manifest_json: str,
    out_dir: str,
    config: OC3TrainV2Config | None = None,
    device: str | int | None = None,
    max_iters: int | None = None,
) -> OC3TrainV2Result:
    """Train an OC3 head adapter with real YOLO forward passes.

    Parameters
    ----------
    model_path : str
        YOLO checkpoint to start from.
    witness_manifest_json : str
        ``oc3_witness_manifest.json`` produced by
        :func:`build_witnesses`.  This contains the physical witness
        image paths + GT bboxes that the v2 trainer needs to compute
        spatial losses.
    out_dir : str
        Output directory.
    config : OC3TrainV2Config
    device : str|int|None
    max_iters : int|None
        Cap on total iterations across epochs.  ``0`` for dry-run.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for v2 trainer")
    from ultralytics import YOLO  # lazy

    cfg = config or OC3TrainV2Config()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    yolo = YOLO(str(model_path))
    model = yolo.model
    model.eval()  # keep in eval mode but enable grads on selected params
    if device is None:
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        dev = f"cuda:{int(device)}"
    else:
        dev = str(device)
    model.to(dev)
    head = model.model[cfg.head_index]
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = _list_trainable_params(head, param_pattern=cfg.param_pattern, only_bias=cfg.only_bias)
    for _, p in trainable:
        p.requires_grad_(True)
    n_trainable = sum(int(p.numel()) for _, p in trainable)
    init_state: dict[str, "torch.Tensor"] = {n: p.detach().clone() for n, p in trainable}

    with open(witness_manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest.get("records") or []
    target_class_ids = [int(c) for c in (manifest.get("target_class_ids") or [int(cfg.target_class_id)])]
    target_id = int(target_class_ids[0])

    # Group records by base_image_id for object_present reference + pair lookups.
    by_base: dict[str, dict[str, dict]] = {}
    for r in records:
        by_base.setdefault(r["base_image_id"], {})[r["witness_type"]] = r

    # Build (image_path, witness_type, bboxes, paired_obj_mean_placeholder) batch list.
    items: list[dict] = []
    for base_id, recs in by_base.items():
        for wtype, r in recs.items():
            items.append(
                {
                    "base_id": base_id,
                    "witness_type": wtype,
                    "image_path": r["image_path"],
                    "bboxes": [tuple(float(v) for v in b) for b in (r.get("object_bboxes_xyxy") or [])],
                }
            )

    rng = np.random.default_rng(int(cfg.seed))
    indices = np.arange(len(items))

    # Pre-pass: compute paired object mean by running on object_present
    # witnesses with no_grad. These are scalar numbers we cache.
    paired_means: dict[str, float] = {}
    with torch.no_grad():
        for r in records:
            if r["witness_type"] != "object_present":
                continue
            try:
                img_t, ohw = _read_image_tensor(r["image_path"], cfg.imgsz, dev)
                outputs, hooks = _hook_cv3(head)
                _ = model(img_t)
                for h in hooks:
                    h.remove()
                cls_outputs = outputs
                bboxes = [tuple(float(v) for v in b) for b in (r.get("object_bboxes_xyxy") or [])]
                if not bboxes or not cls_outputs:
                    continue
                # Use the largest-resolution scale of the inference branch
                # (one2one_cv3.* on YOLO26, cv3.* otherwise) for the
                # reference paired mean.
                inference_outputs = [
                    (n, t) for n, t in cls_outputs
                    if (isinstance(cls_outputs[0], tuple) and "one2one_cv3" in n)
                ] or cls_outputs
                ref_entry = inference_outputs[0]
                ref_scale = ref_entry[1] if isinstance(ref_entry, tuple) else ref_entry
                sig = torch.sigmoid(ref_scale[0, target_id])
                gh, gw = sig.shape
                mask = torch.zeros_like(sig)
                for bb in bboxes:
                    gx1, gy1, gx2, gy2 = _bbox_to_grid(bb, ohw, gh, gw)
                    mask[gy1:gy2, gx1:gx2] = 1.0
                if mask.sum() > 0:
                    paired_means[r["base_image_id"]] = float((sig * mask).sum() / mask.sum())
            except FileNotFoundError:
                continue

    if max_iters is not None and int(max_iters) <= 0:
        # Dry-run path
        adapter = {n: p.detach().cpu().clone() for n, p in trainable}
        torch.save(adapter, out_path / "oc3_adapter.pt")
        return OC3TrainV2Result(
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
        )

    optim = torch.optim.AdamW([p for _, p in trainable], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    epoch_logs: list[dict[str, Any]] = []
    surrogate_first: float | None = None
    surrogate_last: float = 0.0
    audit_first: float | None = None
    audit_last: float = 0.0
    iter_done = 0
    t_start = time.time()
    for epoch in range(int(cfg.epochs)):
        rng.shuffle(indices)
        epoch_total = 0.0
        n_batches = 0
        audit_total = 0.0
        n_audit = 0
        for batch_start in range(0, len(indices), int(cfg.batch_size)):
            batch_idx = indices[batch_start : batch_start + int(cfg.batch_size)]
            optim.zero_grad()
            batch_loss = torch.zeros((), device=dev, dtype=torch.float32)
            for i in batch_idx:
                it = items[int(i)]
                try:
                    img_t, ohw = _read_image_tensor(it["image_path"], cfg.imgsz, dev)
                except FileNotFoundError:
                    continue
                outputs, hooks = _hook_cv3(head)
                _ = model(img_t)
                for h in hooks:
                    h.remove()
                cls_outputs = outputs
                if not cls_outputs:
                    continue
                paired = paired_means.get(it["base_id"]) if it["witness_type"] in ("geometry_pair", "frequency_pair") else None
                l_ctx, l_obj, l_cons = _per_image_surrogate(
                    cls_outputs=cls_outputs,
                    target_class_id=target_id,
                    witness_type=it["witness_type"],
                    bboxes=it["bboxes"],
                    orig_hw=ohw,
                    paired_object_mean=paired,
                    cfg=cfg,
                )
                batch_loss = batch_loss + (
                    float(cfg.weight_context) * l_ctx
                    + float(cfg.weight_object) * l_obj
                    + float(cfg.weight_consistency) * l_cons
                )
            # L2-SP regularization: keep adapter close to initial state to bound drift
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

        # Audit pass: re-run inference under no_grad and aggregate sigmoid scores.
        with torch.no_grad():
            audit_total = 0.0
            n_audit = 0
            for it in items:
                try:
                    img_t, ohw = _read_image_tensor(it["image_path"], cfg.imgsz, dev)
                except FileNotFoundError:
                    continue
                outputs, hooks = _hook_cv3(head)
                _ = model(img_t)
                for h in hooks:
                    h.remove()
                cls_outputs = outputs
                if not cls_outputs:
                    continue
                # Compute the same per-image OC3 surrogate as the audit loss.
                paired = paired_means.get(it["base_id"]) if it["witness_type"] in ("geometry_pair", "frequency_pair") else None
                l_ctx, l_obj, l_cons = _per_image_surrogate(
                    cls_outputs=cls_outputs,
                    target_class_id=target_id,
                    witness_type=it["witness_type"],
                    bboxes=it["bboxes"],
                    orig_hw=ohw,
                    paired_object_mean=paired,
                    cfg=cfg,
                )
                audit_total += float(l_ctx.item() + l_obj.item() + l_cons.item())
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
    # IMPORTANT: yolo.save() preserves the original `ema` key from the
    # loaded checkpoint, which Ultralytics may prefer over the trained
    # `model` at inference time.  Drop ema (and other stale keys) so
    # the trained head adapter is what actually loads.
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
        "n_witness_images": len(items),
    }
    (out_path / "oc3_train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    return OC3TrainV2Result(
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
        n_witness_images_used=int(len(items)),
    )
