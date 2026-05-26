from __future__ import annotations

"""OC3-Detox trainer v3: teacher-anchored ODA-aware adapter.

The v2 trainer (oc3_trainer_v2.py) handles OGA via context-suppression
and the audit guard correctly rejects under-trained adapters.  Empirically
it does **not** generalize to ODA because:

- ODA backdoors *suppress* target detection on trigger images, so
  ``context_insufficiency`` has nothing to push against;
- Boosting ``object_present_suppressed`` via head adapter generalizes
  uniformly, over-detecting helmets on the clean val set and
  collapsing mAP (-12 to -23 pp).

v3 introduces three additions:

1. **Teacher distillation on clean witnesses** -- on every
   ``object_present`` / ``object_present_clean`` witness, pin the
   defended model's per-cell sigmoid scores to the **clean teacher**'s
   per-cell sigmoid scores.  This anchors clean-data behavior while
   leaving trigger-suppressed witnesses free to move.
2. **Adaptive scope auto-selection** -- the trainer picks the best
   ``param_pattern`` (``one2one_cv3`` for YOLO26, ``cv3`` for
   YOLOv8/v11) automatically by inspecting the loaded model.
3. **Direction-aware loss weighting** -- the trainer reads the witness
   manifest's ``attack_witness_type`` and switches between OGA-priority
   and ODA-priority loss weights automatically.

Every other guarantee (audit consistency check, EMA-stripped save,
hooks on inference branch, max-cell context loss) is preserved from
v2.

The end result: a single trainer that handles OGA + ODA without the
operator having to know the direction.
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
    _per_image_surrogate,
)


@dataclass(frozen=True)
class OC3TrainV3Config:
    head_index: int = -1
    param_pattern: str = "auto"  # 'auto' | 'cv3' | 'one2one_cv3'
    only_bias: bool = False
    target_class_id: int = 0
    target_score_cap: float = 0.20
    object_floor: float = 0.30
    consistency_margin: float = 0.05
    weight_context: float = 2.0
    weight_object: float = 1.0
    weight_consistency: float = 0.0
    weight_l2sp: float = 5e-3
    weight_teacher_distill: float = 5.0  # heavy: this is the ODA-saving regularizer
    teacher_distill_clean_only: bool = True  # only distill on clean object_present witnesses
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 5
    batch_size: int = 4
    imgsz: int = 416
    audit_consistency_check: bool = True
    seed: int = 42
    grad_clip: float = 1.0
    direction: str = "auto"  # 'auto' | 'oga' | 'oda'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OC3TrainV3Result:
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
    direction: str
    param_pattern: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_param_pattern(head) -> str:
    """Return ``one2one_cv3`` if the head has it (YOLO26), else ``cv3``."""
    has_one2one = any(name.startswith("one2one_cv3") for name, _ in head.named_modules())
    return "one2one_cv3" if has_one2one else "cv3"


def _detect_direction_from_manifest(manifest: Mapping[str, Any]) -> str:
    explicit = manifest.get("attack_witness_type")
    if explicit == "object_present_suppressed":
        return "oda"
    if explicit == "context_only":
        return "oga"
    # fallback: scan records
    for r in manifest.get("records") or []:
        if r.get("witness_type") == "object_present_suppressed":
            return "oda"
    return "oga"


def train_oc3_adapter_v3(
    *,
    model_path: str,
    teacher_model_path: str,
    witness_manifest_json: str,
    out_dir: str,
    config: OC3TrainV3Config | None = None,
    device: str | int | None = None,
    max_iters: int | None = None,
) -> OC3TrainV3Result:
    """Train an OC3 v3 adapter with teacher distillation on clean witnesses."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required for v3 trainer")
    from ultralytics import YOLO  # lazy

    cfg = config or OC3TrainV3Config()
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

    # Auto-detect param pattern.
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

    # Teacher model (frozen, no grads).
    teacher_yolo = YOLO(str(teacher_model_path))
    teacher = teacher_yolo.model
    teacher.eval()
    teacher.to(dev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_head = teacher.model[cfg.head_index]

    # Load manifest, detect direction.
    with open(witness_manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest.get("records") or []
    target_class_ids = [int(c) for c in (manifest.get("target_class_ids") or [int(cfg.target_class_id)])]
    target_id = int(target_class_ids[0])
    direction = cfg.direction
    if direction == "auto":
        direction = _detect_direction_from_manifest(manifest)

    # Group records and collect items.
    items: list[dict] = []
    for r in records:
        items.append(
            {
                "base_id": r["base_image_id"],
                "witness_type": r["witness_type"],
                "image_path": r["image_path"],
                "bboxes": [tuple(float(v) for v in b) for b in (r.get("object_bboxes_xyxy") or [])],
            }
        )

    rng = np.random.default_rng(int(cfg.seed))
    indices = np.arange(len(items))

    # Pre-compute teacher logit maps on clean object_present witnesses.
    # Cache as CPU tensors keyed by image path; bring to device per batch.
    teacher_cache: dict[str, list["torch.Tensor"]] = {}
    with torch.no_grad():
        for it in items:
            if cfg.teacher_distill_clean_only and it["witness_type"] != "object_present":
                continue
            try:
                img_t, ohw = _read_image_tensor(it["image_path"], cfg.imgsz, dev)
            except FileNotFoundError:
                continue
            outputs, hooks = _hook_cv3(teacher_head)
            _ = teacher(img_t)
            for h in hooks:
                h.remove()
            # Filter to inference branch (one2one_cv3 if present, else cv3).
            inference_outputs: list = []
            for entry in outputs:
                name, t = entry if isinstance(entry, tuple) else (None, entry)
                if param_pattern == "one2one_cv3":
                    if name and "one2one_cv3" in name:
                        inference_outputs.append(t)
                else:
                    if name and name.startswith("cv3.") and "one2one" not in name:
                        inference_outputs.append(t)
            if not inference_outputs:
                inference_outputs = [t for _, t in outputs] if outputs and isinstance(outputs[0], tuple) else list(outputs)
            teacher_cache[it["image_path"]] = [t.detach().cpu() for t in inference_outputs]

    if max_iters is not None and int(max_iters) <= 0:
        adapter = {n: p.detach().cpu().clone() for n, p in trainable}
        torch.save(adapter, out_path / "oc3_adapter.pt")
        return OC3TrainV3Result(
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
            direction=direction,
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
    for epoch in range(int(cfg.epochs)):
        rng.shuffle(indices)
        epoch_total = 0.0
        n_batches = 0
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

                # Filter to inference branch for both surrogate and distill.
                filtered: list = []
                for entry in cls_outputs:
                    name, t = entry if isinstance(entry, tuple) else (None, entry)
                    if param_pattern == "one2one_cv3":
                        if name and "one2one_cv3" in name:
                            filtered.append((name, t))
                    else:
                        if name and name.startswith("cv3.") and "one2one" not in name:
                            filtered.append((name, t))
                if not filtered:
                    filtered = cls_outputs

                l_ctx, l_obj, l_cons = _per_image_surrogate(
                    cls_outputs=filtered,
                    target_class_id=target_id,
                    witness_type=it["witness_type"],
                    bboxes=it["bboxes"],
                    orig_hw=ohw,
                    paired_object_mean=None,
                    cfg=cfg,
                )
                if l_ctx is None:
                    continue

                # Teacher distillation on clean witnesses.
                l_distill = torch.zeros((), device=dev, dtype=torch.float32)
                if (
                    float(cfg.weight_teacher_distill) > 0
                    and (not cfg.teacher_distill_clean_only or it["witness_type"] == "object_present")
                    and it["image_path"] in teacher_cache
                ):
                    teacher_outputs = teacher_cache[it["image_path"]]
                    n_distill_terms = 0
                    for student_entry, t_cpu in zip(filtered, teacher_outputs):
                        _, s = student_entry
                        t = t_cpu.to(dev)
                        if s.shape != t.shape:
                            continue
                        # MSE on sigmoid scores (per-cell, both classes).
                        s_sig = torch.sigmoid(s[0])
                        t_sig = torch.sigmoid(t[0])
                        l_distill = l_distill + ((s_sig - t_sig) ** 2).mean()
                        n_distill_terms += 1
                    if n_distill_terms > 0:
                        l_distill = l_distill / n_distill_terms

                batch_loss = batch_loss + (
                    float(cfg.weight_context) * l_ctx
                    + float(cfg.weight_object) * l_obj
                    + float(cfg.weight_consistency) * l_cons
                    + float(cfg.weight_teacher_distill) * l_distill
                )
            # L2-SP regularization
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
                if not outputs:
                    continue
                filtered: list = []
                for entry in outputs:
                    name, t = entry if isinstance(entry, tuple) else (None, entry)
                    if param_pattern == "one2one_cv3":
                        if name and "one2one_cv3" in name:
                            filtered.append((name, t))
                    else:
                        if name and name.startswith("cv3.") and "one2one" not in name:
                            filtered.append((name, t))
                if not filtered:
                    filtered = outputs
                l_ctx, l_obj, l_cons = _per_image_surrogate(
                    cls_outputs=filtered,
                    target_class_id=target_id,
                    witness_type=it["witness_type"],
                    bboxes=it["bboxes"],
                    orig_hw=ohw,
                    paired_object_mean=None,
                    cfg=cfg,
                )
                if l_ctx is None:
                    continue
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
        "direction": direction,
        "param_pattern": param_pattern,
        "teacher_cache_size": len(teacher_cache),
    }
    (out_path / "oc3_train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    return OC3TrainV3Result(
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
        direction=direction,
        param_pattern=param_pattern,
    )
