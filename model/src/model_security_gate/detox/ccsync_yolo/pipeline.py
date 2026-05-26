"""End-to-end CCSync on YOLO pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    HAS_TORCH = False

from ..ccsync.purify import purify_weights, channel_alpha_from_sync
from ..ccsync.schema import CCSyncConfig, CCSyncReport
from ..ccsync.sync_score import (
    compute_excess_correlation,
    sync_score,
    cohort_topk,
)


@dataclass(frozen=True)
class CCSyncYoloConfig:
    """Hyperparameters for CCSync on YOLO."""

    target_class_id: int = 0
    imgsz: int = 416
    fire_threshold: float = 0.10  # min YOLO target sigmoid for "this cell is interesting"
    max_cells_per_pool: int = 8000
    n_baseline_images: int = 80   # clean target-absent images for clean pool

    # CCSync hyperparameters
    tau: float = 0.10
    alpha_max: float = 0.5
    beta_softmax: float = 0.3

    head_index: int = -1
    cv3_pattern: str = "cv3"

    # Whether to also adjust the cls module's bias on the same channels
    # (recommended; biases are channel-aligned).
    adjust_bias: bool = True


@dataclass
class CCSyncYoloResult:
    """Result of one CCSync on YOLO purification run."""

    accepted: bool
    purified_path: str
    config_path: str
    report_path: str
    n_scales: int
    in_channels_per_scale: List[int]
    n_trig_cells_per_scale: List[int]
    n_clean_cells_per_scale: List[int]
    sigma_max_per_scale: List[float]
    n_active_per_scale: List[int]
    alpha_max_per_scale: List[float]
    pathway_specificity_per_scale: List[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_device(device: Optional[str | int]) -> str:
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return f"cuda:{int(device)}"
    return str(device)


def _load_image_tensor(path: str, imgsz: int, device: str) -> Tensor:
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h0, w0 = img.shape[:2]
    s = min(imgsz / w0, imgsz / h0)
    nw, nh = int(round(w0 * s)), int(round(h0 * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


class _YoloHookContext:
    """Forward hook helper.  Captures (input feature, output cls map)
    pairs for each cls-head module on every forward."""

    def __init__(self, yolo_model, cls_modules: List[nn.Module]):
        self.cls_modules = cls_modules
        self.cls_inputs: List[Tensor] = []
        self.cls_outputs: List[Tensor] = []
        self.handles = []

        def _make_hook(idx):
            def _hook(module, inputs, output):
                self.cls_inputs.append(inputs[0])
                self.cls_outputs.append(output)
            return _hook

        for i, mod in enumerate(cls_modules):
            self.handles.append(mod.register_forward_hook(_make_hook(i)))

    def reset(self):
        self.cls_inputs.clear()
        self.cls_outputs.clear()

    def release(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _find_cls_modules(yolo_model, *, head_index: int = -1,
                      cv3_pattern: str = "cv3") -> Tuple[List[nn.Module], List[str]]:
    head = yolo_model.model[head_index]
    cls_modules: List[nn.Module] = []
    cls_names: List[str] = []
    for name, mod in head.named_modules():
        if (name.startswith(cv3_pattern + ".") and "one2one" not in name
            and len(name.split(".")) == 2):
            cls_modules.append(mod)
            cls_names.append(name)
    if not cls_modules:
        raise RuntimeError(f"could not find {cv3_pattern}.* in YOLO head")
    return cls_modules, cls_names


def _collect_pool(
    yolo_model,
    cls_modules: List[nn.Module],
    image_paths: List[str],
    cfg: CCSyncYoloConfig,
    device: str,
    *,
    only_firing_cells: bool = False,
) -> List[Tensor]:
    """Run YOLO forward on each image, harvest per-FPN-scale per-cell
    activations.  If ``only_firing_cells`` is True, keep cells whose
    YOLO target sigmoid > cfg.fire_threshold (this is the trigger
    pool -- we want the cells responsible for the backdoor).
    Otherwise keep all cells (clean-baseline pool).

    Returns a list of (N, C) tensors, one per scale.
    """
    ctx = _YoloHookContext(yolo_model, cls_modules)
    n_scales = len(cls_modules)
    pool: List[List[Tensor]] = [[] for _ in range(n_scales)]
    try:
        for img_p in image_paths:
            try:
                img_t = _load_image_tensor(img_p, cfg.imgsz, device)
            except FileNotFoundError:
                continue
            ctx.reset()
            with torch.no_grad():
                _ = yolo_model.model(img_t)
            for s in range(n_scales):
                feat_map = ctx.cls_inputs[s]      # (1, C, H, W)
                cls_map = ctx.cls_outputs[s]      # (1, n_classes, H, W)
                _, C, H, W = feat_map.shape
                feat_flat = feat_map[0].permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
                if only_firing_cells:
                    tgt_sigmoid = torch.sigmoid(cls_map[0, cfg.target_class_id])
                    fire_mask = (tgt_sigmoid.reshape(-1) > cfg.fire_threshold)
                    cells = feat_flat[fire_mask]
                else:
                    cells = feat_flat
                if cells.shape[0] > 0:
                    pool[s].append(cells.detach().cpu())
    finally:
        ctx.release()
    out = []
    for s in range(n_scales):
        if pool[s]:
            cat = torch.cat(pool[s], dim=0)
            if cat.shape[0] > cfg.max_cells_per_pool:
                idx = torch.randperm(cat.shape[0])[: cfg.max_cells_per_pool]
                cat = cat[idx]
            out.append(cat)
        else:
            out.append(torch.empty(0))
    return out


def _find_cls_module_in_state(
    state_dict_p: Dict[str, Tensor],
    state_dict_c: Dict[str, Tensor],
    cls_name: str,
    *,
    head_index: int = -1,
) -> List[Tuple[str, str]]:
    """Find the (Conv2d weight, Conv2d bias) keys in both checkpoints
    that correspond to a head.cv3.X module.

    YOLO Ultralytics state dicts use keys like
    'model.23.cv3.0.0.weight' for cv3.0's conv blocks.  We look for
    ANY conv weight whose name contains '.cv3.{idx}.' and is a Conv2d
    weight tensor (4D), restricted to the chosen head index.

    Returns a list of (weight_key, bias_key_or_empty) pairs in the
    same order they would be found by named_modules().
    """
    # Identify the head module index.  In Ultralytics, head_index=-1
    # means 'last module', which corresponds to model.{last_idx}.cv3.X
    # We discover the actual numerical head index by scanning keys.
    cv3_weights = [k for k in state_dict_p if ".cv3." in k and k.endswith(".weight")
                    and state_dict_p[k].dim() == 4]
    # The head is typically model.X where X is the LAST integer with .cv3.
    head_indices = sorted({int(k.split(".")[1]) for k in cv3_weights
                            if k.split(".")[1].isdigit()})
    if not head_indices:
        return []
    head_idx_num = head_indices[-1]  # last head index; -1 means "last"
    pairs: List[Tuple[str, str]] = []
    # Within head, cls_name = "cv3.X"; find ALL conv weights belonging
    # to that branch (typically cv3.X.0 / cv3.X.1 / cv3.X.2 in
    # YOLO26/v8 style).
    cv3_x = cls_name  # e.g. "cv3.0"
    # Scan keys in order
    for k in sorted(state_dict_p.keys()):
        if k.startswith(f"model.{head_idx_num}.{cv3_x}.") and k.endswith(".weight"):
            if state_dict_p[k].dim() == 4 and k in state_dict_c:
                bias_k = k[: -len(".weight")] + ".bias"
                bias_in_both = bias_k in state_dict_p and bias_k in state_dict_c
                pairs.append((k, bias_k if bias_in_both else ""))
    return pairs


def purify_yolo_with_ccsync(
    *,
    yolo_poisoned_path: str,
    yolo_clean_path: str,
    trigger_image_paths: List[str],
    baseline_image_paths: List[str],
    out_dir: str,
    cfg: Optional[CCSyncYoloConfig] = None,
    device: Optional[str | int] = None,
) -> CCSyncYoloResult:
    """Compute CCSync sync scores on the poisoned YOLO and emit a
    purified checkpoint that interpolates poisoned <-> clean weights
    on a per-channel basis driven by σ.

    The poisoned model's CONV-WEIGHT tensors of the cls head are
    interpolated.  All other weights are kept from the poisoned model
    (CCSync targets ONLY the head's classification path; other
    weights stay frozen).

    Args:
        yolo_poisoned_path: path to poisoned .pt
        yolo_clean_path:    path to clean-baseline .pt
        trigger_image_paths: list of triggered-attack image paths
                              (target-absent images that fire on the
                              poisoned model)
        baseline_image_paths: list of clean target-absent images that
                              do NOT fire on the poisoned model (or
                              whose firing rate is the natural
                              background rate)
        out_dir: where to save the purified checkpoint + reports
        cfg: hyperparameters
        device: torch device

    Returns:
        CCSyncYoloResult with paths and per-scale diagnostic numbers.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required")
    cfg = cfg or CCSyncYoloConfig()
    device = _resolve_device(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    yolo = YOLO(yolo_poisoned_path)
    m = yolo.model
    m.eval()
    m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)

    cls_modules, cls_names = _find_cls_modules(
        m, head_index=cfg.head_index, cv3_pattern=cfg.cv3_pattern,
    )
    n_scales = len(cls_modules)
    print(f"[CCSync-YOLO] discovered {n_scales} cls modules: {cls_names}")

    # ---- pool collection ----
    print(f"[CCSync-YOLO] collecting trigger pool from {len(trigger_image_paths)} images ...")
    t0 = time.time()
    trig_pool = _collect_pool(yolo, cls_modules, trigger_image_paths,
                               cfg, device, only_firing_cells=True)
    print(f"[CCSync-YOLO] collecting clean pool from {len(baseline_image_paths)} images ...")
    clean_pool = _collect_pool(yolo, cls_modules,
                                 baseline_image_paths[: cfg.n_baseline_images],
                                 cfg, device, only_firing_cells=False)
    print(f"[CCSync-YOLO] pool collection elapsed: {time.time() - t0:.1f}s")
    in_channels_per_scale = [int(t.shape[1]) if t.dim() == 2 else 0
                              for t in trig_pool]
    n_trig_cells = [int(t.shape[0]) if t.dim() == 2 else 0 for t in trig_pool]
    n_clean_cells = [int(t.shape[0]) if t.dim() == 2 else 0 for t in clean_pool]
    print(f"[CCSync-YOLO] scales channels={in_channels_per_scale} "
          f"trig_cells={n_trig_cells} clean_cells={n_clean_cells}")

    # ---- per-scale sync + alpha ----
    scale_alpha: List[Tensor] = []
    scale_reports: List[Dict[str, Any]] = []
    for s in range(n_scales):
        tp = trig_pool[s]
        cp = clean_pool[s]
        if tp.shape[0] < 5 or cp.shape[0] < 5:
            print(f"[CCSync-YOLO] scale {s}: insufficient pool "
                  f"(trig={tp.shape[0]}, clean={cp.shape[0]}); skipping")
            scale_alpha.append(torch.zeros(in_channels_per_scale[s] or 1))
            scale_reports.append({"skipped": True})
            continue
        S = compute_excess_correlation(tp, cp, standardise=True)
        sigma = sync_score(S, tau=cfg.tau)
        alpha = channel_alpha_from_sync(
            sigma, alpha_max=cfg.alpha_max, beta_softmax=cfg.beta_softmax,
        )
        # diagnostics
        sigma_np = sigma.numpy()
        topk = cohort_topk(sigma, k=max(1, len(sigma_np) // 8))
        non_cohort = [i for i in range(len(sigma_np)) if i not in set(topk)]
        sigma_p95_other = float(np.percentile(sigma_np[non_cohort], 95)) if non_cohort else 0.0
        sigma_min_cohort = float(min(sigma_np[i] for i in topk))
        specificity = sigma_min_cohort / max(1e-6, sigma_p95_other)
        scale_alpha.append(alpha)
        scale_reports.append({
            "skipped": False,
            "sigma_max": float(sigma.max().item()),
            "sigma_median": float(sigma.median().item()),
            "sigma_p95_other": sigma_p95_other,
            "sigma_min_cohort": sigma_min_cohort,
            "pathway_specificity": specificity,
            "n_active": int((sigma > sigma_p95_other).sum().item()),
            "alpha_max": float(alpha.max().item()),
            "alpha_mean": float(alpha.mean().item()),
            "alpha_min": float(alpha.min().item()),
            "cohort_indices_top16":
                cohort_topk(sigma, k=min(16, len(sigma_np))),
        })
        print(f"[CCSync-YOLO] scale {s}: sigma max={float(sigma.max()):.3f} "
              f"specificity={specificity:.2f} "
              f"alpha [min={float(alpha.min()):.3f}, "
              f"mean={float(alpha.mean()):.3f}, "
              f"max={float(alpha.max()):.3f}]")

    # ---- weight purification ----
    print(f"[CCSync-YOLO] applying per-channel weight interpolation ...")
    state_p = m.state_dict()
    # Load clean state
    yolo_clean = YOLO(yolo_clean_path)
    state_c = yolo_clean.model.state_dict()

    purified_state = {k: v.clone() for k, v in state_p.items()}
    for s, cls_name in enumerate(cls_names):
        if scale_reports[s].get("skipped"):
            continue
        alpha = scale_alpha[s].to(device)
        pairs = _find_cls_module_in_state(state_p, state_c, cls_name,
                                            head_index=cfg.head_index)
        if not pairs:
            print(f"[CCSync-YOLO] scale {s}: no matching weight keys found")
            continue
        # The output channel of the FIRST conv in the cv3.X chain has
        # the same C as feat_map's input channel?  Actually the FEATURE
        # we hook is the cv3.X module's INPUT, whose channel count is
        # the BACKBONE/NECK output channel count (= input_channels of
        # cv3.X.0).  Conv2d's weight has shape (out, in, kh, kw).
        # We interpolate along axis=1 (input channel) for the FIRST
        # conv, and along axis=0 (output channel) for the LAST conv.
        # In practice we apply per-channel α along the INPUT axis of
        # each conv that consumes the captured feature.  The first
        # conv in the cv3.X branch has weight shape (out, C, kh, kw)
        # where C matches feat_map's channel count.
        for w_key, b_key in pairs:
            W_p = state_p[w_key].to(device)
            W_c = state_c[w_key].to(device)
            if W_p.shape != W_c.shape:
                print(f"[CCSync-YOLO] shape mismatch on {w_key}: "
                      f"{W_p.shape} vs {W_c.shape}; skipping")
                continue
            # Decide channel axis: if input-channel axis (1) matches
            # alpha length, interpolate along 1; else if output-axis
            # (0) matches, along 0; else broadcast uniform alpha mean.
            in_C = W_p.shape[1]
            out_C = W_p.shape[0]
            if alpha.shape[0] == in_C:
                W_purified = purify_weights(W_p, W_c, alpha, channel_axis=1)
                axis_used = 1
            elif alpha.shape[0] == out_C:
                W_purified = purify_weights(W_p, W_c, alpha, channel_axis=0)
                axis_used = 0
            else:
                # Fallback: use mean alpha as a global Soup-like blend
                a_mean = float(alpha.mean().item())
                W_purified = (1 - a_mean) * W_p + a_mean * W_c
                axis_used = -1
            purified_state[w_key] = W_purified.to(state_p[w_key].dtype).cpu()
            # Bias: only adjust if the bias's shape matches alpha (output channel)
            if cfg.adjust_bias and b_key:
                b_p = state_p[b_key].to(device)
                b_c = state_c[b_key].to(device)
                if alpha.shape[0] == b_p.shape[0]:
                    b_purified = (1 - alpha.clamp(0.0, 1.0)) * b_p + alpha.clamp(0.0, 1.0) * b_c
                    purified_state[b_key] = b_purified.to(state_p[b_key].dtype).cpu()
            print(f"[CCSync-YOLO]   purified {w_key} along axis {axis_used}")

    # ---- save merged checkpoint ----
    purified_pt = out_path / "ccsync_purified.pt"
    # Build a full Ultralytics-compatible blob: copy the original .pt
    # blob (with metadata + model config) and overwrite state_dict.
    ckpt_p = torch.load(yolo_poisoned_path, map_location="cpu", weights_only=False)
    if "model" in ckpt_p and hasattr(ckpt_p["model"], "state_dict"):
        ckpt_p["model"].load_state_dict(purified_state)
    else:
        ckpt_p["state_dict"] = purified_state
    # Annotate
    ckpt_p["ccsync_metadata"] = {
        "purified_from": str(yolo_poisoned_path),
        "clean_anchor": str(yolo_clean_path),
        "tau": float(cfg.tau),
        "alpha_max": float(cfg.alpha_max),
        "beta_softmax": float(cfg.beta_softmax),
    }
    torch.save(ckpt_p, purified_pt)
    print(f"[CCSync-YOLO] purified checkpoint -> {purified_pt}")

    # ---- save reports ----
    cfg_path = out_path / "ccsync_config.json"
    cfg_path.write_text(json.dumps({
        "config": asdict(cfg),
        "yolo_poisoned_path": str(yolo_poisoned_path),
        "yolo_clean_path": str(yolo_clean_path),
        "in_channels_per_scale": in_channels_per_scale,
        "n_trig_cells_per_scale": n_trig_cells,
        "n_clean_cells_per_scale": n_clean_cells,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rep_path = out_path / "ccsync_report.json"
    rep_path.write_text(json.dumps({
        "scales": scale_reports,
        "alpha_per_scale": [a.tolist() for a in scale_alpha],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    accepted = all(
        not r.get("skipped") and r.get("pathway_specificity", 0) > 1.5
        for r in scale_reports
    )

    return CCSyncYoloResult(
        accepted=bool(accepted),
        purified_path=str(purified_pt),
        config_path=str(cfg_path),
        report_path=str(rep_path),
        n_scales=int(n_scales),
        in_channels_per_scale=in_channels_per_scale,
        n_trig_cells_per_scale=n_trig_cells,
        n_clean_cells_per_scale=n_clean_cells,
        sigma_max_per_scale=[r.get("sigma_max", 0.0) for r in scale_reports],
        n_active_per_scale=[r.get("n_active", 0) for r in scale_reports],
        alpha_max_per_scale=[r.get("alpha_max", 0.0) for r in scale_reports],
        pathway_specificity_per_scale=[
            r.get("pathway_specificity", 0.0) for r in scale_reports
        ],
    )
