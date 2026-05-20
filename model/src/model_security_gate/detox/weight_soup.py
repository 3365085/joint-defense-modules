from __future__ import annotations

"""Weight-space last-mile recovery for rollback-safe YOLO detox.

This module implements a conservative *last-mile* candidate generator.  It is
intended for the common state seen in the mask_bd_v2 run: an accepted detox
checkpoint already passes the external ASR smoke gate, but clean mAP is slightly
above a stricter certificate tolerance.  Instead of doing another free-form
fine-tune that can drift back into the trigger basin, we generate small
weight-space interpolations from the defended checkpoint toward a trusted clean
anchor and let the existing external hard-suite / clean mAP gates select.

The module deliberately does not claim that interpolation itself is a proof;
its job is to produce low-drift candidates that are cheap to evaluate and safe
to roll back.
"""

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from model_security_gate.utils.io import write_json
from model_security_gate.utils.torchvision_compat import patch_torchvision_nms_fallback


@dataclass
class WeightSoupCandidate:
    alpha: float
    anchor_model: str
    output_model: str
    n_total_tensors: int
    n_interpolated_tensors: int
    n_skipped_tensors: int
    n_filtered_tensors: int = 0
    include_key_patterns: list[str] = field(default_factory=list)
    exclude_key_patterns: list[str] = field(default_factory=list)
    skipped_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeightSoupBuildResult:
    base_model: str
    candidates: list[WeightSoupCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {"base_model": self.base_model, "candidates": [c.to_dict() for c in self.candidates]}


def _load_yolo_module(path: str | Path):
    """Load a YOLO wrapper only when needed.

    Importing Ultralytics at module import time makes lightweight tests and CI
    brittle.  Keep it lazy so pure state-dict tests can run without the heavy
    dependency.
    """

    patch_torchvision_nms_fallback()
    from ultralytics import YOLO

    yolo = YOLO(str(path))
    if not hasattr(yolo, "model") or not isinstance(yolo.model, torch.nn.Module):
        raise TypeError(f"{path} did not load as an Ultralytics model with a torch module")
    return yolo


def load_state_dict_any(path: str | Path) -> dict[str, torch.Tensor]:
    """Return a state dict from either a YOLO checkpoint or a plain torch file."""

    path = Path(path)
    # Try Ultralytics first for real YOLO .pt files.  It knows how to resolve
    # serialized DetectionModel classes.
    try:
        yolo = _load_yolo_module(path)
        return {str(k): v.detach().cpu().clone() for k, v in yolo.model.state_dict().items()}
    except Exception:
        pass

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return {str(k): v.detach().cpu().clone() for k, v in obj.state_dict().items()}
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict"):
            maybe = obj.get(key)
            if isinstance(maybe, Mapping):
                return {str(k): v.detach().cpu().clone() for k, v in maybe.items() if torch.is_tensor(v)}
        for key in ("ema", "model"):
            maybe = obj.get(key)
            if isinstance(maybe, torch.nn.Module):
                return {str(k): v.detach().cpu().clone() for k, v in maybe.state_dict().items()}
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return {str(k): v.detach().cpu().clone() for k, v in obj.items()}
    raise TypeError(f"Unsupported checkpoint format for {path}")


def interpolate_state_dicts(
    base_state: Mapping[str, torch.Tensor],
    anchor_state: Mapping[str, torch.Tensor],
    alpha: float,
    include_key_patterns: Sequence[str] | None = None,
    exclude_key_patterns: Sequence[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Interpolate matching floating-point tensors.

    ``alpha=0`` returns the base state; ``alpha=1`` returns the anchor state for
    matched tensors.  Non-floating or shape-mismatched tensors are copied from
    the base state to avoid corrupting BatchNorm counters, anchors, or metadata.
    Optional include/exclude patterns limit interpolation to selected keys.  A
    pattern with shell wildcards uses ``fnmatch``; otherwise it is treated as a
    substring.  Filtered keys are also copied from the base state.
    """

    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    include_patterns = tuple(str(p) for p in (include_key_patterns or ()) if str(p))
    exclude_patterns = tuple(str(p) for p in (exclude_key_patterns or ()) if str(p))
    out: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    n_interp = 0
    n_filtered = 0
    for key, base_tensor in base_state.items():
        anchor_tensor = anchor_state.get(key)
        filtered = not _key_is_selected(str(key), include_patterns, exclude_patterns)
        if (
            not filtered
            and anchor_tensor is not None
            and torch.is_tensor(base_tensor)
            and torch.is_tensor(anchor_tensor)
            and base_tensor.shape == anchor_tensor.shape
            and torch.is_floating_point(base_tensor)
            and torch.is_floating_point(anchor_tensor)
        ):
            out[key] = ((1.0 - a) * base_tensor.float() + a * anchor_tensor.float()).to(dtype=base_tensor.dtype)
            n_interp += 1
        else:
            out[key] = base_tensor.detach().cpu().clone()
            if filtered:
                n_filtered += 1
            if len(skipped) < 20:
                skipped.append(str(key))
    return out, {
        "n_total_tensors": len(base_state),
        "n_interpolated_tensors": n_interp,
        "n_skipped_tensors": len(base_state) - n_interp,
        "n_filtered_tensors": n_filtered,
        "include_key_patterns": list(include_patterns),
        "exclude_key_patterns": list(exclude_patterns),
        "skipped_examples": skipped,
    }


def _pattern_matches(key: str, pattern: str) -> bool:
    if any(ch in pattern for ch in "*?[]"):
        return fnmatchcase(key, pattern)
    return pattern in key


def _key_is_selected(key: str, include_patterns: Sequence[str], exclude_patterns: Sequence[str]) -> bool:
    included = not include_patterns or any(_pattern_matches(key, pattern) for pattern in include_patterns)
    excluded = any(_pattern_matches(key, pattern) for pattern in exclude_patterns)
    return included and not excluded


def save_state_into_yolo_template(
    template_model: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    out_path: str | Path,
    strict: bool = False,
) -> Path:
    """Save ``state_dict`` using ``template_model``'s Ultralytics wrapper.

    Real YOLO checkpoints contain metadata and model classes that are easiest to
    preserve through ``YOLO.save``.  Tests that do not have Ultralytics can call
    ``save_plain_state_dict`` instead.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yolo = _load_yolo_module(template_model)
    missing, unexpected = yolo.model.load_state_dict(dict(state_dict), strict=bool(strict))
    if strict and (missing or unexpected):
        raise RuntimeError(f"state_dict load mismatch: missing={missing}, unexpected={unexpected}")
    yolo.save(str(out_path))
    return out_path


def save_plain_state_dict(state_dict: Mapping[str, torch.Tensor], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({str(k): v.detach().cpu().clone() for k, v in state_dict.items()}, out_path)
    return out_path


def build_weight_soup_candidates(
    base_model: str | Path,
    anchor_models: Sequence[str | Path],
    out_dir: str | Path,
    alphas: Sequence[float] = (0.01, 0.02, 0.04, 0.06, 0.08),
    use_yolo_template: bool = True,
    include_key_patterns: Sequence[str] | None = None,
    exclude_key_patterns: Sequence[str] | None = None,
    candidate_suffix: str = "",
) -> WeightSoupBuildResult:
    """Build a grid of low-drift interpolation candidates.

    The YOLO wrapper is loaded at most once.  Re-loading the template for every
    alpha can make a 5-10 candidate sweep painfully slow on CPU-only machines.
    """

    base_model = Path(base_model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_state = load_state_dict_any(base_model)
    template_yolo = None
    if use_yolo_template:
        try:
            template_yolo = _load_yolo_module(base_model)
        except Exception:
            template_yolo = None
    candidates: list[WeightSoupCandidate] = []
    for anchor_index, anchor in enumerate(anchor_models, 1):
        anchor = Path(anchor)
        anchor_state = load_state_dict_any(anchor)
        for alpha in alphas:
            state, meta = interpolate_state_dicts(
                base_state,
                anchor_state,
                float(alpha),
                include_key_patterns=include_key_patterns,
                exclude_key_patterns=exclude_key_patterns,
            )
            suffix = f"_{candidate_suffix}" if candidate_suffix else ""
            stem = f"anchor{anchor_index:02d}_{anchor.stem}{suffix}_alpha{str(float(alpha)).replace('.', 'p')}"
            out_path = out_dir / f"{stem}.pt"
            if template_yolo is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                template_yolo.model.load_state_dict(dict(state), strict=False)
                template_yolo.save(str(out_path))
            else:
                save_plain_state_dict(state, out_path)
            candidates.append(
                WeightSoupCandidate(
                    alpha=float(alpha),
                    anchor_model=str(anchor),
                    output_model=str(out_path),
                    n_total_tensors=int(meta["n_total_tensors"]),
                    n_interpolated_tensors=int(meta["n_interpolated_tensors"]),
                    n_skipped_tensors=int(meta["n_skipped_tensors"]),
                    n_filtered_tensors=int(meta.get("n_filtered_tensors", 0)),
                    include_key_patterns=list(meta.get("include_key_patterns") or []),
                    exclude_key_patterns=list(meta.get("exclude_key_patterns") or []),
                    skipped_examples=list(meta.get("skipped_examples") or []),
                )
            )
    result = WeightSoupBuildResult(base_model=str(base_model), candidates=candidates)
    write_json(out_dir / "weight_soup_candidates_manifest.json", result.to_dict())
    return result


def parse_alpha_grid(spec: str | Sequence[float]) -> list[float]:
    if not isinstance(spec, str):
        return [float(x) for x in spec]
    raw = spec.strip()
    if not raw:
        return []
    if ":" in raw:
        parts = [float(x) for x in raw.split(":")]
        if len(parts) != 3:
            raise ValueError("alpha range must be start:stop:step")
        start, stop, step = parts
        if step <= 0:
            raise ValueError("alpha step must be positive")
        out = []
        x = start
        while x <= stop + 1e-12:
            out.append(round(float(x), 10))
            x += step
        return out
    return [float(x.strip()) for x in raw.split(",") if x.strip()]
