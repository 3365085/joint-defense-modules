"""A3b edge-NPU prototype — pure-tensor signals for static-media / screen
spoof detection.

Each signal is implemented as a pytorch function that takes GPU tensors
and returns a GPU tensor (no CPU round-trip, no Python loops over
candidates). Signals:

  (1) Moire FFT ring — screen replay detection via frequency-domain peaks
  (2) Planar flow gap — ROI-inside vs ROI-outside optical flow consistency
  (3) Color uniformity — saturation / hue distribution
  (4) Gradient orientation concentration — rectangular-edge anisotropy

The prototype runs against the 7 sample clips and reports per-signal
distributions on clean vs attacked to decide which signals to keep.

Usage::
    python 探索/a3b_edge_prototype.py

Output lands in ``探索/a3b_edge_prototype_report.json``.

Design rule: every function here must be convertible to ONNX/RKNN with
only static-shape, batch-tensor ops. No ``.item()``, no ``.cpu()``, no
Python branching based on tensor values.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1] / "模块A"
sys.path.insert(0, str(PKG_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from defense.module_a.backends import create_detector_backend  # noqa: E402


# ---------------------------------------------------------------------------
# Shared Sobel kernels (reused from existing detector logic but bound here).
# ---------------------------------------------------------------------------

def _sobel_kernels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3) / 8.0
    return kx, ky


# ---------------------------------------------------------------------------
# (1) Moire FFT ring
# ---------------------------------------------------------------------------

def moire_score(patch: torch.Tensor) -> torch.Tensor:
    """Per-patch frequency-domain moire score.

    patch: (B, 1, H, W) float in [0, 1].
    Returns: (B,) score in [0, 1]. High = screen-like high-frequency peaks.

    Algorithm:
      * rfft2 on the patch.
      * Take log1p magnitude.
      * Compute mean energy in a mid-high-frequency annulus (radius 0.3–0.7
        of Nyquist). Real photos have ~monotonic decay; screens have local
        peaks in this band from Bayer × sub-pixel aliasing.
      * Normalise by the low-band mean so scene luminance doesn't confound.
    """
    B, C, H, W = patch.shape
    # Centered DC: shift is unnecessary for ring energy.
    spec = torch.fft.rfft2(patch, norm="forward")  # (B, C, H, W//2+1)
    mag = torch.log1p(torch.abs(spec))[:, 0]  # (B, H, W//2+1)

    # Build a cached radial distance map.
    yy = torch.arange(H, device=patch.device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(W // 2 + 1, device=patch.device, dtype=torch.float32).view(1, -1)
    # Put DC at (0, 0) corner; FFT is not centred.
    ry = torch.minimum(yy, H - yy) / (H / 2.0)
    r = torch.sqrt(ry * ry + (xx / (W / 2.0)) ** 2)  # (H, W//2+1) in [0, ~sqrt(2)]

    low_band = (r > 0.05) & (r <= 0.3)
    mid_band = (r > 0.3) & (r <= 0.7)

    low_band_f = low_band.float().unsqueeze(0)  # (1, H, W//2+1)
    mid_band_f = mid_band.float().unsqueeze(0)
    low_e = (mag * low_band_f).sum(dim=(-2, -1)) / (low_band_f.sum() + 1e-6)
    mid_e = (mag * mid_band_f).sum(dim=(-2, -1)) / (mid_band_f.sum() + 1e-6)

    # High-band-mean / low-band-mean: screens have elevated mid-band → ratio
    # closer to 1.0 (or even > 1). Natural scenes typically drop by ~3-10×.
    ratio = mid_e / (low_e + 1e-6)  # (B,)
    # Squash to [0, 1] — 0.3 is typical natural decay midpoint, 0.7+ suspicious.
    return torch.clamp((ratio - 0.3) / 0.4, 0.0, 1.0)


# ---------------------------------------------------------------------------
# (2) Planar flow gap
# ---------------------------------------------------------------------------

def planar_flow_gap(
    prev_gray: torch.Tensor,
    curr_gray: torch.Tensor,
    rois_mask: torch.Tensor,
) -> torch.Tensor:
    """ROI-inside vs ROI-outside motion consistency.

    prev_gray, curr_gray: (1, 1, H, W) float in [0, 255].
    rois_mask: (B, 1, H, W) binary — one mask per candidate.

    Returns: (B,) score in [0, 1]. High = motion inside ROI is strongly
    decoupled from motion outside, suggesting a rigid planar object being
    moved in front of a distinct background.

    Algorithm:
      * Compute abs frame diff (cheap).
      * For each ROI mask, compute mean diff inside + mean diff in an
        expanded outside ring.
      * flow_gap = |inside - outside| / (outside + 1e-4).
    """
    diff = torch.abs(curr_gray - prev_gray) / 255.0  # (1, 1, H, W)
    B = rois_mask.shape[0]

    inside_sums = (diff * rois_mask).sum(dim=(-3, -2, -1))  # (B,)
    inside_counts = rois_mask.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    inside_mean = inside_sums / inside_counts

    # Outside ring: dilate the mask by ~20 px via max_pool2d and subtract.
    kernel = 21
    dilated = F.max_pool2d(rois_mask, kernel_size=kernel, stride=1, padding=kernel // 2)
    ring = dilated - rois_mask  # 1 where expanded but not inside
    outside_sums = (diff * ring).sum(dim=(-3, -2, -1))
    outside_counts = ring.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    outside_mean = outside_sums / outside_counts

    # The "planar flow gap" is high when the object moves but the
    # background doesn't (or vice versa). We care about the magnitude of
    # the gap normalised by the overall motion.
    total = (inside_mean + outside_mean + 1e-4)
    gap = torch.abs(inside_mean - outside_mean) / total
    return torch.clamp(gap, 0.0, 1.0)


# ---------------------------------------------------------------------------
# (3) Color uniformity
# ---------------------------------------------------------------------------

def color_uniformity(bgr: torch.Tensor, rois_mask: torch.Tensor) -> torch.Tensor:
    """Saturation concentration inside each ROI.

    bgr: (1, 3, H, W) float in [0, 1]. rois_mask: (B, 1, H, W).

    Returns: (B,) score in [0, 1]. High = ROI has a narrow hue band and
    elevated saturation, typical of a self-emitting display.
    """
    # Compute min/max over channels for V / chroma.
    max_c, _ = bgr.max(dim=1, keepdim=True)  # (1, 1, H, W)
    min_c, _ = bgr.min(dim=1, keepdim=True)
    chroma = max_c - min_c
    saturation = chroma / (max_c + 1e-6)

    sat_t = saturation  # (1, 1, H, W)
    B = rois_mask.shape[0]
    # Mean saturation inside each ROI.
    inside_sums = (sat_t * rois_mask).sum(dim=(-3, -2, -1))
    inside_counts = rois_mask.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    sat_mean = inside_sums / inside_counts

    # Saturation above 0.45 is strong signal; clamp.
    return torch.clamp((sat_mean - 0.25) / 0.35, 0.0, 1.0)


# ---------------------------------------------------------------------------
# (4) Gradient orientation concentration
# ---------------------------------------------------------------------------

def gradient_orientation_score(
    gray: torch.Tensor, rois_mask: torch.Tensor
) -> torch.Tensor:
    """Fraction of gradient magnitude aligned with 0° / 90° axes.

    gray: (1, 1, H, W) float in [0, 1]. rois_mask: (B, 1, H, W).

    Returns: (B,) score in [0, 1]. High = ROI edges are concentrated on
    horizontal / vertical axes, typical of a rectangular screen/paper.
    """
    kx, ky = _sobel_kernels(gray.device)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    # Alignment score: |gx * gy| is high on diagonals, low on horiz/vert.
    # axis_score = 1 - |gx*gy| / (gx^2 + gy^2).
    axis_score = 1.0 - torch.abs(gx * gy) / (gx * gx + gy * gy + 1e-6)  # (1,1,H,W)
    weight = mag * rois_mask  # weighted by edge strength inside ROI
    num = (axis_score * weight).sum(dim=(-3, -2, -1))
    den = weight.sum(dim=(-3, -2, -1)).clamp_min(1e-4)
    return torch.clamp(num / den, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ROI → dense mask builder (no Python loop over pixels, only over ROIs).
# ---------------------------------------------------------------------------

def rois_to_masks(
    rois_xyxy: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    """Build (B, 1, H, W) binary masks, one per ROI. Vectorised."""
    if not rois_xyxy:
        return torch.zeros((0, 1, h, w), device=device, dtype=torch.float32)
    boxes = torch.tensor(rois_xyxy, device=device, dtype=torch.float32)  # (B, 4)
    ys = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)
    x1 = boxes[:, 0].view(-1, 1, 1, 1)
    y1 = boxes[:, 1].view(-1, 1, 1, 1)
    x2 = boxes[:, 2].view(-1, 1, 1, 1)
    y2 = boxes[:, 3].view(-1, 1, 1, 1)
    masks = ((xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)).float()  # (B,1,H,W)
    return masks


def crop_patches(
    gray: torch.Tensor,
    rois_xyxy: list[tuple[int, int, int, int]],
    size: int = 64,
) -> torch.Tensor:
    """Crop (B, 1, size, size) patches from ROIs via grid_sample, no loop."""
    if not rois_xyxy:
        return torch.zeros((0, 1, size, size), device=gray.device)
    B = len(rois_xyxy)
    _, _, H, W = gray.shape
    # Build an affine grid per ROI.
    theta = torch.zeros((B, 2, 3), device=gray.device, dtype=torch.float32)
    for i, (x1, y1, x2, y2) in enumerate(rois_xyxy):
        cx = (x1 + x2) * 0.5 / W * 2.0 - 1.0
        cy = (y1 + y2) * 0.5 / H * 2.0 - 1.0
        sx = (x2 - x1) / W
        sy = (y2 - y1) / H
        theta[i, 0, 0] = sx
        theta[i, 1, 1] = sy
        theta[i, 0, 2] = cx
        theta[i, 1, 2] = cy
    grid = F.affine_grid(theta, (B, 1, size, size), align_corners=False)
    batched_gray = gray.expand(B, -1, -1, -1)
    return F.grid_sample(batched_gray, grid, align_corners=False)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


@dataclass
class _ClipStats:
    clip: str
    frames: int
    n_rois_mean: float
    moire_mean: float
    flow_gap_mean: float
    color_mean: float
    gradient_mean: float
    moire_max: float
    flow_gap_max: float
    color_max: float
    gradient_max: float
    forward_ms_mean: float
    forward_ms_p95: float


def run_clip(
    backend: Any,
    clip_path: Path,
    device: torch.device,
    max_rois: int = 4,
) -> _ClipStats:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip_path}")

    moire_vals: list[float] = []
    flow_vals: list[float] = []
    color_vals: list[float] = []
    grad_vals: list[float] = []
    roi_counts: list[int] = []
    forward_ms: list[float] = []
    prev_gray = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_640 = cv2.resize(frame, (640, 640))
            detections = backend.predict(frame_640)
            rois_xyxy = detections.boxes[:max_rois]
            # If no YOLO detections, use 4 quadrant pseudo-ROIs so signals
            # always have something to aggregate. Edge runtime will do the
            # same (fixed static grid).
            if not rois_xyxy:
                rois_xyxy = [
                    (0, 0, 320, 320),
                    (320, 0, 640, 320),
                    (0, 320, 320, 640),
                    (320, 320, 640, 640),
                ]

            bgr_t = (
                torch.from_numpy(frame_640).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            )  # (1, 3, 640, 640)
            gray_t = (0.114 * bgr_t[:, 0:1] + 0.587 * bgr_t[:, 1:2] + 0.299 * bgr_t[:, 2:3])
            gray_255 = gray_t * 255.0

            started = time.perf_counter()
            masks = rois_to_masks(rois_xyxy, 640, 640, device)
            patches = crop_patches(gray_t, rois_xyxy, size=64)

            moire = moire_score(patches)
            if prev_gray is not None:
                flow_gap = planar_flow_gap(prev_gray, gray_255, masks)
            else:
                flow_gap = torch.zeros(masks.shape[0], device=device)
            color = color_uniformity(bgr_t, masks)
            grad = gradient_orientation_score(gray_t, masks)
            torch.cuda.synchronize()
            forward_ms.append((time.perf_counter() - started) * 1000.0)

            # Aggregate max-across-ROIs for per-frame signature.
            moire_vals.append(float(moire.max().item()))
            flow_vals.append(float(flow_gap.max().item()))
            color_vals.append(float(color.max().item()))
            grad_vals.append(float(grad.max().item()))
            roi_counts.append(len(rois_xyxy))
            prev_gray = gray_255
            frame_idx += 1
    finally:
        cap.release()

    def _mean(v):
        return float(np.mean(v)) if v else 0.0

    def _p95(v):
        return float(np.percentile(v, 95)) if v else 0.0

    return _ClipStats(
        clip=clip_path.name,
        frames=frame_idx,
        n_rois_mean=_mean(roi_counts),
        moire_mean=_mean(moire_vals),
        flow_gap_mean=_mean(flow_vals),
        color_mean=_mean(color_vals),
        gradient_mean=_mean(grad_vals),
        moire_max=max(moire_vals) if moire_vals else 0.0,
        flow_gap_max=max(flow_vals) if flow_vals else 0.0,
        color_max=max(color_vals) if color_vals else 0.0,
        gradient_max=max(grad_vals) if grad_vals else 0.0,
        forward_ms_mean=_mean(forward_ms),
        forward_ms_p95=_p95(forward_ms),
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    device = torch.device("cuda:0")

    config_path = PKG_ROOT / "experiments" / "configs" / "module_a_baseline.yaml"
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    backend = create_detector_backend(config, PKG_ROOT)

    report: dict[str, Any] = {"signals": {}, "clips": []}
    samples_dir = PKG_ROOT / "samples"
    for clip in sorted(samples_dir.glob("*.mp4")):
        print(f"[run] {clip.name}", flush=True)
        stats = run_clip(backend, clip, device)
        report["clips"].append(stats.__dict__)

    # Quick separation scoring: for each signal, compute
    # (mean of attacked clips) - (clean clip mean). Positive = discriminative.
    clean = [c for c in report["clips"] if c["clip"] == "clean_baseline.mp4"]
    if clean:
        clean_means = {
            "moire": clean[0]["moire_mean"],
            "flow_gap": clean[0]["flow_gap_mean"],
            "color": clean[0]["color_mean"],
            "gradient": clean[0]["gradient_mean"],
        }
        attack_clips = [c for c in report["clips"] if c["clip"] != "clean_baseline.mp4"]
        for sig in ("moire", "flow_gap", "color", "gradient"):
            attack_mean = float(np.mean([c[f"{sig}_mean"] for c in attack_clips]))
            report["signals"][sig] = {
                "clean_mean": clean_means[sig],
                "attack_mean": attack_mean,
                "separation": attack_mean - clean_means[sig],
            }

    out = Path(__file__).resolve().parent / "a3b_edge_prototype_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Pretty-print
    print("\n" + "=" * 80)
    print(f"{'clip':<42} {'moire':>8} {'flow':>8} {'color':>8} {'grad':>8} {'fwd_p95':>8}")
    for c in report["clips"]:
        print(
            f"{c['clip']:<42} "
            f"{c['moire_mean']:>8.3f} {c['flow_gap_mean']:>8.3f} "
            f"{c['color_mean']:>8.3f} {c['gradient_mean']:>8.3f} "
            f"{c['forward_ms_p95']:>8.2f}"
        )
    if report["signals"]:
        print("\nsignal separation (attack_mean − clean_mean):")
        for sig, v in report["signals"].items():
            print(f"  {sig:<10} {v['separation']:+.3f}  "
                  f"(clean={v['clean_mean']:.3f}, attack={v['attack_mean']:.3f})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
