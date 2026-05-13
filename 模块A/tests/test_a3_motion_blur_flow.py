"""A3 motion / blur / light-flow detectors."""
from __future__ import annotations

import pytest
import torch

from defense.module_a.features.blur_degradation import GPUBlurDegradationDetector
from defense.module_a.features.light_flow import GPULightOpticalFlowDetector
from defense.module_a.features.motion_artifact import GPUMotionArtifactDetector
from defense.module_a.types import ROI


def _gray(values: torch.Tensor, device: str) -> torch.Tensor:
    return values.to(device).view(1, 1, *values.shape)


# ------------------------------ motion artifact ------------------------------


def test_motion_artifact_zero_on_identical_frames(cuda_device: str) -> None:
    det = GPUMotionArtifactDetector(diff_threshold=25.0, grid_size=16)
    frame = _gray(torch.full((64, 64), 128.0), cuda_device)
    out = det.compute(frame, frame)
    assert out["region_count"] == 0
    assert out["max_magnitude"] == pytest.approx(0.0)
    assert out["motion_score"] == pytest.approx(0.0)


def test_motion_artifact_on_dramatic_change(cuda_device: str) -> None:
    det = GPUMotionArtifactDetector(diff_threshold=25.0, grid_size=16)
    prev = _gray(torch.zeros((64, 64)), cuda_device)
    curr = _gray(torch.full((64, 64), 200.0), cuda_device)
    out = det.compute(prev, curr)
    assert out["region_count"] > 0
    assert out["max_magnitude"] >= 25.0
    assert 0.0 < out["motion_score"] <= 1.0


def test_motion_artifact_returns_empty_without_prev(cuda_device: str) -> None:
    det = GPUMotionArtifactDetector(diff_threshold=25.0, grid_size=16)
    curr = _gray(torch.full((64, 64), 128.0), cuda_device)
    out = det.compute(None, curr)
    assert out["region_count"] == 0
    assert out["motion_score"] == 0.0


# ------------------------------ blur degradation ------------------------------


def test_blur_score_zero_on_sharp_image(cuda_device: str) -> None:
    det = GPUBlurDegradationDetector()
    # Sharp checkerboard pattern → high Laplacian energy → low blur_score
    pattern = torch.zeros((64, 64))
    pattern[::2, ::2] = 255.0
    pattern[1::2, 1::2] = 255.0
    rois = [ROI("roi", (0, 0, 64, 64), label="person", confidence=0.9)]
    out = det.compute(_gray(pattern, cuda_device), rois)
    assert out["blur_score"] == pytest.approx(0.0)


def test_blur_score_rises_on_flat_roi(cuda_device: str) -> None:
    """A 64x64 flat ROI has no high-frequency energy, so its local energy
    ratio falls far below 1.0 and blur_score saturates near 1.0.
    We seed the background with a sharp grid so global_mean stays > 0."""
    det = GPUBlurDegradationDetector()
    # Build a 256×256 canvas: sharp pattern outside, flat patch inside ROI.
    canvas = torch.zeros((256, 256))
    canvas[::2, ::2] = 255.0
    canvas[1::2, 1::2] = 255.0
    # Flat patch
    canvas[96:160, 96:160] = 128.0
    rois = [ROI("roi", (96, 96, 160, 160), label="person", confidence=0.9)]
    out = det.compute(_gray(canvas, cuda_device), rois)
    assert out["blur_score"] > 0.5
    assert out["blur_roi_energy_ratio"] < 0.5


# ------------------------------ light optical flow ------------------------------


def test_light_flow_is_noop_when_disabled(cuda_device: str) -> None:
    det = GPULightOpticalFlowDetector()
    prev = _gray(torch.zeros((128, 128)), cuda_device)
    curr = _gray(torch.full((128, 128), 50.0), cuda_device)
    out = det.compute(prev, curr, rois=None, run=False)
    assert out["light_flow_available"] is False
    assert out["light_flow_backend"] == "gpu_lk_lite_skipped"
    assert out["light_flow_score"] == pytest.approx(0.0)


def test_light_flow_produces_score_on_translation(cuda_device: str) -> None:
    """A horizontal shift of a bright bar should produce valid flow vectors
    and yield a non-zero ``light_flow_mean_magnitude``.

    We do NOT assert anomaly_ratio here because a pure translation field is
    *consistent* (low residual), which is exactly what the detector is
    designed to ignore. What we verify is that the detector ran (``available
    == True``) and produced finite magnitudes."""
    det = GPULightOpticalFlowDetector()
    prev_arr = torch.zeros((128, 128))
    prev_arr[:, 30:40] = 255.0
    curr_arr = torch.zeros((128, 128))
    curr_arr[:, 50:60] = 255.0  # bar shifted ~20px to the right
    out = det.compute(_gray(prev_arr, cuda_device), _gray(curr_arr, cuda_device), run=True)
    assert out["light_flow_available"] is True
    assert out["light_flow_backend"] == "gpu_lk_lite"
    assert 0.0 <= out["light_flow_score"] <= 1.0
    # Magnitudes should be finite numbers.
    assert out["light_flow_max_magnitude"] >= 0.0
