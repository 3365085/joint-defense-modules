"""A1 overexposure detector — binary glare indicator.

Contract being validated (架构说明 §三 A1):
  * ``ratio`` is the share of pixels >= 250 in the grayscale tensor.
  * ``underexposed_ratio`` is the share of pixels <= 5.
  * ``is_glare`` flips to True iff ``ratio >= threshold``.

Tested on fabricated tensors rather than real frames: the detector only
reads saturation statistics, so a small synthetic canvas is enough to
exercise both "clean", "edge case" and "glare" regimes.
"""
from __future__ import annotations

import pytest
import torch

from defense.module_a.features.overexposure import GPUOverexposureDetector


@pytest.fixture
def det(cuda_device: str) -> GPUOverexposureDetector:
    return GPUOverexposureDetector(threshold=0.06)


def _gray(values: torch.Tensor, device: str) -> torch.Tensor:
    return values.to(device).view(1, 1, *values.shape)


def test_uniform_midgray_has_zero_glare(cuda_device: str, det: GPUOverexposureDetector) -> None:
    gray = _gray(torch.full((128, 128), 128.0), cuda_device)
    out = det.compute(gray)
    assert out["ratio"] == pytest.approx(0.0)
    assert out["underexposed_ratio"] == pytest.approx(0.0)
    assert out["is_glare"] is False


def test_half_saturated_triggers_glare(cuda_device: str, det: GPUOverexposureDetector) -> None:
    # 50% of pixels at 255 (>= 250) → ratio = 0.5, well above 0.06 threshold
    flat = torch.zeros((128, 128))
    flat[:64, :] = 255.0
    gray = _gray(flat, cuda_device)
    out = det.compute(gray)
    assert out["ratio"] == pytest.approx(0.5, abs=1e-4)
    assert out["is_glare"] is True


def test_just_below_threshold_keeps_glare_off(
    cuda_device: str, det: GPUOverexposureDetector
) -> None:
    # 5% of pixels saturated → below the 6% threshold, must NOT trigger.
    flat = torch.zeros((100, 100))
    flat[:5, :] = 255.0  # 5% of 10000 = 500 pixels
    gray = _gray(flat, cuda_device)
    out = det.compute(gray)
    assert out["ratio"] == pytest.approx(0.05, abs=1e-4)
    assert out["is_glare"] is False


def test_underexposed_side_is_reported_separately(
    cuda_device: str, det: GPUOverexposureDetector
) -> None:
    # All black frame → under=1.0, over=0.0, no glare.
    gray = _gray(torch.zeros((64, 64)), cuda_device)
    out = det.compute(gray)
    assert out["ratio"] == pytest.approx(0.0)
    assert out["underexposed_ratio"] == pytest.approx(1.0)
    assert out["is_glare"] is False


def test_threshold_is_exposed(det: GPUOverexposureDetector) -> None:
    assert det.compute.__self__.threshold == pytest.approx(0.06)
