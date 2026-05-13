"""A2 LBP texture + temporal-texture analysers.

Contract:
  * ``GPULBPTextureAnalyzer.compute_lbp`` emits a dense LBP code map for
    a grayscale tensor; ``summarize`` returns ``delta_h`` / ``local_max``
    bounded in [0, 1].
  * ``GPUTemporalTextureAnalyzer.compute`` returns zeros on first frame
    (no ``prev_lbp``) and strictly positive ``change_t`` when the LBP
    code map changes, triggering when change exceeds the configured
    threshold.
"""
from __future__ import annotations

import pytest
import torch

from defense.module_a.features.lbp_texture import GPULBPTextureAnalyzer
from defense.module_a.features.temporal_texture import GPUTemporalTextureAnalyzer


def _frame(tensor: torch.Tensor, device: str) -> torch.Tensor:
    return tensor.to(device).view(1, 1, *tensor.shape)


def test_uniform_frame_has_minimal_texture(cuda_device: str) -> None:
    analyzer = GPULBPTextureAnalyzer(radius=3, grid_size=16)
    flat = _frame(torch.full((128, 128), 120.0), cuda_device)
    lbp = analyzer.compute_lbp(flat)
    summary = analyzer.summarize(lbp)
    assert 0.0 <= summary["delta_h"] <= 1.0
    assert 0.0 <= summary["local_max"] <= 1.0
    # Uniform flat image → LBP comparisons collapse to a single code everywhere
    assert summary["delta_h"] <= 0.02


def test_noisy_frame_raises_delta_h(cuda_device: str) -> None:
    analyzer = GPULBPTextureAnalyzer(radius=3, grid_size=16)
    torch.manual_seed(0)
    noise = torch.rand((128, 128), device=cuda_device) * 255.0
    lbp = analyzer.compute_lbp(noise.view(1, 1, 128, 128))
    summary = analyzer.summarize(lbp)
    assert summary["delta_h"] > 0.0
    assert summary["local_max"] >= summary["delta_h"]


def test_temporal_texture_returns_zeros_without_prev(cuda_device: str) -> None:
    analyzer_t = GPUTemporalTextureAnalyzer(threshold=0.25, grid_size=16)
    curr_lbp = torch.zeros((1, 1, 32, 32), device=cuda_device)
    out = analyzer_t.compute(None, curr_lbp)
    assert out["change_t"] == 0.0
    assert out["local_max"] == 0.0
    assert out["triggered"] is False


def test_temporal_texture_detects_change(cuda_device: str) -> None:
    """Forced difference across the whole LBP map should lift change_t to the
    maximum normalized value (1.0) and flip the ``triggered`` flag."""
    analyzer_t = GPUTemporalTextureAnalyzer(threshold=0.10, grid_size=16)
    prev = torch.zeros((1, 1, 32, 32), device=cuda_device)
    curr = torch.full((1, 1, 32, 32), 255.0, device=cuda_device)
    out = analyzer_t.compute(prev, curr)
    assert out["change_t"] == pytest.approx(1.0, abs=1e-6)
    assert out["local_max"] == pytest.approx(1.0, abs=1e-6)
    assert out["triggered"] is True


def test_temporal_texture_quiet_frames_stay_under_threshold(cuda_device: str) -> None:
    analyzer_t = GPUTemporalTextureAnalyzer(threshold=0.25, grid_size=16)
    prev = torch.full((1, 1, 32, 32), 100.0, device=cuda_device)
    curr = torch.full((1, 1, 32, 32), 101.0, device=cuda_device)
    out = analyzer_t.compute(prev, curr)
    assert out["change_t"] < 0.25
    assert out["triggered"] is False


# --- P1-A-4 adaptive noise-suppression regression tests ----------------------


def test_temporal_adaptive_baseline_not_ready_during_warmup(cuda_device: str) -> None:
    """Until 30 samples accumulate the adaptive gate cannot fire."""
    analyzer = GPUTemporalTextureAnalyzer(
        threshold=0.25, adaptive_baseline=True, adaptive_ema_alpha=0.02
    )
    prev = torch.zeros((1, 1, 32, 32), device=cuda_device)
    curr = torch.full((1, 1, 32, 32), 1.0, device=cuda_device)  # tiny change
    for _ in range(5):
        out = analyzer.compute(prev, curr)
    assert out["adaptive_baseline_active"] is False
    assert out["noise_suppressed"] is False
    # Raw values must always be exposed regardless of suppression state.
    assert "change_t_raw" in out and "change_t_baseline" in out


def test_temporal_adaptive_baseline_suppresses_noise_after_warmup(cuda_device: str) -> None:
    analyzer = GPUTemporalTextureAnalyzer(
        threshold=0.25,
        adaptive_baseline=True,
        adaptive_ema_alpha=0.1,
        adaptive_multiplier=2.0,
        adaptive_floor=0.015,
    )
    prev = torch.zeros((1, 1, 32, 32), device=cuda_device)
    # Tiny steady change: change_t ≈ 1/255 ≈ 0.0039
    curr = torch.full((1, 1, 32, 32), 1.0, device=cuda_device)
    for _ in range(35):
        out = analyzer.compute(prev, curr)
    # EMA is warm, change_raw (~0.004) < adaptive_floor (0.015) → suppressed.
    assert out["adaptive_baseline_active"] is True
    assert out["noise_suppressed"] is True
    # Exposed values are capped just below rule_fusion's defaults.
    assert out["change_t"] <= 0.029
    assert out["local_max"] <= 0.044


def test_temporal_real_spike_survives_suppression(cuda_device: str) -> None:
    """A large real change must always surface above the rule_fusion floor."""
    analyzer = GPUTemporalTextureAnalyzer(
        threshold=0.25,
        adaptive_baseline=True,
        adaptive_ema_alpha=0.1,
        adaptive_multiplier=2.0,
        adaptive_floor=0.015,
    )
    prev = torch.zeros((1, 1, 32, 32), device=cuda_device)
    quiet = torch.full((1, 1, 32, 32), 1.0, device=cuda_device)
    spike = torch.full((1, 1, 32, 32), 200.0, device=cuda_device)
    for _ in range(35):
        analyzer.compute(prev, quiet)
    out = analyzer.compute(prev, spike)
    # Huge change → rule_fusion temporal_trigger (0.03) must pass through.
    assert out["change_t"] > 0.5
    assert out["noise_suppressed"] is False


def test_temporal_reset_clears_adaptive_state(cuda_device: str) -> None:
    analyzer = GPUTemporalTextureAnalyzer(
        threshold=0.25, adaptive_baseline=True, adaptive_ema_alpha=0.1
    )
    prev = torch.zeros((1, 1, 16, 16), device=cuda_device)
    curr = torch.full((1, 1, 16, 16), 1.0, device=cuda_device)
    for _ in range(35):
        analyzer.compute(prev, curr)
    assert analyzer._ema_samples >= 30
    analyzer.reset()
    assert analyzer._ema_samples == 0
    assert analyzer._change_ema == 0.0
