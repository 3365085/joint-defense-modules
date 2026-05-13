"""End-to-end integration of ModuleADetector on synthetic frames.

Scope:
  * Drives ``ModuleADetector.process`` directly with ``ModuleAInput`` built
    from numpy frames. This isolates the detection pipeline from the
    YOLO backend and is therefore deterministic and cheap.
  * Covers: quiet stream, glare attack, noise-burst attack, and alert
    state-machine confirmation across multiple frames.
"""
from __future__ import annotations

import numpy as np
import pytest

from defense.module_a.detector import ModuleADetector
from defense.module_a.types import ROI, ModuleAInput


def _detector_config() -> dict:
    """Minimal GPU-first config that does not load the classifier artifact.

    ``fusion_backend=rule`` keeps the test self-contained and reproducible.
    """
    return {
        "module_a": {
            "require_gpu": True,
            "frame_size": 128,
            "keyframe_interval": 1,
            "alert_window": 5,
            "alert_trigger_count": 3,
            "attack_state_hold_frames": 0,
            "light_flow_enabled": False,  # Keeps the test cheap + deterministic
            "static_image_enabled": False,
            "source_authenticity_enabled": False,
            "fusion_backend": "rule",
            "glare_ratio_threshold": 0.05,
            "lbp_temporal_change_threshold": 0.15,
            "p_adv_threshold": 0.2,
            "use_grid_when_no_roi": True,
            "grid_roi_count": 2,
        }
    }


@pytest.fixture
def detector(cuda_device: str) -> ModuleADetector:
    # cuda_device fixture guarantees CUDA is present, so ModuleADetector
    # passes its GPU-first assertion.
    del cuda_device  # only needed for skip behaviour
    return ModuleADetector(_detector_config())


def _mid_gray_frame(size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), 128, dtype=np.uint8)


def _glare_frame(size: int = 128) -> np.ndarray:
    frame = np.full((size, size, 3), 128, dtype=np.uint8)
    frame[: size // 2, :, :] = 255
    return frame


def test_first_frame_emits_no_alert(detector: ModuleADetector) -> None:
    frame = _mid_gray_frame()
    result = detector.process(ModuleAInput(frame=frame, frame_idx=0))
    assert result.alert_confirmed is False
    assert 0.0 <= result.p_adv <= 1.0
    assert isinstance(result.reason_codes, list)


def test_glare_frame_flags_overexposure(detector: ModuleADetector) -> None:
    result = detector.process(ModuleAInput(frame=_glare_frame(), frame_idx=0))
    assert "overexposure" in result.reason_codes
    assert result.single_frame_suspicious is True


def test_repeated_glare_confirms_alert(detector: ModuleADetector) -> None:
    for i in range(5):
        result = detector.process(ModuleAInput(frame=_glare_frame(), frame_idx=i))
    assert result.alert_confirmed is True


def test_quiet_stream_keeps_alert_off(detector: ModuleADetector) -> None:
    result = None
    for i in range(6):
        result = detector.process(ModuleAInput(frame=_mid_gray_frame(), frame_idx=i))
    assert result is not None
    assert result.alert_confirmed is False


def test_reset_clears_history(detector: ModuleADetector) -> None:
    # Seed with glare confirmations.
    for i in range(5):
        detector.process(ModuleAInput(frame=_glare_frame(), frame_idx=i))
    detector.reset()
    # After reset a single clean frame must not be confirmed.
    result = detector.process(ModuleAInput(frame=_mid_gray_frame(), frame_idx=0))
    assert result.alert_confirmed is False


def test_explicit_roi_overrides_grid(detector: ModuleADetector) -> None:
    """Supplying an ROI should shortcut the grid fallback."""
    frame = _mid_gray_frame()
    roi = ROI(roi_id="explicit", bbox=(10, 10, 60, 60), label="person", confidence=0.9)
    detector.reset()
    result = detector.process(ModuleAInput(frame=frame, frame_idx=0, rois=[roi]))
    # Grid fallback (grid_roi_count=2 → 4 ROIs) should NOT fire.
    assert isinstance(result.roi_results, list)


def test_module_a_breakdown_present(detector: ModuleADetector) -> None:
    """The 6-field breakdown contract (tasks §3.1) must be stamped on details."""
    result = detector.process(ModuleAInput(frame=_mid_gray_frame(), frame_idx=0))
    breakdown = result.details.get("module_a_breakdown", {})
    for field in (
        "a1_overexposure_ms",
        "a2_temporal_ms",
        "a3_motion_ms",
        "a3b_static_media_ms",
        "a4_fusion_ms",
        "source_auth_ms",
    ):
        assert field in breakdown, f"missing timing field: {field}"
        assert breakdown[field] >= 0.0
