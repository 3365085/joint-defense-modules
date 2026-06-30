"""A4 rule fusion — 5-dim weighted fusion of A1/A2/A3/A3b."""
from __future__ import annotations

import pytest

from defense.module_a.fusion.rule_fusion import GPURuleFusion


def _empty(**overrides):
    base = {
        "texture": {"delta_h": 0.0, "local_max": 0.0},
        "temporal": {"change_t": 0.0, "local_max": 0.0, "roi_results": []},
        "motion": {
            "motion_score": 0.0,
            "local_max_ratio": 0.0,
            "static_image_score": 0.0,
            "static_image_triggered": False,
            "light_flow_available": False,
            "light_flow_score": 0.0,
            "light_flow_local_anomaly_ratio": 0.0,
        },
        "overexposure": {"ratio": 0.0, "is_glare": False},
        "blur": {"blur_score": 0.0},
        "track": {"track_score": 0.0},
    }
    base.update(overrides)
    return base


def test_clean_frame_is_not_suspicious(cuda_device: str) -> None:
    fusion = GPURuleFusion(device=cuda_device)
    out = fusion.compute(**_empty())
    assert out["p_adv"] == pytest.approx(0.0)
    assert out["is_suspicious"] is False
    assert out["reason_codes"] == []


def test_overexposure_alone_triggers_suspicion(cuda_device: str) -> None:
    fusion = GPURuleFusion(device=cuda_device)
    scenario = _empty(overexposure={"ratio": 0.25, "is_glare": True})
    out = fusion.compute(**scenario)
    assert out["is_suspicious"] is True
    assert "overexposure" in out["reason_codes"]


def test_static_image_triggers_suspicion(cuda_device: str) -> None:
    fusion = GPURuleFusion(device=cuda_device)
    scenario = _empty()
    scenario["motion"] = dict(scenario["motion"])
    scenario["motion"]["static_image_score"] = 0.9
    scenario["motion"]["static_image_triggered"] = True
    out = fusion.compute(**scenario)
    assert out["is_suspicious"] is True
    assert "static_image_spoof" in out["reason_codes"]


def test_paired_track_blur_temporal_triggers(cuda_device: str) -> None:
    fusion = GPURuleFusion(device=cuda_device)
    scenario = _empty(
        temporal={"change_t": 0.05, "local_max": 0.30, "roi_results": []},
        blur={"blur_score": 0.60},
        track={"track_score": 0.80},
    )
    out = fusion.compute(**scenario)
    assert out["paired_track_triggered"] is True
    assert out["is_suspicious"] is True
    assert "paired_track_consistency_drop" in out["reason_codes"]


def test_p_adv_threshold_alone_triggers(cuda_device: str) -> None:
    """If the linear-weighted score crosses threshold, suspicion fires."""
    fusion = GPURuleFusion(device=cuda_device, threshold=0.10)
    scenario = _empty(
        texture={"delta_h": 0.30, "local_max": 0.4},
        temporal={"change_t": 0.20, "local_max": 0.4, "roi_results": []},
    )
    out = fusion.compute(**scenario)
    assert out["p_adv"] > 0.10
    assert out["p_adv_triggered"] is True
    assert "p_adv" in out["reason_codes"]
    assert out["is_suspicious"] is True


def test_weights_length_validated() -> None:
    with pytest.raises(ValueError):
        GPURuleFusion(device="cpu", weights=(0.5, 0.5))
