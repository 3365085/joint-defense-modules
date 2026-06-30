from __future__ import annotations

from defense.module_a.fusion.target_anchored import TargetAnchoredAnalyzer
from defense.module_a.types import ROI


def _targets() -> list[ROI]:
    return [
        ROI("person_1", (10, 10, 80, 110), label="person", confidence=0.9),
        ROI("helmet_1", (20, 10, 60, 50), label="helmet", confidence=0.8),
    ]


def _grid_rois() -> list[ROI]:
    return [ROI("grid_0_0", (0, 0, 64, 64), label="grid")]


def _evaluate(
    analyzer: TargetAnchoredAnalyzer,
    rois: list[ROI],
    *,
    overexposure: dict | None = None,
    blur: dict | None = None,
    temporal: dict | None = None,
    motion: dict | None = None,
) -> dict:
    return analyzer.evaluate(
        rois=rois,
        overexposure=overexposure or {"ratio": 0.0, "temporal_flash": False},
        blur=blur or {"blur_score": 0.0, "blur_low_energy_ratio": 0.0},
        track={"track_score": 0.0, "confidence_drop_score": 0.0},
        temporal=temporal or {"local_max": 0.0, "change_t": 0.0},
        motion=motion or {"motion_score": 0.0, "local_max_ratio": 0.0},
        static_image={"triggered": False, "score": 0.0},
    )


def test_grid_rois_are_not_target_anchors() -> None:
    analyzer = TargetAnchoredAnalyzer()

    out = _evaluate(
        analyzer,
        _grid_rois(),
        blur={"blur_score": 0.9, "blur_low_energy_ratio": 0.8},
        temporal={"local_max": 0.7, "change_t": 0.2},
        motion={"motion_score": 1.0, "local_max_ratio": 0.9},
    )

    assert out["has_targets"] is False
    assert out["suspicious"] is False
    assert out["reason_codes"] == []


def test_no_target_exposure_transition_is_suppressed() -> None:
    analyzer = TargetAnchoredAnalyzer()
    _evaluate(analyzer, _targets())

    out = _evaluate(
        analyzer,
        _grid_rois(),
        overexposure={"ratio": 0.02, "temporal_flash": False},
        blur={"blur_score": 0.8, "blur_low_energy_ratio": 0.7},
        temporal={"local_max": 0.6, "change_t": 0.3},
        motion={"motion_score": 1.0, "local_max_ratio": 0.75},
    )

    assert out["suspicious"] is False
    assert out["reason_codes"] == []


def test_recent_target_disappearance_with_blur_evidence_triggers() -> None:
    analyzer = TargetAnchoredAnalyzer()
    _evaluate(analyzer, _targets())

    out = _evaluate(
        analyzer,
        _grid_rois(),
        overexposure={"ratio": 0.0, "temporal_flash": False},
        blur={"blur_score": 0.18, "blur_low_energy_ratio": 0.2},
        temporal={"local_max": 0.35, "change_t": 0.1},
        motion={"motion_score": 0.0, "local_max_ratio": 0.04},
    )

    assert out["suspicious"] is True
    assert out["reason_codes"] == ["targets_disappeared_with_blur_degradation"]


def test_no_target_global_camera_blur_is_suppressed() -> None:
    analyzer = TargetAnchoredAnalyzer()
    _evaluate(analyzer, _targets())

    out = _evaluate(
        analyzer,
        _grid_rois(),
        overexposure={"ratio": 0.0, "temporal_flash": False},
        blur={"blur_score": 0.56, "blur_low_energy_ratio": 0.2},
        temporal={"local_max": 0.42, "change_t": 0.1},
        motion={"motion_score": 1.0, "local_max_ratio": 0.36},
    )

    assert out["suspicious"] is False
    assert out["reason_codes"] == []


def test_recent_target_disappearance_with_occlusion_evidence_triggers() -> None:
    analyzer = TargetAnchoredAnalyzer()
    _evaluate(analyzer, _targets())

    out = _evaluate(
        analyzer,
        _grid_rois(),
        overexposure={"ratio": 0.0, "temporal_flash": False},
        blur={"blur_score": 0.05, "blur_low_energy_ratio": 0.7},
        temporal={"local_max": 0.55, "change_t": 0.12},
        motion={"motion_score": 1.0, "local_max_ratio": 0.72},
    )

    assert out["suspicious"] is True
    assert out["reason_codes"] == ["targets_disappeared_with_occlusion_degradation"]


def test_target_flow_local_anomaly_is_suppressed_during_exposure_drift() -> None:
    analyzer = TargetAnchoredAnalyzer()

    out = _evaluate(
        analyzer,
        _targets(),
        overexposure={"ratio": 0.03, "temporal_flash": False, "is_glare": False},
        temporal={"local_max": 0.55, "change_t": 0.2},
        motion={"motion_score": 1.0, "local_max_ratio": 0.9},
    )

    assert out["suspicious"] is False
    assert "flow_local_anomaly" not in out["reason_codes"]


def test_target_track_motion_is_suppressed_during_exposure_drift() -> None:
    analyzer = TargetAnchoredAnalyzer()

    out = analyzer.evaluate(
        rois=_targets(),
        overexposure={"ratio": 0.03, "temporal_flash": False, "is_glare": False},
        blur={"blur_score": 0.2, "blur_low_energy_ratio": 0.0},
        track={"track_score": 1.0, "confidence_drop_score": 1.0},
        temporal={"local_max": 0.58, "change_t": 0.2},
        motion={"motion_score": 1.0, "local_max_ratio": 0.8},
        static_image={"triggered": False, "score": 0.0},
    )

    assert out["suspicious"] is False
    assert "target_track_consistency_drop" not in out["reason_codes"]


def test_strong_static_glare_survives_natural_exposure_suppression() -> None:
    analyzer = TargetAnchoredAnalyzer()

    out = _evaluate(
        analyzer,
        _targets(),
        overexposure={"ratio": 0.11, "temporal_flash": False, "is_glare": True},
        temporal={"local_max": 0.32, "change_t": 0.12},
        motion={"motion_score": 0.0, "local_max_ratio": 0.05},
    )

    assert out["suspicious"] is True
    assert "overexposure" in out["reason_codes"]
    assert "natural_exposure_suppressed" not in out["reason_codes"]


def test_no_target_fallback_expires_after_short_event_window() -> None:
    analyzer = TargetAnchoredAnalyzer(no_target_fallback_window_frames=2)
    _evaluate(analyzer, _targets())
    _evaluate(analyzer, _grid_rois())
    _evaluate(analyzer, _grid_rois())

    out = _evaluate(
        analyzer,
        _grid_rois(),
        blur={"blur_score": 0.9, "blur_low_energy_ratio": 0.8},
        temporal={"local_max": 0.7, "change_t": 0.2},
        motion={"motion_score": 1.0, "local_max_ratio": 0.9},
    )

    assert out["suspicious"] is False
    assert out["reason_codes"] == []
