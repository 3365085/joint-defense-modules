from __future__ import annotations

import copy

import pytest

from defense.runtime.config import load_runtime_config
from defense.runtime.config_schema import ConfigValidationError, validate_runtime_config


def _base_config() -> dict:
    return load_runtime_config(profile="default")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("module_a", "detector_impl"), "rebuidl"),
        (("a3b", "observed_threshold"), 1.1),
        (("a3b", "trigger_threshold"), -0.1),
        (("ppe_tracking", "iou_match_threshold"), 2.0),
        (
            ("module_a", "rebuilt_a3b_soft_gate_candidate_tolerance"),
            "invalid",
        ),
        (
            ("module_a", "rebuilt_alert_hold_refresh_on_padv"),
            "true",
        ),
        (
            ("module_a", "rebuilt_sustained_adv_seconds"),
            0,
        ),
        (
            ("module_a", "rebuilt_blind_confirm_ratio"),
            1.1,
        ),
        (("module_a", "temporal_detector_reuse_max_gap"), 4),
        (
            ("module_a", "temporal_detector_reuse_max_source_time_gap_s"),
            0.5,
        ),
        (
            (
                "module_a",
                "rebuilt_sustained_adv_require_physical_support",
            ),
            1,
        ),
    ],
)
def test_invalid_enums_and_probabilities_fail_fast(path: tuple[str, str], value: object) -> None:
    cfg = copy.deepcopy(_base_config())
    cfg[path[0]][path[1]] = value
    with pytest.raises(ConfigValidationError):
        validate_runtime_config(cfg)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: cfg["module_a"].update(
            rebuilt_a3b_gate_edge_min=0.9,
            rebuilt_a3b_gate_edge_max=0.1,
        ),
        lambda cfg: cfg["a3b"].update(window_size=2, min_window_hits=9),
        lambda cfg: cfg["a3b"].update(observed_threshold=0.8, trigger_threshold=0.7),
        lambda cfg: cfg["a3b"].update(
            trigger_threshold=0.8,
            strong_single_frame_threshold=0.7,
        ),
        lambda cfg: cfg["module_a"].update(
            rebuilt_a3b_soft_gate_aspect_ratio_min=3.0,
            rebuilt_a3b_soft_gate_aspect_ratio_max=1.0,
        ),
        lambda cfg: cfg["ppe_tracking"].update(alert_window=2, alert_trigger_count=3),
    ],
)
def test_cross_field_strategy_invariants_fail_fast(mutate) -> None:
    cfg = copy.deepcopy(_base_config())
    mutate(cfg)
    with pytest.raises(ConfigValidationError):
        validate_runtime_config(cfg)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("desktop_rtx", 60),
        ("high_quality", 60),
    ],
)
def test_profile_process_cap_is_the_effective_detector_cap(profile: str, expected: int) -> None:
    cfg = load_runtime_config(profile=profile)
    assert cfg["runtime"]["process_fps_cap"] == expected
    assert cfg["runtime"]["detector_process_fps_cap"] == expected


def test_disabled_stale_custom_model_does_not_override_desktop_rtx_backend() -> None:
    cfg = load_runtime_config(
        profile="desktop_rtx",
        custom_model={
            "enabled": False,
            "path": "baseline_training/runs/baseline_yolov8_three_put/best.pt",
            "backend": "auto",
            "model_family": "auto",
        },
    )

    assert cfg["inference"]["backend"] == "tensorrt"
    assert cfg["inference"]["model_family"] == "yolov8"
    assert cfg["inference"]["artifacts"]["engine"] == [
        "runtime/artifacts/yolo/"
        "4d7a23d3866ac2d9db6e59ae537da1274d988bd53ca6c7d519297fcbb96626f8/"
        "mask_bd_v4_clean_baseline.engine"
    ]


def test_source_keywords_are_not_detection_configuration() -> None:
    cfg = load_runtime_config(profile="default")
    assert cfg["a3b"]["observed_only_source_keywords"] == []
    assert cfg["a3b"]["trigger_source_keywords"] == []


def test_total_module_a_behavior_defaults_are_explicit() -> None:
    module_a = _base_config()["module_a"]

    assert module_a["rebuilt_theta_adv"] == 0.65
    assert module_a["rebuilt_theta_blind"] == 0.55
    assert module_a["rebuilt_blind_confirm_ratio"] == 0.50
    assert module_a["rebuilt_alert_hold_frames"] == 12
    assert module_a["rebuilt_a3b_alert_hold_frames"] == 90
    assert module_a["rebuilt_alert_hold_refresh_on_padv"] is True
    assert module_a["rebuilt_scene_baseline"] is True
    assert module_a["rebuilt_sustained_adv_escalation"] is True
    assert module_a["rebuilt_sustained_adv_seconds"] == 2.0
    assert module_a["rebuilt_sustained_adv_run_mult"] == 1.6
    assert module_a["rebuilt_sustained_adv_require_target"] is False
    assert (
        module_a[
            "rebuilt_sustained_adv_require_physical_support"
        ]
        is False
    )
    assert module_a["rebuilt_blind_sustained_escalation"] is True
    assert module_a["rebuilt_blind_sustained_floor"] == 12
    assert module_a["analysis_max_hz"] == pytest.approx(25.0)
    assert module_a["temporal_detector_reuse_max_gap"] == 2
    assert module_a["temporal_detector_reuse_max_source_time_gap_s"] == pytest.approx(
        0.04
    )


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("desktop_rtx", 25),
        ("high_quality", 25),
    ],
)
def test_profile_preview_fps_is_the_effective_render_cap(profile: str, expected: int) -> None:
    cfg = load_runtime_config(profile=profile)
    assert cfg["runtime"]["preview_render_fps"] == expected


def test_evidence_writer_runtime_defaults_are_bounded_and_explicit() -> None:
    cfg = load_runtime_config(profile="default")
    runtime = cfg["runtime"]

    assert runtime["evidence_writer_queue_capacity"] == 256
    assert runtime["evidence_writer_enqueue_timeout_s"] == 0.02
    assert runtime["evidence_writer_drain_timeout_s"] == 10.0


@pytest.mark.parametrize(
    "profile",
    [
        "default",
        "desktop_rtx",
        "high_quality",
    ],
)
def test_enabled_flow_profiles_do_not_periodically_delete_a3_evidence(profile: str) -> None:
    cfg = load_runtime_config(profile=profile)
    assert cfg["module_a"]["light_flow_enabled"] is True
    assert cfg["module_a"]["light_flow_interval"] == 1
