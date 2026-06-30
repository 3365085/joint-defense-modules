from __future__ import annotations

from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState
from defense.runtime.config import apply_custom_model, apply_feature_options, normalize_custom_model_options


def test_a3b_sensitivity_feature_option_applies_threshold_preset() -> None:
    config = {
        "module_a": {"static_image_enabled": True, "static_image_interval": 4},
        "a3b": {
            "observed_threshold": 0.42,
            "trigger_threshold": 0.62,
            "strong_single_frame_threshold": 0.78,
        },
    }

    apply_feature_options(config, {"static_image_enabled": True, "a3b_sensitivity": "high"})

    assert config["module_a"]["static_image_enabled"] is True
    assert config["module_a"]["static_image_interval"] == 2
    assert config["a3b"]["sensitivity"] == "high"
    assert config["a3b"]["observed_threshold"] == 0.34
    assert config["a3b"]["trigger_threshold"] == 0.54
    assert config["a3b"]["strong_single_frame_threshold"] == 0.70
    assert config["a3b"]["min_window_hits"] == 2
    assert config["a3b"]["observed_only_min_window_hits"] == 2
    assert config["a3b"]["min_consecutive_hits"] == 2


def test_invalid_a3b_sensitivity_feature_option_is_ignored() -> None:
    config = {
        "module_a": {"static_image_enabled": True, "static_image_interval": 4},
        "a3b": {
            "sensitivity": "balanced",
            "observed_threshold": 0.42,
            "trigger_threshold": 0.62,
            "strong_single_frame_threshold": 0.78,
        },
    }

    apply_feature_options(config, {"a3b_sensitivity": "not-a-real-level"})

    assert config["module_a"]["static_image_interval"] == 4
    assert config["a3b"]["sensitivity"] == "balanced"
    assert config["a3b"]["observed_threshold"] == 0.42
    assert config["a3b"]["trigger_threshold"] == 0.62


def test_high_a3b_sensitivity_warns_after_two_observed_only_hits() -> None:
    config = {
        "module_a": {"static_image_enabled": True, "static_image_interval": 4},
        "a3b": {
            "observed_threshold": 0.42,
            "trigger_threshold": 0.62,
            "strong_single_frame_threshold": 0.78,
            "observed_only_warning_threshold": 0.50,
            "observed_only_track_threshold": 0.50,
            "observed_only_min_window_hits": 3,
        },
    }
    apply_feature_options(config, {"a3b_sensitivity": "high"})
    state = A3BSoftTriggerState(config["a3b"])

    result = None
    for score in [0.43, 0.44]:
        result = state.update(
            {
                "live_score": score,
                "score": score,
                "p_media": score,
                "p_media_scores": {"track": 0.43},
                "p_media_border_state": {"suppressed": False},
                "p_media_camera_motion_state": {"suppressed": False},
                "p_media_physical_motion_state": {"suppressed": False},
                "source_path": r"D:\security_project_d\素材\视频中出现干扰视频\case.mp4",
            }
        )

    assert result is not None
    assert result["triggered"] is True
    assert result["triggered_source"] == "observed_window"
    assert result["state"] == "suspect"
    assert result["debug"]["observed_only_window_hits"] == 2


def test_custom_engine_auto_backend_resolves_to_tensorrt() -> None:
    config = {"inference": {"backend": "onnx", "artifacts": {}}}
    custom = normalize_custom_model_options(
        {
            "enabled": True,
            "path": r"D:\联合防御模块\model\run_model\yolov8\best.engine",
            "backend": "auto",
            "model_family": "auto",
        }
    )

    resolved = apply_custom_model(config, custom)

    assert resolved["backend"] == "tensorrt"
    assert config["inference"]["backend"] == "tensorrt"
    assert config["inference"]["artifacts"]["engine"] == [r"D:\联合防御模块\model\run_model\yolov8\best.engine"]
