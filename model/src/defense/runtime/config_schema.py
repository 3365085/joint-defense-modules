from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when runtime configuration has invalid structure or values."""


@dataclass(frozen=True)
class FieldRule:
    path: str
    types: tuple[type, ...]
    minimum: float | None = None
    maximum: float | None = None
    allowed: set[str] | None = None


def _get(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _validate_rule(config: dict[str, Any], rule: FieldRule, errors: list[str]) -> None:
    value = _get(config, rule.path)
    if value is None:
        return
    if not isinstance(value, rule.types):
        type_names = ', '.join(t.__name__ for t in rule.types)
        errors.append(f"{rule.path}: expected {type_names}, got {type(value).__name__}={value!r}")
        return
    if rule.minimum is not None:
        try:
            if float(value) < rule.minimum:
                errors.append(f"{rule.path}: expected >= {rule.minimum}, got {value!r}")
        except Exception:
            errors.append(f"{rule.path}: expected numeric value, got {value!r}")
    if rule.maximum is not None:
        try:
            if float(value) > rule.maximum:
                errors.append(f"{rule.path}: expected <= {rule.maximum}, got {value!r}")
        except Exception:
            errors.append(f"{rule.path}: expected numeric value, got {value!r}")
    if rule.allowed is not None and str(value).lower() not in rule.allowed:
        errors.append(f"{rule.path}: expected one of {sorted(rule.allowed)}, got {value!r}")


RULES = (
    FieldRule('inference.backend', (str,), allowed={'tensorrt','onnx','pytorch','ultralytics'}),
    FieldRule('inference.model_family', (str,), allowed={'yolov5','yolov8','ultralytics'}),
    FieldRule('inference.device', (str,)),
    FieldRule('inference.confidence', (int, float), minimum=0, maximum=1),
    FieldRule('module_a.detector_impl', (str,), allowed={'legacy', 'rebuilt'}),
    FieldRule('module_a.frame_size', (int,), minimum=32),
    FieldRule('module_a.keyframe_interval', (int,), minimum=1),
    FieldRule('module_a.analysis_max_hz', (int, float), minimum=1, maximum=60),
    FieldRule('module_a.light_flow_interval', (int,), minimum=1),
    FieldRule(
        'module_a.temporal_detector_reuse_threshold',
        (int, float),
        minimum=0,
        maximum=1,
    ),
    FieldRule(
        'module_a.temporal_detector_reuse_max_gap',
        (int,),
        minimum=1,
        maximum=3,
    ),
    FieldRule(
        'module_a.temporal_detector_reuse_max_source_time_gap_s',
        (int, float),
        minimum=0.001,
        maximum=0.1,
    ),
    FieldRule('module_a.static_image_interval', (int,), minimum=1),
    FieldRule('module_a.static_image_worker_timeout_s', (int, float), minimum=0.01),
    FieldRule('module_a.static_image_result_lease_s', (int, float), minimum=0.01),
    FieldRule('module_a.static_image_max_retired_workers', (int,), minimum=0),
    FieldRule('module_a.static_image_global_worker_limit', (int,), minimum=1),
    FieldRule('module_a.rebuilt_theta_media', (int, float), minimum=0, maximum=1),
    FieldRule('module_a.rebuilt_theta_media_raw', (int, float), minimum=0, maximum=1),
    FieldRule('module_a.rebuilt_theta_adv', (int, float), minimum=0, maximum=1),
    FieldRule('module_a.rebuilt_theta_blind', (int, float), minimum=0, maximum=1),
    FieldRule(
        'module_a.rebuilt_blind_confirm_ratio',
        (int, float),
        minimum=0,
        maximum=1,
    ),
    FieldRule('module_a.rebuilt_alert_hold_frames', (int,), minimum=0),
    FieldRule('module_a.rebuilt_a3b_alert_hold_frames', (int,), minimum=0),
    FieldRule('module_a.rebuilt_alert_hold_refresh_on_padv', (bool,)),
    FieldRule('module_a.rebuilt_scene_baseline', (bool,)),
    FieldRule('module_a.rebuilt_sustained_adv_escalation', (bool,)),
    FieldRule(
        'module_a.rebuilt_sustained_adv_seconds',
        (int, float),
        minimum=0.01,
    ),
    FieldRule(
        'module_a.rebuilt_sustained_adv_run_mult',
        (int, float),
        minimum=1,
    ),
    FieldRule(
        'module_a.rebuilt_sustained_adv_benign_decay',
        (int, float),
        minimum=0,
        maximum=1,
    ),
    FieldRule('module_a.rebuilt_sustained_adv_require_target', (bool,)),
    FieldRule(
        'module_a.rebuilt_sustained_adv_require_physical_support',
        (bool,),
    ),
    FieldRule(
        'module_a.rebuilt_sustained_adv_exclude_static_bg',
        (bool,),
    ),
    FieldRule(
        'module_a.rebuilt_sustained_adv_recent_target_min',
        (int,),
        minimum=1,
    ),
    FieldRule('module_a.rebuilt_blind_sustained_escalation', (bool,)),
    FieldRule(
        'module_a.rebuilt_blind_sustained_floor',
        (int,),
        minimum=1,
    ),
    FieldRule(
        'module_a.rebuilt_blind_sustained_degrade_min',
        (int, float),
        minimum=0,
        maximum=1,
    ),
    FieldRule(
        'module_a.rebuilt_blind_sustained_established_min',
        (int,),
        minimum=1,
    ),
    FieldRule('module_a.rebuilt_a3b_gate_candidate_min', (int, float), minimum=0, maximum=1),
    FieldRule('module_a.rebuilt_a3b_gate_edge_min', (int, float), minimum=0, maximum=1),
    FieldRule('module_a.rebuilt_a3b_gate_edge_max', (int, float), minimum=0, maximum=1),
    FieldRule('module_a.rebuilt_a3b_gate_border_contrast_min', (int, float), minimum=0, maximum=1),
    FieldRule(
        'module_a.rebuilt_a3b_soft_gate_candidate_tolerance',
        (int, float),
        minimum=0,
        maximum=0.05,
    ),
    FieldRule(
        'module_a.rebuilt_a3b_soft_gate_aspect_ratio_min',
        (int, float),
        minimum=0.01,
        maximum=10,
    ),
    FieldRule(
        'module_a.rebuilt_a3b_soft_gate_aspect_ratio_max',
        (int, float),
        minimum=0.01,
        maximum=10,
    ),
    FieldRule('module_a.p_adv_threshold', (int, float), minimum=0, maximum=1),
    FieldRule('module_a.a3b_suppression_hold_s', (int, float), minimum=0.01),
    FieldRule('module_a.a3b_suppression_stale_bridge_s', (int, float), minimum=0),
    FieldRule('module_a.alert_window', (int,), minimum=1),
    FieldRule('module_a.alert_trigger_count', (int,), minimum=1),
    FieldRule('runtime.process_fps_cap', (int, float), minimum=0),
    FieldRule('runtime.detector_process_fps_cap', (int, float), minimum=0),
    FieldRule('runtime.preview_render_fps', (int, float), minimum=1),
    FieldRule(
        'runtime.video_decoder_preference',
        (str,),
        allowed={'auto', 'nvdec', 'opencv'},
    ),
    FieldRule('runtime.video_decoder_allow_cpu_fallback', (bool,)),
    FieldRule('runtime.video_decoder_gpu_id', (int,), minimum=0),
    FieldRule(
        'runtime.evidence_writer_queue_capacity',
        (int,),
        minimum=8,
    ),
    FieldRule(
        'runtime.evidence_writer_enqueue_timeout_s',
        (int, float),
        minimum=0,
    ),
    FieldRule(
        'runtime.evidence_writer_drain_timeout_s',
        (int, float),
        minimum=0.1,
    ),
    FieldRule('ppe_tracking.business_min_confidence', (int, float), minimum=0, maximum=1),
    FieldRule('ppe_tracking.temporal_candidate_min_confidence', (int, float), minimum=0, maximum=1),
    FieldRule('ppe_tracking.iou_match_threshold', (int, float), minimum=0, maximum=1),
    FieldRule('ppe_tracking.head_helmet_mutex_iou', (int, float), minimum=0, maximum=1),
    FieldRule('ppe_tracking.head_helmet_mutex_center_distance', (int, float), minimum=0, maximum=1),
    FieldRule('ppe_tracking.head_helmet_mutex_min_helmet_confidence', (int, float), minimum=0, maximum=1),
    FieldRule('ppe_tracking.max_missed_frames', (int,), minimum=0),
    FieldRule('ppe_tracking.alert_window', (int,), minimum=1),
    FieldRule('ppe_tracking.alert_trigger_count', (int,), minimum=1),
    FieldRule('ppe_tracking.fast_alert_window', (int,), minimum=1),
    FieldRule('ppe_tracking.fast_alert_trigger_count', (int,), minimum=1),
    FieldRule('ppe_tracking.fast_alert_min_head_confidence', (int, float), minimum=0, maximum=1),
    FieldRule('a3b.observed_threshold', (int, float), minimum=0, maximum=1),
    FieldRule('a3b.trigger_threshold', (int, float), minimum=0, maximum=1),
    FieldRule('a3b.strong_single_frame_threshold', (int, float), minimum=0, maximum=1),
    FieldRule('a3b.observed_only_warning_threshold', (int, float), minimum=0, maximum=1),
    FieldRule('a3b.observed_only_track_threshold', (int, float), minimum=0, maximum=1),
    FieldRule('a3b.window_size', (int,), minimum=1),
    FieldRule('a3b.min_window_hits', (int,), minimum=1),
    FieldRule('a3b.observed_only_min_window_hits', (int,), minimum=1),
    FieldRule('a3b.min_consecutive_hits', (int,), minimum=1),
    FieldRule('a3b.decay', (int, float), minimum=0, maximum=1),
    FieldRule('model_security.enabled', (bool,)),
    FieldRule('model_security.startup_policy', (str,), allowed={'hash_trust','always_scan','off'}),
    FieldRule('model_security.unknown_model_policy', (str,), allowed={'warn','block'}),
    FieldRule('model_security.background_scan_unknown', (bool,)),
    FieldRule('model_security.max_layers', (int,), minimum=1),
    FieldRule('model_security.max_probes', (int,), minimum=1),
    FieldRule('model_security.batch_size', (int,), minimum=1),
    FieldRule('model_security.time_budget_s', (int, float), minimum=1),
)


def _validate_lte(
    config: dict[str, Any],
    lower_path: str,
    upper_path: str,
    errors: list[str],
) -> None:
    lower = _get(config, lower_path)
    upper = _get(config, upper_path)
    if lower is None or upper is None:
        return
    try:
        if float(lower) > float(upper):
            errors.append(
                f"{lower_path}: expected <= {upper_path} ({upper!r}), got {lower!r}"
            )
    except Exception:
        return


def _validate_count_within_window(
    config: dict[str, Any],
    count_path: str,
    window_path: str,
    errors: list[str],
) -> None:
    _validate_lte(config, count_path, window_path, errors)


def _validate_invariants(config: dict[str, Any], errors: list[str]) -> None:
    light_flow_enabled = _get(config, 'module_a.light_flow_enabled')
    light_flow_interval = _get(config, 'module_a.light_flow_interval')
    if (
        light_flow_enabled is True
        and light_flow_interval is not None
        and int(light_flow_interval) != 1
    ):
        errors.append(
            "module_a.light_flow_interval: expected 1 while light_flow_enabled=true; "
            "interval sampling currently removes A2/A3 evidence"
        )
    _validate_lte(
        config,
        'module_a.a3b_suppression_stale_bridge_s',
        'module_a.a3b_suppression_hold_s',
        errors,
    )
    _validate_lte(
        config,
        'module_a.rebuilt_theta_media_raw',
        'module_a.rebuilt_theta_media',
        errors,
    )
    _validate_lte(
        config,
        'module_a.rebuilt_a3b_gate_edge_min',
        'module_a.rebuilt_a3b_gate_edge_max',
        errors,
    )
    _validate_lte(
        config,
        'module_a.rebuilt_a3b_soft_gate_aspect_ratio_min',
        'module_a.rebuilt_a3b_soft_gate_aspect_ratio_max',
        errors,
    )
    _validate_count_within_window(
        config,
        'module_a.alert_trigger_count',
        'module_a.alert_window',
        errors,
    )
    _validate_lte(config, 'a3b.observed_threshold', 'a3b.trigger_threshold', errors)
    _validate_lte(config, 'a3b.trigger_threshold', 'a3b.strong_single_frame_threshold', errors)
    _validate_count_within_window(config, 'a3b.min_window_hits', 'a3b.window_size', errors)
    _validate_count_within_window(
        config,
        'a3b.observed_only_min_window_hits',
        'a3b.window_size',
        errors,
    )
    _validate_count_within_window(
        config,
        'a3b.min_consecutive_hits',
        'a3b.window_size',
        errors,
    )
    _validate_count_within_window(
        config,
        'ppe_tracking.alert_trigger_count',
        'ppe_tracking.alert_window',
        errors,
    )
    _validate_count_within_window(
        config,
        'ppe_tracking.fast_alert_trigger_count',
        'ppe_tracking.fast_alert_window',
        errors,
    )


def _validate_artifacts(config: dict[str, Any], errors: list[str]) -> None:
    artifacts = _get(config, 'inference.artifacts')
    if artifacts is None:
        return
    if not isinstance(artifacts, dict):
        errors.append(f"inference.artifacts: expected dict, got {type(artifacts).__name__}")
        return
    for key, value in artifacts.items():
        if isinstance(value, str):
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            continue
        errors.append(f"inference.artifacts.{key}: expected string or list[str], got {value!r}")


def validate_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigValidationError(f"runtime config root must be dict, got {type(config).__name__}")
    errors: list[str] = []
    for rule in RULES:
        _validate_rule(config, rule, errors)
    _validate_artifacts(config, errors)
    _validate_invariants(config, errors)
    profiles = config.get('profiles')
    if profiles is not None and not isinstance(profiles, dict):
        errors.append(f"profiles: expected dict, got {type(profiles).__name__}")
    if errors:
        raise ConfigValidationError('Invalid runtime configuration:\n- ' + '\n- '.join(errors))
    return config
