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
    if rule.allowed is not None and str(value).lower() not in rule.allowed:
        errors.append(f"{rule.path}: expected one of {sorted(rule.allowed)}, got {value!r}")


RULES = (
    FieldRule('inference.backend', (str,), allowed={'tensorrt','onnx','pytorch','ultralytics'}),
    FieldRule('inference.model_family', (str,), allowed={'yolov5','yolov8','ultralytics'}),
    FieldRule('inference.device', (str,)),
    FieldRule('module_a.frame_size', (int,), minimum=32),
    FieldRule('module_a.keyframe_interval', (int,), minimum=1),
    FieldRule('module_a.light_flow_interval', (int,), minimum=1),
    FieldRule('module_a.static_image_interval', (int,), minimum=1),
    FieldRule('runtime.process_fps_cap', (int, float), minimum=0),
    FieldRule('runtime.detector_process_fps_cap', (int, float), minimum=0),
    FieldRule('runtime.preview_render_fps', (int, float), minimum=1),
    FieldRule('ppe_tracking.iou_match_threshold', (int, float), minimum=0),
    FieldRule('ppe_tracking.max_missed_frames', (int,), minimum=0),
    FieldRule('a3b.window_size', (int,), minimum=1),
    FieldRule('a3b.min_window_hits', (int,), minimum=1),
    FieldRule('model_security.enabled', (bool,)),
    FieldRule('model_security.startup_policy', (str,), allowed={'hash_trust','always_scan','off'}),
    FieldRule('model_security.unknown_model_policy', (str,), allowed={'warn','block'}),
    FieldRule('model_security.background_scan_unknown', (bool,)),
    FieldRule('model_security.max_layers', (int,), minimum=1),
    FieldRule('model_security.max_probes', (int,), minimum=1),
    FieldRule('model_security.batch_size', (int,), minimum=1),
    FieldRule('model_security.time_budget_s', (int, float), minimum=1),
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
    profiles = config.get('profiles')
    if profiles is not None and not isinstance(profiles, dict):
        errors.append(f"profiles: expected dict, got {type(profiles).__name__}")
    if errors:
        raise ConfigValidationError('Invalid runtime configuration:\n- ' + '\n- '.join(errors))
    return config
