from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .authoritative_model import (
    AuthoritativeModelValidationError,
    validate_production_model_config,
)
from .config_schema import ConfigValidationError, validate_runtime_config

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is part of runtime deps, fallback keeps tests light.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "module_a_runtime.yaml"
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_MATERIAL_ROOT = DEFAULT_WORKSPACE_ROOT / "素材"


def project_root() -> Path:
    return PROJECT_ROOT


def workspace_root() -> Path:
    """Return the outer workspace that owns shared assets and the Pixi env."""
    for env_name in ("MODULE_A_WORKSPACE_ROOT", "SECURITY_PROJECT_ROOT"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser()

    parent = DEFAULT_WORKSPACE_ROOT
    workspace_markers = (".pixi", "素材", "训练素材", "模型和素材")
    if any((parent / marker).exists() for marker in workspace_markers):
        return parent
    return PROJECT_ROOT


def workspace_material_root() -> Path:
    return workspace_root() / "素材"


def workspace_asset_roots() -> list[Path]:
    workspace = workspace_root()
    roots = [
        workspace / "素材",
        PROJECT_ROOT,
        workspace,
        workspace / "模型和素材",
        workspace / "训练素材",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root.absolute())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        data = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError("需要安装 PyYAML 才能读取 YAML 配置")
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件根节点必须是对象: {path}")
    return data


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into ``base`` and return ``base``."""
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def get_nested(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_nested(data: dict[str, Any], dotted: str, value: Any) -> None:
    node = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def list_profiles(config_path: str | Path | None = None) -> list[str]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _read_mapping(path)
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict):
        return ["default"]
    names = ["default"] + sorted(str(name) for name in profiles.keys())
    return list(dict.fromkeys(names))


def load_runtime_config(
    *,
    config_path: str | Path | None = None,
    profile: str = "default",
    feature_options: dict[str, Any] | None = None,
    custom_model: dict[str, Any] | None = None,
    allow_test_custom_model: bool = False,
) -> dict[str, Any]:
    """Load Module A runtime config with profile and UI overrides applied.

    The profile system is intentionally shallow for operators: base config lives
    in ``configs/module_a_runtime.yaml``; each profile only overrides changed
    keys. The Web adapter never mutates this object directly.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _read_mapping(path)
    profiles = raw.pop("profiles", {}) if isinstance(raw.get("profiles", {}), dict) else {}
    cfg = copy.deepcopy(raw)
    selected_profile: dict[str, Any] | None = None
    if profile and profile != "default":
        if profile not in profiles:
            raise KeyError(f"未知运行档位: {profile}; 可用档位: {', '.join(sorted(profiles))}")
        selected_profile = profiles[profile]
        deep_merge(cfg, selected_profile)
        profile_runtime = selected_profile.get("runtime", {}) if isinstance(selected_profile, dict) else {}
        if (
            isinstance(profile_runtime, dict)
            and "process_fps_cap" in profile_runtime
            and "detector_process_fps_cap" not in profile_runtime
        ):
            cfg.setdefault("runtime", {})["detector_process_fps_cap"] = profile_runtime[
                "process_fps_cap"
            ]
        if (
            isinstance(profile_runtime, dict)
            and "preview_fps" in profile_runtime
            and "preview_render_fps" not in profile_runtime
        ):
            cfg.setdefault("runtime", {})["preview_render_fps"] = profile_runtime[
                "preview_fps"
            ]

    # Environment override remains useful when the package is embedded in a
    # larger project whose weights/configs live outside the delivery folder.
    env_device = os.environ.get("MODULE_A_DEVICE")
    if env_device:
        set_nested(cfg, "module_a.device", env_device)
        set_nested(cfg, "inference.device", env_device)

    apply_feature_options(cfg, feature_options or {})
    normalized_custom = normalize_custom_model_options(custom_model)
    if bool(normalized_custom.get("enabled", False)):
        runtime = cfg.setdefault("runtime", {})
        runtime["production_unique_model"] = False
        if allow_test_custom_model:
            runtime["test_custom_model_bypass"] = True
    resolved_custom = apply_custom_model(cfg, normalized_custom)
    cfg.setdefault("runtime", {})["profile"] = profile or "default"
    cfg.setdefault("runtime", {})["custom_model"] = resolved_custom
    validate_runtime_config(cfg)
    try:
        validate_production_model_config(cfg, PROJECT_ROOT)
    except AuthoritativeModelValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc
    return cfg


def apply_feature_options(config: dict[str, Any], options: dict[str, Any]) -> None:
    module_a = config.setdefault("module_a", {})
    if "static_image_enabled" in options:
        module_a["static_image_enabled"] = bool(options["static_image_enabled"])
    sensitivity = normalize_a3b_sensitivity(options.get("a3b_sensitivity", ""))
    if sensitivity:
        apply_a3b_sensitivity(config, sensitivity)


def normalize_a3b_sensitivity(value: Any) -> str:
    sensitivity = str(value or "").strip().lower()
    return sensitivity if sensitivity in A3B_SENSITIVITY_PRESETS else ""


A3B_SENSITIVITY_PRESETS: dict[str, dict[str, Any]] = {
    "conservative": {
        "observed_threshold": 0.46,
        "trigger_threshold": 0.66,
        "strong_single_frame_threshold": 0.82,
        "observed_only_warning_threshold": 0.55,
        "observed_only_track_threshold": 0.55,
        "static_image_interval": 4,
    },
    "balanced": {
        "observed_threshold": 0.42,
        "trigger_threshold": 0.62,
        "strong_single_frame_threshold": 0.78,
        "observed_only_warning_threshold": 0.50,
        "observed_only_track_threshold": 0.50,
        "static_image_interval": 4,
    },
    "sensitive": {
        "observed_threshold": 0.38,
        "trigger_threshold": 0.58,
        "strong_single_frame_threshold": 0.74,
        "observed_only_warning_threshold": 0.46,
        "observed_only_track_threshold": 0.46,
        "observed_only_min_window_hits": 2,
        "static_image_interval": 3,
    },
    "high": {
        "observed_threshold": 0.34,
        "trigger_threshold": 0.54,
        "strong_single_frame_threshold": 0.70,
        "observed_only_warning_threshold": 0.42,
        "observed_only_track_threshold": 0.42,
        "min_window_hits": 2,
        "observed_only_min_window_hits": 2,
        "min_consecutive_hits": 2,
        "static_image_interval": 2,
    },
}


def apply_a3b_sensitivity(config: dict[str, Any], sensitivity: str) -> None:
    sensitivity = normalize_a3b_sensitivity(sensitivity)
    preset = A3B_SENSITIVITY_PRESETS.get(sensitivity)
    if preset is None:
        return
    a3b = config.setdefault("a3b", {})
    for key, value in preset.items():
        if key == "static_image_interval":
            module_a = config.setdefault("module_a", {})
            module_a[key] = int(value)
        else:
            a3b[key] = value
    a3b["sensitivity"] = sensitivity


def infer_backend_from_model_path(path: Path, fallback: str = "onnx") -> str:
    suffix = path.suffix.lower()
    if suffix == ".engine":
        return "tensorrt"
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pt", ".pth"}:
        return "pytorch"
    return fallback


def _canonical_model_family(value: str, fallback: str = "ultralytics") -> str:
    family = str(value or fallback).strip().lower()
    if family == "yolov8":
        return "ultralytics"
    if family in {"auto", "yolov5", "ultralytics"}:
        return family
    return fallback


def _infer_model_family_from_path_hint(path: Path) -> str | None:
    hint_text = " ".join(part.lower() for part in path.parts)
    if "yolov5" in hint_text:
        return "yolov5"
    if any(token in hint_text for token in ("yolov8", "yolo8", "yolo11", "ultralytics")):
        return "ultralytics"
    return None


def _infer_model_family_from_output_dims(dims: list[int]) -> str | None:
    if len(dims) != 3:
        return None
    _, axis_a, axis_b = dims
    if 0 < axis_a <= 128 and axis_b > axis_a:
        return "ultralytics"
    if axis_a > axis_b and 0 < axis_b <= 128:
        return "yolov5"
    return None


def _ensure_yolov5_base_importable() -> None:
    yolov5_root = PROJECT_ROOT / "src" / "defense" / "model_bases" / "yolov5_official"
    if yolov5_root.exists() and str(yolov5_root) not in sys.path:
        sys.path.insert(0, str(yolov5_root))


def _infer_pt_model_family(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        import torch
    except Exception:
        return None

    _ensure_yolov5_base_importable()
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(path, **load_kwargs, weights_only=False)
    except TypeError:
        try:
            checkpoint = torch.load(path, **load_kwargs)
        except Exception:
            return None
    except Exception:
        return None

    model = checkpoint
    if isinstance(checkpoint, dict):
        model = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
    module_name = type(model).__module__.lower()
    class_name = type(model).__name__.lower()
    if module_name.startswith("ultralytics.") or "ultralytics" in module_name:
        return "ultralytics"
    if module_name.startswith("models.") or "yolov5" in module_name:
        return "yolov5"
    if "detectionmodel" in class_name and hasattr(model, "yaml"):
        yaml_data = getattr(model, "yaml", {})
        if isinstance(yaml_data, dict) and "anchors" in yaml_data:
            return "yolov5"
    return None


def _infer_onnx_model_family(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        import onnx
    except Exception:
        return None
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception:
        return None

    producer = str(getattr(model, "producer_name", "") or "").lower()
    if "ultralytics" in producer:
        return "ultralytics"
    metadata = {
        str(item.key).lower(): str(item.value).lower()
        for item in getattr(model, "metadata_props", [])
    }
    metadata_text = " ".join([producer, *metadata.keys(), *metadata.values()])
    if "yolov5" in metadata_text:
        return "yolov5"
    if "yolov8" in metadata_text or "yolo11" in metadata_text or "ultralytics" in metadata_text:
        return "ultralytics"

    for output in model.graph.output:
        dims = [int(dim.dim_value or 0) for dim in output.type.tensor_type.shape.dim]
        family = _infer_model_family_from_output_dims(dims)
        if family:
            return family
    return None


def _infer_tensorrt_model_family(path: Path) -> str | None:
    path_hint = _infer_model_family_from_path_hint(path)
    if path_hint:
        return path_hint
    if not path.exists():
        return None
    try:
        import tensorrt as trt
    except Exception:
        return None
    try:
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
    except Exception:
        return None
    if engine is None:
        return None

    output_shapes: list[tuple[int, ...]] = []
    try:
        if hasattr(engine, "num_io_tensors"):
            for index in range(int(engine.num_io_tensors)):
                name = engine.get_tensor_name(index)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    output_shapes.append(tuple(int(v) for v in engine.get_tensor_shape(name)))
        else:
            for index in range(int(engine.num_bindings)):
                if not engine.binding_is_input(index):
                    output_shapes.append(tuple(int(v) for v in engine.get_binding_shape(index)))
    except Exception:
        return None

    for shape in output_shapes:
        family = _infer_model_family_from_output_dims([int(v) for v in shape])
        if family:
            return family
    return None


def infer_model_family_from_model_path(path: Path, fallback: str = "ultralytics") -> str:
    fallback = _canonical_model_family(fallback, "ultralytics")
    if fallback == "auto":
        fallback = "ultralytics"
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return _infer_pt_model_family(path) or fallback
    if suffix == ".onnx":
        return _infer_onnx_model_family(path) or fallback
    if suffix == ".engine":
        return _infer_tensorrt_model_family(path) or fallback
    return fallback


def normalize_class_names_option(value: Any) -> list[str] | dict[int, str] | None:
    if isinstance(value, dict):
        normalized: dict[int, str] = {}
        for key, name in value.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            text = str(name or "").strip()
            if text:
                normalized[idx] = text
        return normalized or None
    if isinstance(value, (list, tuple)):
        names = [str(name or "").strip() for name in value if str(name or "").strip()]
        return names or None
    if isinstance(value, str):
        names = [part.strip() for part in re.split(r"[,/|;\s]+", value) if part.strip()]
        return names or None
    return None


def normalize_custom_model_options(custom_model: dict[str, Any] | None) -> dict[str, Any]:
    custom_model = custom_model or {}
    enabled = bool(custom_model.get("enabled", False))
    path = str(custom_model.get("path", "") or "").strip()
    backend = str(custom_model.get("backend", "auto") or "auto").strip().lower()
    model_family = _canonical_model_family(
        str(custom_model.get("model_family", "auto") or "auto"),
        "auto",
    )
    if backend not in {"auto", "tensorrt", "onnx", "pytorch"}:
        backend = "auto"
    normalized = {
        "enabled": enabled and bool(path),
        "path": path,
        "backend": backend,
        "model_family": model_family,
        "source_pt_path": str(custom_model.get("source_pt_path", "") or "").strip(),
    }
    class_names = normalize_class_names_option(custom_model.get("class_names"))
    if class_names is not None:
        normalized["class_names"] = class_names
    return normalized


def apply_custom_model(config: dict[str, Any], custom_model: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(custom_model)
    if not resolved.get("enabled"):
        return resolved
    path = Path(str(resolved.get("path", ""))).expanduser()
    if path.suffix.lower() in {".pt", ".pth"}:
        source_pt_path = str(resolved.get("source_pt_path", "") or "").strip()
        if not source_pt_path or Path(source_pt_path).expanduser() != path:
            resolved["source_pt_path"] = str(path)
    backend = str(resolved.get("backend", "auto"))
    inferred_backend = infer_backend_from_model_path(path, "")
    if backend == "auto":
        backend = inferred_backend or str(get_nested(config, "inference.backend", "onnx"))
    elif inferred_backend and backend != inferred_backend:
        raise ValueError(
            f"Custom model backend does not match file suffix: {path.suffix or '<no suffix>'} "
            f"should use {inferred_backend}, got {backend}"
        )
    resolved["backend"] = backend
    inference = config.setdefault("inference", {})
    model_family = _canonical_model_family(str(resolved.get("model_family", "auto")), "auto")
    if model_family == "auto":
        model_family = infer_model_family_from_model_path(path, "ultralytics")
        resolved["model_family_auto_detected"] = True
    else:
        resolved["model_family_auto_detected"] = False
    resolved["model_family"] = model_family
    inference["backend"] = backend
    inference["model_family"] = model_family
    class_names = normalize_class_names_option(resolved.get("class_names"))
    if class_names is not None:
        inference["class_names"] = class_names
        resolved["class_names"] = class_names
    artifacts = inference.setdefault("artifacts", {})
    key = "engine" if backend == "tensorrt" else backend
    artifacts[key] = [str(path)]
    return resolved


def public_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Return non-sensitive config fields for /api/status."""
    inference = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    module_a = config.get("module_a", {}) if isinstance(config.get("module_a"), dict) else {}
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    return {
        "profile": runtime.get("profile", "default"),
        "backend": inference.get("backend"),
        "device": inference.get("device", module_a.get("device")),
        "model_family": inference.get("model_family", inference.get("family")),
        "class_names": inference.get("class_names", inference.get("names", inference.get("labels"))),
        "frame_size": module_a.get("frame_size", 640),
        "light_flow_interval": module_a.get("light_flow_interval"),
        "static_image_interval": module_a.get("static_image_interval"),
    }


def write_config_snapshot(config: dict[str, Any], target_dir: str | Path) -> Path:
    """Write the full resolved config to a JSON snapshot in *target_dir*.

    Returns the snapshot path.  Callers that don't have a *target_dir* yet
    can skip this — the function is designed to be called once per
    :meth:`MonitorEngine.start` after the evidence session is created.
    """
    snapshot = {"$schema": "runtime_config_snapshot", "config": _strip_large_arrays(config)}
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    out = target / "config_snapshot.json"
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def _strip_large_arrays(d: Any, max_items: int = 100) -> Any:
    """Recursively truncate large lists/arrays to keep the snapshot readable."""
    if isinstance(d, dict):
        return {k: _strip_large_arrays(v, max_items) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        seq = [_strip_large_arrays(v, max_items) for v in d]
        return seq[:max_items] if len(seq) > max_items else seq
    return d
