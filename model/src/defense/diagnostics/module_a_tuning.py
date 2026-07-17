from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.module_a.backends import create_detector_backend
from defense.module_a.backends.detector_backend import configured_class_names
from defense.module_a.result_contract import adapt_a3b_result
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline
from defense.runtime.config import (
    DEFAULT_CONFIG_PATH,
    deep_merge,
    load_runtime_config,
    project_root,
)
from defense.runtime.config_schema import validate_runtime_config
from defense.runtime.pipeline_factory import EmptyDetectorBackend, configure_runtime_threads


PipelineFactory = Callable[[dict[str, Any]], Any]
CaptureFactory = Callable[[str], Any]

_ROOT_CONFIG_KEYS = frozenset(
    {
        "a3b",
        "inference",
        "module_a",
        "ppe_tracking",
        "runtime",
        "web",
    }
)


def parse_tuning_patch(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    """Parse a JSON/YAML path, inline JSON object, or mapping into a config patch."""

    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))

    text_value = str(value).strip()
    candidate = Path(text_value).expanduser()
    try:
        candidate_is_file = candidate.is_file()
    except OSError:
        candidate_is_file = False
    if candidate_is_file:
        text = candidate.read_text(encoding="utf-8-sig")
        suffix = candidate.suffix.lower()
        if suffix == ".json":
            data = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - Pixi environment includes PyYAML.
                raise RuntimeError("读取 YAML tuning patch 需要 PyYAML") from exc
            data = yaml.safe_load(text) or {}
    else:
        try:
            data = json.loads(text_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--tuning 必须是 JSON 对象，或指向 JSON/YAML 文件"
            ) from exc

    if not isinstance(data, Mapping):
        raise ValueError("tuning patch 根节点必须是对象")
    return copy.deepcopy(dict(data))


def normalize_tuning_patch(patch: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep nested runtime patches intact and wrap legacy flat tuning keys."""

    normalized = copy.deepcopy(dict(patch or {}))
    if not normalized:
        return {}
    if "profiles" in normalized:
        raise ValueError("tuning patch 不能修改 profiles；请修改已解析的 effective config")
    if any(key in _ROOT_CONFIG_KEYS for key in normalized):
        return normalized
    return {"module_a": normalized}


def build_effective_config(
    *,
    config_path: str | Path | None = None,
    profile: str = "desktop_rtx",
    tuning_patch: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the production runtime config and recursively apply an explicit patch."""

    config = load_runtime_config(
        config_path=config_path,
        profile=str(profile or "default"),
        feature_options=None,
    )
    normalized_patch = normalize_tuning_patch(tuning_patch)
    deep_merge(config, normalized_patch)
    validate_runtime_config(config)

    detector_impl = str(config.get("module_a", {}).get("detector_impl", "rebuilt")).lower()
    if detector_impl != "rebuilt":
        raise ValueError(
            "Module A 调优/诊断工具仅运行生产 rebuilt pipeline；"
            f"当前 effective detector_impl={detector_impl!r}"
        )
    return config, normalized_patch


def create_production_pipeline(config: dict[str, Any]) -> VideoDefensePipeline:
    """Construct the same detector backend and VideoDefensePipeline used in production."""

    configure_runtime_threads()
    runtime_config = config.get("runtime", {})
    allow_empty = bool(
        runtime_config.get("allow_empty_backend", False)
        if isinstance(runtime_config, Mapping)
        else False
    )
    backend = (
        EmptyDetectorBackend(configured_class_names(config))
        if allow_empty
        else create_detector_backend(config, project_root())
    )
    return VideoDefensePipeline(backend, config=config)


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _first_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is not None:
            return _as_float(value, default)
    return float(default)


def authoritative_frame_row(
    *,
    video_path: Path,
    frame_idx: int,
    source_time_s: float,
    info: Mapping[str, Any],
    pipeline_call_wall_ms: float,
) -> dict[str, Any]:
    """Flatten only fields already decided or emitted by the production pipeline."""

    info_map = _as_mapping(info)
    details = _as_mapping(info_map.get("details"))
    latency = _as_mapping(info_map.get("latency_breakdown"))
    timing = _as_mapping(details.get("timing"))
    detections = _as_mapping(details.get("detections"))

    a1 = _as_mapping(details.get("a1"))
    a2 = _as_mapping(details.get("a2"))
    a3 = _as_mapping(details.get("a3"))
    a4 = _as_mapping(details.get("a4"))
    a3b = _as_mapping(details.get("a3b"))
    joint = _as_mapping(details.get("joint_decision"))
    scene = _as_mapping(details.get("scene_context"))
    flow = _as_mapping(details.get("flow_context"))

    # Compatibility-only fallback for historical result layouts. No thresholds
    # or local decision rules are evaluated here.
    legacy_features = _as_mapping(details.get("module_a_features"))
    if not a1:
        a1 = _as_mapping(legacy_features.get("overexposure"))
    if not a2:
        a2 = _as_mapping(legacy_features.get("temporal"))
    if not a3:
        a3 = _as_mapping(legacy_features.get("flow"))
    if not a4:
        a4 = _as_mapping(legacy_features.get("classifier"))

    a3b_contract = adapt_a3b_result(info_map)
    if not a3b:
        a3b = _as_mapping(a3b_contract)
    module_a_breakdown = _as_mapping(latency.get("module_a_breakdown"))
    if not module_a_breakdown:
        module_a_breakdown = {
            str(key): value
            for key, value in timing.items()
            if key
            not in {
                "detector_ms",
                "module_a_ms",
                "pipeline_ms",
                "total",
                "total_ms",
            }
        }
    reuse = _as_mapping(latency.get("detector_reuse"))
    reuse_counters = _as_mapping(latency.get("detector_reuse_counters"))

    reason_codes = [str(code) for code in info_map.get("reason_codes", []) or []]
    return {
        "video": str(video_path),
        "frame_idx": int(frame_idx),
        "source_time_s": float(source_time_s),
        "alert_confirmed": bool(info_map.get("alert_confirmed", False)),
        "single_frame_suspicious": bool(
            info_map.get(
                "single_frame_suspicious",
                joint.get("single_frame_candidate", False),
            )
        ),
        "attack_state_active": bool(info_map.get("attack_state_active", False)),
        "is_attack": bool(info_map.get("is_attack", False)),
        "p_adv": _as_float(info_map.get("p_adv")),
        "p_adv_display": _first_float(
            info_map.get("p_adv_display"),
            info_map.get("p_adv"),
        ),
        "a1_score": _first_float(
            a1.get("a1_feature_score"),
            legacy_features.get("a1_score"),
        ),
        "a2_score": _first_float(
            a2.get("a2_feature_score"),
            legacy_features.get("a2_score"),
        ),
        "a3_score": _first_float(
            a3.get("a3_feature_score"),
            legacy_features.get("a3_score"),
        ),
        "a4_score": _first_float(a4.get("p_adv"), info_map.get("p_adv")),
        "p_media": _first_float(
            a3b.get("p_media_policy"),
            a3b.get("p_media"),
            a3b_contract.get("p_media"),
        ),
        "a3b_confirmed": bool(
            a3b.get(
                "media_confirmed",
                a3b_contract.get("confirmed", a3b_contract.get("triggered", False)),
            )
        ),
        "a3b_result_seq": a3b.get(
            "a3b_result_seq",
            a3b_contract.get("a3b_result_seq"),
        ),
        "candidate_source": str(
            joint.get("candidate_source", legacy_features.get("candidate_source", "none"))
        ),
        "primary_channel": str(
            joint.get("primary_channel", legacy_features.get("primary_channel", "none"))
        ),
        "public_reason": str(
            joint.get("public_reason", legacy_features.get("public_reason", ""))
        ),
        "suppressed_reason": str(
            joint.get("suppressed_reason", a3b.get("suppressed_reason", "none"))
        ),
        "reason_codes": reason_codes,
        "roi_count": int(detections.get("roi_count", 0) or 0),
        "detection_count": len(detections.get("boxes", []) or []),
        "flow_backend": str(
            flow.get("backend", flow.get("flow_backend", a3.get("flow_backend", "unknown")))
        ),
        "flow_sampled": (
            bool(flow.get("flow_sampled", a3.get("flow_sampled")))
            if flow.get("flow_sampled", a3.get("flow_sampled")) is not None
            else None
        ),
        "overexposure_ratio": _first_float(
            scene.get("overexposure_ratio"),
            a1.get("overexposure_ratio"),
            a1.get("ratio"),
        ),
        "frame_diff_global": _as_float(scene.get("frame_diff_global")),
        "pipeline_call_wall_ms": _as_float(pipeline_call_wall_ms),
        "pipeline_ms": _first_float(
            timing.get("pipeline_ms"),
            latency.get("e2e_ms"),
            info_map.get("timing_ms"),
        ),
        "backend_inference_ms": _first_float(
            info_map.get("detector_inference_ms"),
            latency.get("detector_ms"),
            timing.get("detector_ms"),
        ),
        "module_a_ms": _first_float(
            info_map.get("module_a_timing_ms"),
            latency.get("module_a_total_ms"),
            timing.get("module_a_ms"),
            timing.get("total_ms"),
            timing.get("total"),
        ),
        "frame_resize_ms": _as_float(latency.get("frame_resize_ms")),
        "detector_reuse_hit": bool(latency.get("detector_reuse_hit", False)),
        "detector_reuse_reason": str(reuse.get("reason", "unavailable")),
        "detector_change_score": _first_float(
            latency.get("detector_change_score"),
            info_map.get("detector_change_score"),
        ),
        "backend_predict_count": int(reuse_counters.get("backend_predict_count", 0) or 0),
        "module_a_breakdown_ms": {
            str(key): _as_float(value)
            for key, value in module_a_breakdown.items()
        },
    }


def _percentile_stats(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "p99": round(float(np.percentile(array, 99)), 4),
        "max": round(float(array.max()), 4),
    }


def summarize_authoritative_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    wall_seconds: float,
    source_fps: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize authoritative signals, decisions, reuse, and measured timings."""

    row_list = [dict(row) for row in rows]
    reason_counts: Counter[str] = Counter()
    reuse_reason_counts: Counter[str] = Counter()
    for row in row_list:
        reason_counts.update(str(code) for code in row.get("reason_codes", []) or [])
        reuse_reason_counts[str(row.get("detector_reuse_reason", "unavailable"))] += 1

    performance_keys = (
        "pipeline_call_wall_ms",
        "pipeline_ms",
        "backend_inference_ms",
        "module_a_ms",
        "frame_resize_ms",
    )
    performance = {
        key: _percentile_stats(
            [_as_float(row.get(key)) for row in row_list]
        )
        for key in performance_keys
    }
    active_backend_values = [
        _as_float(row.get("backend_inference_ms"))
        for row in row_list
        if not bool(row.get("detector_reuse_hit", False))
    ]
    performance["backend_active_inference_ms"] = _percentile_stats(active_backend_values)

    breakdown_keys = sorted(
        {
            str(key)
            for row in row_list
            for key in _as_mapping(row.get("module_a_breakdown_ms"))
        }
    )
    performance["module_a_breakdown_ms"] = {
        key: _percentile_stats(
            [
                _as_float(_as_mapping(row.get("module_a_breakdown_ms")).get(key))
                for row in row_list
            ]
        )
        for key in breakdown_keys
    }

    signal_keys = ("p_adv", "a1_score", "a2_score", "a3_score", "a4_score", "p_media")
    signals = {
        key: _percentile_stats([_as_float(row.get(key)) for row in row_list])
        for key in signal_keys
    }

    frame_count = len(row_list)
    reuse_hits = sum(bool(row.get("detector_reuse_hit", False)) for row in row_list)
    final_backend_predict_count = (
        int(row_list[-1].get("backend_predict_count", 0) or 0)
        if row_list
        else 0
    )
    runtime_config = _as_mapping(config.get("runtime"))
    configured_process_fps = _first_float(
        runtime_config.get("detector_process_fps_cap"),
        runtime_config.get("process_fps_cap"),
        default=0.0,
    )
    effective_expected_process_fps = (
        min(configured_process_fps, float(source_fps))
        if configured_process_fps > 0.0 and source_fps > 0.0
        else max(configured_process_fps, float(source_fps), 0.0)
    )
    configured_budget_ms = (
        1000.0 / effective_expected_process_fps
        if effective_expected_process_fps > 0.0
        else 0.0
    )
    pipeline_p95 = _as_float(_as_mapping(performance["pipeline_ms"]).get("p95"))

    return {
        "frames": frame_count,
        "wall_seconds": round(float(wall_seconds), 4),
        "throughput_fps": round(frame_count / wall_seconds, 4)
        if wall_seconds > 0.0
        else 0.0,
        "source_fps": round(float(source_fps), 4),
        "source_realtime_factor": round(
            (frame_count / wall_seconds) / source_fps,
            4,
        )
        if wall_seconds > 0.0 and source_fps > 0.0
        else 0.0,
        "configured_detector_process_fps_cap": configured_process_fps,
        "effective_expected_process_fps": round(
            effective_expected_process_fps,
            4,
        ),
        "configured_frame_budget_ms": round(configured_budget_ms, 4),
        "pipeline_p95_budget_ratio": round(
            pipeline_p95 / configured_budget_ms,
            4,
        )
        if configured_budget_ms > 0.0
        else 0.0,
        "decisions": {
            "alert_frames": sum(bool(row.get("alert_confirmed", False)) for row in row_list),
            "suspicious_frames": sum(
                bool(row.get("single_frame_suspicious", False)) for row in row_list
            ),
            "attack_state_frames": sum(
                bool(row.get("attack_state_active", False)) for row in row_list
            ),
            "a3b_confirmed_frames": sum(
                bool(row.get("a3b_confirmed", False)) for row in row_list
            ),
            "reason_code_counts": dict(reason_counts.most_common()),
        },
        "detector_reuse": {
            "hit_frames": int(reuse_hits),
            "hit_rate": round(reuse_hits / frame_count, 4) if frame_count else 0.0,
            "backend_predict_count": final_backend_predict_count,
            "reason_counts": dict(reuse_reason_counts.most_common()),
        },
        "performance_ms": performance,
        "signals": signals,
    }


def _valid_source_fps(capture: Any) -> float:
    source_fps = _as_float(capture.get(cv2.CAP_PROP_FPS), 30.0)
    return source_fps if 0.1 <= source_fps <= 240.0 else 30.0


def _run_video_with_pipeline(
    pipeline: Any,
    video_path: Path,
    *,
    config: Mapping[str, Any],
    max_frames: int,
    capture_factory: CaptureFactory,
) -> dict[str, Any]:
    capture = capture_factory(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开视频: {video_path}")

    source_fps = _valid_source_fps(capture)
    rows: list[dict[str, Any]] = []
    frame_idx = 0
    started = time.perf_counter()
    pipeline.reset()
    try:
        while max_frames <= 0 or frame_idx < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            source_time_s = frame_idx / source_fps
            call_started = time.perf_counter()
            _, _, info = pipeline.process_frame(
                frame,
                timestamp=source_time_s,
                source_fps=source_fps,
                source_frame_idx=frame_idx,
            )
            pipeline_call_wall_ms = (time.perf_counter() - call_started) * 1000.0
            rows.append(
                authoritative_frame_row(
                    video_path=video_path,
                    frame_idx=frame_idx,
                    source_time_s=source_time_s,
                    info=info,
                    pipeline_call_wall_ms=pipeline_call_wall_ms,
                )
            )
            frame_idx += 1
    finally:
        capture.release()

    wall_seconds = time.perf_counter() - started
    return {
        "video": str(video_path),
        "source_fps": source_fps,
        "summary": summarize_authoritative_rows(
            rows,
            wall_seconds=wall_seconds,
            source_fps=source_fps,
            config=config,
        ),
        "frames": rows,
    }


def run_module_a_videos(
    video_paths: Sequence[str | Path],
    *,
    config_path: str | Path | None = None,
    profile: str = "desktop_rtx",
    tuning_patch: Mapping[str, Any] | None = None,
    max_frames: int = 0,
    warmup: bool = True,
    pipeline_factory: PipelineFactory | None = None,
    capture_factory: CaptureFactory | None = None,
) -> dict[str, Any]:
    """Run one production rebuilt pipeline over one or more videos."""

    paths = [Path(path).expanduser().resolve() for path in video_paths]
    if not paths:
        raise ValueError("至少需要一个 --video")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"视频不存在: {path}")

    config_started = time.perf_counter()
    config, normalized_patch = build_effective_config(
        config_path=config_path,
        profile=profile,
        tuning_patch=tuning_patch,
    )
    config_load_ms = (time.perf_counter() - config_started) * 1000.0

    construct_started = time.perf_counter()
    pipeline = (pipeline_factory or create_production_pipeline)(config)
    pipeline_construct_ms = (time.perf_counter() - construct_started) * 1000.0
    detector_backend = getattr(pipeline, "detector_backend", None)
    backend_metadata = {
        "name": str(getattr(detector_backend, "backend", "unknown")),
        "artifact_path": str(getattr(detector_backend, "artifact_path", "")),
        "device": str(getattr(detector_backend, "device", "unknown")),
    }
    warmup_ms = 0.0
    warmup_frames = int(getattr(pipeline, "warmup_frames", 0) or 0) if warmup else 0
    try:
        if warmup_frames > 0:
            warmup_started = time.perf_counter()
            pipeline.warmup(warmup_frames)
            warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

        video_reports = [
            _run_video_with_pipeline(
                pipeline,
                path,
                config=config,
                max_frames=max(0, int(max_frames)),
                capture_factory=capture_factory or cv2.VideoCapture,
            )
            for path in paths
        ]
    finally:
        pipeline.close()

    return {
        "schema_version": 1,
        "tool": "module_a_tuning_offline",
        "pipeline_contract": "VideoDefensePipeline/rebuilt",
        "configuration": {
            "config_path": str(
                Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
            ),
            "profile": str(profile or "default"),
            "requested_tuning_patch": normalized_patch,
            "effective_config": config,
            "detector_impl": str(
                config.get("module_a", {}).get("detector_impl", "rebuilt")
            ),
        },
        "backend": backend_metadata,
        "initialization_ms": {
            "config_load": round(config_load_ms, 4),
            "pipeline_construct": round(pipeline_construct_ms, 4),
            "warmup": round(warmup_ms, 4),
            "warmup_frames": warmup_frames,
        },
        "videos": video_reports,
    }


def run_module_a_tuning(
    video_path: str | Path,
    tuning: str | Path | Mapping[str, Any] | None = None,
    max_frames: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility-oriented single-video entry point for the CLI wrapper."""

    return run_module_a_videos(
        [video_path],
        tuning_patch=parse_tuning_patch(tuning),
        max_frames=max_frames,
        **kwargs,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_module_a_reports(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
    stem: str = "module_a_tuning",
) -> dict[str, str]:
    """Write a compact JSON summary and a JSONL frame report."""

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{stem}_report.json"
    frames_path = out_dir / f"{stem}_frames.jsonl"

    compact = copy.deepcopy(dict(report))
    frame_rows: list[dict[str, Any]] = []
    for video in compact.get("videos", []) or []:
        if not isinstance(video, dict):
            continue
        frame_rows.extend(
            dict(row)
            for row in video.pop("frames", []) or []
            if isinstance(row, Mapping)
        )
    compact["frame_report"] = str(frames_path)
    report_path.write_text(
        json.dumps(_jsonable(compact), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with frames_path.open("w", encoding="utf-8") as fp:
        for row in frame_rows:
            fp.write(json.dumps(_jsonable(row), ensure_ascii=False))
            fp.write("\n")
    return {
        "report": str(report_path),
        "frames": str(frames_path),
    }
