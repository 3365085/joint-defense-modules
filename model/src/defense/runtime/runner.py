from __future__ import annotations

import math
import os
import platform
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlparse

import cv2

from defense.pipelines.video_decoder import VideoDecoder
from defense.pipelines.video_decoder_factory import create_video_decoder
from defense.visualization import encode_jpeg, render_preview
from defense.web.overlay_timeline import interpolate_overlay

from .authoritative_model import validate_artifact_binding
from .config import (
    load_runtime_config,
    normalize_custom_model_options,
    project_root,
    workspace_asset_roots,
    workspace_material_root,
    workspace_root,
    write_config_snapshot,
)
from .evidence import EvidenceSession
from .frame_processor import FrameProcessor, build_branch_cards, ProcessedFrame
from .backend_pipeline import (
    DetectionBus,
    FramePacket,
    PreviewBus,
    SharedFrameLease,
)
from .overlay_records import (
    annotate_alert_display_context,
    build_overlay_record,
    preview_module_info_from_overlay,
)
from .pipeline_factory import PipelineCache


_INVISIBLE_PATH_CHARS = {
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\u00a0",
}
_DETECTOR_FPS_WINDOW_FRAMES = 240


def _empty_decoder_status(requested_backend: str = "nvdec") -> dict[str, Any]:
    return {
        "requested_backend": str(requested_backend or "nvdec"),
        "backend": "not_started",
        "effective_backend": "not_started",
        "codec": "not_started",
        "gpu_device": "cuda:0",
        "output_format": "not_started",
        "frame_device": "not_started",
        "decode_p50_ms": 0.0,
        "decode_p95_ms": 0.0,
        "decode_ms": {"p50": 0.0, "p95": 0.0},
        "d2d_copy_p50_ms": 0.0,
        "d2d_copy_p95_ms": 0.0,
        "d2d_copy_ms": {"p50": 0.0, "p95": 0.0},
        "gpu_to_cpu_copy_p50_ms": 0.0,
        "gpu_to_cpu_copy_p95_ms": 0.0,
        "gpu_to_cpu_copy_ms": {"p50": 0.0, "p95": 0.0},
        "frames_decoded": 0,
        "bytes_decoded": 0,
        "fallback_count": 0,
        "fallback_reason": "not_started",
        "fallback_reasons": [],
        "close_error": "not_started",
        "closed": False,
        "eof": False,
        "surface_clone_policy": "not_started",
        "source_alias_mode": "not_started",
        "source_alias_cleanup_error": "",
        "decode_source": None,
        "derived_cache_used": False,
        "derived_cache_validation": "not_used",
        "source_sha256": None,
        "decode_source_sha256": None,
        "derived_metadata_path": None,
        "derived_metadata_sha256": None,
        "source_asset_id": None,
        "source_role": None,
        "source_label": None,
        "source_attack_type": None,
        "source_codec": None,
        "derived_codec": None,
        "derived_profile_id": None,
        "derived_profile_sha256": None,
        "derived_expected_frame_count": 0,
        "derived_expected_duration_s": 0.0,
        "transcode_decode_backend": None,
        "transcode_encode_backend": None,
        "derived_frame_parity": False,
        "derived_frame_count_match": False,
        "derived_fps_match": False,
    }


def _decoder_status_contract(snapshot: dict[str, Any]) -> dict[str, Any]:
    requested = str(snapshot.get("requested_backend") or "nvdec")
    effective = str(
        snapshot.get("effective_backend")
        or snapshot.get("backend")
        or "not_started"
    )
    decode_p50 = float(snapshot.get("decode_ms_p50") or 0.0)
    decode_p95 = float(snapshot.get("decode_ms_p95") or 0.0)
    d2d_p50 = float(snapshot.get("d2d_copy_ms_p50") or 0.0)
    d2d_p95 = float(snapshot.get("d2d_copy_ms_p95") or 0.0)
    d2h_p50 = float(snapshot.get("d2h_copy_ms_p50") or 0.0)
    d2h_p95 = float(snapshot.get("d2h_copy_ms_p95") or 0.0)
    fallback_reason = str(snapshot.get("fallback_reason") or "none")
    close_error = str(snapshot.get("close_error") or "none")
    frame_device = str(snapshot.get("frame_device") or "host")
    gpu_device = snapshot.get("gpu_device")
    return {
        **snapshot,
        "requested_backend": requested,
        "backend": effective,
        "effective_backend": effective,
        "codec": str(snapshot.get("codec") or "unknown"),
        "gpu_device": str(gpu_device or ("cpu" if frame_device == "host" else "unknown")),
        "output_format": str(snapshot.get("output_format") or "unknown"),
        "frame_device": frame_device,
        "decode_p50_ms": decode_p50,
        "decode_p95_ms": decode_p95,
        "decode_ms": {"p50": decode_p50, "p95": decode_p95},
        "d2d_copy_p50_ms": d2d_p50,
        "d2d_copy_p95_ms": d2d_p95,
        "d2d_copy_ms": {"p50": d2d_p50, "p95": d2d_p95},
        "gpu_to_cpu_copy_p50_ms": d2h_p50,
        "gpu_to_cpu_copy_p95_ms": d2h_p95,
        "gpu_to_cpu_copy_ms": {"p50": d2h_p50, "p95": d2h_p95},
        "frames_decoded": int(snapshot.get("frames_decoded") or 0),
        "bytes_decoded": int(snapshot.get("bytes_decoded") or 0),
        "fallback_count": int(snapshot.get("fallback_count") or 0),
        "fallback_reason": fallback_reason,
        "fallback_reasons": list(snapshot.get("fallback_reasons") or []),
        "close_error": close_error,
        "closed": bool(snapshot.get("closed", False)),
        "eof": bool(snapshot.get("eof", False)),
        "decode_source": snapshot.get("decode_source"),
        "derived_cache_used": bool(
            snapshot.get("derived_cache_used", False)
        ),
        "derived_cache_validation": str(
            snapshot.get("derived_cache_validation") or "not_used"
        ),
        "source_sha256": snapshot.get("source_sha256"),
        "decode_source_sha256": snapshot.get("decode_source_sha256"),
        "derived_metadata_path": snapshot.get("derived_metadata_path"),
        "derived_metadata_sha256": snapshot.get(
            "derived_metadata_sha256"
        ),
        "source_asset_id": snapshot.get("source_asset_id"),
        "source_role": snapshot.get("source_role"),
        "source_label": snapshot.get("source_label"),
        "source_attack_type": snapshot.get("source_attack_type"),
        "source_codec": snapshot.get("source_codec"),
        "derived_codec": snapshot.get("derived_codec"),
        "derived_profile_id": snapshot.get("derived_profile_id"),
        "derived_profile_sha256": snapshot.get(
            "derived_profile_sha256"
        ),
        "derived_expected_frame_count": max(
            0,
            int(snapshot.get("derived_expected_frame_count") or 0),
        ),
        "derived_expected_duration_s": max(
            0.0,
            float(snapshot.get("derived_expected_duration_s") or 0.0),
        ),
        "transcode_decode_backend": snapshot.get(
            "transcode_decode_backend"
        ),
        "transcode_encode_backend": snapshot.get(
            "transcode_encode_backend"
        ),
        "derived_frame_parity": bool(
            snapshot.get("derived_frame_parity", False)
        ),
        "derived_frame_count_match": bool(
            snapshot.get("derived_frame_count_match", False)
        ),
        "derived_fps_match": bool(
            snapshot.get("derived_fps_match", False)
        ),
    }


def _timing_distribution(samples: deque[float]) -> dict[str, float | int]:
    values = [
        float(value)
        for value in samples
        if math.isfinite(float(value)) and float(value) >= 0.0
    ]
    if not values:
        return {
            "count": 0,
            "latest": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(values)

    def percentile(ratio: float) -> float:
        position = (len(ordered) - 1) * min(1.0, max(0.0, ratio))
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return (
            ordered[lower] * (1.0 - weight)
            + ordered[upper] * weight
        )

    return {
        "count": len(values),
        "latest": values[-1],
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _evidence_writer_status_contract(
    evidence: EvidenceSession | Any | None = None,
    *,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = dict(snapshot or {})
    if evidence is not None and snapshot is None:
        writer_status = getattr(evidence, "writer_status", None)
        if callable(writer_status):
            try:
                data = dict(writer_status() or {})
            except Exception as exc:
                data = {
                    "enabled": bool(getattr(evidence, "enabled", False)),
                    "alive": False,
                    "failed": 1,
                    "last_error": (
                        "evidence_writer_status_failed:"
                        f"{type(exc).__name__}:{exc}"
                    ),
                }
        else:
            data = {
                "enabled": bool(getattr(evidence, "enabled", False)),
                "alive": False,
            }
    normalized = {
        "enabled": bool(data.get("enabled", False)),
        "alive": bool(data.get("alive", False)),
        "queue_capacity": max(0, int(data.get("queue_capacity") or 0)),
        "pending": max(0, int(data.get("pending") or 0)),
        "completed": max(0, int(data.get("completed") or 0)),
        "failed": max(0, int(data.get("failed") or 0)),
        "queue_full": max(0, int(data.get("queue_full") or 0)),
        "drain_ms": max(0.0, float(data.get("drain_ms") or 0.0)),
        "last_error": str(data.get("last_error") or ""),
    }
    return {
        "evidence_writer": normalized,
        **{
            f"evidence_writer_{key}": value
            for key, value in normalized.items()
        },
    }


def _authoritative_status_contract(
    authoritative: dict[str, Any],
    *,
    fallback_backend: str = "tensorrt",
) -> dict[str, Any]:
    artifacts = (
        authoritative.get("artifacts", {})
        if isinstance(authoritative.get("artifacts"), dict)
        else {}
    )
    source = (
        dict(authoritative.get("source") or {})
        if isinstance(authoritative.get("source"), dict)
        else {}
    )
    engine = (
        dict(artifacts.get("engine") or {})
        if isinstance(artifacts.get("engine"), dict)
        else {}
    )
    onnx = (
        dict(artifacts.get("onnx") or {})
        if isinstance(artifacts.get("onnx"), dict)
        else {}
    )
    source_sha256 = source.get("sha256")
    if source_sha256:
        engine.setdefault("source_sha256", source_sha256)
        onnx.setdefault("source_sha256", source_sha256)
    return {
        "model_id": authoritative.get(
            "model_id",
            "mask_bd_v4_clean_baseline",
        ),
        "locked": True,
        "metadata_valid": bool(
            authoritative.get("metadata_valid", False)
        ),
        "source": source or None,
        "engine": engine or None,
        "onnx": onnx or None,
        "metadata_path": authoritative.get("metadata_path"),
        "class_names": authoritative.get(
            "class_names",
            ["helmet", "head"],
        ),
        "image_size": authoritative.get("image_size", 640),
        "backend": authoritative.get("backend", fallback_backend),
        "half": bool(authoritative.get("half", True)),
    }


def _native_status_contract(snapshot: dict[str, Any]) -> dict[str, Any]:
    fallback_reason = str(
        snapshot.get("fallback_reason")
        or snapshot.get("load_error")
        or "none"
    )
    enabled_stages = list(snapshot.get("enabled_stages") or [])
    return {
        **snapshot,
        "version": str(
            snapshot.get("crate_version")
            or snapshot.get("version")
            or "unknown"
        ),
        "enabled_stages": enabled_stages,
        "fallback_reason": fallback_reason,
    }


def _empty_a3b_health() -> dict[str, Any]:
    return {
        "a3b_background_enabled": False,
        "a3b_generation": 0,
        "a3b_active_worker_count": 0,
        "a3b_retired_worker_count": 0,
        "a3b_live_worker_count": 0,
        "a3b_global_live_worker_count": 0,
        "a3b_global_worker_limit": 0,
        "a3b_worker_limit_scope": "process",
        "a3b_worker_timeout_s": 0.0,
        "a3b_max_retired_workers": 0,
        "a3b_active_worker_started_at": None,
        "a3b_active_worker_age_s": 0.0,
        "a3b_active_worker_frame_idx": None,
        "a3b_active_worker_timestamp": None,
        "a3b_timed_out_worker_count": 0,
        "a3b_worker_rejected_count": 0,
        "a3b_last_worker_rejected_at": None,
        "a3b_schedule_blocked": False,
        "a3b_schedule_blocked_reason": "none",
        "a3b_error_count": 0,
        "a3b_last_error": None,
        "a3b_last_error_at": None,
        "a3b_last_success_at": None,
        "a3b_source_frame_idx": None,
        "a3b_source_timestamp": None,
        "a3b_source_fps": 0.0,
        "a3b_source_interval_frames": 0,
        "media_source_frame_units": 0,
        "media_tighten_aspect_ratio": 0.0,
        "media_tighten_aspect_pass": False,
        "a3b_last_attempt_frame_idx": None,
        "a3b_last_attempt_timestamp": None,
        "a3b_result_published_at": None,
        "a3b_result_age_s": 0.0,
        "a3b_result_lease_s": 0.0,
        "a3b_result_fresh": False,
        "a3b_result_expired_count": 0,
        "a3b_result_seq": 0,
    }


def _empty_module_a_effective_config() -> dict[str, Any]:
    return {
        "detector_impl": None,
        "analysis_max_hz": None,
        "detector_process_fps_cap": None,
        "a3b_sensitivity": None,
        "a3b_source_keyword_policy": "diagnostic_only",
        "a3b_source_keyword_match_required": False,
        "a3b_observed_only_source_keywords": [],
        "a3b_trigger_source_keywords": [],
        "static_image_enabled": None,
        "static_image_interval": None,
        "static_image_worker_timeout_s": None,
        "static_image_result_lease_s": None,
        "static_image_max_retired_workers": None,
        "static_image_global_worker_limit": None,
        "rebuilt_theta_media_raw": None,
        "rebuilt_theta_media": None,
        "rebuilt_theta_adv": None,
        "rebuilt_theta_blind": None,
        "rebuilt_blind_confirm_ratio": None,
        "rebuilt_alert_hold_frames": None,
        "rebuilt_a3b_alert_hold_frames": None,
        "rebuilt_alert_hold_refresh_on_padv": None,
        "rebuilt_adv_candidate_bridge_frames": None,
        "rebuilt_a4_classifier_rescue_underexposed_max": None,
        "rebuilt_sustained_adv_escalation": None,
        "rebuilt_sustained_adv_seconds": None,
        "rebuilt_sustained_adv_run_mult": None,
        "rebuilt_sustained_adv_benign_decay": None,
        "rebuilt_sustained_adv_require_target": None,
        "rebuilt_sustained_adv_require_physical_support": None,
        "rebuilt_sustained_adv_exclude_static_bg": None,
        "rebuilt_sustained_adv_recent_target_min": None,
        "rebuilt_blind_sustained_escalation": None,
        "rebuilt_blind_sustained_floor": None,
        "rebuilt_blind_sustained_degrade_min": None,
        "rebuilt_blind_sustained_established_min": None,
        "rebuilt_a3b_independent_trigger": None,
        "rebuilt_a3b_tighten_gate": None,
        "rebuilt_a3b_gate_candidate_min": None,
        "rebuilt_a3b_gate_edge_min": None,
        "rebuilt_a3b_gate_edge_max": None,
        "rebuilt_a3b_gate_border_contrast_min": None,
        "rebuilt_a3b_soft_gate_candidate_tolerance": None,
        "rebuilt_a3b_soft_gate_aspect_ratio_min": None,
        "rebuilt_a3b_soft_gate_aspect_ratio_max": None,
        "rebuilt_a3b_media_run_floor": None,
        "rebuilt_a3b_media_run_gap_tol": None,
        "flow_requested_device": None,
        "flow_effective_device": None,
        "flow_backend": None,
        "flow_fallback_reason": None,
        "a4_classifier_configured": None,
        "a4_classifier_loaded": None,
        "a4_classifier_error": None,
        "a4_classifier_fallback_reason": None,
        "a4_classifier_alarm_window": None,
        "a4_classifier_alarm_required_hits": None,
    }


def normalize_source_text(source: str) -> str:
    text = str(source or "")
    text = "".join(ch for ch in text if ch not in _INVISIBLE_PATH_CHARS).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if text.lower().startswith("file:"):
        parsed = urlparse(text)
        if parsed.scheme.lower() == "file":
            if parsed.netloc and parsed.netloc not in {"localhost", "127.0.0.1"}:
                text = f"//{parsed.netloc}{parsed.path}"
            else:
                text = parsed.path
            text = unquote(text)
            if platform.system().lower().startswith("win") and len(text) >= 3 and text[0] == "/" and text[2] == ":":
                text = text[1:]
    else:
        text = unquote(text)
    return text.strip()


def _path_parts_for_alias(path: Path, source_text: str = "") -> tuple[str, ...]:
    text = normalize_source_text(source_text or str(path))
    windows_like = "\\" in text or (len(text) >= 2 and text[1] == ":")
    if windows_like:
        return tuple(str(part) for part in PureWindowsPath(text).parts)
    return tuple(str(part) for part in path.parts)


def _path_alias_candidates(path: Path, source_text: str = "") -> list[Path]:
    candidates: list[Path] = []
    parts = _path_parts_for_alias(path, source_text)
    if len(parts) >= 3:
        # Old documents and UI screenshots may point at a former top-level
        # material root such as D:\联合防御模块训练素材\... .  The current project
        # keeps those same relative folders under D:\security_project_d\素材.
        tail_after_top = Path(*parts[2:])
        for root in workspace_asset_roots():
            candidates.append(root / tail_after_top)

    marker_names = {
        "素材",
        "训练素材",
        "联合防御模块训练素材",
        "模型和素材",
    }
    for index, part in enumerate(parts):
        if part in marker_names and index + 1 < len(parts):
            tail = Path(*parts[index + 1 :])
            for root in workspace_asset_roots():
                candidates.append(root / tail)

    max_tail = min(5, max(0, len(parts) - 1))
    for tail_len in range(2, max_tail + 1):
        tail_parts = parts[-tail_len:]
        if any(part.endswith(":") or part.endswith(":\\") for part in tail_parts):
            continue
        tail = Path(*tail_parts)
        for root in workspace_asset_roots():
            candidates.append(root / tail)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_source_path(source: str) -> Path:
    text = normalize_source_text(source)
    path = Path(text).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        for candidate in _path_alias_candidates(path, text):
            if candidate.exists():
                return candidate
        return path
    fallback = project_root() / path
    if fallback.exists():
        return fallback
    for root in workspace_asset_roots():
        candidate = root / path
        if candidate.exists():
            return candidate
    for candidate in _path_alias_candidates(path, text):
        if candidate.exists():
            return candidate
    return fallback


def _decoder_zero_frame_message(
    decoder: VideoDecoder,
    source: str | Path,
) -> str:
    try:
        snapshot = dict(decoder.status_snapshot())
    except Exception as exc:
        return (
            "decoder_zero_frame_eof:"
            f"source={source}:status_error={type(exc).__name__}:{exc}"
        )
    return (
        "decoder_zero_frame_eof:"
        f"source={source}:"
        f"backend={snapshot.get('effective_backend') or snapshot.get('backend')}:"
        f"codec={snapshot.get('codec')}:"
        f"frames_decoded={int(snapshot.get('frames_decoded') or 0)}:"
        f"derived_cache_used={bool(snapshot.get('derived_cache_used', False))}:"
        f"derived_cache_validation={snapshot.get('derived_cache_validation')}:"
        f"fallback_count={int(snapshot.get('fallback_count') or 0)}:"
        f"fallback_reason={snapshot.get('fallback_reason') or 'none'}"
    )


def validate_file_source(
    source: str,
    *,
    preference: str = "nvdec",
    allow_cpu_fallback: bool = True,
    gpu_id: int = 0,
) -> Path:
    path = resolve_source_path(source)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"视频文件不存在或不可访问: {path}")
    decoder: VideoDecoder | None = None
    lease = None
    try:
        decoder = create_video_decoder(
            path,
            preference=str(preference or "nvdec"),
            allow_cpu_fallback=bool(allow_cpu_fallback),
            gpu_id=max(0, int(gpu_id)),
        )
        lease = decoder.read()
        if lease is None:
            raise RuntimeError(_decoder_zero_frame_message(decoder, path))
    finally:
        if lease is not None:
            lease.release()
        if decoder is not None:
            decoder.close()
    return path


def _probe_tcp_reachable(url: str, timeout_ms: int = 3000) -> None:
    """对 http(s) 流地址做一次短超时 TCP 连接探活。不可达立即抛 RuntimeError,
    避免 OpenCV/FFMPEG 在设备离线时走内部重试阻塞十几秒。"""
    import socket

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    sock = None
    try:
        sock = socket.create_connection((host, int(port)), timeout=max(0.5, timeout_ms / 1000.0))
    except OSError as exc:
        raise RuntimeError(f"网络流不可达 {host}:{port} ({exc})") from exc
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def open_capture(source_type: str, source: str, timeout_ms: int = 5000) -> cv2.VideoCapture:
    source_type = str(source_type or "file").lower()
    if source_type == "camera":
        try:
            camera_text = normalize_source_text(source)
            if camera_text.lower().startswith("camera:"):
                camera_text = camera_text.split(":", 1)[1].strip()
            index = int(camera_text)
        except ValueError as exc:
            raise ValueError("摄像头输入必须是编号，例如 0 或 1") from exc
        if platform.system().lower().startswith("win"):
            # Windows 依次尝试 DSHOW → MSMF → 默认后端, 命中第一个能打开的。
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            for backend in (getattr(cv2, "CAP_MSMF", cv2.CAP_ANY), cv2.CAP_ANY):
                if cap.isOpened():
                    break
                cap.release()
                cap = cv2.VideoCapture(index, backend)
        else:
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    elif source_type == "file":
        path = resolve_source_path(source)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {path}")
        cap = cv2.VideoCapture(str(path))
    elif source_type == "rtsp":
        url = normalize_source_text(source)
        scheme = urlparse(url).scheme.lower()
        # HTTP(S) MJPEG 网络摄像头(如 192.168.x.x:8081/video)用默认后端即可秒开;
        # FFMPEG 后端对 HTTP multipart 流会阻塞卡死, 只对真正的 rtsp/rtmp 用它。
        if scheme in {"http", "https"}:
            # HTTP MJPEG 网络摄像头在线时默认后端秒开; 但设备离线/拒连时
            # OpenCV 会走 FFMPEG tcp 重试卡 ~14s, 故先做一次短超时 TCP 探活,
            # 不可达则立即抛错, 避免整条链路阻塞。
            _probe_tcp_reachable(url, timeout_ms=timeout_ms)
            cap = cv2.VideoCapture()
            # 设读超时: 摄像头 TCP 保持但停止推帧(设备假死)时, 让 read() 有上界返回,
            # 采集循环的 5s 无帧重连才能真正兜底, 不会永久阻塞在 read()。
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_ms))
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_ms))
            except Exception:
                pass
            cap.open(url, cv2.CAP_ANY)
        else:
            # 真 rtsp/rtmp: OPEN_TIMEOUT_MSEC 对 FFMPEG 后端常不生效(实测不可达地址
            # 阻塞 ~30s), 故 (1) 先短超时 TCP 探活秒判不可达, (2) 用环境变量向 FFMPEG
            # 注入 stimeout(微秒)兜底连接/读取超时。
            _probe_tcp_reachable(url, timeout_ms=timeout_ms)
            stimeout_us = int(max(1, timeout_ms) * 1000)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"stimeout;{stimeout_us}|rtsp_transport;tcp"
            cap = cv2.VideoCapture()
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_ms))
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_ms))
            except Exception:
                pass
            cap.open(url, cv2.CAP_FFMPEG)
    else:
        raise ValueError(f"不支持的输入类型: {source_type}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开输入源: {source}")
    return cap


def sample_sources() -> list[dict[str, str]]:
    """Return known demo sources under the current project layout.

    The current tree keeps code under ``Model A`` and media/model bundles in
    sibling workspace directories.  Return absolute file paths so browser and
    server side clocks resolve the same source regardless of working directory.
    """
    root = project_root()
    workspace = workspace_root()
    material_root = workspace_material_root()
    candidates = [
        material_root,
        root / "samples",
        workspace / "samples",
        workspace / "训练素材",
        workspace / "模型和素材",
    ]
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for base in candidates:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.mp4")):
            try:
                resolved_key = str(path.resolve())
            except OSError:
                resolved_key = str(path.absolute())
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            try:
                label = str(path.relative_to(base))
            except ValueError:
                label = path.name
            label = label.replace("\\", "/")
            out.append({"label": label, "source_type": "file", "source": str(path)})
            if len(out) >= 200:
                return out
    return out


def scan_camera_devices(max_index: int = 8) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for index in range(max(0, int(max_index))):
        cap = None
        try:
            cap = open_capture("camera", str(index), timeout_ms=800)
            ok = False
            frame = None
            for _ in range(8):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.06)
            if ok and frame is not None:
                devices.append(
                    {
                        "index": index,
                        "label": f"Camera {index}",
                        "source": str(index),
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
    return devices


def _is_preview_scene_cut(previous_frame: Any, frame: Any) -> bool:
    if previous_frame is None or frame is None:
        return False
    if previous_frame.shape[:2] != frame.shape[:2]:
        return True
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    previous_small = cv2.resize(previous_gray, (64, 64), interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current_gray, (64, 64), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(previous_small, current_small)
    return bool(float(diff.mean()) >= 45.0 and float((diff >= 25).mean()) >= 0.65)


def _file_realtime_wait_s(
    *,
    playback_anchor_wall: float,
    playback_anchor_frame: float,
    next_frame: float,
    fps: float,
    speed: float,
    now: float | None = None,
) -> float:
    """Return absolute-clock wait time for the next realtime file frame.

    Per-iteration sleeps permanently accumulate any GIL/decoder scheduling
    delay. Anchoring each frame to the source clock lets a delayed capture
    iteration catch up on the next frame without skipping source frames.
    """

    current = time.perf_counter() if now is None else float(now)
    frame_delta = max(0.0, float(next_frame) - float(playback_anchor_frame))
    target_wall = float(playback_anchor_wall) + frame_delta / (
        max(1.0, float(fps)) * max(0.1, float(speed))
    )
    return max(0.0, target_wall - current)


def _detector_completion_fps(completion_times: deque[float]) -> float:
    if len(completion_times) < 2:
        return 0.0
    elapsed = completion_times[-1] - completion_times[0]
    if elapsed <= 1e-6:
        return 0.0
    return (len(completion_times) - 1) / elapsed


def _detector_can_follow_file_source(
    detector_process_fps_cap: float,
    effective_capture_fps: float,
) -> bool:
    source_fps = max(0.0, float(effective_capture_fps or 0.0))
    detector_cap = max(0.0, float(detector_process_fps_cap or 0.0))
    if source_fps <= 0.0 or detector_cap <= 0.0:
        return False
    # Nominal 60 FPS files often report 60.49/60.50. A cap within one percent
    # is the same intended cadence; strict comparison otherwise alternates
    # submissions and incorrectly collapses coverage to roughly 50 percent.
    return detector_cap >= source_fps * 0.99


def _adaptive_file_overlay_bridge_s(
    status: dict[str, Any],
    records: list[dict[str, Any]],
) -> float:
    """Return a bounded causal hold window for asynchronous file preview.

    The configured detector cap is only an admission ceiling, not measured
    throughput.  Using it as the overlay cadence made a nominal 60 FPS cap
    collapse the bridge to the fixed 200 ms minimum even when an expensive
    detector/A3b cycle took longer.  Derive the live bridge from completed
    detector cadence and recent source-time gaps, while keeping the configured
    bridge and ``overlay_max_age_ms`` as lower/hard upper bounds.
    """

    detector_cap = max(
        1.0,
        float(status.get("detector_process_fps_cap") or 15.0),
    )
    bridge_frames = max(
        1.0,
        float(status.get("file_realtime_overlay_bridge_frames") or 3.2),
    )
    bridge_min_s = max(
        0.0,
        float(status.get("file_realtime_overlay_bridge_min_s") or 0.20),
    )
    configured_max_s = max(
        bridge_min_s,
        float(status.get("file_realtime_overlay_bridge_max_s") or 0.36),
    )
    max_age_s = max(
        bridge_min_s,
        float(status.get("overlay_max_age_ms") or 950.0) / 1000.0,
    )
    configured_bridge_s = min(
        configured_max_s,
        max(bridge_min_s, bridge_frames / detector_cap),
    )

    playback_speed = max(0.1, float(status.get("playback_speed") or 1.0))
    preview_fps = max(
        1.0,
        float(status.get("preview_render_fps") or 25.0),
    )
    observed_periods_s: list[float] = []

    measured_fps = float(status.get("fps") or 0.0)
    if math.isfinite(measured_fps) and measured_fps > 0.0:
        observed_periods_s.append(playback_speed / measured_fps)

    cycle_distribution = status.get("detector_cycle_ms_distribution")
    if isinstance(cycle_distribution, dict):
        cycle_p95_ms = float(cycle_distribution.get("p95") or 0.0)
        if math.isfinite(cycle_p95_ms) and cycle_p95_ms > 0.0:
            observed_periods_s.append(
                playback_speed * cycle_p95_ms / 1000.0
            )

    recent_times = [
        float(item.get("video_time_s") or 0.0)
        for item in records[-33:]
    ]
    recent_gaps = [
        right - left
        for left, right in zip(recent_times, recent_times[1:])
        if math.isfinite(left)
        and math.isfinite(right)
        and 0.0 < right - left <= max_age_s
    ]
    if recent_gaps:
        ordered_gaps = sorted(recent_gaps)
        p95_index = int(math.ceil(0.95 * len(ordered_gaps))) - 1
        observed_periods_s.append(
            ordered_gaps[max(0, min(p95_index, len(ordered_gaps) - 1))]
        )

    if not observed_periods_s:
        return min(max_age_s, configured_bridge_s)

    preview_period_s = playback_speed / preview_fps
    adaptive_bridge_s = (
        max(observed_periods_s) * bridge_frames + preview_period_s
    )
    return min(
        max_age_s,
        max(configured_bridge_s, adaptive_bridge_s),
    )


def _causal_interpolated_overlay(
    previous: dict[str, Any],
    interpolated: dict[str, Any],
    source_time_s: float,
) -> dict[str, Any]:
    """Interpolate geometry without applying a future discrete alarm state."""

    causal = dict(previous)
    causal.update(
        {
            "video_time_s": float(source_time_s),
            "source_time_s": float(source_time_s),
            "ppe_tracks": [
                dict(track)
                for track in interpolated.get("ppe_tracks", []) or []
            ],
            "interpolated": True,
        }
    )
    return causal


class MonitorEngine:
    """Runtime service boundary for capture, clocking, inference and evidence.

    MP4, RTSP and camera inputs all use the backend source pipeline. Preview
    frames are published by the source clock, while inference consumes a
    latest-only queue so slow detection can drop stale work without stalling the
    displayed stream.
    """

    def _overlay_timeline_maxlen(self) -> int:
        """Dynamic timeline capacity based on expected runtime (2026-06-11 架构修复).

        Base 1000 frames + 120 seconds at source fps = covers up to 2 minutes
        for long runs at any typical frame rate.
        """
        src_fps = getattr(self, '_source_fps', 25.0)
        return max(1000, int(src_fps * 120))

    def __init__(self, cache: PipelineCache) -> None:
        self.cache = cache
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.process_thread: threading.Thread | None = None
        self.preview_thread: threading.Thread | None = None
        self.thread_join_timeout_s = 2.0
        self.preview_bus: PreviewBus | None = None
        self.detection_bus: DetectionBus | None = None
        self.run_id = 0
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.latest_jpeg_meta: dict[str, Any] = {}
        self.preview_publish_times: deque[float] = deque(maxlen=60)
        # A paced 25 FPS file can only complete at roughly 25 FPS. Keep the
        # completion-rate horizon aligned with cycle distributions so a single
        # end-of-file scheduling outlier cannot turn full zero-drop coverage
        # into a false 24.8/24.9 FPS failure.
        self.detect_times: deque[float] = deque(
            maxlen=_DETECTOR_FPS_WINDOW_FRAMES
        )
        self.detector_cycle_samples: deque[float] = deque(maxlen=240)
        self.evidence_update_samples: deque[float] = deque(maxlen=240)
        self.overlay_publish_samples: deque[float] = deque(maxlen=240)
        self.process_done_event = threading.Event()
        self.detector_drain_timeout_s = 10.0
        self.overlay_timeline: deque[dict[str, Any]] = deque(maxlen=self._overlay_timeline_maxlen())
        self.overlay_seq = 0
        self.status: dict[str, Any] = self._empty_status()
        self.display_options: dict[str, bool] = {
            "show_boxes": True,
            "show_person_boxes": True,
            "show_module_hud": True,
            "show_ppe_hud": True,
        }
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=20)
        self.recent_ppe_events: deque[dict[str, Any]] = deque(maxlen=20)
        self.recent_source_events: deque[dict[str, Any]] = deque(maxlen=20)
        self._source_fps = 25.0
        self._source_duration_s = 0.0
        self._source_frame_count = 0
        self._capture_done = False
        self._source_epoch = 0
        self._seek_request_s: float | None = None
        self._playback_paused = False
        self._playback_speed = 1.0
        self._preview_last_overlay: dict[str, Any] | None = None
        self._release_pipeline_cache_when_stopped = False
        self._active_file_decoder: VideoDecoder | None = None
        self._hydrate_idle_production_status()

    def _hydrate_idle_production_status(self) -> None:
        """Expose static production bindings before the first video starts."""

        if not isinstance(self.cache, PipelineCache):
            return
        idle_errors: list[str] = []
        try:
            config_path = getattr(self.cache, "config_path", None)
            root = getattr(self.cache, "root", project_root())
            config = load_runtime_config(
                config_path=config_path,
                profile="default",
            )
            inference = (
                config.get("inference", {})
                if isinstance(config.get("inference"), dict)
                else {}
            )
            runtime = (
                config.get("runtime", {})
                if isinstance(config.get("runtime"), dict)
                else {}
            )
            authoritative = validate_artifact_binding(config, root)
            authoritative_status = _authoritative_status_contract(
                authoritative,
                fallback_backend=str(
                    inference.get("backend") or "tensorrt"
                ),
            )
            engine = authoritative_status.get("engine")
            artifact_path = (
                str(engine.get("path"))
                if isinstance(engine, dict) and engine.get("path")
                else None
            )
            decoder_requested = str(
                runtime.get("video_decoder_preference", "nvdec")
                or "nvdec"
            )
            decoder_status = _empty_decoder_status(decoder_requested)
            decoder_status["gpu_device"] = (
                f"cuda:{int(runtime.get('video_decoder_gpu_id', 0) or 0)}"
            )
            self.status.update(
                {
                    "backend": str(
                        inference.get("backend") or "tensorrt"
                    ),
                    "model_family": str(
                        inference.get("model_family") or "ultralytics"
                    ),
                    "artifact": artifact_path,
                    "authoritative_model": authoritative_status,
                    "decoder": decoder_status,
                }
            )
        except Exception as exc:
            idle_errors.append(
                "idle_production_binding_failed:"
                f"{type(exc).__name__}:{exc}"
            )

        try:
            from defense.module_a.native_bridge import status as native_status

            self.status["native"] = _native_status_contract(native_status())
        except Exception as exc:
            idle_errors.append(
                "idle_native_status_failed:"
                f"{type(exc).__name__}:{exc}"
            )
            self.status["native"] = {
                "available": False,
                "version": "unknown",
                "binary_sha256": None,
                "enabled_stages": [],
                "fallback_reason": (
                    f"{type(exc).__name__}:{exc}"
                ),
            }

        if idle_errors:
            self.status["idle_status_errors"] = idle_errors
        self.status["branch_cards"] = build_branch_cards(self.status)

    @staticmethod
    def _normalize_start_feature_options(
        feature_options: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return only options explicitly supplied by the caller."""
        if feature_options is None:
            return {}, {}
        normalized: dict[str, Any] = {}
        if "static_image_enabled" in feature_options:
            normalized["static_image_enabled"] = bool(
                feature_options.get("static_image_enabled")
            )
        if "a3b_sensitivity" in feature_options:
            normalized["a3b_sensitivity"] = str(
                feature_options.get("a3b_sensitivity") or "balanced"
            )
        return dict(normalized), dict(normalized)

    @staticmethod
    def _empty_status() -> dict[str, Any]:
        a3b_health = _empty_a3b_health()
        status = {
            "run_id": 0,
            "running": False,
            "source_type": None,
            "source": None,
            "profile": "default",
            "backend": None,
            "model_family": None,
            "artifact": None,
            "authoritative_model": {
                "model_id": "mask_bd_v4_clean_baseline",
                "locked": True,
                "metadata_valid": False,
                "source": None,
                "engine": None,
                "onnx": None,
                "class_names": ["helmet", "head"],
                "image_size": 640,
                "backend": "tensorrt",
            },
            "decoder": _empty_decoder_status(),
            "frame_idx": 0,
            "fps": 0.0,
            "preview_fps": 0.0,
            "preview_seq": 0,
            "source_fps": 0.0,
            "source_duration_s": 0.0,
            "source_frame_count": 0,
            "dropped_frames": 0,
            "timing_ms": 0.0,
            "processing_ms": 0.0,
            "detector_cycle_ms": 0.0,
            "detector_compute_fps": 0.0,
            "evidence_update_ms": 0.0,
            "overlay_status_publish_ms": 0.0,
            "detector_cycle_ms_distribution": _timing_distribution(deque()),
            "evidence_update_ms_distribution": _timing_distribution(deque()),
            "overlay_status_publish_ms_distribution": _timing_distribution(
                deque()
            ),
            "detector_inference_ms": 0.0,
            "detector_preprocess_ms": 0.0,
            "detector_input_device": "not_started",
            "detector_input_format": "not_started",
            "frame_materialization_ms": 0.0,
            "previous_frame_materialization_ms": 0.0,
            "module_a_timing_ms": 0.0,
            "timing_frame_idx": 0,
            "latency_frame_idx": 0,
            "a3b_static_media_ms": 0.0,
            "target_frame_budget_ms": 0.0,
            "processing_budget_ok": True,
            "p_adv": None,
            "p_adv_display": 0.0,
            "p_adv_missing_reason": "not_started",
            "alert_confirmed": False,
            "physical_alert_confirmed": False,
            "module_a_alert_confirmed": False,
            "module_a_alert_channel": "none",
            "attack_detected": False,
            "physical_attack_detected": False,
            "module_a_attack_detected": False,
            "attack_state_active": False,
            "physical_attack_state_active": False,
            "module_a_attack_state_active": False,
            "reason": "",
            "a3b_score": 0.0,
            "a3b_confidence": 0.0,
            "a3b_observed_score": 0.0,
            "a3b_confirmed_score": 0.0,
            "a3b_display_score": 0.0,
            "a3b_event_score": 0.0,
            "a3b_state": "normal",
            "a3b_triggered": False,
            "a3b_confirmed_alert": False,
            "a3b_triggered_source": "none",
            "a3b_reason": "",
            "a3b_debug": dict(a3b_health),
            **a3b_health,
            "module_a_effective_config": _empty_module_a_effective_config(),
            "ppe_warning": False,
            "ppe_candidate": False,
            "ppe_confirmed": False,
            "ppe_confirmed_source": "",
            "ppe_event_active": False,
            "ppe_event_hold_remaining": 0,
            "ppe_event_last_reason": "",
            "ppe_event_last_confirmed_source": "",
            "ppe_person_count": 0,
            "ppe_raw_person_count": 0,
            "ppe_inferred_person_count": 0,
            "ppe_person_context_count": 0,
            "ppe_weak_person_count": 0,
            "ppe_promoted_person_count": 0,
            "ppe_effective_person_count": 0,
            "ppe_helmet_count": 0,
            "ppe_raw_helmet_count": 0,
            "ppe_weak_helmet_count": 0,
            "ppe_promoted_helmet_count": 0,
            "ppe_effective_helmet_count": 0,
            "ppe_head_count": 0,
            "ppe_raw_head_count": 0,
            "ppe_weak_head_count": 0,
            "ppe_promoted_head_count": 0,
            "ppe_effective_head_count": 0,
            "ppe_missing_helmet_count": 0,
            "ppe_has_person_class": False,
            "ppe_evidence_mode": "",
            "ppe_uncertain": False,
            "ppe_reason": "",
            "ppe_tracks": [],
            "feature_options": {"static_image_enabled": True},
            "custom_model": normalize_custom_model_options(None),
            "display_options": {
                "show_boxes": True,
                "show_person_boxes": True,
                "show_module_hud": True,
                "show_ppe_hud": True,
            },
            "evidence_session_dir": None,
            "evidence_manifest_path": None,
            "evidence_saved_event_count": 0,
            **_evidence_writer_status_contract(),
            "recent_events": [],
            "recent_ppe_events": [],
            "recent_source_auth_events": [],
            "started_at": None,
            "stopped_at": None,
            "error": "",
            "secondary_errors": [],
            "restart_blocked_reason": "",
            "stop_threads_pending": [],
            "pipeline_cache_release_deferred": False,
            "preview_mode": "idle",
            "initializing": False,
            "prewarming": False,
            "detector_ready": False,
            "first_detection_ready": False,
            "ready_for_preview": False,
            "preview_started": False,
            "preview_seekable": False,
            "source_ended": False,
            "source_eof_reached": False,
            "process_done": False,
            "detector_drain_active": False,
            "detector_drain_completed": False,
            "detector_drain_timed_out": False,
            "detector_drain_timeout_s": 0.0,
            "detector_drain_ms": 0.0,
            "detector_drain_failed_reason": "",
            "evidence_drain_active": False,
            "evidence_drain_completed": False,
            "evidence_drain_failed": False,
            "evidence_drain_ms": 0.0,
            "evidence_drain_error": "",
            "stream_reconnects": 0,
            "stream_last_frame_age_ms": 0.0,
            "source_decode_recoveries": 0,
            "source_decode_last_recovery_frame": 0,
            "preview_start_time_s": 0.0,
            "preview_render_fps": 0.0,
            "preview_max_side": 960,
            "preview_width": 0,
            "preview_height": 0,
            "capture_max_side": 1280,
            "file_source_fps_cap": 0.0,
            "source_frame_step": 1.0,
            "source_frame_skip_enabled": False,
            "detector_submit_every_file_frame": False,
            "source_frame_width": 0,
            "source_frame_height": 0,
            "capture_frame_width": 0,
            "capture_frame_height": 0,
            "capture_resized": False,
            "preview_never_wait_for_detection": True,
            "init_ms": 0.0,
            "pipeline_cache_hit": False,
            "pipeline_cache_get_ms": 0.0,
            "pipeline_config_load_ms": 0.0,
            "pipeline_backend_create_ms": 0.0,
            "pipeline_construct_ms": 0.0,
            "pipeline_warmup_ms": 0.0,
            "pipeline_warmup_frames": 0,
            "pipeline_reset_ms": 0.0,
            "release_pipeline_cache_on_file_end": False,
            "detector_thread_warmup_ms": 0.0,
            "detector_thread_warmup_frames": 0,
            "first_detection_processing_ms": 0.0,
            "first_detection_timing_ms": 0.0,
            "first_detection_detector_inference_ms": 0.0,
            "first_detection_module_a_timing_ms": 0.0,
            "first_detection_frame_idx": 0,
            "first_detection_source_time_s": 0.0,
            "video_time_s": 0.0,
            "source_time_s": 0.0,
            "source_epoch": 0,
            "overlay_seq": 0,
            "overlay_buffered_until_s": 0.0,
            "detector_lookahead_s": 0.0,
            "clock_skew_ms": 0.0,
            "overlay_match_window_ms": 180.0,
            "overlay_hold_ms": 550.0,
            "overlay_interpolate_ms": 400.0,
            "file_realtime_overlay_bridge_frames": 3.2,
            "file_realtime_overlay_bridge_min_s": 0.20,
            "file_realtime_overlay_bridge_max_s": 0.36,
            "detector_pipeline_mode": "idle",
            "detector_queue_policy": "latest_only",
            "detector_process_fps_cap": 0.0,
            "dropped_detection_frames": 0,
            "source_frames_skipped_for_realtime": 0,
            "capture_frames_published": 0,
            "detector_submission_count": 0,
            "processed_detection_frames": 0,
            "detection_source_coverage_ratio": 0.0,
            "backend_source_pipeline": True,
            "playback_paused": False,
            "playback_speed": 1.0,
            "overlay_max_age_ms": 800.0,
            "stale_overlay_dropped": 0,
            "raw_boxes_count": 0,
            "ppe_boxes_count": 0,
            "tracked_boxes_count": 0,
            "render_boxes_count": 0,
            "ppe_file_realtime_max_render_misses": 2,
            "ppe_roi_redetect_budget_ok": True,
            "ppe_roi_redetect_triggered": False,
            "ppe_roi_redetect_count": 0,
            "ppe_roi_redetect_ms": 0.0,
        }
        status["branch_cards"] = build_branch_cards(status)
        return status

    def start(
        self,
        *,
        source_type: str,
        source: str,
        profile: str = "default",
        realtime: bool = True,
        feature_options: dict[str, Any] | None = None,
        custom_model: dict[str, Any] | None = None,
        allow_test_custom_model: bool = False,
    ) -> int:
        source_type = str(source_type or "file").lower()
        # File preview is always clock-locked native playback. The old checkbox
        # no longer switches to the unsynchronized MJPEG path.
        if source_type == "file":
            realtime = True
            # Validate the new file before stopping the current session, so a
            # bad pasted path cannot kill a running monitor.
            probe_config = load_runtime_config(
                config_path=getattr(self.cache, "config_path", None),
                profile=profile,
            )
            probe_runtime = (
                probe_config.get("runtime", {})
                if isinstance(probe_config.get("runtime"), dict)
                else {}
            )
            source = str(
                validate_file_source(
                    source,
                    preference=str(
                        probe_runtime.get(
                            "video_decoder_preference",
                            "nvdec",
                        )
                        or "nvdec"
                    ),
                    allow_cpu_fallback=bool(
                        probe_runtime.get(
                            "video_decoder_allow_cpu_fallback",
                            True,
                        )
                    ),
                    gpu_id=int(
                        probe_runtime.get("video_decoder_gpu_id", 0)
                        or 0
                    ),
                )
            )
        elif source_type == "camera":
            camera_text = normalize_source_text(source)
            source = camera_text.split(":", 1)[1].strip() if camera_text.lower().startswith("camera:") else camera_text
        self.stop(release_pipeline_cache=False)
        pending_threads = [
            thread.name
            for thread in (self.capture_thread, self.process_thread, self.preview_thread)
            if thread is not None and thread.is_alive()
        ]
        if pending_threads:
            message = (
                "monitor start blocked: previous worker threads are still running: "
                + ", ".join(pending_threads)
            )
            with self.condition:
                self.status["restart_blocked_reason"] = message
                self.status["warning"] = "restart_blocked_by_pending_threads"
                if not str(self.status.get("error") or "").strip():
                    self.status["error"] = message
                self.condition.notify_all()
            raise RuntimeError(message)
        cache_feature_options, feature_options = self._normalize_start_feature_options(feature_options)
        custom_model_options = normalize_custom_model_options(custom_model)
        self.stop_event.clear()
        self.process_done_event.clear()
        with self.condition:
            self.run_id += 1
            run_id = self.run_id
            self.latest_jpeg = None
            self.latest_jpeg_seq = 0
            self.latest_jpeg_meta = {}
            self.overlay_timeline.clear()
            self.overlay_seq = 0
            self.preview_publish_times.clear()
            self.detect_times.clear()
            self.detector_cycle_samples.clear()
            self.evidence_update_samples.clear()
            self.overlay_publish_samples.clear()
            self.recent_events.clear()
            self.recent_ppe_events.clear()
            self.recent_source_events.clear()
            self._capture_done = False
            self._source_duration_s = 0.0
            self._source_frame_count = 0
            self.status = self._empty_status()
            self.status.update(
                {
                    "run_id": run_id,
                    "running": True,
                    "source_type": source_type,
                    "source": source,
                    "profile": profile,
                    "realtime": bool(realtime),
                    "feature_options": dict(feature_options),
                    "custom_model": dict(custom_model_options),
                    "test_custom_model_bypass": bool(allow_test_custom_model),
                    "display_options": dict(self.display_options),
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "preview_mode": "initializing_detector",
                    "initializing": True,
                    "prewarming": True,
                    "detector_ready": False,
                    "first_detection_ready": False,
                    "ready_for_preview": False,
                    "preview_started": False,
                    "preview_seekable": False,
                    "source_ended": False,
                    "stream_reconnects": 0,
                    "stream_last_frame_age_ms": 0.0,
                }
            )
            self.condition.notify_all()

        init_started = time.perf_counter()
        try:
            preload_bundle = self.cache.get(
                profile=profile,
                feature_options=cache_feature_options,
                custom_model=custom_model_options,
                allow_test_custom_model=bool(allow_test_custom_model),
            )
            init_ms = (time.perf_counter() - init_started) * 1000.0
            with self.condition:
                if run_id == self.run_id:
                    authoritative = (
                        preload_bundle.config.get("runtime", {}).get(
                            "authoritative_model", {}
                        )
                        if isinstance(preload_bundle.config.get("runtime"), dict)
                        else {}
                    )
                    try:
                        from defense.module_a.native_bridge import (
                            status as native_status,
                        )

                        native_runtime_status = _native_status_contract(
                            native_status()
                        )
                    except Exception as exc:
                        native_runtime_status = {
                            "available": False,
                            "version": "unknown",
                            "binary_sha256": None,
                            "enabled_stages": [],
                            "fallback_reason": (
                                f"{type(exc).__name__}:{exc}"
                            ),
                        }
                    self.status.update(
                        {
                            "backend": preload_bundle.backend,
                            "model_family": preload_bundle.model_family,
                            "artifact": preload_bundle.artifact_path,
                            "authoritative_model": (
                                _authoritative_status_contract(
                                    authoritative,
                                    fallback_backend=preload_bundle.backend,
                                )
                            ),
                            "native": native_runtime_status,
                            "custom_model": dict(
                                preload_bundle.config.get("runtime", {}).get(
                                    "custom_model", custom_model_options
                                )
                            ),
                            "initializing": False,
                            "prewarming": True,
                            "detector_ready": False,
                            "init_ms": init_ms,
                            "pipeline_cache_hit": bool(preload_bundle.cache_hit),
                            "pipeline_cache_get_ms": float(preload_bundle.cache_get_ms),
                            "pipeline_config_load_ms": float(preload_bundle.config_load_ms),
                            "pipeline_backend_create_ms": float(preload_bundle.backend_create_ms),
                            "pipeline_construct_ms": float(preload_bundle.pipeline_construct_ms),
                            "pipeline_warmup_ms": float(preload_bundle.warmup_ms),
                            "pipeline_warmup_frames": int(preload_bundle.warmup_frames),
                            "pipeline_reset_ms": float(preload_bundle.pipeline_reset_ms),
                            "warmup_error": preload_bundle.warmup_error,
                            "preview_mode": "detector_thread_prewarming",
                        }
                    )
                    self.condition.notify_all()
        except BaseException as exc:
            self._set_error(str(exc), run_id)
            raise

        runtime_config = preload_bundle.config.get("runtime", {}) if isinstance(preload_bundle.config.get("runtime"), dict) else {}
        self.detector_drain_timeout_s = max(
            0.1,
            float(
                runtime_config.get("detector_drain_timeout_s", 10.0)
                or 10.0
            ),
        )
        preview_render_fps = float(
            runtime_config.get("preview_render_fps", runtime_config.get("preview_fps", 25)) or 25
        )
        preview_max_side = int(runtime_config.get("preview_max_side", 960) or 960)
        detector_process_fps_cap = float(
            runtime_config.get("detector_process_fps_cap", runtime_config.get("process_fps_cap", 15)) or 15
        )
        file_realtime_overlay_bridge_frames = float(
            runtime_config.get("file_realtime_overlay_bridge_frames", 3.2) or 3.2
        )
        file_realtime_overlay_bridge_min_s = float(
            runtime_config.get("file_realtime_overlay_bridge_min_s", 0.20) or 0.20
        )
        file_realtime_overlay_bridge_max_s = float(
            runtime_config.get("file_realtime_overlay_bridge_max_s", 0.36) or 0.36
        )
        capture_max_side = int(runtime_config.get("capture_max_side", preview_max_side) or preview_max_side)
        file_source_fps_cap = float(
            runtime_config.get(
                "file_source_fps_cap",
                0.0,
            )
            or 0.0
        )
        video_decoder_preference = str(
            runtime_config.get("video_decoder_preference", "nvdec") or "nvdec"
        )
        video_decoder_allow_cpu_fallback = bool(
            runtime_config.get("video_decoder_allow_cpu_fallback", True)
        )
        video_decoder_gpu_id = int(
            runtime_config.get("video_decoder_gpu_id", 0) or 0
        )
        preview_bus = PreviewBus()
        detection_bus = DetectionBus()
        with self.condition:
            if run_id == self.run_id:
                self.preview_bus = preview_bus
                self.detection_bus = detection_bus
                self._source_epoch += 1
                self._seek_request_s = None
                self._playback_paused = False
                self._playback_speed = 1.0
                self._preview_last_overlay = None
                self.status.update(
                    {
                        "preview_mode": "backend_source_pipeline",
                        "backend_source_pipeline": True,
                        "detector_pipeline_mode": "backend_latest_only",
                        "detector_queue_policy": "latest_only",
                        "detector_process_fps_cap": detector_process_fps_cap,
                        "detector_drain_timeout_s": float(
                            self.detector_drain_timeout_s
                        ),
                        "file_realtime_overlay_bridge_frames": file_realtime_overlay_bridge_frames,
                        "file_realtime_overlay_bridge_min_s": file_realtime_overlay_bridge_min_s,
                        "file_realtime_overlay_bridge_max_s": file_realtime_overlay_bridge_max_s,
                        "preview_render_fps": preview_render_fps,
                        "preview_max_side": preview_max_side,
                        "capture_max_side": capture_max_side,
                        "file_source_fps_cap": file_source_fps_cap,
                        "decoder": _empty_decoder_status(
                            video_decoder_preference
                        ),
                        "preview_never_wait_for_detection": True,
                        "ready_for_preview": False,
                        "first_detection_ready": False,
                        "preview_seekable": source_type == "file",
                        "source_epoch": self._source_epoch,
                        "playback_paused": False,
                        "playback_speed": 1.0,
                        "source_ended": False,
                        "stream_reconnects": 0,
                        "stream_last_frame_age_ms": 0.0,
                    }
                )
                self.condition.notify_all()

        process_args = (
            run_id,
            preview_bus,
            detection_bus,
            source_type,
            source,
            profile,
            bool(realtime),
            cache_feature_options,
            feature_options,
            custom_model_options,
        )
        capture_args = (
            run_id,
            preview_bus,
            detection_bus,
            source_type,
            source,
            profile,
            bool(realtime),
            feature_options,
            custom_model_options,
            float(preview_render_fps),
            float(detector_process_fps_cap),
            int(capture_max_side),
            float(file_source_fps_cap),
            video_decoder_preference,
            video_decoder_allow_cpu_fallback,
            video_decoder_gpu_id,
        )
        self.process_thread = threading.Thread(target=self._backend_process_loop, args=process_args, name="module-a-detector", daemon=True)
        self.preview_thread = threading.Thread(
            target=self._preview_render_loop,
            args=(run_id, preview_bus, float(preview_render_fps)),
            name="module-a-preview",
            daemon=True,
        )
        self.process_thread.start()
        detector_ready_deadline = time.perf_counter() + float(runtime_config.get("detector_thread_warmup_timeout_s", 30.0) or 30.0)
        start_error = ""
        with self.condition:
            while (
                run_id == self.run_id
                and not self.stop_event.is_set()
                and bool(self.status.get("running"))
                and bool(self.status.get("prewarming", False))
                and time.perf_counter() < detector_ready_deadline
            ):
                self.condition.wait(timeout=0.05)
            if self.status.get("error"):
                start_error = str(self.status.get("error"))
            elif run_id != self.run_id or self.stop_event.is_set():
                start_error = "monitor start was interrupted"
            elif bool(self.status.get("prewarming", False)):
                start_error = "detector thread warmup timed out"
        if start_error:
            self._set_error(start_error, run_id)
            raise RuntimeError(start_error)

        self.capture_thread = threading.Thread(target=self._backend_capture_loop, args=capture_args, name="module-a-source", daemon=True)
        self.capture_thread.start()
        self.preview_thread.start()
        return run_id

    def wait_detector_ready(self, run_id: int, timeout: float = 30.0) -> dict[str, Any]:
        deadline = time.perf_counter() + max(0.0, timeout)
        with self.condition:
            while run_id == self.run_id:
                status = dict(self.status)
                if status.get("detector_ready") or status.get("error") or not status.get("running"):
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(0.05, remaining))
        return self.get_status()

    def wait_ready_for_preview(self, run_id: int, timeout: float = 30.0) -> dict[str, Any]:
        deadline = time.perf_counter() + max(0.0, timeout)
        with self.condition:
            while run_id == self.run_id:
                status = dict(self.status)
                if status.get("ready_for_preview") or status.get("error") or not status.get("running"):
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(0.05, remaining))
        return self.get_status()

    def update_display_options(self, options: dict[str, Any]) -> dict[str, bool]:
        allowed = {"show_boxes", "show_person_boxes", "show_module_hud", "show_ppe_hud"}
        with self.condition:
            for key, value in (options or {}).items():
                if key in allowed:
                    self.display_options[key] = bool(value)
            self.status["display_options"] = dict(self.display_options)
            self.condition.notify_all()
            return dict(self.display_options)

    def stop(self, *, release_pipeline_cache: bool = True) -> None:
        thread_items = [
            ("capture_thread", self.capture_thread),
            ("process_thread", self.process_thread),
            ("preview_thread", self.preview_thread),
        ]
        self.stop_event.set()
        active_decoder = self._active_file_decoder
        request_decoder_cancel = getattr(
            active_decoder,
            "request_cancel",
            None,
        )
        if callable(request_decoder_cancel):
            try:
                request_decoder_cancel()
            except Exception as exc:
                with self.condition:
                    errors = list(
                        self.status.get("secondary_errors") or []
                    )
                    errors.append(
                        "decoder_cancel_failed:"
                        f"{type(exc).__name__}:{exc}"
                    )
                    self.status["secondary_errors"] = errors
        if self.preview_bus is not None:
            self.preview_bus.close()
        if self.detection_bus is not None:
            self.detection_bus.close()
        current = threading.current_thread()
        for _, thread in thread_items:
            if thread is None:
                continue
            if thread is current:
                continue
            if thread.ident is None:
                continue
            thread.join(timeout=self.thread_join_timeout_s)
        alive_after_join: list[str] = []
        with self.condition:
            for attr_name, thread in thread_items:
                if thread is None:
                    continue
                if thread.is_alive():
                    alive_after_join.append(thread.name)
                elif getattr(self, attr_name) is thread:
                    setattr(self, attr_name, None)
            if alive_after_join:
                self._release_pipeline_cache_when_stopped = bool(
                    self._release_pipeline_cache_when_stopped or release_pipeline_cache
                )
                release_cache = False
            else:
                self.capture_thread = None
                self.process_thread = None
                self.preview_thread = None
                self.preview_bus = None
                self.detection_bus = None
                release_cache = bool(release_pipeline_cache or self._release_pipeline_cache_when_stopped)
                self._release_pipeline_cache_when_stopped = False
        if release_cache:
            self._release_pipeline_cache()
        with self.condition:
            if self.status.get("running"):
                self.status["running"] = False
                self.status["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self.status["preview_started"] = False
            self.status["ready_for_preview"] = False
            self.status["first_detection_ready"] = False
            self.status["playback_paused"] = False
            self.status["source_ended"] = False
            self.status["preview_fps"] = 0.0
            self.status["preview_mode"] = "stopped"
            self.status["detector_pipeline_mode"] = "idle"
            self.status["stop_threads_pending"] = alive_after_join
            self.status["pipeline_cache_release_deferred"] = bool(
                self._release_pipeline_cache_when_stopped
            )
            if alive_after_join:
                self.status["warning"] = "worker_threads_did_not_stop"
            else:
                self.status["restart_blocked_reason"] = ""
                if self.status.get("warning") in {
                    "worker_threads_did_not_stop",
                    "restart_blocked_by_pending_threads",
                }:
                    self.status.pop("warning", None)
            self.condition.notify_all()

    def _release_pipeline_cache(self) -> None:
        clear_cache = getattr(self.cache, "clear", None)
        if callable(clear_cache):
            clear_cache()

    def _should_release_finished_file_pipeline(
        self,
        *,
        run_id: int,
        source_type: str,
        runtime_config: dict[str, Any],
    ) -> bool:
        return bool(
            int(run_id) == int(self.run_id)
            and str(source_type or "").lower() == "file"
            and bool(
                self.status.get("source_ended")
                or (
                    self.status.get("source_eof_reached")
                    and self.status.get("process_done")
                )
            )
            and bool(runtime_config.get("release_pipeline_cache_on_file_end", False))
            and not self.stop_event.is_set()
        )

    def control_run(self, run_id: int, action: str, **payload: Any) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        with self.condition:
            if int(run_id or 0) != self.run_id:
                raise RuntimeError("run_id does not match current run")
            if not bool(self.status.get("running")) or bool(self.status.get("source_ended")):
                raise RuntimeError("run is not active")
            if action == "play":
                self._playback_paused = False
                self.status["playback_paused"] = False
                self.status["preview_started"] = True
            elif action == "pause":
                self._playback_paused = True
                self.status["playback_paused"] = True
            elif action == "seek":
                if str(self.status.get("source_type") or "").lower() != "file":
                    raise RuntimeError("seek is only available for MP4/file sources")
                target = max(0.0, float(payload.get("source_time_s", payload.get("video_time_s", payload.get("time_s", 0.0))) or 0.0))
                duration = float(self.status.get("source_duration_s") or self._source_duration_s or 0.0)
                if duration > 0:
                    target = min(target, max(0.0, duration - 0.001))
                self._seek_request_s = target
                self._reset_source_epoch_state_locked(source_time_s=target)
                self.status.update(
                    {
                        "overlay_seq": int(self.overlay_seq),
                    }
                )
            elif action == "set_speed":
                speed = float(payload.get("speed", payload.get("playback_speed", 1.0)) or 1.0)
                self._playback_speed = max(0.1, min(4.0, speed))
                self.status["playback_speed"] = self._playback_speed
            else:
                raise RuntimeError(f"unsupported control action: {action}")
            self.condition.notify_all()
        return self.get_status()

    def _reset_source_epoch_state_locked(self, *, source_time_s: float | None = None) -> int:
        """Advance source epoch and discard source-scoped buffered state.

        The caller must hold ``self.condition``.  Sequence counters stay
        monotonic; only buffered frames and temporal/display state are cleared.
        """
        self._source_epoch = max(
            int(self._source_epoch),
            int(self.status.get("source_epoch") or self._source_epoch or 0),
        ) + 1
        self.overlay_timeline.clear()
        self.latest_jpeg = None
        self.latest_jpeg_meta = {}
        self.preview_publish_times.clear()
        self.detect_times.clear()
        self._preview_last_overlay = None
        if self.preview_bus is not None:
            self.preview_bus.clear()
        if self.detection_bus is not None:
            self.detection_bus.clear()
        status_update: dict[str, Any] = {
            "source_epoch": self._source_epoch,
            "first_detection_ready": False,
            "source_ended": False,
            "preview_seq": int(self.latest_jpeg_seq),
            "preview_fps": 0.0,
        }
        if source_time_s is not None:
            status_update["source_time_s"] = float(source_time_s)
            status_update["video_time_s"] = float(source_time_s)
        self.status.update(status_update)
        return self._source_epoch

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            payload = dict(self.status)
            processed_detection_frames = int(self.overlay_seq)
            source_fps = float(payload.get("source_fps") or 0.0)
            source_time_s = float(
                payload.get(
                    "source_time_s",
                    payload.get("video_time_s", 0.0),
                )
                or 0.0
            )
            source_frame_idx = int(payload.get("frame_idx") or 0)
            # Detector status is asynchronous and may describe an older source
            # frame than the JPEG currently being displayed.  Coverage must use
            # the furthest observed preview/source clock rather than silently
            # treating detector progress as source progress.
            preview_source_time_s = float(
                self.latest_jpeg_meta.get("source_time_s") or 0.0
            )
            preview_frame_idx = int(
                self.latest_jpeg_meta.get("frame_idx") or 0
            )
            source_time_s = max(source_time_s, preview_source_time_s)
            source_frame_idx = max(source_frame_idx, preview_frame_idx)
            observed_source_frames = max(
                source_frame_idx + 1 if processed_detection_frames > 0 else 0,
                (
                    int(round(source_time_s * source_fps)) + 1
                    if source_fps > 0.0 and source_time_s > 0.0
                    else 0
                ),
            )
            if bool(payload.get("source_ended")):
                observed_source_frames = max(
                    observed_source_frames,
                    int(payload.get("source_frame_count") or 0),
                )
            payload["processed_detection_frames"] = processed_detection_frames
            payload["detection_source_coverage_ratio"] = (
                float(processed_detection_frames / observed_source_frames)
                if observed_source_frames > 0
                else 0.0
            )
            payload["display_options"] = dict(self.display_options)
            payload["recent_events"] = list(self.recent_events)
            payload["recent_ppe_events"] = list(self.recent_ppe_events)
            payload["recent_source_auth_events"] = list(self.recent_source_events)
            payload["branch_cards"] = build_branch_cards(payload)
            return payload

    def get_overlay(self, since_seq: int = 0) -> dict[str, Any]:
        with self.lock:
            since_seq = int(since_seq or 0)
            records = [dict(item) for item in self.overlay_timeline if int(item.get("overlay_seq", 0)) > since_seq]
            return {
                "run_id": self.run_id,
                "records": records,
                "latest_seq": int(self.overlay_seq),
                "running": bool(self.status.get("running")),
                "ready_for_preview": bool(self.status.get("ready_for_preview")),
                "preview_started": bool(self.status.get("preview_started")),
            }

    def wait_latest_jpeg(self, last_seq: int, timeout: float = 0.5) -> tuple[int, bytes | None, bool]:
        deadline = time.perf_counter() + max(0.0, timeout)
        with self.condition:
            if not self.status.get("running"):
                return self.latest_jpeg_seq, None, False
            while self.latest_jpeg_seq == last_seq and self.status.get("running") and time.perf_counter() < deadline:
                self.condition.wait(timeout=min(0.05, max(0.0, deadline - time.perf_counter())))
            if not self.status.get("running"):
                return self.latest_jpeg_seq, None, False
            if self.latest_jpeg_seq <= last_seq:
                return self.latest_jpeg_seq, None, True
            return self.latest_jpeg_seq, self.latest_jpeg, bool(self.status.get("running"))

    def wait_latest_jpeg_snapshot(
        self,
        last_seq: int,
        timeout: float = 0.5,
    ) -> tuple[int, bytes | None, bool, dict[str, Any]]:
        deadline = time.perf_counter() + max(0.0, timeout)
        with self.condition:
            if not self.status.get("running"):
                return self.latest_jpeg_seq, None, False, {}
            while (
                self.latest_jpeg_seq == last_seq
                and self.status.get("running")
                and time.perf_counter() < deadline
            ):
                self.condition.wait(
                    timeout=min(
                        0.05,
                        max(0.0, deadline - time.perf_counter()),
                    )
                )
            if not self.status.get("running"):
                return self.latest_jpeg_seq, None, False, {}
            if self.latest_jpeg_seq <= last_seq:
                return self.latest_jpeg_seq, None, True, {}
            return (
                self.latest_jpeg_seq,
                self.latest_jpeg,
                bool(self.status.get("running")),
                dict(self.latest_jpeg_meta),
            )

    @staticmethod
    def _read_capture_meta(cap: cv2.VideoCapture) -> tuple[float, float, int]:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps < 1.0 or fps > 120.0:
            fps = 25.0
        fps = min(fps, 60.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = (frame_count / fps) if frame_count > 0 and fps > 0 else 0.0
        return fps, duration_s, frame_count

    @staticmethod
    def _resize_capture_frame(frame: Any, max_side: int) -> tuple[Any, bool]:
        if max_side <= 0:
            return frame, False
        height, width = frame.shape[:2]
        longest = max(int(width), int(height))
        if longest <= max_side:
            return frame, False
        scale = float(max_side) / float(longest)
        resized = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, True

    @staticmethod
    def _file_frame_step(source_fps: float, file_source_fps_cap: float, preview_render_fps: float, detector_fps: float) -> float:
        fps = max(1.0, float(source_fps or 0.0))
        cap = float(file_source_fps_cap or 0.0)
        if cap <= 0.0:
            return 1.0
        return max(1.0, fps / max(1.0, min(fps, cap)))

    @staticmethod
    def _near_file_eof(cap: cv2.VideoCapture, frame_count: int, *, tolerance: int = 3) -> bool:
        if frame_count <= 0:
            return False
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        return pos >= max(0, int(frame_count) - int(tolerance))

    def _recover_file_decode_gap(
        self,
        cap: cv2.VideoCapture,
        frame_count: int,
        run_id: int,
        *,
        max_skip_frames: int = 3,
    ) -> Any | None:
        if self._near_file_eof(cap, frame_count):
            return None
        base_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        for offset in range(1, max(1, int(max_skip_frames)) + 1):
            if self.stop_event.is_set():
                return None
            target = max(0, base_pos + offset)
            if frame_count > 0 and target >= frame_count:
                return None
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                with self.condition:
                    if run_id == self.run_id:
                        recoveries = int(self.status.get("source_decode_recoveries") or 0) + 1
                        self.status["source_decode_recoveries"] = recoveries
                        self.status["source_decode_last_recovery_frame"] = target
                        self.condition.notify_all()
                return frame
        return None

    def _refresh_active_decoder_status(self, run_id: int) -> None:
        decoder = self._active_file_decoder
        if decoder is None:
            return
        try:
            decoder_status = _decoder_status_contract(decoder.status_snapshot())
        except Exception as exc:
            decoder_status = {
                **_empty_decoder_status(),
                "fallback_reason": (
                    "decoder_status_snapshot_failed:"
                    f"{type(exc).__name__}:{exc}"
                ),
            }
        with self.condition:
            if run_id == self.run_id:
                self.status["decoder"] = decoder_status
                self.condition.notify_all()

    def _record_detector_cycle_metrics(
        self,
        *,
        run_id: int,
        source_epoch: int,
        frame_idx: int,
        detector_cycle_ms: float,
        evidence_update_ms: float,
        overlay_status_publish_ms: float,
    ) -> None:
        detector_cycle_ms = max(0.0, float(detector_cycle_ms))
        evidence_update_ms = max(0.0, float(evidence_update_ms))
        overlay_status_publish_ms = max(
            0.0,
            float(overlay_status_publish_ms),
        )
        self.detector_cycle_samples.append(detector_cycle_ms)
        self.evidence_update_samples.append(evidence_update_ms)
        self.overlay_publish_samples.append(overlay_status_publish_ms)
        cycle_distribution = _timing_distribution(
            self.detector_cycle_samples
        )
        cycle_mean_ms = float(cycle_distribution.get("mean") or 0.0)
        detector_compute_fps = (
            1000.0 / cycle_mean_ms if cycle_mean_ms > 1e-6 else 0.0
        )
        evidence_distribution = _timing_distribution(
            self.evidence_update_samples
        )
        overlay_distribution = _timing_distribution(
            self.overlay_publish_samples
        )
        with self.condition:
            if run_id != self.run_id:
                return
            if int(self.status.get("source_epoch") or 0) != int(source_epoch):
                return
            metrics = {
                "detector_cycle_ms": detector_cycle_ms,
                "detector_compute_fps": detector_compute_fps,
                "evidence_update_ms": evidence_update_ms,
                "overlay_status_publish_ms": overlay_status_publish_ms,
                "detector_cycle_ms_distribution": cycle_distribution,
                "evidence_update_ms_distribution": evidence_distribution,
                "overlay_status_publish_ms_distribution": (
                    overlay_distribution
                ),
            }
            self.status.update(metrics)
            if self.overlay_timeline:
                latest_overlay = self.overlay_timeline[-1]
                if (
                    int(latest_overlay.get("source_epoch") or 0)
                    == int(source_epoch)
                    and int(latest_overlay.get("frame_idx") or 0)
                    == int(frame_idx)
                ):
                    latest_overlay.update(metrics)
            self.condition.notify_all()

    def _finalize_capture_run(
        self,
        *,
        run_id: int,
        preview_bus: PreviewBus,
        detection_bus: DetectionBus,
        source_type: str,
        source_ended_candidate: bool,
        final_status_updates: dict[str, Any] | None = None,
    ) -> None:
        """Close source buses, drain detector/evidence, then publish EOF.

        ``source_ended`` is a completion guarantee, not merely a decoder EOF
        marker.  It becomes true only after the process thread consumed its
        final packet and completed ``EvidenceSession.close()``.
        """

        drain_started = time.perf_counter()
        candidate = bool(
            source_type == "file"
            and source_ended_candidate
            and not self.stop_event.is_set()
        )
        with self.condition:
            self._capture_done = True
            if run_id == self.run_id:
                if final_status_updates:
                    self.status.update(final_status_updates)
                self.status.update(
                    {
                        "source_eof_reached": candidate,
                        "source_ended": False,
                        "detector_drain_active": candidate,
                        "detector_drain_completed": False,
                        "detector_drain_timed_out": False,
                        "detector_drain_ms": 0.0,
                        "detector_drain_failed_reason": "",
                        "process_done": self.process_done_event.is_set(),
                        "preview_started": False,
                        "ready_for_preview": False,
                    }
                )
                self.latest_jpeg = None
                self.latest_jpeg_meta = {}
                if candidate:
                    self.status["preview_mode"] = "source_eof_drain"
                    self.status["detector_pipeline_mode"] = "draining"
                else:
                    self.status["running"] = False
            self.condition.notify_all()

        preview_bus.close()
        detection_bus.close()

        process_done = self.process_done_event.is_set()
        timed_out = False
        if candidate and not process_done:
            deadline = drain_started + max(
                0.1,
                float(self.detector_drain_timeout_s),
            )
            while not process_done and not self.stop_event.is_set():
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    timed_out = True
                    break
                process_done = self.process_done_event.wait(
                    timeout=min(0.05, remaining)
                )

        if candidate:
            preview_bus.clear()
            detection_bus.clear()
        drain_ms = (time.perf_counter() - drain_started) * 1000.0
        with self.condition:
            if run_id != self.run_id:
                return
            process_done = self.process_done_event.is_set()
            error_text = str(self.status.get("error") or "").strip()
            source_ended = bool(
                candidate
                and process_done
                and not timed_out
                and not self.stop_event.is_set()
                and not error_text
            )
            failed_reason = ""
            if timed_out:
                failed_reason = (
                    "detector_drain_timeout:"
                    f"{float(self.detector_drain_timeout_s):.3f}s"
                )
                errors = list(self.status.get("secondary_errors") or [])
                if failed_reason not in errors:
                    errors.append(failed_reason)
                self.status["secondary_errors"] = errors
                self.status["warning"] = "detector_drain_timeout"
            elif candidate and not process_done:
                failed_reason = (
                    "stopped_during_detector_drain"
                    if self.stop_event.is_set()
                    else "detector_drain_incomplete"
                )
            elif error_text:
                failed_reason = error_text

            self.status.update(
                {
                    "running": False,
                    "source_ended": source_ended,
                    "process_done": process_done,
                    "detector_drain_active": False,
                    "detector_drain_completed": process_done,
                    "detector_drain_timed_out": timed_out,
                    "detector_drain_ms": drain_ms,
                    "detector_drain_failed_reason": failed_reason,
                    "preview_fps": 0.0,
                    "ready_for_preview": False,
                    "preview_started": False,
                    "stopped_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
            )
            if source_ended:
                final_time_s = float(
                    self.status.get("source_time_s") or 0.0
                )
                if self._source_duration_s > 0:
                    final_time_s = min(
                        max(final_time_s, 0.0),
                        float(self._source_duration_s),
                    )
                self.status.update(
                    {
                        "source_time_s": final_time_s,
                        "video_time_s": final_time_s,
                        "preview_mode": "source_ended",
                        "detector_pipeline_mode": "ended",
                    }
                )
            elif candidate:
                self.status["preview_mode"] = "source_drain_failed"
                self.status["detector_pipeline_mode"] = "drain_failed"
            self.condition.notify_all()

    def _backend_file_decoder_loop(
        self,
        *,
        run_id: int,
        preview_bus: PreviewBus,
        detection_bus: DetectionBus,
        source: str,
        realtime: bool,
        preview_render_fps: float,
        detector_process_fps_cap: float,
        capture_max_side: int,
        file_source_fps_cap: float,
        decoder_preference: str,
        allow_cpu_fallback: bool,
        gpu_id: int,
    ) -> None:
        """Production file-source loop backed by the unified decoder adapter.

        File frames remain in the decoder-owned representation until preview or
        detection explicitly materializes the size it needs. NVDEC therefore
        does not immediately download a full-resolution BGR frame.
        """

        decoder: VideoDecoder | None = None
        packet_seq = 0
        last_detection_push = 0.0
        source_frames_skipped_for_realtime = 0
        capture_frames_published = 0
        detector_submission_count = 0
        next_read_frame = 0.0
        playback_anchor_wall = time.perf_counter()
        playback_anchor_frame = 0.0
        playback_clock_speed = 1.0
        playback_was_paused = False
        temporal_previous_lease = None
        temporal_previous_owner: SharedFrameLease | None = None
        temporal_previous_frame_idx: int | None = None
        temporal_previous_source_time_s: float | None = None
        pending_owner: SharedFrameLease | None = None
        source_ended_normally = False

        try:
            decoder = create_video_decoder(
                source,
                preference=decoder_preference,
                allow_cpu_fallback=allow_cpu_fallback,
                gpu_id=gpu_id,
            )
            self._active_file_decoder = decoder
            stream_info = decoder.info
            fps = float(stream_info.fps or 0.0)
            if not math.isfinite(fps) or fps < 1.0 or fps > 120.0:
                fps = 25.0
            frame_count = max(0, int(stream_info.frame_count or 0))
            duration_s = float(stream_info.duration_s or 0.0)
            if duration_s <= 0.0 and frame_count > 0:
                duration_s = frame_count / fps
            self._source_fps = fps
            self._source_duration_s = duration_s
            self._source_frame_count = frame_count
            frame_step = self._file_frame_step(
                fps,
                file_source_fps_cap,
                preview_render_fps,
                detector_process_fps_cap,
            )
            source_frame_skip_enabled = frame_step > 1.0
            detect_interval = 1.0 / max(
                1.0,
                min(detector_process_fps_cap, fps),
            )
            effective_capture_fps = fps / max(1.0, frame_step)
            detector_submit_every_file_frame = (
                _detector_can_follow_file_source(
                    detector_process_fps_cap,
                    effective_capture_fps,
                )
            )
            decoder_status = _decoder_status_contract(decoder.status_snapshot())
            with self.condition:
                if run_id == self.run_id:
                    self.status.update(
                        {
                            "source_fps": fps,
                            "source_duration_s": duration_s,
                            "source_frame_count": frame_count,
                            "ready_for_preview": True,
                            "preview_started": True,
                            "preview_seekable": True,
                            "preview_mode": "backend_source_pipeline",
                            "detector_pipeline_mode": "backend_latest_only",
                            "capture_max_side": int(capture_max_side),
                            "file_source_fps_cap": float(
                                file_source_fps_cap or 0.0
                            ),
                            "source_frame_step": float(frame_step),
                            "source_frame_skip_enabled": bool(
                                source_frame_skip_enabled
                            ),
                            "detector_submit_every_file_frame": (
                                detector_submit_every_file_frame
                            ),
                            "decoder": decoder_status,
                        }
                    )
                    self.condition.notify_all()

            playback_anchor_wall = time.perf_counter()
            playback_anchor_frame = 0.0
            while run_id == self.run_id and not self.stop_event.is_set():
                _loop_started = time.perf_counter()
                with self.condition:
                    paused = bool(self._playback_paused)
                    speed = max(0.1, float(self._playback_speed or 1.0))
                    seek_request = self._seek_request_s
                    if seek_request is not None:
                        self._seek_request_s = None
                    current_epoch = int(self._source_epoch)

                if seek_request is not None:
                    if temporal_previous_owner is not None:
                        temporal_previous_owner.release()
                        temporal_previous_owner = None
                    temporal_previous_lease = None
                    temporal_previous_frame_idx = None
                    temporal_previous_source_time_s = None
                    old_decoder = decoder
                    try:
                        old_decoder.close()
                    except Exception as exc:
                        with self.condition:
                            if run_id == self.run_id:
                                errors = list(
                                    self.status.get("secondary_errors") or []
                                )
                                errors.append(
                                    "decoder_seek_close_failed:"
                                    f"{type(exc).__name__}:{exc}"
                                )
                                self.status["secondary_errors"] = errors
                    decoder = create_video_decoder(
                        source,
                        preference=decoder_preference,
                        allow_cpu_fallback=allow_cpu_fallback,
                        gpu_id=gpu_id,
                    )
                    self._active_file_decoder = decoder
                    decoder.seek_time(max(0.0, float(seek_request)))
                    next_read_frame = max(0.0, float(seek_request) * fps)
                    playback_anchor_wall = time.perf_counter()
                    playback_anchor_frame = next_read_frame
                    playback_clock_speed = speed
                    playback_was_paused = False
                    with self.condition:
                        if run_id == self.run_id:
                            self.status["source_epoch"] = current_epoch
                            self.status["decoder"] = _decoder_status_contract(
                                decoder.status_snapshot()
                            )
                            self.condition.notify_all()
                    continue

                if paused:
                    playback_was_paused = True
                    if self.stop_event.wait(timeout=0.03):
                        break
                    continue

                if realtime and (
                    playback_was_paused
                    or abs(float(speed) - float(playback_clock_speed)) > 1e-3
                ):
                    playback_anchor_wall = time.perf_counter()
                    playback_anchor_frame = float(next_read_frame)
                    playback_clock_speed = float(speed)
                    playback_was_paused = False

                lease = decoder.read()
                if lease is None:
                    if capture_frames_published == 0:
                        raise RuntimeError(
                            _decoder_zero_frame_message(decoder, source)
                        )
                    source_ended_normally = True
                    break
                pending_owner = SharedFrameLease(lease)
                pending_owner.acquire()

                original_w = int(lease.width)
                original_h = int(lease.height)
                current_frame_idx = max(0, int(lease.frame_idx))
                source_time_s = max(0.0, float(lease.pts_s))
                if source_time_s <= 0.0 and current_frame_idx > 0:
                    source_time_s = current_frame_idx / fps
                temporal_predecessor_valid = bool(
                    temporal_previous_lease is not None
                    and temporal_previous_frame_idx is not None
                    and current_frame_idx == temporal_previous_frame_idx + 1
                    and int(temporal_previous_lease.width) == original_w
                    and int(temporal_previous_lease.height) == original_h
                )
                packet_seq += 1
                capture_frames_published += 1
                packet = FramePacket(
                    seq=packet_seq,
                    frame_idx=current_frame_idx,
                    source_time_s=source_time_s,
                    wall_time_ms=time.time() * 1000.0,
                    epoch=current_epoch,
                    frame=None,
                    width=original_w,
                    height=original_h,
                    fps=fps,
                    flags={
                        "original_width": original_w,
                        "original_height": original_h,
                        "capture_resized": False,
                        "temporal_predecessor_available": (
                            temporal_predecessor_valid
                        ),
                        "decoder_backend": decoder_status.get(
                            "effective_backend"
                        ),
                        "frame_device": lease.storage,
                        "pixel_format": lease.pixel_format,
                    },
                    previous_frame_idx=(
                        temporal_previous_frame_idx
                        if temporal_predecessor_valid
                        else None
                    ),
                    previous_source_time_s=(
                        temporal_previous_source_time_s
                        if temporal_predecessor_valid
                        else None
                    ),
                    decoder_lease=lease,
                    previous_decoder_lease=(
                        temporal_previous_lease
                        if temporal_predecessor_valid
                        else None
                    ),
                    decoder_lease_owner=pending_owner,
                    previous_decoder_lease_owner=(
                        temporal_previous_owner
                        if temporal_predecessor_valid
                        else None
                    ),
                )
                now = time.perf_counter()
                decoder_status = _decoder_status_contract(
                    decoder.status_snapshot()
                )
                should_submit_detection = bool(
                    detector_submit_every_file_frame
                    or now - last_detection_push >= detect_interval
                )
                with self.condition:
                    if run_id != self.run_id or self.stop_event.is_set():
                        pending_owner.release()
                        pending_owner = None
                        break
                    if (
                        int(current_epoch) != int(self._source_epoch)
                        or self._seek_request_s is not None
                    ):
                        pending_owner.release()
                        pending_owner = None
                        continue
                    old_temporal_owner = temporal_previous_owner
                    temporal_previous_lease = lease
                    temporal_previous_owner = pending_owner
                    pending_owner = None
                    temporal_previous_frame_idx = current_frame_idx
                    temporal_previous_source_time_s = source_time_s
                    preview_bus.publish(packet)
                    if should_submit_detection:
                        last_detection_push = now
                        detection_bus.push(packet)
                        detector_submission_count += 1
                    self.status.update(
                        {
                            "source_time_s": source_time_s,
                            "video_time_s": source_time_s,
                            "source_epoch": current_epoch,
                            "preview_started": True,
                            "ready_for_preview": True,
                            "source_frame_width": original_w,
                            "source_frame_height": original_h,
                            "capture_frame_width": original_w,
                            "capture_frame_height": original_h,
                            "capture_resized": False,
                            "dropped_detection_frames": int(
                                detection_bus.dropped
                            ),
                            "source_frames_skipped_for_realtime": int(
                                source_frames_skipped_for_realtime
                            ),
                            "capture_frames_published": int(
                                capture_frames_published
                            ),
                            "detector_submission_count": int(
                                detector_submission_count
                            ),
                            "stream_last_frame_age_ms": 0.0,
                            "decoder": decoder_status,
                        }
                    )
                    self.condition.notify_all()
                if old_temporal_owner is not None:
                    old_temporal_owner.release()

                if realtime and source_frame_skip_enabled:
                    clock_target = playback_anchor_frame + max(
                        0.0,
                        time.perf_counter() - playback_anchor_wall,
                    ) * float(speed) * max(1.0, fps)
                    next_read_frame = max(
                        float(packet.frame_idx) + frame_step,
                        clock_target,
                    )
                    next_index = int(round(next_read_frame))
                    skip_count = max(
                        0,
                        next_index - (int(packet.frame_idx) + 1),
                    )
                    skipped = 0
                    latest_skipped_lease = None
                    for _ in range(min(skip_count, 120)):
                        if self.stop_event.is_set():
                            break
                        candidate = decoder.read()
                        if candidate is None:
                            source_ended_normally = True
                            break
                        if latest_skipped_lease is not None:
                            latest_skipped_lease.release()
                        latest_skipped_lease = candidate
                        skipped += 1
                    source_frames_skipped_for_realtime += skipped
                    if latest_skipped_lease is not None:
                        if temporal_previous_owner is not None:
                            temporal_previous_owner.release()
                        temporal_previous_owner = SharedFrameLease(
                            latest_skipped_lease
                        )
                        temporal_previous_owner.acquire()
                        temporal_previous_lease = latest_skipped_lease
                        temporal_previous_frame_idx = int(
                            latest_skipped_lease.frame_idx
                        )
                        temporal_previous_source_time_s = max(
                            0.0,
                            float(latest_skipped_lease.pts_s),
                        )
                    if source_ended_normally:
                        break
                    if skipped > 0:
                        with self.condition:
                            if run_id == self.run_id:
                                self.status[
                                    "source_frames_skipped_for_realtime"
                                ] = int(source_frames_skipped_for_realtime)
                                self.status["decoder"] = (
                                    _decoder_status_contract(
                                        decoder.status_snapshot()
                                    )
                                )
                                self.condition.notify_all()

                if realtime:
                    next_scheduled_frame = float(packet.frame_idx) + frame_step
                    if source_frame_skip_enabled:
                        next_scheduled_frame = max(
                            next_scheduled_frame,
                            float(next_read_frame),
                        )
                    wait_s = _file_realtime_wait_s(
                        playback_anchor_wall=playback_anchor_wall,
                        playback_anchor_frame=playback_anchor_frame,
                        next_frame=next_scheduled_frame,
                        fps=fps,
                        speed=speed,
                    )
                    if wait_s > 0 and self.stop_event.wait(timeout=wait_s):
                        break
        except BaseException as exc:
            if not self.stop_event.is_set():
                self._set_error(str(exc), run_id)
        finally:
            if pending_owner is not None:
                pending_owner.release()
                pending_owner = None
            if temporal_previous_owner is not None:
                temporal_previous_owner.release()
                temporal_previous_owner = None
            final_decoder_status = _empty_decoder_status(
                decoder_preference
            )
            if decoder is not None:
                try:
                    decoder.close()
                except Exception as exc:
                    with self.condition:
                        if run_id == self.run_id:
                            errors = list(
                                self.status.get("secondary_errors") or []
                            )
                            errors.append(
                                "decoder_close_failed:"
                                f"{type(exc).__name__}:{exc}"
                            )
                            self.status["secondary_errors"] = errors
                try:
                    final_decoder_status = _decoder_status_contract(
                        decoder.status_snapshot()
                    )
                except Exception as exc:
                    final_decoder_status = {
                        **_empty_decoder_status(decoder_preference),
                        "close_error": (
                            "decoder_status_after_close_failed:"
                            f"{type(exc).__name__}:{exc}"
                        ),
                    }
            if self._active_file_decoder is decoder:
                self._active_file_decoder = None
            self._finalize_capture_run(
                run_id=run_id,
                preview_bus=preview_bus,
                detection_bus=detection_bus,
                source_type="file",
                source_ended_candidate=source_ended_normally,
                final_status_updates={"decoder": final_decoder_status},
            )

    def _backend_capture_loop(
        self,
        run_id: int,
        preview_bus: PreviewBus,
        detection_bus: DetectionBus,
        source_type: str,
        source: str,
        profile: str,
        realtime: bool,
        feature_options: dict[str, Any],
        custom_model: dict[str, Any],
        preview_render_fps: float,
        detector_process_fps_cap: float,
        capture_max_side: int,
        file_source_fps_cap: float,
        video_decoder_preference: str | None = None,
        video_decoder_allow_cpu_fallback: bool = True,
        video_decoder_gpu_id: int = 0,
    ) -> None:
        if source_type == "file" and video_decoder_preference is not None:
            self._backend_file_decoder_loop(
                run_id=run_id,
                preview_bus=preview_bus,
                detection_bus=detection_bus,
                source=source,
                realtime=realtime,
                preview_render_fps=preview_render_fps,
                detector_process_fps_cap=detector_process_fps_cap,
                capture_max_side=capture_max_side,
                file_source_fps_cap=file_source_fps_cap,
                decoder_preference=video_decoder_preference,
                allow_cpu_fallback=video_decoder_allow_cpu_fallback,
                gpu_id=video_decoder_gpu_id,
            )
            return
        cap = None
        packet_seq = 0
        last_detection_push = 0.0
        source_frames_skipped_for_realtime = 0
        capture_frames_published = 0
        detector_submission_count = 0
        last_frame_seen = time.perf_counter()
        reconnects = 0
        next_read_frame = 0.0
        playback_anchor_wall = time.perf_counter()
        playback_anchor_frame = 0.0
        playback_clock_speed = 1.0
        playback_was_paused = False
        temporal_previous_frame = None
        temporal_previous_frame_idx: int | None = None
        temporal_previous_source_time_s: float | None = None
        try:
            cap = open_capture(source_type, source)
            fps, duration_s, frame_count = self._read_capture_meta(cap)
            self._source_fps = fps
            self._source_duration_s = duration_s
            self._source_frame_count = frame_count
            frame_step = (
                self._file_frame_step(fps, file_source_fps_cap, preview_render_fps, detector_process_fps_cap)
                if source_type == "file"
                else 1.0
            )
            source_frame_skip_enabled = (
                source_type == "file" and frame_step > 1.0
            )
            detect_interval = 1.0 / max(1.0, min(detector_process_fps_cap, fps if source_type == "file" else detector_process_fps_cap))
            effective_capture_fps = fps / max(1.0, frame_step)
            detector_submit_every_file_frame = bool(
                source_type == "file"
                and _detector_can_follow_file_source(
                    detector_process_fps_cap,
                    effective_capture_fps,
                )
            )
            requested_live_decoder = str(
                video_decoder_preference or "opencv"
            )
            live_decoder_status = _empty_decoder_status(
                requested_live_decoder
            )
            live_decoder_status.update(
                {
                    "backend": "opencv",
                    "effective_backend": "opencv",
                    "codec": "capture_backend_managed",
                    "gpu_device": "cpu",
                    "output_format": "bgr24",
                    "frame_device": "host",
                    "fallback_count": (
                        0
                        if requested_live_decoder.strip().lower()
                        in {"opencv", "cpu"}
                        else 1
                    ),
                    "fallback_reason": (
                        "none"
                        if requested_live_decoder.strip().lower()
                        in {"opencv", "cpu"}
                        else (
                            f"{source_type}_nvdec_adapter_unavailable:"
                            "using_opencv_capture"
                        )
                    ),
                    "fallback_reasons": (
                        []
                        if requested_live_decoder.strip().lower()
                        in {"opencv", "cpu"}
                        else [
                            (
                                f"{source_type}_nvdec_adapter_unavailable:"
                                "using_opencv_capture"
                            )
                        ]
                    ),
                    "closed": False,
                }
            )
            with self.condition:
                if run_id == self.run_id:
                    preview_can_start = True
                    self.status.update(
                        {
                            "source_fps": fps,
                            "source_duration_s": duration_s,
                            "source_frame_count": frame_count,
                            "ready_for_preview": preview_can_start,
                            "preview_started": preview_can_start,
                            "preview_seekable": source_type == "file",
                            "preview_mode": "backend_source_pipeline",
                            "detector_pipeline_mode": "backend_latest_only",
                            "capture_max_side": int(capture_max_side),
                            "file_source_fps_cap": float(file_source_fps_cap or 0.0),
                            "source_frame_step": float(frame_step),
                            "source_frame_skip_enabled": bool(
                                source_frame_skip_enabled
                            ),
                            "detector_submit_every_file_frame": (
                                detector_submit_every_file_frame
                            ),
                            "decoder": live_decoder_status,
                        }
                    )
                    self.condition.notify_all()

            while run_id == self.run_id and not self.stop_event.is_set():
                _loop_started = time.perf_counter()
                with self.condition:
                    paused = bool(self._playback_paused)
                    speed = max(0.1, float(self._playback_speed or 1.0))
                    seek_request = self._seek_request_s
                    if seek_request is not None:
                        self._seek_request_s = None
                        current_epoch = int(self._source_epoch)
                    else:
                        current_epoch = int(self._source_epoch)
                if seek_request is not None and source_type == "file":
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(seek_request)) * 1000.0)
                    next_read_frame = max(0.0, float(seek_request) * fps)
                    playback_anchor_wall = time.perf_counter()
                    playback_anchor_frame = next_read_frame
                    playback_clock_speed = speed
                    playback_was_paused = False
                    temporal_previous_frame = None
                    temporal_previous_frame_idx = None
                    temporal_previous_source_time_s = None
                    with self.condition:
                        if run_id == self.run_id:
                            self.status["source_epoch"] = current_epoch
                    continue
                if paused and source_type == "file":
                    playback_was_paused = True
                    if self.stop_event.wait(timeout=0.03):
                        break
                    continue
                if (
                    source_type == "file"
                    and realtime
                    and source_frame_skip_enabled
                ):
                    current_index_for_anchor = float(cap.get(cv2.CAP_PROP_POS_FRAMES) or next_read_frame)
                    if playback_was_paused or abs(float(speed) - float(playback_clock_speed)) > 1e-3:
                        playback_anchor_wall = time.perf_counter()
                        playback_anchor_frame = current_index_for_anchor
                        playback_clock_speed = float(speed)
                        playback_was_paused = False

                ok, frame = cap.read()
                if not ok or frame is None:
                    if source_type == "file":
                        recovered_frame = self._recover_file_decode_gap(cap, frame_count, run_id)
                        if recovered_frame is None:
                            break
                        frame = recovered_frame
                    now = time.perf_counter()
                    if source_type != "file" and now - last_frame_seen >= 5.0:
                        reconnects += 1
                        with self.condition:
                            if run_id == self.run_id:
                                self.status["stream_reconnects"] = reconnects
                                self.status["stream_last_frame_age_ms"] = (now - last_frame_seen) * 1000.0
                                self.status["preview_mode"] = "source_waiting_for_frames"
                                self.condition.notify_all()
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = open_capture(source_type, source)
                        with self.condition:
                            if run_id != self.run_id or self.stop_event.is_set():
                                break
                            current_epoch = self._reset_source_epoch_state_locked()
                            self.status["stream_reconnects"] = reconnects
                            self.status["preview_mode"] = "source_reconnected"
                            self.condition.notify_all()
                        last_frame_seen = time.perf_counter()
                        last_detection_push = 0.0
                        temporal_previous_frame = None
                        temporal_previous_frame_idx = None
                        temporal_previous_source_time_s = None
                    if self.stop_event.wait(timeout=0.03):
                        break
                    if source_type != "file":
                        continue

                last_frame_seen = time.perf_counter()
                original_h, original_w = frame.shape[:2]
                frame, capture_resized = self._resize_capture_frame(frame, int(capture_max_side))
                h, w = frame.shape[:2]
                if source_type == "file":
                    # 本地文件: 用容器时间戳/帧号定位, 支持 seek。
                    source_time_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or packet_seq + 1)
                    if source_time_s <= 0.0 and fps > 0:
                        source_time_s = max(0, frame_idx - 1) / fps
                    current_frame_idx = max(0, frame_idx - 1)
                else:
                    # 实时流(摄像头/网络流): POS_FRAMES/POS_MSEC 不可靠(常年返回 0),
                    # 用单调递增的抓帧序号做帧号, 时间用挂钟相对起点估算。
                    current_frame_idx = packet_seq
                    source_time_s = (packet_seq / fps) if fps > 0 else 0.0
                temporal_predecessor_valid = bool(
                    temporal_previous_frame is not None
                    and temporal_previous_frame_idx is not None
                    and current_frame_idx == temporal_previous_frame_idx + 1
                    and temporal_previous_frame.shape[:2] == frame.shape[:2]
                )
                packet_seq += 1
                capture_frames_published += 1
                packet = FramePacket(
                    seq=packet_seq,
                    frame_idx=current_frame_idx,
                    source_time_s=max(0.0, source_time_s),
                    wall_time_ms=time.time() * 1000.0,
                    epoch=current_epoch,
                    frame=frame,
                    width=int(w),
                    height=int(h),
                    fps=float(fps),
                    flags={
                        "original_width": int(original_w),
                        "original_height": int(original_h),
                        "capture_resized": bool(capture_resized),
                        "temporal_predecessor_available": temporal_predecessor_valid,
                    },
                    previous_frame=temporal_previous_frame if temporal_predecessor_valid else None,
                    previous_frame_idx=(
                        temporal_previous_frame_idx if temporal_predecessor_valid else None
                    ),
                    previous_source_time_s=(
                        temporal_previous_source_time_s if temporal_predecessor_valid else None
                    ),
                )
                now = time.perf_counter()
                should_submit_detection = bool(
                    detector_submit_every_file_frame
                    or now - last_detection_push >= detect_interval
                )
                with self.condition:
                    if run_id != self.run_id or self.stop_event.is_set():
                        break
                    if (
                        int(current_epoch) != int(self._source_epoch)
                        or self._seek_request_s is not None
                    ):
                        # A seek/source reset may occur while cap.read() is
                        # blocked. Never let that in-flight old frame republish
                        # after the epoch was advanced and buffers were cleared.
                        continue
                    temporal_previous_frame = frame
                    temporal_previous_frame_idx = current_frame_idx
                    temporal_previous_source_time_s = max(0.0, source_time_s)
                    preview_bus.publish(packet)
                    if should_submit_detection:
                        last_detection_push = now
                        detection_bus.push(packet)
                        detector_submission_count += 1
                    preview_can_start = True
                    self.status.update(
                        {
                            "source_time_s": packet.source_time_s,
                            "video_time_s": packet.source_time_s,
                            "source_epoch": packet.epoch,
                            "preview_started": preview_can_start,
                            "ready_for_preview": preview_can_start,
                            "source_frame_width": int(original_w),
                            "source_frame_height": int(original_h),
                            "capture_frame_width": int(w),
                            "capture_frame_height": int(h),
                            "capture_resized": bool(capture_resized),
                            "dropped_detection_frames": int(detection_bus.dropped),
                            "source_frames_skipped_for_realtime": int(
                                source_frames_skipped_for_realtime
                            ),
                            "capture_frames_published": int(
                                capture_frames_published
                            ),
                            "detector_submission_count": int(
                                detector_submission_count
                            ),
                            "stream_last_frame_age_ms": 0.0,
                        }
                    )
                    self.condition.notify_all()
                if source_type == "file" and realtime:
                    clock_target = playback_anchor_frame + max(0.0, time.perf_counter() - playback_anchor_wall) * float(speed) * max(1.0, fps)
                    next_read_frame = max(float(packet.frame_idx) + frame_step, clock_target)
                    next_index = int(round(next_read_frame))
                    current_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or (packet.frame_idx + 1))
                    skip_count = max(0, next_index - current_index)
                    grabbed = 0
                    for _ in range(min(skip_count, 120)):
                        if not cap.grab():
                            break
                        grabbed += 1
                    source_frames_skipped_for_realtime += grabbed
                    if grabbed > 0:
                        retrieved, skipped_frame = cap.retrieve()
                        if retrieved and skipped_frame is not None:
                            skipped_frame, _ = self._resize_capture_frame(
                                skipped_frame,
                                int(capture_max_side),
                            )
                            skipped_idx = max(
                                0,
                                int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1,
                            )
                            skipped_time_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                            if skipped_time_s <= 0.0 and fps > 0:
                                skipped_time_s = skipped_idx / fps
                            temporal_previous_frame = skipped_frame
                            temporal_previous_frame_idx = skipped_idx
                            temporal_previous_source_time_s = max(0.0, skipped_time_s)
                        else:
                            temporal_previous_frame = None
                            temporal_previous_frame_idx = None
                            temporal_previous_source_time_s = None
                        with self.condition:
                            if run_id == self.run_id:
                                self.status[
                                    "source_frames_skipped_for_realtime"
                                ] = int(source_frames_skipped_for_realtime)
                if source_type == "file" and realtime:
                    next_scheduled_frame = float(packet.frame_idx) + frame_step
                    if source_frame_skip_enabled:
                        next_scheduled_frame = max(
                            next_scheduled_frame,
                            float(next_read_frame),
                        )
                    wait_s = _file_realtime_wait_s(
                        playback_anchor_wall=playback_anchor_wall,
                        playback_anchor_frame=playback_anchor_frame,
                        next_frame=next_scheduled_frame,
                        fps=fps,
                        speed=speed,
                    )
                    if wait_s > 0 and self.stop_event.wait(timeout=wait_s):
                        break
        except BaseException as exc:
            self._set_error(str(exc), run_id)
        finally:
            if cap is not None:
                cap.release()
            self._finalize_capture_run(
                run_id=run_id,
                preview_bus=preview_bus,
                detection_bus=detection_bus,
                source_type=source_type,
                source_ended_candidate=(
                    source_type == "file"
                    and not self.stop_event.is_set()
                ),
            )

    def _backend_process_loop(
        self,
        run_id: int,
        preview_bus: PreviewBus,
        detection_bus: DetectionBus,
        source_type: str,
        source: str,
        profile: str,
        realtime: bool,
        cache_feature_options: dict[str, Any],
        feature_options: dict[str, Any],
        custom_model: dict[str, Any],
    ) -> None:
        evidence: EvidenceSession | None = None
        runtime_config: dict[str, Any] = {}
        target_frame_budget_ms: float | None = None
        effective_source_fps = 0.0
        last_seq = 0
        last_epoch: int | None = None
        active_packet: FramePacket | None = None
        try:
            bundle = self.cache.get(
                profile=profile,
                feature_options=cache_feature_options,
                custom_model=custom_model,
            )
            runtime_config = bundle.config.get("runtime", {}) if isinstance(bundle.config.get("runtime"), dict) else {}
            thread_warmup_frames = int(
                runtime_config.get(
                    "detector_thread_warmup_frames",
                    getattr(bundle.pipeline, "warmup_frames", 0) or 0,
                )
                or 0
            )
            thread_warmup_started = time.perf_counter()
            thread_warmup_error = ""
            try:
                bundle.pipeline.warmup(thread_warmup_frames)
            except Exception as exc:
                thread_warmup_error = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    bundle.pipeline.reset()
                except Exception as exc:
                    thread_warmup_error = thread_warmup_error or f"{type(exc).__name__}: {exc}"
                thread_warmup_ms = (time.perf_counter() - thread_warmup_started) * 1000.0
            processor = FrameProcessor(bundle, jpeg_quality=int(runtime_config.get("jpeg_quality", 82)))
            process_fps_cap = float(runtime_config.get("detector_process_fps_cap", runtime_config.get("process_fps_cap", 15)) or 15)
            with self.condition:
                if run_id == self.run_id:
                    self.status.update(
                        {
                            "backend": bundle.backend,
                            "model_family": bundle.model_family,
                            "artifact": bundle.artifact_path,
                            "custom_model": dict(
                                bundle.config.get("runtime", {}).get("custom_model", custom_model)
                            ),
                            "evidence_session_dir": None,
                            "evidence_manifest_path": None,
                            "detector_ready": True,
                            "initializing": False,
                            "prewarming": False,
                            "detector_thread_warmup_ms": float(thread_warmup_ms),
                            "detector_thread_warmup_frames": int(thread_warmup_frames),
                            "warmup_error": thread_warmup_error or str(bundle.warmup_error or ""),
                            "detector_process_fps_cap": process_fps_cap,
                            "detector_queue_policy": "latest_only",
                            "release_pipeline_cache_on_file_end": bool(
                                runtime_config.get("release_pipeline_cache_on_file_end", False)
                            ),
                            "preview_mode": "mp4_clock_prepare"
                            if source_type == "file"
                            else "detector_ready_wait_first_frame",
                        }
                    )
                    self.condition.notify_all()

            while run_id == self.run_id and not self.stop_event.is_set():
                packet = detection_bus.pop_latest(last_seq, timeout=0.2)
                if packet is None:
                    if detection_bus.closed:
                        break
                    continue
                active_packet = packet
                detector_cycle_started = time.perf_counter()
                last_seq = packet.seq
                if last_epoch is None:
                    last_epoch = int(packet.epoch)
                elif int(packet.epoch) != last_epoch:
                    if evidence is None:
                        raise RuntimeError(
                            "evidence session is unavailable during source epoch reset"
                        )
                    self._reset_process_epoch_state(
                        run_id=run_id,
                        processor=processor,
                        evidence=evidence,
                        source_epoch=int(packet.epoch),
                    )
                    last_epoch = int(packet.epoch)
                if evidence is None:
                    packet_fps = float(packet.fps or 0.0)
                    if not math.isfinite(packet_fps) or packet_fps <= 0.0:
                        packet_fps = float(self._source_fps or 0.0)
                    effective_source_fps = max(1.0, packet_fps)
                    evidence_fps = max(
                        1,
                        int(runtime_config.get("evidence_fps", 15) or 15),
                    )
                    evidence = EvidenceSession(
                        source_type=source_type,
                        source=source,
                        profile=profile,
                        run_id=run_id,
                        source_epoch=int(packet.epoch),
                        enabled=bool(runtime_config.get("evidence_enabled", True)),
                        pre_frames=int(runtime_config.get("evidence_pre_frames", 12)),
                        post_frames=int(runtime_config.get("evidence_post_frames", 18)),
                        sample_every=max(
                            1,
                            int(effective_source_fps / evidence_fps),
                        ),
                        max_frames_per_event=int(
                            runtime_config.get("evidence_max_frames_per_event", 40)
                        ),
                        clip_fps=int(runtime_config.get("evidence_clip_fps", 6)),
                        writer_queue_capacity=int(
                            runtime_config.get(
                                "evidence_writer_queue_capacity",
                                256,
                            )
                            or 256
                        ),
                        writer_enqueue_timeout_s=float(
                            runtime_config.get(
                                "evidence_writer_enqueue_timeout_s",
                                0.02,
                            )
                            or 0.0
                        ),
                        writer_drain_timeout_s=float(
                            runtime_config.get(
                                "evidence_writer_drain_timeout_s",
                                10.0,
                            )
                            or 10.0
                        ),
                    )
                    if evidence.enabled and evidence.session_dir is not None:
                        snapshot_config = dict(bundle.config)
                        snapshot_config["_runtime_context"] = {
                            "run_id": int(run_id),
                            "source_epoch": int(packet.epoch),
                        }
                        write_config_snapshot(snapshot_config, evidence.session_dir)
                    target_frame_budget_ms = 1000.0 / max(
                        1.0,
                        min(effective_source_fps, process_fps_cap),
                    )
                    with self.condition:
                        if run_id == self.run_id:
                            self.status.update(
                                {
                                    "evidence_session_dir": (
                                        str(evidence.session_dir)
                                        if evidence.session_dir is not None
                                        else None
                                    ),
                                    "evidence_manifest_path": (
                                        str(evidence.manifest_path)
                                        if evidence.manifest_path is not None
                                        else None
                                    ),
                                    **_evidence_writer_status_contract(
                                        evidence
                                    ),
                                    "source_fps": effective_source_fps,
                                    "target_frame_budget_ms": target_frame_budget_ms,
                                }
                            )
                            self.condition.notify_all()
                with self.lock:
                    display_options = dict(self.display_options)
                processed = processor.process(
                    packet.frame,
                    frame_idx=packet.frame_idx,
                    source_type=source_type,
                    source=source,
                    profile=profile,
                    realtime=realtime,
                    video_time_s=packet.source_time_s,
                    source_fps=effective_source_fps,
                    dropped_frames=detection_bus.dropped,
                    display_options=display_options,
                    feature_options=feature_options,
                    custom_model=custom_model,
                    target_frame_budget_ms=float(target_frame_budget_ms),
                    temporal_previous_frame=packet.previous_frame,
                    temporal_previous_frame_idx=packet.previous_frame_idx,
                    temporal_previous_source_time_s=packet.previous_source_time_s,
                    decoded_frame_lease=packet.decoder_lease,
                    temporal_previous_decoded_frame=(
                        packet.previous_decoder_lease
                    ),
                )
                if packet.decoder_lease is not None:
                    self._refresh_active_decoder_status(run_id)
                # 实测检测帧率(YOLO+模块A 全链路, 与 demo detect_fps 同口径)。
                self.detect_times.append(time.perf_counter())
                detect_fps = _detector_completion_fps(self.detect_times)
                after_processed_timings = self._after_processed(
                    run_id,
                    processed,
                    evidence,
                    preview_mode="backend_source_pipeline",
                    extra_status={
                        "fps": round(float(detect_fps), 1),
                        "source_fps": effective_source_fps,
                        "source_duration_s": self._source_duration_s,
                        "source_frame_count": self._source_frame_count,
                        "source_time_s": packet.source_time_s,
                        "source_epoch": packet.epoch,
                        "backend_source_pipeline": True,
                        "detector_pipeline_mode": "backend_latest_only",
                        "detector_queue_policy": "latest_only",
                        "dropped_detection_frames": int(detection_bus.dropped),
                    },
                    publish_jpeg=False,
                )
                packet.release_lease_refs()
                active_packet = None
                detector_cycle_ms = (
                    time.perf_counter() - detector_cycle_started
                ) * 1000.0
                self._record_detector_cycle_metrics(
                    run_id=run_id,
                    source_epoch=int(packet.epoch),
                    frame_idx=int(packet.frame_idx),
                    detector_cycle_ms=detector_cycle_ms,
                    evidence_update_ms=float(
                        after_processed_timings.get(
                            "evidence_update_ms",
                            0.0,
                        )
                    ),
                    overlay_status_publish_ms=float(
                        after_processed_timings.get(
                            "overlay_status_publish_ms",
                            0.0,
                        )
                    ),
                )
        except BaseException as exc:
            self._set_error(str(exc), run_id)
            detection_bus.close()
        finally:
            if active_packet is not None:
                active_packet.release_lease_refs()
                active_packet = None
            evidence_drain_started = time.perf_counter()
            evidence_drain_error = ""
            evidence_writer_status = _evidence_writer_status_contract(
                evidence
            )
            with self.condition:
                if run_id == self.run_id:
                    self.status.update(
                        {
                            "evidence_drain_active": evidence is not None,
                            "evidence_drain_completed": False,
                            "evidence_drain_failed": False,
                            "evidence_drain_error": "",
                            **evidence_writer_status,
                        }
                    )
                    self.condition.notify_all()
            if evidence is not None:
                try:
                    self._merge_completed_events(
                        evidence.close(),
                        run_id=run_id,
                    )
                except Exception as exc:
                    evidence_drain_error = (
                        f"{type(exc).__name__}:{exc}"
                    )
                    self._set_error(
                        f"evidence close failed: {exc}",
                        run_id,
                    )
                evidence_writer_status = (
                    _evidence_writer_status_contract(evidence)
                )
                writer_failed = int(
                    evidence_writer_status.get(
                        "evidence_writer_failed",
                        0,
                    )
                    or 0
                )
                writer_pending = int(
                    evidence_writer_status.get(
                        "evidence_writer_pending",
                        0,
                    )
                    or 0
                )
                writer_last_error = str(
                    evidence_writer_status.get(
                        "evidence_writer_last_error",
                        "",
                    )
                    or ""
                )
                writer_alive = bool(
                    evidence_writer_status.get(
                        "evidence_writer_alive",
                        False,
                    )
                )
                if (
                    not evidence_drain_error
                    and (
                        writer_failed > 0
                        or writer_pending > 0
                        or writer_alive
                        or writer_last_error
                    )
                ):
                    evidence_drain_error = (
                        "evidence_writer_unhealthy_after_drain:"
                        f"failed={writer_failed},"
                        f"pending={writer_pending},"
                        f"alive={str(writer_alive).lower()},"
                        f"last_error={writer_last_error or 'none'}"
                    )
                    self._set_error(
                        f"evidence writer drain failed: {evidence_drain_error}",
                        run_id,
                    )
            evidence_drain_ms = (
                time.perf_counter() - evidence_drain_started
            ) * 1000.0
            with self.condition:
                if run_id == self.run_id:
                    self.status.update(
                        {
                            "evidence_drain_active": False,
                            "evidence_drain_completed": (
                                not evidence_drain_error
                            ),
                            "evidence_drain_failed": bool(
                                evidence_drain_error
                            ),
                            "evidence_drain_ms": evidence_drain_ms,
                            "evidence_drain_error": (
                                evidence_drain_error
                            ),
                            **evidence_writer_status,
                            "process_done": True,
                        }
                    )
                    self.condition.notify_all()
            self.process_done_event.set()
            with self.condition:
                release_finished_file_pipeline = self._should_release_finished_file_pipeline(
                    run_id=run_id,
                    source_type=source_type,
                    runtime_config=runtime_config,
                )
            if release_finished_file_pipeline:
                self._release_pipeline_cache()

    def _reset_process_epoch_state(
        self,
        *,
        run_id: int,
        processor: FrameProcessor,
        evidence: EvidenceSession,
        source_epoch: int,
    ) -> None:
        reset_evidence = getattr(evidence, "reset", None)
        if callable(reset_evidence):
            completed = reset_evidence(
                reason="source_epoch_changed",
                source_epoch=int(source_epoch),
            )
            self._merge_completed_events(completed, run_id=run_id)
        processor.reset()

    @staticmethod
    def _preview_target_size(
        width: int,
        height: int,
        max_side: int,
    ) -> tuple[int, int]:
        width = max(1, int(width))
        height = max(1, int(height))
        if max_side <= 0 or max(width, height) <= max_side:
            return width, height
        scale = float(max_side) / float(max(width, height))
        return (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )

    def _materialize_preview_frame(
        self,
        packet: FramePacket,
        *,
        max_side: int,
    ) -> Any:
        target_size = self._preview_target_size(
            packet.width,
            packet.height,
            max_side,
        )
        if packet.decoder_lease is not None:
            return packet.decoder_lease.materialize_host_bgr(
                size=target_size
            )
        frame = packet.frame
        if frame is None:
            raise RuntimeError("frame_packet_has_no_preview_materializer")
        src_height, src_width = frame.shape[:2]
        if (int(src_width), int(src_height)) == target_size:
            return frame
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def _preview_render_loop(self, run_id: int, preview_bus: PreviewBus, preview_render_fps: float) -> None:
        interval = 1.0 / max(1.0, float(preview_render_fps or 25.0))
        last_seq = 0
        scene_cut_frame_idx: int | None = None
        scene_cut_epoch: int | None = None
        previous_display_frame = None
        while run_id == self.run_id and not self.stop_event.is_set():
            started = time.perf_counter()
            packet = preview_bus.wait_for_frame(last_seq, timeout=interval)
            if packet is None:
                if preview_bus.closed:
                    break
                continue
            if packet.seq <= last_seq:
                packet.release_lease_refs()
                continue
            last_seq = packet.seq
            with self.condition:
                if run_id != self.run_id:
                    packet.release_lease_refs()
                    break
                if int(self.status.get("source_epoch") or 0) != int(packet.epoch):
                    packet.release_lease_refs()
                    continue
            if scene_cut_epoch != packet.epoch:
                scene_cut_epoch = packet.epoch
                scene_cut_frame_idx = None
                previous_display_frame = None
            with self.condition:
                waiting_for_first_file_detection = (
                    str(self.status.get("source_type") or "").lower() == "file"
                    and bool(self.status.get("realtime", True))
                    and not bool(self.status.get("preview_never_wait_for_detection", True))
                    and not bool(self.status.get("first_detection_ready"))
                )
            if waiting_for_first_file_detection:
                packet.release_lease_refs()
                if self.stop_event.wait(timeout=min(interval, 0.03)):
                    break
                continue
            try:
                max_side = int(
                    self.get_status().get("preview_max_side") or 960
                )
                display_frame = self._materialize_preview_frame(
                    packet,
                    max_side=max_side,
                )
                if _is_preview_scene_cut(
                    previous_display_frame,
                    display_frame,
                ):
                    scene_cut_frame_idx = packet.frame_idx
                previous_display_frame = display_frame
                overlay = self._select_preview_overlay(
                    packet.source_time_s,
                    packet.epoch,
                    display_frame_idx=packet.frame_idx,
                    display_scene_cut_frame_idx=scene_cut_frame_idx,
                )
                rendered = self._render_backend_preview(
                    packet,
                    overlay,
                    display_frame=display_frame,
                )
                self._publish_preview(
                    encode_jpeg(rendered, quality=82),
                    run_id,
                    source_epoch=packet.epoch,
                    source_time_s=packet.source_time_s,
                    frame_idx=packet.frame_idx,
                )
                if packet.decoder_lease is not None:
                    self._refresh_active_decoder_status(run_id)
                packet.release_lease_refs()
            except BaseException as exc:
                packet.release_lease_refs()
                self._set_error(str(exc), run_id)
                break
            elapsed = time.perf_counter() - started
            if elapsed < interval and self.stop_event.wait(timeout=interval - elapsed):
                break

    def _select_preview_overlay(
        self,
        source_time_s: float,
        epoch: int,
        *,
        display_frame_idx: int | None = None,
        display_scene_cut_frame_idx: int | None = None,
    ) -> dict[str, Any] | None:
        with self.lock:
            records = [
                dict(item)
                for item in self.overlay_timeline
                if int(item.get("source_epoch", item.get("epoch", epoch)) or 0) == int(epoch)
            ]
            match_s = float(self.status.get("overlay_match_window_ms") or 180.0) / 1000.0
            hold_s = float(self.status.get("overlay_hold_ms") or 550.0) / 1000.0
            interp_s = float(self.status.get("overlay_interpolate_ms") or 400.0) / 1000.0
            max_age_s = float(self.status.get("overlay_max_age_ms") or 950.0) / 1000.0
            file_realtime_preview = str(self.status.get("source_type") or "").lower() == "file" and bool(
                self.status.get("realtime", True)
            )
            if file_realtime_preview:
                # File preview follows the native source clock while detection
                # completes asynchronously.  Bridge from measured cadence, not
                # the configured FPS ceiling; otherwise a slow A3b cycle creates
                # a periodic empty-overlay frame even though inference is alive.
                bridge_s = _adaptive_file_overlay_bridge_s(
                    self.status,
                    records,
                )
                match_s = min(match_s, bridge_s)
                hold_s = min(max_age_s, bridge_s)
                interp_s = min(
                    max_age_s,
                    max(interp_s, bridge_s * 2.0),
                )
        if not records:
            return None
        records.sort(key=lambda item: float(item.get("video_time_s") or 0.0))
        prev = None
        next_item = None
        for item in records:
            item_time = float(item.get("video_time_s") or 0.0)
            if item_time <= source_time_s:
                prev = item
            elif next_item is None:
                next_item = item
                break
        if prev is not None and next_item is not None:
            if float(next_item.get("video_time_s") or 0.0) - float(prev.get("video_time_s") or 0.0) <= interp_s:
                next_suppresses_tracks = bool(
                    next_item.get("ppe_source_auth_media_suppressed")
                    or next_item.get("ppe_source_auth_temporal_reset")
                )
                mixed = interpolate_overlay(
                    prev,
                    next_item,
                    source_time_s,
                    keep_unmatched_tracks=(
                        not file_realtime_preview
                        or not next_suppresses_tracks
                    ),
                )
                if mixed is not None:
                    # Box coordinates may be interpolated, but alarm/A3b/PPE
                    # state is discrete and must remain causal until the next
                    # detector timestamp.  Midpoint state switching made HUD
                    # messages clear half an interval early (and appear early).
                    mixed = _causal_interpolated_overlay(
                        prev,
                        mixed,
                        source_time_s,
                    )
                    mixed["overlay_display_source"] = "interpolated"
                    mixed = annotate_alert_display_context(
                        mixed,
                        records,
                        display_frame_idx=display_frame_idx,
                        display_scene_cut_frame_idx=display_scene_cut_frame_idx,
                    )
                    self._preview_last_overlay = mixed
                    return mixed
        nearest = min(records, key=lambda item: abs(float(item.get("video_time_s") or 0.0) - source_time_s))
        if abs(float(nearest.get("video_time_s") or 0.0) - source_time_s) <= match_s:
            nearest["overlay_display_source"] = "nearest"
            nearest = annotate_alert_display_context(
                nearest,
                records,
                display_frame_idx=display_frame_idx,
                display_scene_cut_frame_idx=display_scene_cut_frame_idx,
            )
            self._preview_last_overlay = nearest
            return nearest
        held = self._preview_last_overlay
        if held is not None:
            dt = source_time_s - float(held.get("video_time_s") or 0.0)
            if 0.0 <= dt <= min(hold_s, max_age_s):
                out = dict(held)
                out["video_time_s"] = source_time_s
                out["held"] = True
                out["overlay_display_source"] = "held"
                tracks = []
                crosses_scene_cut = bool(
                    display_frame_idx is not None
                    and display_scene_cut_frame_idx is not None
                    and held.get("frame_idx") is not None
                    and int(held.get("frame_idx") or 0)
                    < int(display_scene_cut_frame_idx)
                    <= int(display_frame_idx)
                )
                for track in out.get("ppe_tracks", []) or []:
                    if crosses_scene_cut:
                        continue
                    if not bool(track.get("hold_eligible", True)):
                        continue
                    clone = dict(track)
                    clone["box"] = list(track.get("box") or [])
                    clone["source"] = "held"
                    tracks.append(clone)
                out["ppe_tracks"] = tracks
                if crosses_scene_cut:
                    # Keep the event/HUD timing context, but never drag a media
                    # or PPE box across an observed scene boundary.
                    out["a3b_bbox"] = None
                return annotate_alert_display_context(
                    out,
                    records,
                    display_frame_idx=display_frame_idx,
                    display_scene_cut_frame_idx=display_scene_cut_frame_idx,
                )
        return None

    def _render_backend_preview(
        self,
        packet: FramePacket,
        overlay: dict[str, Any] | None,
        *,
        display_frame: Any | None = None,
    ) -> Any:
        status = self.get_status()
        display_options = status.get("display_options", dict(self.display_options))
        max_side = int(status.get("preview_max_side") or 960)
        if display_frame is None:
            display_frame = self._materialize_preview_frame(
                packet,
                max_side=max_side,
            )
        height, width = display_frame.shape[:2]
        with self.condition:
            if packet.epoch == self.status.get("source_epoch"):
                self.status["preview_width"] = int(width)
                self.status["preview_height"] = int(height)
        overlay = overlay or {}
        scaled_tracks = []
        for track in overlay.get("ppe_tracks", []) or []:
            box = track.get("box") or []
            if len(box) < 4:
                continue
            copy = dict(track)
            copy["box"] = [
                float(box[0]) * width / 640.0,
                float(box[1]) * height / 640.0,
                float(box[2]) * width / 640.0,
                float(box[3]) * height / 640.0,
            ]
            scaled_tracks.append(copy)
        info = preview_module_info_from_overlay(overlay)
        info["detect_fps"] = float(status.get("fps", 0.0) or 0.0)
        ppe = {
            "warning": bool(overlay.get("ppe_warning")),
            "confirmed": bool(overlay.get("ppe_confirmed")),
            "event_active": bool(overlay.get("ppe_event_active")),
            "event_hold_remaining": int(overlay.get("ppe_event_hold_remaining") or 0),
            "event_last_reason": overlay.get("ppe_event_last_reason") or "",
            "event_last_confirmed_source": overlay.get("ppe_event_last_confirmed_source") or "",
            "person_count": int(overlay.get("ppe_person_count") or 0),
            "raw_person_count": int(overlay.get("ppe_raw_person_count", overlay.get("ppe_person_count")) or 0),
            "inferred_person_count": int(overlay.get("ppe_inferred_person_count", overlay.get("ppe_person_count")) or 0),
            "person_context_count": int(overlay.get("ppe_person_context_count", overlay.get("ppe_person_count")) or 0),
            "weak_person_count": int(overlay.get("ppe_weak_person_count") or 0),
            "promoted_person_count": int(overlay.get("ppe_promoted_person_count") or 0),
            "effective_person_count": int(overlay.get("ppe_effective_person_count", overlay.get("ppe_person_count")) or 0),
            "helmet_count": int(overlay.get("ppe_helmet_count") or 0),
            "raw_helmet_count": int(overlay.get("ppe_raw_helmet_count", overlay.get("ppe_helmet_count")) or 0),
            "weak_helmet_count": int(overlay.get("ppe_weak_helmet_count") or 0),
            "promoted_helmet_count": int(overlay.get("ppe_promoted_helmet_count") or 0),
            "effective_helmet_count": int(overlay.get("ppe_effective_helmet_count", overlay.get("ppe_helmet_count")) or 0),
            "head_count": int(overlay.get("ppe_head_count") or 0),
            "raw_head_count": int(overlay.get("ppe_raw_head_count", overlay.get("ppe_head_count")) or 0),
            "weak_head_count": int(overlay.get("ppe_weak_head_count") or 0),
            "promoted_head_count": int(overlay.get("ppe_promoted_head_count") or 0),
            "effective_head_count": int(overlay.get("ppe_effective_head_count", overlay.get("ppe_head_count")) or 0),
            "missing_helmet_count": int(overlay.get("ppe_missing_helmet_count") or 0),
            "uncertain": bool(overlay.get("ppe_uncertain")),
            "reason": overlay.get("ppe_reason") or "",
        }
        return render_preview(
            display_frame,
            info=info,
            ppe=ppe,
            ppe_tracks=scaled_tracks,
            display_options=display_options,
            frame_idx=packet.frame_idx,
        )

    def _after_processed(
        self,
        run_id: int,
        processed: ProcessedFrame,
        evidence: EvidenceSession,
        *,
        preview_mode: str,
        extra_status: dict[str, Any] | None = None,
        publish_jpeg: bool,
    ) -> dict[str, float]:
        empty_timings = {
            "evidence_update_ms": 0.0,
            "overlay_status_publish_ms": 0.0,
        }
        with self.lock:
            display_options = dict(self.display_options)
        if publish_jpeg:
            # For MJPEG output the boxes are baked into the JPEG, so re-render
            # with the latest display options.
            rendered = render_preview(
                processed.frame_640,
                info=processed.info,
                ppe=processed.ppe,
                ppe_tracks=processed.ppe_tracks,
                display_options=display_options,
                frame_idx=processed.frame_idx,
            )
            source_epoch = None
            if extra_status and "source_epoch" in extra_status:
                try:
                    source_epoch = int(extra_status["source_epoch"])
                except (TypeError, ValueError):
                    source_epoch = None
            self._publish_preview(encode_jpeg(rendered, quality=82), run_id, source_epoch=source_epoch)
        processed_epoch = None
        if extra_status and "source_epoch" in extra_status:
            try:
                processed_epoch = int(extra_status["source_epoch"])
            except (TypeError, ValueError):
                processed_epoch = None
        with self.condition:
            if run_id != self.run_id or self.stop_event.is_set():
                return empty_timings
            current_epoch = int(self.status.get("source_epoch") or 0)
            if processed_epoch is not None and processed_epoch != current_epoch:
                return empty_timings
        processed.status["timing_frame_idx"] = int(processed.frame_idx)
        processed.status["latency_frame_idx"] = int(processed.frame_idx)
        evidence_status = dict(processed.status)
        if extra_status:
            evidence_status.update(extra_status)
        evidence_status["run_id"] = int(run_id)
        evidence_started = time.perf_counter()
        completed = evidence.update(
            frame_idx=processed.frame_idx,
            frame=processed.frame_640,
            info=processed.info,
            ppe=processed.ppe,
            status=evidence_status,
        )
        self._merge_completed_events(
            completed,
            run_id=run_id,
            source_epoch=processed_epoch,
        )
        evidence_update_ms = (
            time.perf_counter() - evidence_started
        ) * 1000.0
        processed.status["evidence_update_ms"] = evidence_update_ms
        processed.status.update(
            _evidence_writer_status_contract(evidence)
        )
        overlay_publish_started = time.perf_counter()
        record_status = dict(processed.status)
        if extra_status:
            record_status.update(extra_status)
        overlay_record = self._build_overlay_record(record_status, processed.ppe_tracks)
        with self.condition:
            if run_id != self.run_id or self.stop_event.is_set():
                return
            current_epoch = int(self.status.get("source_epoch") or 0)
            if processed_epoch is not None and processed_epoch != current_epoch:
                return
            self.overlay_seq += 1
            overlay_record["overlay_seq"] = self.overlay_seq
            self.overlay_timeline.append(overlay_record)
            processed.status["overlay_seq"] = self.overlay_seq
            processed.status["evidence_session_dir"] = (
                str(evidence.session_dir)
                if evidence.session_dir is not None
                else None
            )
            processed.status["evidence_manifest_path"] = (
                str(evidence.manifest_path)
                if evidence.manifest_path is not None
                else None
            )
            processed.status["evidence_saved_event_count"] = evidence.saved_event_count
            processed.status["recent_events"] = list(self.recent_events)
            processed.status["recent_ppe_events"] = list(self.recent_ppe_events)
            processed.status["recent_source_auth_events"] = list(self.recent_source_events)
            processed.status["preview_mode"] = preview_mode
            processed.status["display_options"] = dict(display_options)
            if extra_status:
                processed.status.update(extra_status)
            first_detection_pending = not bool(self.status.get("first_detection_ready"))
            if first_detection_pending:
                processed.status["first_detection_processing_ms"] = float(
                    processed.status.get("processing_ms") or 0.0
                )
                processed.status["first_detection_timing_ms"] = float(processed.status.get("timing_ms") or 0.0)
                processed.status["first_detection_detector_inference_ms"] = float(
                    processed.status.get("detector_inference_ms") or 0.0
                )
                processed.status["first_detection_module_a_timing_ms"] = float(
                    processed.status.get("module_a_timing_ms") or 0.0
                )
                processed.status["first_detection_frame_idx"] = int(processed.status.get("frame_idx") or 0)
                processed.status["first_detection_source_time_s"] = float(
                    processed.status.get("source_time_s", processed.status.get("video_time_s", 0.0)) or 0.0
                )
            was_running = bool(self.status.get("running"))
            source_ended = bool(self.status.get("source_ended"))
            source_eof_reached = bool(
                self.status.get("source_eof_reached")
            )
            ended_status = {
                "running": False,
                "source_ended": True,
                "source_time_s": self.status.get("source_time_s", processed.status.get("source_time_s", 0.0)),
                "video_time_s": self.status.get("video_time_s", processed.status.get("video_time_s", 0.0)),
                "preview_fps": 0.0,
                "preview_mode": "source_ended",
                "detector_pipeline_mode": "ended",
                "ready_for_preview": False,
                "preview_started": False,
            }
            self.status.update(processed.status)
            self.status["run_id"] = run_id
            self.status["running"] = was_running
            if source_ended:
                self.status.update(ended_status)
            self.status["initializing"] = False
            self.status["prewarming"] = False
            self.status["detector_ready"] = True
            self.status["first_detection_ready"] = True
            if was_running and not source_eof_reached:
                self.status["ready_for_preview"] = True
            elif source_eof_reached:
                self.status["ready_for_preview"] = False
                self.status["preview_started"] = False
                self.status["preview_mode"] = "source_eof_drain"
                self.status["detector_pipeline_mode"] = "draining"
            if "preview_seekable" not in processed.status:
                self.status["preview_seekable"] = str(processed.status.get("source_type") or "").lower() == "file"
            self.condition.notify_all()
        return {
            "evidence_update_ms": evidence_update_ms,
            "overlay_status_publish_ms": (
                time.perf_counter() - overlay_publish_started
            )
            * 1000.0,
        }

    def _build_overlay_record(self, status: dict[str, Any], ppe_tracks: list[dict[str, Any]]) -> dict[str, Any]:
        return build_overlay_record(
            status=status,
            ppe_tracks=ppe_tracks,
            run_id=self.run_id,
            display_options=self.display_options,
        )

    def _publish_preview(
        self,
        jpeg: bytes,
        run_id: int,
        *,
        source_epoch: int | None = None,
        source_time_s: float | None = None,
        frame_idx: int | None = None,
    ) -> None:
        now = time.perf_counter()
        with self.condition:
            if run_id != self.run_id:
                return
            if source_epoch is not None and int(self.status.get("source_epoch") or 0) != int(source_epoch):
                return
            if self.status.get("source_ended"):
                self.latest_jpeg = None
                self.latest_jpeg_meta = {}
                self.status["preview_fps"] = 0.0
                self.status["preview_started"] = False
                self.status["ready_for_preview"] = False
                self.status["detector_pipeline_mode"] = "ended"
                self.condition.notify_all()
                return
            self.latest_jpeg = jpeg
            self.latest_jpeg_seq += 1
            self.latest_jpeg_meta = {
                "preview_seq": int(self.latest_jpeg_seq),
                "source_epoch": (
                    int(source_epoch)
                    if source_epoch is not None
                    else int(self.status.get("source_epoch") or 0)
                ),
                "source_time_s": (
                    float(source_time_s)
                    if source_time_s is not None
                    else float(self.status.get("source_time_s") or 0.0)
                ),
                "frame_idx": (
                    int(frame_idx)
                    if frame_idx is not None
                    else int(self.status.get("frame_idx") or 0)
                ),
            }
            self.preview_publish_times.append(now)
            preview_fps = 0.0
            if len(self.preview_publish_times) >= 2:
                elapsed = self.preview_publish_times[-1] - self.preview_publish_times[0]
                if elapsed > 0:
                    preview_fps = (len(self.preview_publish_times) - 1) / elapsed
            self.status["preview_seq"] = self.latest_jpeg_seq
            self.status["preview_fps"] = preview_fps
            self.condition.notify_all()

    def _merge_completed_events(
        self,
        events: list[dict[str, Any]] | dict[str, Any],
        *,
        run_id: int | None = None,
        source_epoch: int | None = None,
    ) -> None:
        if isinstance(events, dict):
            events = [events]
        if not events:
            return
        with self.condition:
            if run_id is not None and int(run_id) != int(self.run_id):
                return
            if source_epoch is not None and int(source_epoch) != int(
                self.status.get("source_epoch") or 0
            ):
                return
            for event in events:
                channel = str(event.get("channel", ""))
                if channel == "ppe":
                    self.recent_ppe_events.appendleft(event)
                elif channel in {"source_auth", "a3b"}:
                    self.recent_source_events.appendleft(event)
                else:
                    self.recent_events.appendleft(event)
            self.condition.notify_all()

    def _set_error(self, message: str, run_id: int) -> None:
        preview_bus: PreviewBus | None = None
        detection_bus: DetectionBus | None = None
        with self.condition:
            if run_id != self.run_id:
                return
            first_error = str(self.status.get("error") or "").strip()
            if not first_error:
                self.status["error"] = message
            elif message and message != first_error:
                secondary_errors = list(self.status.get("secondary_errors") or [])
                if message not in secondary_errors:
                    secondary_errors.append(message)
                self.status["secondary_errors"] = secondary_errors
            self.stop_event.set()
            preview_bus = self.preview_bus
            detection_bus = self.detection_bus
            self.status["running"] = False
            self.status["initializing"] = False
            self.status["prewarming"] = False
            self.status["ready_for_preview"] = False
            self.status["preview_started"] = False
            self.status["preview_fps"] = 0.0
            self.status["preview_mode"] = "error"
            self.status["detector_pipeline_mode"] = "error"
            self.latest_jpeg = None
            self.latest_jpeg_meta = {}
            self.status["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self.condition.notify_all()
        if preview_bus is not None:
            preview_bus.close()
        if detection_bus is not None:
            detection_bus.close()
