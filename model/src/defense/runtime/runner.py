from __future__ import annotations

import platform
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlparse

import cv2

from defense.visualization import encode_jpeg, render_preview
from defense.web.overlay_timeline import interpolate_overlay

from .config import normalize_custom_model_options, project_root, workspace_asset_roots, workspace_material_root, workspace_root
from .evidence import EvidenceSession
from .frame_processor import FrameProcessor, prepare_frame_640, build_branch_cards, ProcessedFrame
from .backend_pipeline import DetectionBus, FramePacket, PreviewBus
from .overlay_records import build_overlay_record
from .pipeline_factory import PipelineCache


_INVISIBLE_PATH_CHARS = {
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\u00a0",
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


def validate_file_source(source: str) -> Path:
    path = resolve_source_path(source)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"视频文件不存在或不可访问: {path}")
    cap = None
    try:
        cap = open_capture("file", str(path))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"视频文件可打开但读不到有效帧: {path}")
    finally:
        if cap is not None:
            cap.release()
    return path


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
            cap = None
            for backend in (cv2.CAP_DSHOW, getattr(cv2, "CAP_MSMF", cv2.CAP_ANY), cv2.CAP_ANY):
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    break
            if cap is None:
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        else:
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    elif source_type == "file":
        path = resolve_source_path(source)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {path}")
        cap = cv2.VideoCapture(str(path))
    elif source_type == "rtsp":
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_ms))
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_ms))
        except Exception:
            pass
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


class MonitorEngine:
    """Runtime service boundary for capture, clocking, inference and evidence.

    MP4, RTSP and camera inputs all use the backend source pipeline. Preview
    frames are published by the source clock, while inference consumes a
    latest-only queue so slow detection can drop stale work without stalling the
    displayed stream.
    """

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
        self.preview_publish_times: deque[float] = deque(maxlen=60)
        self.overlay_timeline: deque[dict[str, Any]] = deque(maxlen=1000)
        self.overlay_seq = 0
        self.status: dict[str, Any] = self._empty_status()
        self.display_options: dict[str, bool] = {
            "show_boxes": True,
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

    @staticmethod
    def _empty_status() -> dict[str, Any]:
        status = {
            "run_id": 0,
            "running": False,
            "source_type": None,
            "source": None,
            "profile": "default",
            "backend": None,
            "model_family": None,
            "artifact": None,
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
            "detector_inference_ms": 0.0,
            "module_a_timing_ms": 0.0,
            "p_adv": None,
            "p_adv_display": 0.0,
            "p_adv_missing_reason": "not_started",
            "alert_confirmed": False,
            "attack_detected": False,
            "attack_state_active": False,
            "reason": "",
            "a3b_score": 0.0,
            "a3b_confidence": 0.0,
            "a3b_observed_score": 0.0,
            "a3b_confirmed_score": 0.0,
            "a3b_display_score": 0.0,
            "a3b_event_score": 0.0,
            "a3b_state": "normal",
            "a3b_triggered": False,
            "a3b_triggered_source": "none",
            "a3b_reason": "",
            "a3b_debug": {},
            "ppe_warning": False,
            "ppe_candidate": False,
            "ppe_confirmed": False,
            "ppe_person_count": 0,
            "ppe_raw_person_count": 0,
            "ppe_inferred_person_count": 0,
            "ppe_person_context_count": 0,
            "ppe_helmet_count": 0,
            "ppe_head_count": 0,
            "ppe_missing_helmet_count": 0,
            "ppe_has_person_class": False,
            "ppe_evidence_mode": "",
            "ppe_uncertain": False,
            "ppe_reason": "",
            "ppe_tracks": [],
            "feature_options": {"static_image_enabled": True},
            "custom_model": normalize_custom_model_options(None),
            "display_options": {"show_boxes": True, "show_module_hud": True, "show_ppe_hud": True},
            "evidence_session_dir": None,
            "evidence_manifest_path": None,
            "evidence_saved_event_count": 0,
            "recent_events": [],
            "recent_ppe_events": [],
            "recent_source_auth_events": [],
            "started_at": None,
            "stopped_at": None,
            "error": "",
            "preview_mode": "idle",
            "initializing": False,
            "detector_ready": False,
            "first_detection_ready": False,
            "ready_for_preview": False,
            "preview_started": False,
            "preview_seekable": False,
            "source_ended": False,
            "stream_reconnects": 0,
            "stream_last_frame_age_ms": 0.0,
            "preview_start_time_s": 0.0,
            "preview_render_fps": 0.0,
            "preview_max_side": 960,
            "preview_width": 0,
            "preview_height": 0,
            "capture_max_side": 1280,
            "file_source_fps_cap": 0.0,
            "source_frame_step": 1.0,
            "source_frame_width": 0,
            "source_frame_height": 0,
            "capture_frame_width": 0,
            "capture_frame_height": 0,
            "capture_resized": False,
            "preview_never_wait_for_detection": True,
            "init_ms": 0.0,
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
            "detector_pipeline_mode": "idle",
            "detector_queue_policy": "latest_only",
            "detector_process_fps_cap": 0.0,
            "dropped_detection_frames": 0,
            "backend_source_pipeline": True,
            "playback_paused": False,
            "playback_speed": 1.0,
            "overlay_max_age_ms": 800.0,
            "stale_overlay_dropped": 0,
            "raw_boxes_count": 0,
            "ppe_boxes_count": 0,
            "tracked_boxes_count": 0,
            "render_boxes_count": 0,
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
    ) -> int:
        source_type = str(source_type or "file").lower()
        # File preview is always clock-locked native playback. The old checkbox
        # no longer switches to the unsynchronized MJPEG path.
        if source_type == "file":
            realtime = True
            # Validate the new file before stopping the current session, so a
            # bad pasted path cannot kill a running monitor.
            source = str(validate_file_source(source))
        elif source_type == "camera":
            camera_text = normalize_source_text(source)
            source = camera_text.split(":", 1)[1].strip() if camera_text.lower().startswith("camera:") else camera_text
        self.stop()
        feature_options = {
            "static_image_enabled": bool((feature_options or {}).get("static_image_enabled", True)),
        }
        custom_model_options = normalize_custom_model_options(custom_model)
        self.stop_event.clear()
        with self.condition:
            self.run_id += 1
            run_id = self.run_id
            self.latest_jpeg = None
            self.latest_jpeg_seq = 0
            self.overlay_timeline.clear()
            self.overlay_seq = 0
            self.preview_publish_times.clear()
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
                    "display_options": dict(self.display_options),
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "preview_mode": "initializing_detector",
                    "initializing": True,
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
                feature_options=feature_options,
                custom_model=custom_model_options,
            )
            init_ms = (time.perf_counter() - init_started) * 1000.0
            with self.condition:
                if run_id == self.run_id:
                    self.status.update(
                        {
                            "backend": preload_bundle.backend,
                            "model_family": preload_bundle.model_family,
                            "artifact": preload_bundle.artifact_path,
                            "custom_model": dict(
                                preload_bundle.config.get("runtime", {}).get(
                                    "custom_model", custom_model_options
                                )
                            ),
                            "initializing": False,
                            "detector_ready": True,
                            "init_ms": init_ms,
                            "warmup_error": preload_bundle.warmup_error,
                            "preview_mode": "mp4_clock_prepare" if source_type == "file" else "detector_ready_wait_first_frame",
                        }
                    )
                    self.condition.notify_all()
        except BaseException as exc:
            self._set_error(str(exc), run_id)
            raise

        runtime_config = preload_bundle.config.get("runtime", {}) if isinstance(preload_bundle.config.get("runtime"), dict) else {}
        preview_render_fps = float(
            runtime_config.get("preview_render_fps", runtime_config.get("preview_fps", 25)) or 25
        )
        preview_max_side = int(runtime_config.get("preview_max_side", 960) or 960)
        detector_process_fps_cap = float(
            runtime_config.get("detector_process_fps_cap", runtime_config.get("process_fps_cap", 15)) or 15
        )
        capture_max_side = int(runtime_config.get("capture_max_side", preview_max_side) or preview_max_side)
        file_source_fps_cap = float(
            runtime_config.get(
                "file_source_fps_cap",
                preview_render_fps if source_type == "file" else 0.0,
            )
            or 0.0
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
                        "preview_render_fps": preview_render_fps,
                        "preview_max_side": preview_max_side,
                        "capture_max_side": capture_max_side,
                        "file_source_fps_cap": file_source_fps_cap,
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
            feature_options,
            custom_model_options,
        )
        capture_args = (
            *process_args,
            float(preview_render_fps),
            float(detector_process_fps_cap),
            int(capture_max_side),
            float(file_source_fps_cap),
        )
        self.capture_thread = threading.Thread(target=self._backend_capture_loop, args=capture_args, name="module-a-source", daemon=True)
        self.process_thread = threading.Thread(target=self._backend_process_loop, args=process_args, name="module-a-detector", daemon=True)
        self.preview_thread = threading.Thread(
            target=self._preview_render_loop,
            args=(run_id, preview_bus, float(preview_render_fps)),
            name="module-a-preview",
            daemon=True,
        )
        self.capture_thread.start()
        self.process_thread.start()
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
        allowed = {"show_boxes", "show_module_hud", "show_ppe_hud"}
        with self.condition:
            for key, value in (options or {}).items():
                if key in allowed:
                    self.display_options[key] = bool(value)
            self.status["display_options"] = dict(self.display_options)
            self.condition.notify_all()
            return dict(self.display_options)

    def stop(self) -> None:
        threads = [thread for thread in (self.capture_thread, self.process_thread, self.preview_thread) if thread is not None]
        self.stop_event.set()
        if self.preview_bus is not None:
            self.preview_bus.close()
        if self.detection_bus is not None:
            self.detection_bus.close()
        current = threading.current_thread()
        alive_after_join: list[str] = []
        for thread in threads:
            if thread is current:
                continue
            thread.join(timeout=self.thread_join_timeout_s)
            if thread.is_alive():
                alive_after_join.append(thread.name)
        self.capture_thread = None
        self.process_thread = None
        self.preview_thread = None
        self.preview_bus = None
        self.detection_bus = None
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
            self.status["stop_threads_pending"] = alive_after_join
            if alive_after_join:
                self.status["warning"] = "worker_threads_did_not_stop"
            self.condition.notify_all()

    def _release_pipeline_cache(self) -> None:
        clear_cache = getattr(self.cache, "clear", None)
        if callable(clear_cache):
            clear_cache()

    def control_run(self, run_id: int, action: str, **payload: Any) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        with self.condition:
            if int(run_id or 0) != self.run_id:
                raise RuntimeError("run_id does not match current run")
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
                self._source_epoch += 1
                self.overlay_timeline.clear()
                self.overlay_seq = 0
                self.latest_jpeg = None
                self.latest_jpeg_seq = 0
                self.preview_publish_times.clear()
                self._preview_last_overlay = None
                if self.detection_bus is not None:
                    self.detection_bus.clear()
                self.status.update(
                    {
                        "source_epoch": self._source_epoch,
                        "source_time_s": target,
                        "video_time_s": target,
                        "overlay_seq": 0,
                        "first_detection_ready": False,
                        "source_ended": False,
                        "preview_seq": 0,
                        "preview_fps": 0.0,
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

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            payload = dict(self.status)
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
            while self.latest_jpeg_seq == last_seq and self.status.get("running") and time.perf_counter() < deadline:
                self.condition.wait(timeout=min(0.05, max(0.0, deadline - time.perf_counter())))
            return self.latest_jpeg_seq, self.latest_jpeg, bool(self.status.get("running"))

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
    ) -> None:
        cap = None
        packet_seq = 0
        last_detection_push = 0.0
        last_frame_seen = time.perf_counter()
        reconnects = 0
        next_read_frame = 0.0
        playback_anchor_wall = time.perf_counter()
        playback_anchor_frame = 0.0
        playback_clock_speed = 1.0
        playback_was_paused = False
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
            frame_period = frame_step / max(1.0, fps)
            detect_interval = 1.0 / max(1.0, min(detector_process_fps_cap, fps if source_type == "file" else detector_process_fps_cap))
            with self.condition:
                if run_id == self.run_id:
                    self.status.update(
                        {
                            "source_fps": fps,
                            "source_duration_s": duration_s,
                            "source_frame_count": frame_count,
                            "ready_for_preview": True,
                            "preview_started": True,
                            "preview_seekable": source_type == "file",
                            "preview_mode": "backend_source_pipeline",
                            "detector_pipeline_mode": "backend_latest_only",
                            "capture_max_side": int(capture_max_side),
                            "file_source_fps_cap": float(file_source_fps_cap or 0.0),
                            "source_frame_step": float(frame_step),
                        }
                    )
                    self.condition.notify_all()

            while run_id == self.run_id and not self.stop_event.is_set():
                loop_started = time.perf_counter()
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
                    with self.condition:
                        if run_id == self.run_id:
                            self.status["source_epoch"] = current_epoch
                    continue
                if paused and source_type == "file":
                    playback_was_paused = True
                    time.sleep(0.03)
                    continue
                if source_type == "file" and realtime:
                    current_index_for_anchor = float(cap.get(cv2.CAP_PROP_POS_FRAMES) or next_read_frame)
                    if playback_was_paused or abs(float(speed) - float(playback_clock_speed)) > 1e-3:
                        playback_anchor_wall = time.perf_counter()
                        playback_anchor_frame = current_index_for_anchor
                        playback_clock_speed = float(speed)
                        playback_was_paused = False

                ok, frame = cap.read()
                if not ok or frame is None:
                    if source_type == "file":
                        break
                    now = time.perf_counter()
                    if now - last_frame_seen >= 5.0:
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
                        last_frame_seen = time.perf_counter()
                    time.sleep(0.03)
                    continue

                last_frame_seen = time.perf_counter()
                original_h, original_w = frame.shape[:2]
                frame, capture_resized = self._resize_capture_frame(frame, int(capture_max_side))
                h, w = frame.shape[:2]
                source_time_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if source_time_s <= 0.0 and fps > 0:
                    source_time_s = packet_seq / fps
                packet_seq += 1
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or packet_seq)
                packet = FramePacket(
                    seq=packet_seq,
                    frame_idx=max(0, frame_idx - 1),
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
                    },
                )
                preview_bus.publish(packet)
                now = time.perf_counter()
                if now - last_detection_push >= detect_interval:
                    last_detection_push = now
                    detection_bus.push(packet)
                with self.condition:
                    if run_id == self.run_id:
                        self.status.update(
                            {
                                "source_time_s": packet.source_time_s,
                                "video_time_s": packet.source_time_s,
                                "source_epoch": packet.epoch,
                                "preview_started": True,
                                "ready_for_preview": True,
                                "source_frame_width": int(original_w),
                                "source_frame_height": int(original_h),
                                "capture_frame_width": int(w),
                                "capture_frame_height": int(h),
                                "capture_resized": bool(capture_resized),
                                "dropped_detection_frames": int(detection_bus.dropped),
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
                    for _ in range(min(skip_count, 120)):
                        if not cap.grab():
                            break
                if source_type == "file" and realtime:
                    elapsed = time.perf_counter() - loop_started
                    wait_s = max(0.0, (frame_period / speed) - elapsed)
                    if wait_s > 0:
                        time.sleep(wait_s)
        except BaseException as exc:
            self._set_error(str(exc), run_id)
        finally:
            if cap is not None:
                cap.release()
            with self.condition:
                self._capture_done = True
                if run_id == self.run_id:
                    self.status["running"] = False
                    source_ended = source_type == "file" and not self.stop_event.is_set()
                    self.status["source_ended"] = source_ended
                    self.status["preview_started"] = False
                    if source_ended:
                        final_time_s = float(self.status.get("source_time_s") or 0.0)
                        if self._source_duration_s > 0:
                            final_time_s = min(max(final_time_s, 0.0), float(self._source_duration_s))
                        self.status.update(
                            {
                                "source_time_s": final_time_s,
                                "video_time_s": final_time_s,
                                "preview_fps": 0.0,
                                "preview_mode": "source_ended",
                                "detector_pipeline_mode": "ended",
                            }
                        )
                    self.status["stopped_at"] = datetime.now().isoformat(timespec="seconds")
                self.condition.notify_all()
            preview_bus.close()
            detection_bus.close()

    def _backend_process_loop(
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
    ) -> None:
        evidence: EvidenceSession | None = None
        last_seq = 0
        last_epoch: int | None = None
        try:
            bundle = self.cache.get(profile=profile, feature_options=feature_options, custom_model=custom_model)
            runtime_config = bundle.config.get("runtime", {}) if isinstance(bundle.config.get("runtime"), dict) else {}
            processor = FrameProcessor(bundle, jpeg_quality=int(runtime_config.get("jpeg_quality", 82)))
            evidence = EvidenceSession(
                source_type=source_type,
                source=source,
                profile=profile,
                enabled=bool(runtime_config.get("evidence_enabled", True)),
                pre_frames=int(runtime_config.get("evidence_pre_frames", 12)),
                post_frames=int(runtime_config.get("evidence_post_frames", 18)),
                sample_every=max(1, int(max(1, self._source_fps) / max(1, int(runtime_config.get("evidence_fps", 15))))),
            )
            process_fps_cap = float(runtime_config.get("detector_process_fps_cap", runtime_config.get("process_fps_cap", 15)) or 15)
            target_frame_budget_ms = 1000.0 / max(1.0, min(self._source_fps, process_fps_cap))
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
                            "evidence_session_dir": str(evidence.session_dir),
                            "evidence_manifest_path": str(evidence.manifest_path),
                            "detector_ready": True,
                            "initializing": False,
                            "detector_process_fps_cap": process_fps_cap,
                            "detector_queue_policy": "latest_only",
                        }
                    )
                    self.condition.notify_all()

            while run_id == self.run_id and not self.stop_event.is_set():
                packet = detection_bus.pop_latest(last_seq, timeout=0.2)
                if packet is None:
                    if detection_bus.closed:
                        break
                    continue
                last_seq = packet.seq
                if last_epoch is None:
                    last_epoch = int(packet.epoch)
                elif int(packet.epoch) != last_epoch:
                    processor.reset()
                    last_epoch = int(packet.epoch)
                display_options = self.get_status().get("display_options", dict(self.display_options))
                processed = processor.process(
                    packet.frame,
                    frame_idx=packet.frame_idx,
                    source_type=source_type,
                    source=source,
                    profile=profile,
                    realtime=realtime,
                    video_time_s=packet.source_time_s,
                    source_fps=self._source_fps,
                    dropped_frames=detection_bus.dropped,
                    display_options=display_options,
                    feature_options=feature_options,
                    custom_model=custom_model,
                    target_frame_budget_ms=target_frame_budget_ms,
                )
                self._after_processed(
                    run_id,
                    processed,
                    evidence,
                    preview_mode="backend_source_pipeline",
                    extra_status={
                        "source_fps": self._source_fps,
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
        except BaseException as exc:
            self._set_error(str(exc), run_id)
            detection_bus.close()
        finally:
            if evidence is not None:
                try:
                    self._merge_completed_events(evidence.close())
                except Exception as exc:
                    self._set_error(f"evidence close failed: {exc}", run_id)
            with self.condition:
                release_finished_file_pipeline = (
                    run_id == self.run_id
                    and source_type == "file"
                    and bool(self.status.get("source_ended"))
                    and not self.stop_event.is_set()
                )
            if release_finished_file_pipeline:
                self._release_pipeline_cache()

    def _preview_render_loop(self, run_id: int, preview_bus: PreviewBus, preview_render_fps: float) -> None:
        interval = 1.0 / max(1.0, float(preview_render_fps or 25.0))
        last_seq = 0
        while run_id == self.run_id and not self.stop_event.is_set():
            started = time.perf_counter()
            packet = preview_bus.latest_packet_if_open()
            if packet is None:
                packet = preview_bus.wait_for_frame(last_seq, timeout=interval)
            if packet is None:
                if preview_bus.closed:
                    break
                continue
            last_seq = packet.seq
            try:
                overlay = self._select_preview_overlay(packet.source_time_s, packet.epoch)
                rendered = self._render_backend_preview(packet, overlay)
                self._publish_preview(encode_jpeg(rendered, quality=82), run_id)
            except BaseException as exc:
                self._set_error(str(exc), run_id)
                break
            elapsed = time.perf_counter() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _select_preview_overlay(self, source_time_s: float, epoch: int) -> dict[str, Any] | None:
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
                mixed = interpolate_overlay(prev, next_item, source_time_s)
                if mixed is not None:
                    mixed["overlay_display_source"] = "interpolated"
                    self._preview_last_overlay = mixed
                    return mixed
        nearest = min(records, key=lambda item: abs(float(item.get("video_time_s") or 0.0) - source_time_s))
        if abs(float(nearest.get("video_time_s") or 0.0) - source_time_s) <= match_s:
            nearest["overlay_display_source"] = "nearest"
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
                for track in out.get("ppe_tracks", []) or []:
                    clone = dict(track)
                    clone["box"] = list(track.get("box") or [])
                    clone["source"] = "held"
                    tracks.append(clone)
                out["ppe_tracks"] = tracks
                return out
        return None

    def _render_backend_preview(self, packet: FramePacket, overlay: dict[str, Any] | None) -> Any:
        frame = packet.frame
        status = self.get_status()
        display_options = status.get("display_options", dict(self.display_options))
        max_side = int(status.get("preview_max_side") or 960)
        src_height, src_width = frame.shape[:2]
        if max_side > 0 and max(src_width, src_height) > max_side:
            scale = float(max_side) / float(max(src_width, src_height))
            width = max(1, int(round(src_width * scale)))
            height = max(1, int(round(src_height * scale)))
            display_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        else:
            display_frame = frame
            height, width = src_height, src_width
        with self.condition:
            if packet.epoch == self.status.get("source_epoch"):
                self.status["preview_width"] = int(width)
                self.status["preview_height"] = int(height)
        if overlay is None:
            return display_frame
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
        info = {
            "p_adv": overlay.get("p_adv"),
            "alert_confirmed": bool(overlay.get("alert_confirmed")),
            "attack_detected": bool(overlay.get("attack_detected")),
            "timing_ms": float(overlay.get("timing_ms") or 0.0),
            "layer_triggered": overlay.get("a3b_triggered_source") or "backend",
            "reason_codes": [overlay.get("a3b_reason")] if overlay.get("a3b_reason") else [],
        }
        ppe = {
            "warning": bool(overlay.get("ppe_warning")),
            "confirmed": bool(overlay.get("ppe_warning")),
            "person_count": int(overlay.get("ppe_person_count") or 0),
            "raw_person_count": int(overlay.get("ppe_raw_person_count", overlay.get("ppe_person_count")) or 0),
            "inferred_person_count": int(overlay.get("ppe_inferred_person_count", overlay.get("ppe_person_count")) or 0),
            "person_context_count": int(overlay.get("ppe_person_context_count", overlay.get("ppe_person_count")) or 0),
            "helmet_count": int(overlay.get("ppe_helmet_count") or 0),
            "head_count": int(overlay.get("ppe_head_count") or 0),
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
    ) -> None:
        display_options = self.get_status().get("display_options", dict(self.display_options))
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
            self._publish_preview(encode_jpeg(rendered, quality=82), run_id)
        completed = evidence.update(
            frame_idx=processed.frame_idx,
            frame=processed.frame_640,
            info=processed.info,
            ppe=processed.ppe,
            status=processed.status,
        )
        self._merge_completed_events(completed)
        record_status = dict(processed.status)
        if extra_status:
            record_status.update(extra_status)
        overlay_record = self._build_overlay_record(record_status, processed.ppe_tracks)
        with self.condition:
            if run_id != self.run_id:
                return
            self.overlay_seq += 1
            overlay_record["overlay_seq"] = self.overlay_seq
            self.overlay_timeline.append(overlay_record)
            processed.status["overlay_seq"] = self.overlay_seq
            processed.status["evidence_session_dir"] = str(evidence.session_dir)
            processed.status["evidence_manifest_path"] = str(evidence.manifest_path)
            processed.status["evidence_saved_event_count"] = evidence.saved_event_count
            processed.status["recent_events"] = list(self.recent_events)
            processed.status["recent_ppe_events"] = list(self.recent_ppe_events)
            processed.status["recent_source_auth_events"] = list(self.recent_source_events)
            processed.status["preview_mode"] = preview_mode
            processed.status["display_options"] = dict(display_options)
            if extra_status:
                processed.status.update(extra_status)
            was_running = bool(self.status.get("running"))
            source_ended = bool(self.status.get("source_ended"))
            ended_status = {
                "running": False,
                "source_ended": True,
                "source_time_s": self.status.get("source_time_s", processed.status.get("source_time_s", 0.0)),
                "video_time_s": self.status.get("video_time_s", processed.status.get("video_time_s", 0.0)),
                "preview_fps": 0.0,
                "preview_mode": "source_ended",
                "detector_pipeline_mode": "ended",
                "preview_started": False,
                "timing_ms": self.status.get("timing_ms", processed.status.get("timing_ms", 0.0)),
                "processing_ms": self.status.get("processing_ms", processed.status.get("processing_ms", 0.0)),
                "detector_inference_ms": self.status.get(
                    "detector_inference_ms", processed.status.get("detector_inference_ms", 0.0)
                ),
                "module_a_timing_ms": self.status.get(
                    "module_a_timing_ms", processed.status.get("module_a_timing_ms", 0.0)
                ),
            }
            self.status.update(processed.status)
            self.status["run_id"] = run_id
            self.status["running"] = was_running
            if source_ended:
                self.status.update(ended_status)
                self.status["timing_ms"] = float(ended_status.get("timing_ms", self.status.get("timing_ms") or 0.0))
                self.status["processing_ms"] = float(
                    ended_status.get("processing_ms", self.status.get("processing_ms") or 0.0)
                )
                self.status["detector_inference_ms"] = float(
                    ended_status.get("detector_inference_ms", self.status.get("detector_inference_ms") or 0.0)
                )
                self.status["module_a_timing_ms"] = float(
                    ended_status.get("module_a_timing_ms", self.status.get("module_a_timing_ms") or 0.0)
                )
            self.status["initializing"] = False
            self.status["detector_ready"] = True
            self.status["first_detection_ready"] = True
            if was_running:
                self.status["ready_for_preview"] = True
            if "preview_seekable" not in processed.status:
                self.status["preview_seekable"] = str(processed.status.get("source_type") or "").lower() == "file"
            self.condition.notify_all()

    def _build_overlay_record(self, status: dict[str, Any], ppe_tracks: list[dict[str, Any]]) -> dict[str, Any]:
        return build_overlay_record(
            status=status,
            ppe_tracks=ppe_tracks,
            run_id=self.run_id,
            display_options=self.display_options,
        )

    def _publish_preview(self, jpeg: bytes, run_id: int) -> None:
        now = time.perf_counter()
        with self.condition:
            if run_id != self.run_id:
                return
            if self.status.get("source_ended"):
                self.status["preview_fps"] = 0.0
                self.status["preview_started"] = False
                self.status["detector_pipeline_mode"] = "ended"
                self.condition.notify_all()
                return
            self.latest_jpeg = jpeg
            self.latest_jpeg_seq += 1
            self.preview_publish_times.append(now)
            preview_fps = 0.0
            if len(self.preview_publish_times) >= 2:
                elapsed = self.preview_publish_times[-1] - self.preview_publish_times[0]
                if elapsed > 0:
                    preview_fps = (len(self.preview_publish_times) - 1) / elapsed
            self.status["preview_seq"] = self.latest_jpeg_seq
            self.status["preview_fps"] = preview_fps
            self.condition.notify_all()

    def _merge_completed_events(self, events: list[dict[str, Any]] | dict[str, Any]) -> None:
        if isinstance(events, dict):
            events = [events]
        if not events:
            return
        with self.condition:
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
        if run_id != self.run_id:
            return
        with self.condition:
            self.status["error"] = message
            self.status["running"] = False
            self.status["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self.condition.notify_all()
