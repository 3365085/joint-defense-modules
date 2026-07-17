from __future__ import annotations

import base64
import copy
import json
import logging
import os
import queue
import re
import secrets
import shutil
import sqlite3
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.runtime.catalog import register_artifact
from defense.visualization import draw_hud, draw_ppe_hud


DEFAULT_EVIDENCE_ROOT = Path(__file__).resolve().parents[3] / "runtime" / "evidence" / "monitor"
LOGGER = logging.getLogger(__name__)


def default_evidence_root() -> Path:
    override = os.environ.get("MODULE_A_EVIDENCE_ROOT")
    return Path(override).expanduser() if override else DEFAULT_EVIDENCE_ROOT


def safe_path_part(text: str, fallback: str = "source", max_len: int = 80) -> str:
    value = re.sub(r"[^0-9A-Za-z_\-.\u4e00-\u9fff]+", "_", str(text or "").strip())
    value = value.strip("._-")
    return (value or fallback)[:max_len]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def _evidence_db_path(root: str | Path | None = None) -> Path:
    return _resolved_root(root) / "evidence_index.sqlite3"


def _ensure_evidence_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_events (
            event_key TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            source_type TEXT,
            source TEXT,
            profile TEXT,
            session_dir TEXT,
            event_dir TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT,
            started_frame INTEGER,
            last_active_frame INTEGER,
            frame_count INTEGER,
            max_score REAL,
            reason TEXT,
            clip_path TEXT,
            representative_path TEXT,
            event_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_events_ended_at ON evidence_events(ended_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_events_channel ON evidence_events(channel)")


def _index_evidence_event(
    summary: dict[str, Any],
    *,
    root: str | Path | None = None,
    raise_errors: bool = False,
) -> bool:
    try:
        event_dir = Path(str(summary.get("event_dir") or "")).resolve()
        if not event_dir:
            return False
        base = _resolved_root(root)
        if not _is_under(event_dir, base):
            return False
        event_key = str(summary.get("evidence_event_key") or _event_key(event_dir, root=base))
        now = datetime.now().isoformat(timespec="seconds")
        db_path = _evidence_db_path(base)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            _ensure_evidence_db(conn)
            conn.execute(
                """
                INSERT INTO evidence_events (
                    event_key, channel, source_type, source, profile, session_dir, event_dir,
                    started_at, ended_at, started_frame, last_active_frame, frame_count,
                    max_score, reason, clip_path, representative_path, event_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_key) DO UPDATE SET
                    channel=excluded.channel,
                    source_type=excluded.source_type,
                    source=excluded.source,
                    profile=excluded.profile,
                    session_dir=excluded.session_dir,
                    event_dir=excluded.event_dir,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    started_frame=excluded.started_frame,
                    last_active_frame=excluded.last_active_frame,
                    frame_count=excluded.frame_count,
                    max_score=excluded.max_score,
                    reason=excluded.reason,
                    clip_path=excluded.clip_path,
                    representative_path=excluded.representative_path,
                    event_json=excluded.event_json,
                    updated_at=excluded.updated_at
                """,
                (
                    event_key,
                    str(summary.get("channel") or ""),
                    str(summary.get("source_type") or ""),
                    str(summary.get("source") or ""),
                    str(summary.get("profile") or ""),
                    str(summary.get("session_dir") or ""),
                    str(summary.get("event_dir") or ""),
                    str(summary.get("started_at") or ""),
                    str(summary.get("ended_at") or ""),
                    _int(summary.get("started_frame")),
                    _int(summary.get("last_active_frame")),
                    _int(summary.get("frame_count")),
                    float(summary.get("max_score") or 0.0),
                    str(summary.get("reason") or summary.get("ppe_event_last_reason") or ""),
                    str(summary.get("evidence_clip_path") or ""),
                    str(summary.get("evidence_representative_path") or summary.get("representative_path") or ""),
                    json.dumps(summary, ensure_ascii=False, default=str),
                    now,
                ),
            )
        catalog_root = _catalog_root_for_evidence(base)
        clip_path = str(summary.get("evidence_clip_path") or "")
        if clip_path:
            register_artifact(
                path=clip_path,
                business_domain="module_a",
                category="evidence",
                artifact_type="event_clip",
                catalog_root=catalog_root,
                fingerprint=event_key,
                source_path=summary.get("source"),
                status=str(summary.get("channel") or ""),
                metadata={
                    "event_key": event_key,
                    "channel": summary.get("channel"),
                    "event_dir": summary.get("event_dir"),
                    "started_frame": summary.get("started_frame"),
                    "last_active_frame": summary.get("last_active_frame"),
                    "reason": summary.get("reason") or summary.get("ppe_event_last_reason"),
                },
            )
        rep_path = str(summary.get("evidence_representative_path") or summary.get("representative_path") or "")
        if rep_path:
            register_artifact(
                path=rep_path,
                business_domain="module_a",
                category="evidence",
                artifact_type="representative_frame",
                catalog_root=catalog_root,
                fingerprint=event_key,
                source_path=summary.get("source"),
                status=str(summary.get("channel") or ""),
                metadata={"event_key": event_key, "channel": summary.get("channel"), "event_dir": summary.get("event_dir")},
            )
        return True
    except Exception:
        if raise_errors:
            raise
        return False


def _resolved_root(root: str | Path | None = None) -> Path:
    return (Path(root) if root else default_evidence_root()).resolve()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _catalog_root_for_evidence(base: Path) -> Path:
    resolved = base.resolve()
    parts = tuple(str(part).lower() for part in resolved.parts)
    if len(parts) >= 3 and parts[-3:] == ("runtime", "evidence", "monitor"):
        return resolved.parents[1]
    if len(parts) >= 2 and parts[-2:] == ("evidence", "monitor"):
        return resolved.parents[1]
    return resolved


def evidence_token_for_path(path: str | Path, *, root: str | Path | None = None) -> str:
    base = _resolved_root(root)
    target = Path(path).resolve()
    if not _is_under(target, base):
        raise ValueError("evidence_path_outside_root")
    rel = target.relative_to(base).as_posix()
    return base64.urlsafe_b64encode(rel.encode("utf-8")).decode("ascii").rstrip("=")


def evidence_path_from_token(token: str, *, root: str | Path | None = None) -> Path:
    text = str(token or "").strip()
    if not text:
        raise ValueError("empty_evidence_token")
    padded = text + ("=" * (-len(text) % 4))
    try:
        rel = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise ValueError("invalid_evidence_token") from exc
    base = _resolved_root(root)
    target = (base / rel).resolve()
    if not _is_under(target, base):
        raise ValueError("evidence_path_outside_root")
    return target


def _event_key(event_dir: Path, *, root: str | Path | None = None) -> str:
    return evidence_token_for_path(event_dir, root=root)


def _file_url(path: str | Path, *, root: str | Path | None = None) -> str:
    return f"/api/evidence/file?token={evidence_token_for_path(path, root=root)}"


def _event_preview_url(event_dir: Path, *, root: str | Path | None = None) -> str:
    return f"/evidence?event={_event_key(event_dir, root=root)}"


def enrich_evidence_summary(summary: dict[str, Any], *, root: str | Path | None = None) -> dict[str, Any]:
    data = dict(summary or {})
    event_dir = Path(str(data.get("event_dir") or ""))
    if not event_dir:
        return data
    base = _resolved_root(root)
    event_dir = event_dir.resolve()
    if not _is_under(event_dir, base):
        return data
    try:
        key = _event_key(event_dir, root=base)
        data["evidence_event_key"] = key
        data["evidence_preview_url"] = _event_preview_url(event_dir, root=base)
    except ValueError:
        return data
    rep = str(data.get("evidence_representative_path") or data.get("representative_path") or "")
    if rep:
        try:
            data["evidence_representative_url"] = _file_url(rep, root=base)
        except ValueError:
            pass
    clip = str(data.get("evidence_clip_path") or "")
    if clip and data.get("evidence_clip_browser_playable", True) is not False:
        try:
            data["evidence_clip_url"] = _file_url(clip, root=base)
        except ValueError:
            pass
    return data


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def _numeric_frame_clip(
    event_dir: Path,
    frame_paths: list[Path],
    *,
    ffmpeg: str,
    fps: int,
) -> tuple[Path | None, dict[str, Any]]:
    tmp_dir = event_dir / "clip_source"
    tmp_path = event_dir / "clip.tmp.mp4"
    out_path = event_dir / "clip.mp4"
    try:
        if tmp_dir.exists():
            for old in tmp_dir.glob("*.jpg"):
                old.unlink(missing_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame_path in enumerate(frame_paths, start=1):
            target = tmp_dir / f"frame_{idx:06d}.jpg"
            try:
                target.hardlink_to(frame_path)
            except Exception:
                shutil.copy2(frame_path, target)
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(max(1, int(fps))),
            "-i",
            str(tmp_dir / "frame_%06d.jpg"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            return None, {
                "evidence_clip_status": "h264_encode_failed",
                "evidence_clip_error": (result.stderr or result.stdout or "").strip()[:800],
                "evidence_clip_codec": "",
                "evidence_clip_browser_playable": False,
            }
        tmp_path.replace(out_path)
        return out_path, {
            "evidence_clip_status": "ok",
            "evidence_clip_codec": "h264",
            "evidence_clip_fps": max(1, int(fps)),
            "evidence_clip_browser_playable": True,
            "evidence_clip_source_frames": len(frame_paths),
        }
    except Exception as exc:
        return None, {
            "evidence_clip_status": "h264_encode_exception",
            "evidence_clip_error": str(exc)[:800],
            "evidence_clip_codec": "",
            "evidence_clip_browser_playable": False,
        }
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if tmp_dir.exists():
                for old in tmp_dir.glob("*.jpg"):
                    old.unlink(missing_ok=True)
                tmp_dir.rmdir()
        except Exception:
            pass


def _write_browser_mp4_from_frames(
    event_dir: Path,
    frame_paths: list[Path],
    *,
    fps: int,
) -> tuple[Path | None, dict[str, Any]]:
    if not frame_paths:
        return None, {"evidence_clip_status": "no_frames", "evidence_clip_browser_playable": False}
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return None, {
            "evidence_clip_status": "ffmpeg_unavailable",
            "evidence_clip_codec": "",
            "evidence_clip_browser_playable": False,
        }
    return _numeric_frame_clip(event_dir, frame_paths, ffmpeg=ffmpeg, fps=max(1, int(fps)))


def _repair_event_clip_if_needed(data: dict[str, Any], *, root: str | Path | None = None) -> dict[str, Any]:
    if data.get("evidence_clip_codec") == "h264" and data.get("evidence_clip_browser_playable", True) is not False:
        return data
    event_dir = Path(str(data.get("event_dir") or ""))
    if not event_dir:
        return data
    base = _resolved_root(root)
    event_dir = event_dir.resolve()
    if not _is_under(event_dir, base):
        return data
    frames_dir = event_dir / "frames"
    frame_paths = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if not frame_paths:
        return data
    fps = _int(data.get("evidence_clip_fps") or data.get("clip_fps") or 6) or 6
    clip_path, clip_meta = _write_browser_mp4_from_frames(event_dir, frame_paths, fps=fps)
    data.update(clip_meta)
    if clip_path:
        data["evidence_clip_path"] = str(clip_path)
    data = enrich_evidence_summary(data, root=base)
    try:
        write_json(event_dir / "event.json", data)
        _index_evidence_event(data, root=base)
    except Exception:
        pass
    return data


def load_evidence_event(key: str, *, root: str | Path | None = None) -> dict[str, Any]:
    event_dir = evidence_path_from_token(key, root=root)
    event_path = event_dir / "event.json"
    if not event_path.exists() or not event_path.is_file():
        raise FileNotFoundError("evidence_event_not_found")
    data = json.loads(event_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid_evidence_event")
    data = enrich_evidence_summary(data, root=root)
    data = _repair_event_clip_if_needed(data, root=root)
    frames_dir = event_dir / "frames"
    frames: list[dict[str, Any]] = []
    if frames_dir.exists() and frames_dir.is_dir():
        for path in sorted(frames_dir.glob("*.jpg")):
            try:
                frames.append({"name": path.name, "url": _file_url(path, root=root)})
            except ValueError:
                continue
    data["frames"] = frames
    data["frame_count"] = int(data.get("frame_count") or len(frames))
    return data


def list_evidence_events(*, root: str | Path | None = None, limit: int = 50) -> dict[str, Any]:
    base = _resolved_root(root)
    events_by_key: dict[str, dict[str, Any]] = {}
    max_limit = max(1, min(int(limit or 50), 200))
    db_path = _evidence_db_path(base)
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path), timeout=5.0) as conn:
                _ensure_evidence_db(conn)
                rows = conn.execute(
                    "SELECT event_json FROM evidence_events ORDER BY ended_at DESC, updated_at DESC LIMIT ?",
                    (max_limit,),
                ).fetchall()
            for (raw,) in rows:
                try:
                    item = json.loads(raw)
                except Exception:
                    continue
                if isinstance(item, dict):
                    enriched = enrich_evidence_summary(item, root=base)
                    key = str(enriched.get("evidence_event_key") or "")
                    if key:
                        events_by_key[key] = enriched
        except Exception:
            events_by_key = {}
    if base.exists():
        event_paths = sorted(base.glob("*/*/event_*/event.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for event_path in event_paths[: max_limit * 2]:
            try:
                item = json.loads(event_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(item, dict):
                enriched = enrich_evidence_summary(item, root=base)
                key = str(enriched.get("evidence_event_key") or "")
                if key and key not in events_by_key:
                    events_by_key[key] = enriched
                _index_evidence_event(enriched, root=base)
    events = sorted(
        events_by_key.values(),
        key=lambda item: str(item.get("ended_at") or item.get("started_at") or ""),
        reverse=True,
    )[:max_limit]
    return {"root": str(base), "database": str(db_path), "count": len(events), "events": events}


@dataclass(slots=True)
class EventState:
    channel: str
    event_id: int
    event_dir: Path
    started_frame: int
    started_at: str
    last_active_frame: int
    post_remaining: int
    frame_count: int = 0
    write_attempt_count: int = 0
    max_score: float = 0.0
    reasons: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    representative_path: str | None = None
    written_frame_indices: set[int] = field(default_factory=set)
    scheduled_frame_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class BufferedEvidenceFrame:
    frame_idx: int
    frame: np.ndarray
    info: dict[str, Any]
    ppe: dict[str, Any]


@dataclass(slots=True)
class _FrameWriteJob:
    event: EventState
    frame_idx: int
    frame: np.ndarray
    info: dict[str, Any]
    ppe: dict[str, Any]


@dataclass(slots=True)
class _FinalizeEventJob:
    event: EventState
    reason: str
    summary: dict[str, Any]


@dataclass(slots=True)
class _ManifestJob:
    opened: bool


@dataclass(slots=True)
class _BarrierJob:
    done: threading.Event


@dataclass(slots=True)
class _StopWriterJob:
    pass


class EvidenceSession:
    """Persist compact alert evidence outside the Web layer.

    It records one directory per alert event with representative JPEGs and JSON
    summaries. The class is intentionally independent from HTTP and can be used
    by command-line evaluation scripts as well.
    """

    def __init__(
        self,
        *,
        source_type: str,
        source: str,
        profile: str,
        run_id: int = 0,
        source_epoch: int = 0,
        root: str | Path | None = None,
        enabled: bool = True,
        pre_frames: int = 12,
        post_frames: int = 18,
        sample_every: int = 3,
        max_frames_per_event: int = 80,
        clip_fps: int = 6,
        writer_queue_capacity: int = 256,
        writer_enqueue_timeout_s: float = 0.02,
        writer_drain_timeout_s: float = 10.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.source_type = str(source_type)
        self.source = str(source)
        self.profile = str(profile)
        self.run_id = int(run_id)
        self.source_epoch = int(source_epoch)
        self.root = Path(root) if root else default_evidence_root()
        self.pre_frames = max(0, int(pre_frames))
        self.post_frames = max(0, int(post_frames))
        self.sample_every = max(1, int(sample_every))
        self.max_frames_per_event = max(1, int(max_frames_per_event))
        self.clip_fps = max(1, int(clip_fps))
        self.writer_queue_capacity = max(8, int(writer_queue_capacity))
        self.writer_enqueue_timeout_s = max(0.0, float(writer_enqueue_timeout_s))
        self.writer_drain_timeout_s = max(0.1, float(writer_drain_timeout_s))
        self._lock = threading.RLock()
        self._closed = False
        self._close_requested = False
        self._prebuffer: deque[BufferedEvidenceFrame] = deque(maxlen=self.pre_frames)
        self._errors: deque[dict[str, Any]] = deque(maxlen=100)
        self.session_dir: Path | None = None
        self.manifest_path: Path | None = None
        self.events_jsonl: Path | None = None
        self._active: dict[str, EventState] = {}
        self._seq: dict[str, int] = {"module_a": 0, "ppe": 0, "a3b": 0, "source_auth": 0}
        self.saved_events: list[dict[str, Any]] = []
        self._writer_queue: queue.Queue[Any] | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_pending = 0
        self._writer_completed = 0
        self._writer_failed = 0
        self._writer_queue_full = 0
        self._writer_drain_ms = 0.0
        self._writer_last_error = ""
        self._writer_fatal_error = ""
        self._writer_critical_error = ""
        if not self.enabled:
            return
        source_part = safe_path_part(Path(source).stem or source)
        readable_prefix = f"{safe_path_part(source_type)}_{source_part}_{safe_path_part(profile)}"
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(10):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            candidate = self.root / f"{stamp}_{readable_prefix}_{secrets.token_hex(3)}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            self.session_dir = candidate
            break
        else:
            raise RuntimeError(f"无法创建唯一 evidence session 目录: {self.root}")
        self.manifest_path = self.session_dir / "manifest.json"
        self.events_jsonl = self.session_dir / "events.jsonl"
        self._write_manifest(opened=True)
        self._writer_queue = queue.Queue(maxsize=self.writer_queue_capacity)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"evidence-writer-{secrets.token_hex(3)}",
            daemon=True,
        )
        self._writer_thread.start()
        if not self._writer_thread.is_alive():
            raise RuntimeError("evidence writer failed to start")

    @property
    def saved_event_count(self) -> int:
        with self._lock:
            return len(self.saved_events)

    def writer_status(self) -> dict[str, Any]:
        with self._lock:
            thread = self._writer_thread
            return {
                "enabled": bool(self.enabled),
                "alive": bool(thread is not None and thread.is_alive()),
                "queue_capacity": self.writer_queue_capacity if self.enabled else 0,
                "pending": int(self._writer_pending),
                "completed": int(self._writer_completed),
                "failed": int(self._writer_failed),
                "queue_full": int(self._writer_queue_full),
                "drain_ms": round(float(self._writer_drain_ms), 3),
                "last_error": str(self._writer_last_error or self._writer_fatal_error),
            }

    def close(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._closed:
                return []
            if not self.enabled:
                self._closed = True
                return []
            self._raise_if_writer_failed()
            self._close_requested = True
            completed = self._finalize_active(reason="session_close")
            self._prebuffer.clear()
        manifest_queued = self._enqueue_job(
            _ManifestJob(opened=False),
            kind="manifest",
            control=True,
            timeout_s=self.writer_drain_timeout_s,
        )
        if not manifest_queued:
            raise RuntimeError(self.writer_status()["last_error"] or "evidence manifest enqueue failed")
        self._drain_writer(timeout_s=self.writer_drain_timeout_s)
        self._stop_writer(timeout_s=min(1.0, self.writer_drain_timeout_s))
        with self._lock:
            self._closed = True
        return completed

    def finalize_active(self, *, reason: str = "manual_finalize") -> list[dict[str, Any]]:
        """Finalize all active events without closing the session."""
        with self._lock:
            if self._closed or self._close_requested:
                return []
            if not self.enabled:
                return []
            self._raise_if_writer_failed()
            completed = self._finalize_active(reason=reason)
        manifest_queued = self._enqueue_job(
            _ManifestJob(opened=True),
            kind="manifest",
            control=True,
            timeout_s=self.writer_drain_timeout_s,
        )
        if not manifest_queued:
            raise RuntimeError(self.writer_status()["last_error"] or "evidence manifest enqueue failed")
        self._drain_writer(timeout_s=self.writer_drain_timeout_s)
        return completed

    def reset(
        self,
        *,
        reason: str = "session_reset",
        source_epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        """Finalize active events and clear source-scoped buffered frames."""
        with self._lock:
            if self._closed or self._close_requested:
                return []
            if not self.enabled:
                if source_epoch is not None:
                    self.source_epoch = int(source_epoch)
                return []
            self._raise_if_writer_failed()
            completed = self._finalize_active(reason=reason)
            self._prebuffer.clear()
            if source_epoch is not None:
                self.source_epoch = int(source_epoch)
        manifest_queued = self._enqueue_job(
            _ManifestJob(opened=True),
            kind="manifest",
            control=True,
            timeout_s=self.writer_drain_timeout_s,
        )
        if not manifest_queued:
            raise RuntimeError(self.writer_status()["last_error"] or "evidence manifest enqueue failed")
        self._drain_writer(timeout_s=self.writer_drain_timeout_s)
        return completed

    def _finalize_active(self, *, reason: str) -> list[dict[str, Any]]:
        completed = [self._finalize(channel, reason=reason) for channel in list(self._active)]
        return [item for item in completed if item]

    def update(
        self,
        *,
        frame_idx: int,
        frame: np.ndarray,
        info: dict[str, Any],
        ppe: dict[str, Any],
        status: dict[str, Any],
    ) -> list[dict[str, Any]]:
        with self._lock:
            if not self.enabled or self._closed or self._close_requested:
                return []
            self._raise_if_writer_failed()
            completed: list[dict[str, Any]] = []
            has_explicit_physical_confirmation = (
                "physical_alert_confirmed" in status
            )
            physical_confirmed = bool(
                status.get("physical_alert_confirmed")
                if has_explicit_physical_confirmation
                else status.get("alert_confirmed")
            )
            physical_hold_active = bool(
                status.get("physical_attack_state_active")
                or status.get("attack_state_active")
            )
            physical_active = (
                physical_confirmed
                or (
                    "module_a" in self._active
                    and physical_hold_active
                )
            )
            has_explicit_a3b_confirmation = (
                "a3b_confirmed_alert" in status
                or "a3b_state" in status
            )
            a3b_confirmed = bool(
                status.get("a3b_confirmed_alert")
                if "a3b_confirmed_alert" in status
                else (
                    status.get("a3b_triggered")
                    and str(status.get("a3b_state") or "").strip().lower()
                    == "confirmed"
                )
                if has_explicit_a3b_confirmation
                else status.get("a3b_triggered")
            )
            a3b_active = bool(
                a3b_confirmed
                or (
                    "a3b" in self._active
                    and status.get("a3b_triggered")
                )
            )
            channels = {
                # ``module_a`` is the historical physical-attack evidence
                # channel. A confirmed A3b result participates in the public
                # Module A umbrella alert, but must not create a duplicate
                # physical evidence event.
                "module_a": physical_active,
                "ppe": bool(status.get("ppe_event_active") or status.get("ppe_warning") or status.get("ppe_confirmed")),
                "a3b": a3b_active,
            }
            for channel, active in channels.items():
                if active:
                    new_event = channel not in self._active
                    event = self._ensure_event(channel, frame_idx, status)
                    if new_event:
                        self._queue_prebuffer(event)
                    event.last_active_frame = int(frame_idx)
                    event.post_remaining = self.post_frames
                    self._update_event_lineage(event, frame_idx, status)
                    score = self._channel_score(channel, status)
                    event.max_score = max(event.max_score, score)
                    reason = self._channel_reason(channel, status)
                    if reason:
                        event.reasons.add(reason)
                    self._merge_event_metadata(event, self._channel_metadata(channel, status, ppe))
                    if (
                        event.write_attempt_count < self.max_frames_per_event
                        and (frame_idx - event.started_frame) % self.sample_every == 0
                    ):
                        self._queue_event_frame(event, frame_idx, frame, info, ppe)
                elif channel in self._active:
                    event = self._active[channel]
                    event.post_remaining -= 1
                    if event.post_remaining <= 0:
                        completed.append(self._finalize(channel, reason="post_window_done"))
            self._buffer_frame(frame_idx, frame, info, ppe)
            return [item for item in completed if item]

    def _buffer_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        info: dict[str, Any],
        ppe: dict[str, Any],
    ) -> None:
        if self.pre_frames <= 0:
            return
        self._prebuffer.append(
            BufferedEvidenceFrame(
                frame_idx=int(frame_idx),
                frame=frame.copy(),
                info=copy.deepcopy(info),
                ppe=copy.deepcopy(ppe),
            )
        )

    def _queue_prebuffer(self, event: EventState) -> None:
        available = max(0, self.max_frames_per_event - event.write_attempt_count - 1)
        if available <= 0:
            return
        buffered_frames = list(self._prebuffer)[-available:]
        for buffered in buffered_frames:
            if event.write_attempt_count >= self.max_frames_per_event - 1:
                break
            self._queue_event_frame(
                event,
                buffered.frame_idx,
                buffered.frame,
                buffered.info,
                buffered.ppe,
                copy_payload=False,
            )

    def _ensure_event(
        self,
        channel: str,
        frame_idx: int,
        status: dict[str, Any],
    ) -> EventState:
        if channel in self._active:
            return self._active[channel]
        if self.session_dir is None:
            raise RuntimeError("evidence session directory is unavailable")
        self._seq[channel] = self._seq.get(channel, 0) + 1
        event_id = self._seq[channel]
        event_dir = self.session_dir / channel / f"event_{event_id:04d}"
        event = EventState(
            channel=channel,
            event_id=event_id,
            event_dir=event_dir,
            started_frame=int(frame_idx),
            started_at=datetime.now().isoformat(timespec="seconds"),
            last_active_frame=int(frame_idx),
            post_remaining=self.post_frames,
        )
        event.metadata.update(self._event_lineage(frame_idx, status))
        self._active[channel] = event
        return event

    def _event_lineage(self, frame_idx: int, status: dict[str, Any]) -> dict[str, Any]:
        run_id_value = status.get("run_id") if "run_id" in status else self.run_id
        if run_id_value is None:
            run_id_value = self.run_id
        source_epoch_value = (
            status.get("source_epoch")
            if "source_epoch" in status
            else self.source_epoch
        )
        if source_epoch_value is None:
            source_epoch_value = self.source_epoch
        source_time_value = (
            status.get("source_time_s")
            if "source_time_s" in status
            else status.get("video_time_s", 0.0)
        )
        if source_time_value is None:
            source_time_value = 0.0
        source_time_s = float(source_time_value)
        return {
            "run_id": int(run_id_value),
            "source_epoch": int(source_epoch_value),
            "source_frame_start": int(frame_idx),
            "source_frame_end": int(frame_idx),
            "source_time_start_s": source_time_s,
            "source_time_end_s": source_time_s,
        }

    def _update_event_lineage(
        self,
        event: EventState,
        frame_idx: int,
        status: dict[str, Any],
    ) -> None:
        lineage = self._event_lineage(frame_idx, status)
        event.metadata["source_frame_end"] = lineage["source_frame_end"]
        event.metadata["source_time_end_s"] = lineage["source_time_end_s"]
        if int(event.metadata.get("run_id", lineage["run_id"])) != int(lineage["run_id"]):
            event.metadata["lineage_conflict"] = "run_id_changed_inside_event"
        if int(event.metadata.get("source_epoch", lineage["source_epoch"])) != int(
            lineage["source_epoch"]
        ):
            event.metadata["lineage_conflict"] = "source_epoch_changed_inside_event"

    def _queue_event_frame(
        self,
        event: EventState,
        frame_idx: int,
        frame: np.ndarray,
        info: dict[str, Any],
        ppe: dict[str, Any],
        *,
        copy_payload: bool = True,
    ) -> Path | None:
        normalized_frame_idx = int(frame_idx)
        if normalized_frame_idx in event.scheduled_frame_indices:
            return event.event_dir / "frames" / f"frame_{normalized_frame_idx:06d}.jpg"
        out = event.event_dir / "frames" / f"frame_{normalized_frame_idx:06d}.jpg"
        event.write_attempt_count += 1
        job = _FrameWriteJob(
            event=event,
            frame_idx=normalized_frame_idx,
            frame=frame.copy() if copy_payload else frame,
            info=copy.deepcopy(info) if copy_payload else info,
            ppe=copy.deepcopy(ppe) if copy_payload else ppe,
        )
        if not self._enqueue_job(
            job,
            kind="frame",
            control=False,
            record_error=False,
        ):
            self._record_frame_write_error(
                event,
                normalized_frame_idx,
                out,
                self._writer_last_error or "evidence writer queue full",
            )
            return None
        event.scheduled_frame_indices.add(normalized_frame_idx)
        if event.representative_path is None:
            event.representative_path = str(out)
        return out

    def _record_frame_write_error(self, event: EventState, frame_idx: int, out: Path, message: str) -> None:
        with self._lock:
            error = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "channel": event.channel,
                "event_id": event.event_id,
                "frame_idx": int(frame_idx),
                "path": str(out),
                "error": str(message),
            }
            self._errors.append(error)
            event.metadata["evidence_write_failed"] = True
            event.metadata["evidence_write_error_count"] = int(
                event.metadata.get("evidence_write_error_count") or 0
            ) + 1
            event.metadata["evidence_last_write_error"] = str(message)
            event.metadata["evidence_last_write_error_frame"] = int(frame_idx)
        LOGGER.error(
            "evidence frame write failed: channel=%s event=%s frame=%s path=%s error=%s",
            event.channel,
            event.event_id,
            frame_idx,
            out,
            message,
        )

    def _finalize(self, channel: str, *, reason: str) -> dict[str, Any]:
        event = self._active.pop(channel, None)
        if event is None:
            return {}
        summary = self._build_event_summary(event, reason=reason, pending=True)
        queued = self._enqueue_job(
            _FinalizeEventJob(
                event=event,
                reason=str(reason),
                summary=summary,
            ),
            kind="finalize",
            control=True,
        )
        if not queued:
            self._mark_summary_persistence_failure(
                summary,
                [self._writer_last_error or "finalize enqueue failed"],
            )
            self.saved_events.append(summary)
        return summary

    def _build_event_summary(
        self,
        event: EventState,
        *,
        reason: str,
        pending: bool,
        frame_paths: list[Path] | None = None,
        clip_path: Path | None = None,
    ) -> dict[str, Any]:
        peak_score = round(float(event.max_score), 6)
        reasons = sorted(event.reasons)
        frames_dir = event.event_dir / "frames"
        if frame_paths is None:
            frame_count = max(event.frame_count, len(event.scheduled_frame_indices))
            representative_path = event.representative_path
            if representative_path is None and event.scheduled_frame_indices:
                first_idx = min(event.scheduled_frame_indices)
                representative_path = str(
                    frames_dir / f"frame_{first_idx:06d}.jpg"
                )
            effective_clip_path = (
                event.event_dir / "clip.mp4"
                if frame_count > 0
                else None
            )
        else:
            frame_count = len(frame_paths)
            representative_path = event.representative_path
            if representative_path and Path(representative_path) not in frame_paths:
                representative_path = None
            if representative_path is None and frame_paths:
                representative_path = str(frame_paths[0])
            effective_clip_path = clip_path
        write_error_count = int(event.metadata.get("evidence_write_error_count") or 0)
        has_saved_frames = frame_count > 0
        evidence_saved = has_saved_frames and write_error_count == 0
        persistence_status = (
            "pending"
            if pending
            else (
                "complete"
                if evidence_saved
                else ("partial" if has_saved_frames else "failed")
            )
        )
        summary = {
            "channel": event.channel,
            "event_id": event.event_id,
            "source_type": self.source_type,
            "source": self.source,
            "profile": self.profile,
            "session_dir": str(self.session_dir or ""),
            "event_dir": str(event.event_dir),
            "started_at": event.started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "started_frame": event.started_frame,
            "last_active_frame": event.last_active_frame,
            "frame_count": frame_count,
            "max_score": peak_score,
            "peak_score": peak_score,
            "peak_p_adv": peak_score if event.channel == "module_a" else 0.0,
            "peak_a3b_score": peak_score if event.channel == "a3b" else 0.0,
            "reasons": reasons,
            "reason": ";".join(reasons),
            "representative_path": representative_path,
            "trigger_frame": event.started_frame,
            "last_alert_frame": event.last_active_frame,
            "last_warning_frame": event.last_active_frame,
            "evidence_saved": evidence_saved,
            "evidence_saved_frame_count": frame_count,
            "evidence_write_attempt_count": event.write_attempt_count,
            "evidence_has_saved_frames": has_saved_frames,
            "evidence_complete": evidence_saved and not pending,
            "evidence_partial": has_saved_frames and write_error_count > 0,
            "evidence_failed": not has_saved_frames and not pending,
            "evidence_write_error_count": write_error_count,
            "evidence_frames_dir": str(frames_dir),
            "evidence_representative_path": representative_path or "",
            "evidence_representative_url": "",
            "evidence_clip_path": str(effective_clip_path) if effective_clip_path else "",
            "evidence_clip_url": "",
            "evidence_persistence_status": persistence_status,
            "evidence_persistence_pending": bool(pending),
            "close_reason": reason,
        }
        summary.update(event.metadata)
        if pending:
            if effective_clip_path:
                summary.update(
                    {
                        "evidence_clip_status": "pending",
                        "evidence_clip_codec": "h264",
                        "evidence_clip_fps": self.clip_fps,
                        "evidence_clip_browser_playable": True,
                        "evidence_clip_source_frames": frame_count,
                    }
                )
            else:
                summary.update(
                    {
                        "evidence_clip_status": "no_frames",
                        "evidence_clip_browser_playable": False,
                    }
                )
        else:
            summary.setdefault(
                "evidence_clip_status",
                "no_frames" if not has_saved_frames else "unavailable",
            )
            summary.setdefault("evidence_clip_codec", "")
            summary.setdefault("evidence_clip_fps", self.clip_fps)
            summary.setdefault("evidence_clip_browser_playable", False)
            summary.setdefault("evidence_clip_source_frames", frame_count)
        summary = enrich_evidence_summary(summary, root=self.root)
        return summary

    def _enqueue_job(
        self,
        job: Any,
        *,
        kind: str,
        control: bool,
        record_error: bool = True,
        timeout_s: float | None = None,
    ) -> bool:
        writer_queue = self._writer_queue
        if writer_queue is None:
            self._record_writer_error(f"evidence_writer_unavailable:{kind}")
            return False
        timeout = (
            self.writer_enqueue_timeout_s
            if timeout_s is None
            else max(0.0, float(timeout_s))
        )
        deadline = time.monotonic() + timeout
        reserve = min(4, max(1, self.writer_queue_capacity // 8))
        frame_limit = max(1, self.writer_queue_capacity - reserve)
        saw_pressure = False
        while True:
            with self._lock:
                if self._writer_fatal_error:
                    self._writer_last_error = self._writer_fatal_error
                    return False
            frame_capacity_reached = (
                not control and writer_queue.qsize() >= frame_limit
            )
            if not frame_capacity_reached:
                try:
                    writer_queue.put_nowait(job)
                except queue.Full:
                    saw_pressure = True
                else:
                    with self._lock:
                        if saw_pressure:
                            self._writer_queue_full += 1
                        self._writer_pending += 1
                    return True
            else:
                saw_pressure = True
            if time.monotonic() >= deadline:
                message = f"evidence_writer_queue_full:{kind}"
                with self._lock:
                    self._writer_queue_full += 1
                    self._writer_failed += 1
                    self._writer_last_error = message
                if record_error:
                    self._record_writer_error(message, count_failure=False)
                return False
            time.sleep(min(0.002, max(0.0, deadline - time.monotonic())))

    def _writer_loop(self) -> None:
        writer_queue = self._writer_queue
        if writer_queue is None:
            return
        try:
            while True:
                job = writer_queue.get()
                kind = self._writer_job_kind(job)
                success = False
                stop = isinstance(job, _StopWriterJob)
                try:
                    success = self._process_writer_job(job)
                except Exception as exc:
                    message = (
                        "evidence_writer_job_failed:"
                        f"{kind}:{type(exc).__name__}:{exc}"
                    )
                    if isinstance(job, _FinalizeEventJob):
                        self._mark_summary_persistence_failure(
                            job.summary,
                            [message],
                        )
                        with self._lock:
                            if not any(
                                item is job.summary
                                for item in self.saved_events
                            ):
                                self.saved_events.append(job.summary)
                    elif isinstance(
                        job,
                        (_ManifestJob, _BarrierJob, _StopWriterJob),
                    ):
                        with self._lock:
                            self._writer_critical_error = message
                    self._record_writer_error(message, count_failure=False)
                    LOGGER.exception("evidence writer job failed: kind=%s", kind)
                finally:
                    with self._lock:
                        self._writer_pending = max(0, self._writer_pending - 1)
                        if success:
                            self._writer_completed += 1
                        else:
                            self._writer_failed += 1
                    writer_queue.task_done()
                if stop:
                    return
        except BaseException as exc:
            message = (
                f"evidence_writer_fatal:{type(exc).__name__}:{exc}"
            )
            with self._lock:
                self._writer_fatal_error = message
                self._writer_last_error = message
            LOGGER.exception("evidence writer terminated unexpectedly")

    @staticmethod
    def _writer_job_kind(job: Any) -> str:
        if isinstance(job, _FrameWriteJob):
            return "frame"
        if isinstance(job, _FinalizeEventJob):
            return "finalize"
        if isinstance(job, _ManifestJob):
            return "manifest"
        if isinstance(job, _BarrierJob):
            return "barrier"
        if isinstance(job, _StopWriterJob):
            return "stop"
        return type(job).__name__

    def _process_writer_job(self, job: Any) -> bool:
        if isinstance(job, _FrameWriteJob):
            return self._persist_frame_job(job)
        if isinstance(job, _FinalizeEventJob):
            return self._persist_finalize_job(job)
        if isinstance(job, _ManifestJob):
            self._write_manifest(opened=job.opened)
            return True
        if isinstance(job, _BarrierJob):
            job.done.set()
            return True
        if isinstance(job, _StopWriterJob):
            return True
        raise TypeError(f"unsupported evidence writer job: {type(job).__name__}")

    def _persist_frame_job(self, job: _FrameWriteJob) -> bool:
        event = job.event
        out = event.event_dir / "frames" / f"frame_{job.frame_idx:06d}.jpg"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            rendered = job.frame.copy()
            if event.channel == "module_a":
                rendered = draw_hud(
                    rendered,
                    job.info,
                    job.frame_idx,
                    effective=True,
                )
            elif event.channel == "ppe":
                rendered = draw_ppe_hud(rendered, job.ppe)
            written = bool(
                cv2.imwrite(
                    str(out),
                    rendered,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                )
            )
            exists = out.is_file() and out.stat().st_size > 0
        except Exception as exc:
            self._record_frame_write_error(
                event,
                job.frame_idx,
                out,
                f"{type(exc).__name__}: {exc}",
            )
            return False
        if not written or not exists:
            try:
                out.unlink(missing_ok=True)
            except OSError:
                pass
            self._record_frame_write_error(
                event,
                job.frame_idx,
                out,
                f"cv2.imwrite returned {written}; file_exists={out.is_file()}",
            )
            return False
        with self._lock:
            event.written_frame_indices.add(job.frame_idx)
            event.frame_count += 1
        return True

    def _persist_finalize_job(self, job: _FinalizeEventJob) -> bool:
        event = job.event
        frames_dir = event.event_dir / "frames"
        frame_paths = [
            path
            for path in sorted(frames_dir.glob("*.jpg"))
            if path.is_file() and path.stat().st_size > 0
        ]
        event.frame_count = len(frame_paths)
        if (
            event.representative_path
            and Path(event.representative_path) not in frame_paths
        ):
            event.representative_path = None
        if event.representative_path is None and frame_paths:
            event.representative_path = str(frame_paths[0])
        clip_path = self._write_event_clip(event)
        final_summary = self._build_event_summary(
            event,
            reason=job.reason,
            pending=False,
            frame_paths=frame_paths,
            clip_path=clip_path,
        )
        failures: list[str] = []
        try:
            write_json(event.event_dir / "event.json", final_summary)
        except Exception as exc:
            failures.append(f"event_json:{type(exc).__name__}:{exc}")
        if self.events_jsonl is not None:
            try:
                append_jsonl(self.events_jsonl, final_summary)
            except Exception as exc:
                failures.append(f"events_jsonl:{type(exc).__name__}:{exc}")
        try:
            indexed = _index_evidence_event(
                final_summary,
                root=self.root,
                raise_errors=True,
            )
            if not indexed:
                failures.append("sqlite_catalog:index_rejected")
        except Exception as exc:
            failures.append(f"sqlite_catalog:{type(exc).__name__}:{exc}")
        if failures:
            self._mark_summary_persistence_failure(final_summary, failures)
            try:
                write_json(event.event_dir / "event.json", final_summary)
            except Exception:
                pass
        with self._lock:
            job.summary.update(final_summary)
            self.saved_events.append(job.summary)
        try:
            self._write_manifest(opened=True)
        except Exception as exc:
            failure = f"manifest:{type(exc).__name__}:{exc}"
            self._mark_summary_persistence_failure(job.summary, [failure])
            self._record_writer_error(failure, count_failure=False)
            try:
                write_json(event.event_dir / "event.json", job.summary)
            except Exception:
                pass
            return False
        if failures:
            self._record_writer_error(
                ";".join(failures),
                count_failure=False,
            )
            return False
        return True

    @staticmethod
    def _mark_summary_persistence_failure(
        summary: dict[str, Any],
        failures: list[str],
    ) -> None:
        has_frames = bool(
            summary.get("evidence_has_saved_frames")
            or summary.get("evidence_saved_frame_count")
        )
        summary.update(
            {
                "evidence_saved": False,
                "evidence_complete": False,
                "evidence_partial": has_frames,
                "evidence_failed": not has_frames,
                "evidence_persistence_status": (
                    "partial" if has_frames else "failed"
                ),
                "evidence_persistence_pending": False,
                "evidence_persistence_error": ";".join(
                    str(item) for item in failures if item
                ),
            }
        )

    def _record_writer_error(
        self,
        message: str,
        *,
        count_failure: bool = False,
    ) -> None:
        with self._lock:
            self._writer_last_error = str(message)
            if count_failure:
                self._writer_failed += 1
            self._errors.append(
                {
                    "timestamp": datetime.now().isoformat(
                        timespec="milliseconds"
                    ),
                    "job": "writer",
                    "error": str(message),
                }
            )
        LOGGER.error("evidence writer error: %s", message)

    def _raise_if_writer_failed(self) -> None:
        if self._writer_fatal_error:
            raise RuntimeError(self._writer_fatal_error)
        if self._writer_critical_error:
            raise RuntimeError(self._writer_critical_error)
        thread = self._writer_thread
        if self.enabled and thread is not None and not thread.is_alive():
            message = self._writer_last_error or "evidence_writer_not_alive"
            raise RuntimeError(message)

    def _drain_writer(self, *, timeout_s: float) -> None:
        started = time.perf_counter()
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        barrier = threading.Event()
        with self._lock:
            self._raise_if_writer_failed()
        remaining = max(0.0, deadline - time.monotonic())
        if not self._enqueue_job(
            _BarrierJob(done=barrier),
            kind="barrier",
            control=True,
            timeout_s=remaining,
        ):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._writer_drain_ms = elapsed_ms
            raise TimeoutError(
                self.writer_status()["last_error"]
                or "evidence writer barrier enqueue timed out"
            )
        remaining = max(0.0, deadline - time.monotonic())
        if not barrier.wait(remaining):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            message = (
                f"evidence_writer_drain_timeout:{elapsed_ms:.3f}ms"
            )
            with self._lock:
                self._writer_drain_ms = elapsed_ms
                self._writer_last_error = message
            raise TimeoutError(message)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._writer_drain_ms = elapsed_ms
            self._raise_if_writer_failed()

    def _stop_writer(self, *, timeout_s: float) -> None:
        thread = self._writer_thread
        if thread is None or not thread.is_alive():
            return
        if not self._enqueue_job(
            _StopWriterJob(),
            kind="stop",
            control=True,
            timeout_s=max(0.1, float(timeout_s)),
        ):
            raise RuntimeError(
                self.writer_status()["last_error"]
                or "evidence writer stop enqueue failed"
            )
        thread.join(max(0.1, float(timeout_s)))
        if thread.is_alive():
            message = "evidence_writer_stop_timeout"
            with self._lock:
                self._writer_last_error = message
            raise TimeoutError(message)

    def _write_event_clip(self, event: EventState) -> Path | None:
        frames_dir = event.event_dir / "frames"
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            event.metadata.update({"evidence_clip_status": "no_frames", "evidence_clip_browser_playable": False})
            return None
        clip_path, clip_meta = _write_browser_mp4_from_frames(event.event_dir, frame_paths, fps=self.clip_fps)
        event.metadata.update(clip_meta)
        return clip_path

    def _write_manifest(self, *, opened: bool) -> None:
        if (
            not self.enabled
            or self.session_dir is None
            or self.manifest_path is None
            or self.events_jsonl is None
        ):
            return
        with self._lock:
            payload = {
                "opened": bool(opened),
                "source_type": self.source_type,
                "source": self.source,
                "profile": self.profile,
                "session_dir": str(self.session_dir),
                "events_jsonl": str(self.events_jsonl),
                "saved_event_count": len(self.saved_events),
                "active_event_count": len(self._active),
                "prebuffer_frame_count": len(self._prebuffer),
                "evidence_error_count": len(self._errors),
                "recent_errors": list(self._errors)[-20:],
                "events": copy.deepcopy(self.saved_events[-50:]),
                "writer": self.writer_status(),
            }
        write_json(self.manifest_path, payload)

    @staticmethod
    def _channel_score(channel: str, status: dict[str, Any]) -> float:
        if channel == "module_a":
            return float(status.get("p_adv") or 0.0)
        if channel == "ppe":
            return float(
                status.get("ppe_window_positive")
                or status.get("ppe_fast_window_positive")
                or status.get("ppe_missing_helmet_count")
                or 0.0
            )
        if channel == "a3b":
            return float(
                status.get("a3b_event_score")
                or status.get("a3b_confidence")
                or status.get("a3b_confirmed_score")
                or status.get("a3b_observed_score")
                or 0.0
            )
        return 0.0

    @staticmethod
    def _channel_reason(channel: str, status: dict[str, Any]) -> str:
        if channel == "module_a":
            return str(status.get("reason") or "")
        if channel == "ppe":
            return str(status.get("ppe_reason") or status.get("ppe_event_last_reason") or "")
        if channel == "a3b":
            return str(status.get("a3b_triggered_source") or "")
        return ""

    @staticmethod
    def _channel_metadata(channel: str, status: dict[str, Any], ppe: dict[str, Any]) -> dict[str, Any]:
        if channel != "ppe":
            return {}
        person_count = _int(status.get("ppe_person_count", ppe.get("person_count")))
        raw_person_count = _int(status.get("ppe_raw_person_count", ppe.get("raw_person_count", person_count)))
        inferred_person_count = _int(
            status.get("ppe_inferred_person_count", ppe.get("inferred_person_count", person_count))
        )
        head_count = _int(status.get("ppe_head_count", ppe.get("head_count")))
        helmet_count = _int(status.get("ppe_helmet_count", ppe.get("helmet_count")))
        missing_helmet_count = _int(status.get("ppe_missing_helmet_count", ppe.get("missing_helmet_count")))
        reason = str(status.get("ppe_reason") or status.get("ppe_event_last_reason") or ppe.get("reason") or "")
        return {
            "person_count": person_count,
            "raw_person_count": raw_person_count,
            "inferred_person_count": inferred_person_count,
            "head_count": head_count,
            "helmet_count": helmet_count,
            "missing_helmet_count": missing_helmet_count,
            "ppe_person_count": person_count,
            "ppe_raw_person_count": raw_person_count,
            "ppe_inferred_person_count": inferred_person_count,
            "ppe_head_count": head_count,
            "ppe_helmet_count": helmet_count,
            "ppe_missing_helmet_count": missing_helmet_count,
            "ppe_confirmed_source": str(status.get("ppe_confirmed_source") or ppe.get("confirmed_source") or ""),
            "ppe_event_last_reason": reason,
            "ppe_event_last_confirmed_source": str(
                status.get("ppe_event_last_confirmed_source") or ppe.get("event_last_confirmed_source") or ""
            ),
        }

    @staticmethod
    def _merge_event_metadata(event: EventState, metadata: dict[str, Any]) -> None:
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str):
                if value:
                    event.metadata[key] = value
                continue
            if key.endswith("_count") or key in {"person_count", "raw_person_count", "inferred_person_count", "head_count", "helmet_count", "missing_helmet_count"}:
                current = _int(event.metadata.get(key))
                incoming = _int(value)
                if key not in event.metadata or incoming > current:
                    event.metadata[key] = incoming
                continue
            event.metadata[key] = value


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
