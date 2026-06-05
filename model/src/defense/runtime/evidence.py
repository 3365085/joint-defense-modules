from __future__ import annotations

import json
import os
import re
import base64
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.runtime.catalog import register_artifact
from defense.visualization import draw_hud, draw_ppe_hud


DEFAULT_EVIDENCE_ROOT = Path(__file__).resolve().parents[3] / "runtime" / "evidence" / "monitor"


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


def _index_evidence_event(summary: dict[str, Any], *, root: str | Path | None = None) -> None:
    try:
        event_dir = Path(str(summary.get("event_dir") or "")).resolve()
        if not event_dir:
            return
        base = _resolved_root(root)
        if not _is_under(event_dir, base):
            return
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
    except Exception:
        return


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
    max_score: float = 0.0
    reasons: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    representative_path: str | None = None


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
        root: str | Path | None = None,
        enabled: bool = True,
        pre_frames: int = 12,
        post_frames: int = 18,
        sample_every: int = 3,
        max_frames_per_event: int = 80,
        clip_fps: int = 6,
    ) -> None:
        self.enabled = bool(enabled)
        self.source_type = str(source_type)
        self.source = str(source)
        self.profile = str(profile)
        self.root = Path(root) if root else default_evidence_root()
        self.pre_frames = max(0, int(pre_frames))
        self.post_frames = max(0, int(post_frames))
        self.sample_every = max(1, int(sample_every))
        self.max_frames_per_event = max(1, int(max_frames_per_event))
        self.clip_fps = max(1, int(clip_fps))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.root / f"{stamp}_{safe_path_part(source_type)}_{safe_path_part(Path(source).stem or source)}_{safe_path_part(profile)}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.session_dir / "manifest.json"
        self.events_jsonl = self.session_dir / "events.jsonl"
        self._active: dict[str, EventState] = {}
        self._seq: dict[str, int] = {"module_a": 0, "ppe": 0, "source_auth": 0}
        self.saved_events: list[dict[str, Any]] = []
        self._write_manifest(opened=True)

    @property
    def saved_event_count(self) -> int:
        return len(self.saved_events)

    def close(self) -> list[dict[str, Any]]:
        completed = []
        for channel in list(self._active.keys()):
            completed.append(self._finalize(channel, reason="session_close"))
        self._write_manifest(opened=False)
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
        if not self.enabled:
            return []
        completed: list[dict[str, Any]] = []
        channels = {
            "module_a": bool(status.get("alert_confirmed") or status.get("attack_state_active")),
            "ppe": bool(status.get("ppe_event_active") or status.get("ppe_warning") or status.get("ppe_confirmed")),
            "a3b": bool(status.get("a3b_triggered")),
        }
        for channel, active in channels.items():
            if active:
                event = self._ensure_event(channel, frame_idx)
                event.last_active_frame = int(frame_idx)
                event.post_remaining = self.post_frames
                score = self._channel_score(channel, status)
                event.max_score = max(event.max_score, score)
                reason = self._channel_reason(channel, status)
                if reason:
                    event.reasons.add(reason)
                self._merge_event_metadata(event, self._channel_metadata(channel, status, ppe))
                if event.frame_count < self.max_frames_per_event and (frame_idx - event.started_frame) % self.sample_every == 0:
                    saved = self._write_event_frame(event, frame_idx, frame, info, ppe)
                    if event.representative_path is None:
                        event.representative_path = str(saved)
            elif channel in self._active:
                event = self._active[channel]
                event.post_remaining -= 1
                if event.post_remaining <= 0:
                    completed.append(self._finalize(channel, reason="post_window_done"))
        return [item for item in completed if item]

    def _ensure_event(self, channel: str, frame_idx: int) -> EventState:
        if channel in self._active:
            return self._active[channel]
        self._seq[channel] = self._seq.get(channel, 0) + 1
        event_id = self._seq[channel]
        event_dir = self.session_dir / channel / f"event_{event_id:04d}"
        (event_dir / "frames").mkdir(parents=True, exist_ok=True)
        event = EventState(
            channel=channel,
            event_id=event_id,
            event_dir=event_dir,
            started_frame=int(frame_idx),
            started_at=datetime.now().isoformat(timespec="seconds"),
            last_active_frame=int(frame_idx),
            post_remaining=self.post_frames,
        )
        self._active[channel] = event
        return event

    def _write_event_frame(
        self,
        event: EventState,
        frame_idx: int,
        frame: np.ndarray,
        info: dict[str, Any],
        ppe: dict[str, Any],
    ) -> Path:
        rendered = frame.copy()
        if event.channel == "module_a":
            rendered = draw_hud(rendered, info, frame_idx, effective=True)
        elif event.channel == "ppe":
            rendered = draw_ppe_hud(rendered, ppe)
        out = event.event_dir / "frames" / f"frame_{int(frame_idx):06d}.jpg"
        cv2.imwrite(str(out), rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        event.frame_count += 1
        return out

    def _finalize(self, channel: str, *, reason: str) -> dict[str, Any]:
        event = self._active.pop(channel, None)
        if event is None:
            return {}
        peak_score = round(float(event.max_score), 6)
        reasons = sorted(event.reasons)
        frames_dir = event.event_dir / "frames"
        clip_path = self._write_event_clip(event)
        summary = {
            "channel": event.channel,
            "event_id": event.event_id,
            "source_type": self.source_type,
            "source": self.source,
            "profile": self.profile,
            "session_dir": str(self.session_dir),
            "event_dir": str(event.event_dir),
            "started_at": event.started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "started_frame": event.started_frame,
            "last_active_frame": event.last_active_frame,
            "frame_count": event.frame_count,
            "max_score": peak_score,
            "peak_score": peak_score,
            "peak_p_adv": peak_score if event.channel == "module_a" else 0.0,
            "peak_a3b_score": peak_score if event.channel == "a3b" else 0.0,
            "reasons": reasons,
            "reason": ";".join(reasons),
            "representative_path": event.representative_path,
            "trigger_frame": event.started_frame,
            "last_alert_frame": event.last_active_frame,
            "last_warning_frame": event.last_active_frame,
            "evidence_saved": True,
            "evidence_saved_frame_count": event.frame_count,
            "evidence_frames_dir": str(frames_dir),
            "evidence_representative_path": event.representative_path or "",
            "evidence_clip_path": str(clip_path) if clip_path else "",
            "close_reason": reason,
        }
        summary.update(event.metadata)
        summary = enrich_evidence_summary(summary, root=self.root)
        write_json(event.event_dir / "event.json", summary)
        append_jsonl(self.events_jsonl, summary)
        _index_evidence_event(summary, root=self.root)
        self.saved_events.append(summary)
        self._write_manifest(opened=True)
        return summary

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
        write_json(
            self.manifest_path,
            {
                "opened": bool(opened),
                "source_type": self.source_type,
                "source": self.source,
                "profile": self.profile,
                "session_dir": str(self.session_dir),
                "events_jsonl": str(self.events_jsonl),
                "saved_event_count": len(self.saved_events),
                "events": self.saved_events[-50:],
            },
        )

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
