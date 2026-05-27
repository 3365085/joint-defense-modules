from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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
        summary = {
            "channel": event.channel,
            "event_id": event.event_id,
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
            "close_reason": reason,
        }
        summary.update(event.metadata)
        write_json(event.event_dir / "event.json", summary)
        append_jsonl(self.events_jsonl, summary)
        self.saved_events.append(summary)
        self._write_manifest(opened=True)
        return summary

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
