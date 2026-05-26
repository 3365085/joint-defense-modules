from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import EvidenceEvent


def load_evidence_events(path: str | Path | None) -> list[EvidenceEvent]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        if "events" in data:
            rows = data.get("events")
        elif "rows" in data:
            rows = data.get("rows")
        else:
            rows = [data]
    else:
        rows = data
    if not isinstance(rows, Sequence):
        return []
    return [EvidenceEvent.from_mapping(row) for row in rows if isinstance(row, Mapping)]


def _parse_timestamp(value: str | None) -> datetime | None:
    """Best-effort ISO-8601 timestamp parsing tolerant of trailing ``Z``."""

    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    # ``datetime.fromisoformat`` accepts ``+00:00`` since Python 3.11; older
    # interpreters reject ``Z`` so we normalize first.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _detect_event_burst(
    events: Sequence[EvidenceEvent],
    *,
    burst_window_minutes: int,
    burst_threshold: int,
) -> tuple[bool, list[dict[str, Any]]]:
    """Return ``(triggered, hits)`` based on a sliding time window per camera/risk.

    Earlier the burst test counted lifetime events per bucket which meant
    accumulated events from days ago could permanently keep deep-checks on.
    Now we slide an explicit ``burst_window_minutes`` window over event
    timestamps; missing timestamps fall back to the original lifetime count
    so we never silently miss bursts when the runtime upstream forgets to
    send timestamps.
    """

    if burst_threshold <= 0:
        return False, []
    window_minutes = max(0, int(burst_window_minutes))
    triggered = False
    hits: list[dict[str, Any]] = []

    timestamped: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    untimestamped_counts: Counter[tuple[str, str]] = Counter()
    for event in events:
        ts = _parse_timestamp(event.timestamp)
        for axis, value in (("camera", event.camera_id or "unknown"), ("risk", event.suspected_risk or "unknown")):
            key = (axis, value)
            if ts is None:
                untimestamped_counts[key] += 1
            else:
                timestamped[key].append(ts)

    for key, stamps in timestamped.items():
        stamps.sort()
        if window_minutes <= 0:
            if len(stamps) >= burst_threshold:
                hits.append({"axis": key[0], "value": key[1], "count": len(stamps), "window_minutes": 0})
                triggered = True
            continue
        # Sliding window over sorted timestamps.
        left = 0
        max_in_window = 0
        for right, ts_right in enumerate(stamps):
            while (ts_right - stamps[left]).total_seconds() > window_minutes * 60:
                left += 1
            count = right - left + 1
            if count > max_in_window:
                max_in_window = count
            if count >= burst_threshold:
                hits.append({"axis": key[0], "value": key[1], "count": count, "window_minutes": window_minutes})
                triggered = True
                break

    for key, count in untimestamped_counts.items():
        if count >= burst_threshold:
            hits.append({"axis": key[0], "value": key[1], "count": count, "window_minutes": None, "no_timestamp": True})
            triggered = True
    return triggered, hits


def summarize_evidence_events(events: Sequence[EvidenceEvent], *, burst_window: int = 5, burst_threshold: int = 3) -> dict[str, Any]:
    by_camera = Counter(e.camera_id or "unknown" for e in events)
    by_risk = Counter(e.suspected_risk or "unknown" for e in events)
    by_model = Counter(e.model_id or "unknown" for e in events)
    scores: dict[str, list[float]] = defaultdict(list)
    for event in events:
        for k, v in event.module_a_scores.items():
            scores[k].append(float(v))
    score_summary = {
        k: {"count": len(v), "mean": sum(v) / len(v), "max": max(v)}
        for k, v in scores.items()
        if v
    }
    triggered, hits = _detect_event_burst(
        events,
        burst_window_minutes=burst_window,
        burst_threshold=burst_threshold,
    )
    return {
        "n_events": len(events),
        "by_camera": dict(by_camera),
        "by_risk": dict(by_risk),
        "by_model": dict(by_model),
        "score_summary": score_summary,
        "event_trigger_deep_check": bool(triggered),
        "burst_hits": hits,
        "burst_window_minutes": int(burst_window),
        "burst_threshold": int(burst_threshold),
        # ``burst_window`` is preserved for backwards compatibility.
        "burst_window": int(burst_window),
    }


def evidence_to_hard_negative_manifest(events: Sequence[EvidenceEvent], out_path: str | Path) -> Path:
    """Write a simple manifest consumable by dataset-building scripts.

    The manifest intentionally stores references, not copied images.  A later
    dataset builder decides whether these events are tuning, validation, or
    held-out material.
    """

    rows = []
    for event in events:
        if event.frame_path:
            rows.append(
                {
                    "event_id": event.event_id,
                    "image_path": event.frame_path,
                    "clip_path": event.clip_path,
                    "camera_id": event.camera_id,
                    "timestamp": event.timestamp,
                    "model_id": event.model_id,
                    "risk": event.suspected_risk,
                    "target_boxes": event.target_boxes,
                    "source": "module_a_evidence_event",
                    "recommended_use": "quarantine_until_split_assignment",
                }
            )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "n_rows": len(rows)}, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
