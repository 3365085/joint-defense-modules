from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clone_track(track: dict[str, Any]) -> dict[str, Any]:
    clone = dict(track)
    clone["box"] = list(track.get("box") or [])
    return clone


def _can_hold_track(track: dict[str, Any]) -> bool:
    return bool(track.get("hold_eligible", True))


def _lerp(a: Any, b: Any, ratio: float) -> float:
    return _num(a) + (_num(b) - _num(a)) * ratio


def interpolate_overlay(
    prev: dict[str, Any],
    next_item: dict[str, Any],
    video_time_s: float,
    *,
    keep_unmatched_tracks: bool = True,
) -> dict[str, Any] | None:
    left = _num(prev.get("video_time_s"), float("nan"))
    right = _num(next_item.get("video_time_s"), float("nan"))
    if right <= left:
        return None
    ratio = max(0.0, min(1.0, (_num(video_time_s) - left) / (right - left)))
    next_by_id = {str(track.get("track_id", track.get("id", ""))): track for track in next_item.get("ppe_tracks", [])}
    tracks: list[dict[str, Any]] = []
    for track in prev.get("ppe_tracks", []):
        key = str(track.get("track_id", track.get("id", "")))
        later = next_by_id.get(key)
        if later is None:
            if not keep_unmatched_tracks or not _can_hold_track(track):
                continue
        clone = _clone_track(track)
        if later and len(track.get("box") or []) >= 4 and len(later.get("box") or []) >= 4:
            clone["box"] = [_lerp(track["box"][idx], later["box"][idx], ratio) for idx in range(4)]
            clone["confidence"] = _lerp(track.get("confidence"), later.get("confidence"), ratio)
            clone["misses"] = round(_lerp(track.get("misses"), later.get("misses"), ratio))
            clone.setdefault("source", "tracked")
        tracks.append(clone)
    out = dict(prev)
    out.update({"video_time_s": float(video_time_s), "ppe_tracks": tracks, "interpolated": True})
    return out


@dataclass
class OverlayTimeline:
    max_items: int = 600
    items: list[dict[str, Any]] = field(default_factory=list)
    last_held: dict[str, Any] | None = None

    def clear(self) -> None:
        self.items.clear()
        self.last_held = None

    def push(self, record: dict[str, Any]) -> None:
        video_time_s = _num(record.get("video_time_s"), float("nan"))
        if video_time_s != video_time_s:
            return
        self.items.append(dict(record))
        self.items.sort(key=lambda item: _num(item.get("video_time_s")))
        if len(self.items) > self.max_items:
            del self.items[: len(self.items) - self.max_items]

    def find_nearest(self, video_time_s: float, window_s: float) -> dict[str, Any] | None:
        if not self.items:
            return None
        best = min(self.items, key=lambda item: abs(_num(item.get("video_time_s")) - video_time_s))
        if abs(_num(best.get("video_time_s")) - video_time_s) > window_s:
            return None
        self.last_held = best
        return best

    def find_bracket(self, video_time_s: float, max_gap_s: float) -> tuple[dict[str, Any], dict[str, Any]] | None:
        prev = None
        next_item = None
        for item in self.items:
            item_time = _num(item.get("video_time_s"))
            if item_time <= video_time_s:
                prev = item
            elif next_item is None:
                next_item = item
                break
        if prev is None or next_item is None:
            return None
        if _num(next_item.get("video_time_s")) - _num(prev.get("video_time_s")) > max_gap_s:
            return None
        return prev, next_item

    def held_overlay_if_fresh(self, video_time_s: float, hold_s: float) -> dict[str, Any] | None:
        if self.last_held is None:
            return None
        dt = video_time_s - _num(self.last_held.get("video_time_s"))
        if dt < 0.0 or dt > hold_s:
            return None
        out = dict(self.last_held)
        out["video_time_s"] = float(video_time_s)
        out["held"] = True
        out["ppe_tracks"] = [
            _clone_track(track) | {"source": "held"}
            for track in self.last_held.get("ppe_tracks", [])
            if _can_hold_track(track)
        ]
        return out

    def select(
        self,
        video_time_s: float,
        *,
        match_window_s: float = 0.18,
        interpolate_s: float = 0.4,
        hold_s: float = 0.55,
        max_age_s: float = 0.95,
        keep_unmatched_tracks: bool = True,
    ) -> dict[str, Any] | None:
        # Prefer interpolation while the display clock is between two detector
        # records.  Nearest-first selection makes boxes jump/drag at detector FPS
        # because the same older record is held until a newer one becomes closer.
        bracket = self.find_bracket(video_time_s, interpolate_s)
        if bracket is not None:
            mixed = interpolate_overlay(
                bracket[0],
                bracket[1],
                video_time_s,
                keep_unmatched_tracks=keep_unmatched_tracks,
            )
            if mixed is not None:
                self.last_held = mixed
                return mixed
        nearest = self.find_nearest(video_time_s, match_window_s)
        if nearest is not None:
            return nearest
        return self.held_overlay_if_fresh(video_time_s, min(hold_s, max_age_s))
