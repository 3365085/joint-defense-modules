from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from defense.module_a.ppe_postprocess import summarize_ppe_from_detections
from defense.module_a.postprocess import PPEDisplayTracker

from .ppe_state import SafetyHelmetState


@dataclass(slots=True)
class PPEBusinessResult:
    """Business-level PPE interpretation for a processed frame."""

    ppe: dict[str, Any]
    tracks: list[dict[str, Any]]


def evaluate_ppe_business(
    detections: Any,
    *,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    ppe_state: SafetyHelmetState,
    ppe_tracker: PPEDisplayTracker,
    tracking_enabled: bool,
    max_render_misses: int | None = None,
) -> PPEBusinessResult:
    """Convert detector boxes into stable safety-helmet business state."""

    ppe_raw = summarize_ppe_from_detections(detections, frame_shape=frame_shape)
    ppe = ppe_state.update(ppe_raw)
    tracks = (
        ppe_tracker.update(
            detections,
            ppe,
            frame_shape=frame_shape,
            max_render_misses=max_render_misses,
        )
        if tracking_enabled
        else []
    )
    tracks = _filter_tracks_for_ppe_counts(tracks, ppe)
    return PPEBusinessResult(ppe=ppe, tracks=tracks)


def _filter_tracks_for_ppe_counts(tracks: list[dict[str, Any]], ppe: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_labels: set[str] = set()
    if int(ppe.get("person_count") or 0) > 0:
        allowed_labels.add("person")
    if int(ppe.get("helmet_count") or 0) > 0:
        allowed_labels.add("helmet")
    if int(ppe.get("head_count") or 0) > 0:
        allowed_labels.add("head")
    if not allowed_labels:
        return []
    return [dict(track) for track in tracks if str(track.get("label") or "") in allowed_labels]
