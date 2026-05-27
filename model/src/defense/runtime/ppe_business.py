from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from defense.module_a.ppe_postprocess import PPEPostprocessConfig, summarize_ppe_from_detections
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
    postprocess_config: PPEPostprocessConfig | None = None,
) -> PPEBusinessResult:
    """Convert detector boxes into stable safety-helmet business state."""

    ppe_raw = summarize_ppe_from_detections(
        detections,
        config=postprocess_config,
        frame_shape=frame_shape,
    )
    render_misses = _effective_max_render_misses(max_render_misses, ppe_raw)
    tracks = (
        ppe_tracker.update(
            detections,
            ppe_raw,
            frame_shape=frame_shape,
            max_render_misses=render_misses,
        )
        if tracking_enabled
        else []
    )
    ppe_input = (
        ppe_tracker.apply_temporal_evidence(ppe_raw, frame_shape)
        if tracking_enabled
        else ppe_raw
    )
    ppe = ppe_state.update(ppe_input)
    tracks = _filter_tracks_for_ppe_counts(tracks, ppe)
    return PPEBusinessResult(ppe=ppe, tracks=tracks)


def _effective_max_render_misses(max_render_misses: int | None, ppe: dict[str, Any]) -> int | None:
    if max_render_misses is None:
        return None
    base = max(0, int(max_render_misses))
    if str(ppe.get("evidence_mode") or "") == "head_helmet_only":
        return max(base, 8)
    return base


def _filter_tracks_for_ppe_counts(tracks: list[dict[str, Any]], ppe: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_labels: set[str] = set()
    if int(ppe.get("person_count") or 0) > 0:
        allowed_labels.add("person")
    if int(ppe.get("helmet_count") or 0) > 0:
        allowed_labels.add("helmet")
    suppression = ppe.get("helmet_fp_suppression", {})
    weak_head_count = 0
    if isinstance(suppression, dict):
        weak_head_count = len(suppression.get("weak_head_indices", []) or [])
    if int(ppe.get("head_count") or 0) > 0 or weak_head_count > 0:
        allowed_labels.add("head")
    if not allowed_labels and str(ppe.get("evidence_mode") or "") == "head_helmet_only":
        held_labels = {
            str(track.get("label") or "")
            for track in tracks
            if int(track.get("misses") or 0) > 0 and str(track.get("label") or "") in {"head", "helmet"}
        }
        allowed_labels.update(held_labels)
    if not allowed_labels:
        return []
    return [dict(track) for track in tracks if str(track.get("label") or "") in allowed_labels]
