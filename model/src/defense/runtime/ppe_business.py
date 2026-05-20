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
) -> PPEBusinessResult:
    """Convert detector boxes into stable safety-helmet business state."""

    ppe_raw = summarize_ppe_from_detections(detections, frame_shape=frame_shape)
    ppe = ppe_state.update(ppe_raw)
    tracks = ppe_tracker.update(detections, ppe, frame_shape=frame_shape) if tracking_enabled else []
    return PPEBusinessResult(ppe=ppe, tracks=tracks)
