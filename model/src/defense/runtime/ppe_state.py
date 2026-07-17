from __future__ import annotations

from collections import deque
from typing import Any


class SafetyHelmetState:
    """Temporal state machine for PPE warnings.

    Raw PPE results are noisy, especially on far targets. This class converts
    single-frame candidates into warning/confirmed states with a 3/5-style
    temporal window, matching Module A alarm semantics.
    """

    def __init__(
        self,
        window: int = 6,
        trigger_count: int = 3,
        hold_frames: int = 12,
        event_hold_frames: int = 45,
        fast_window: int = 3,
        fast_trigger_count: int = 2,
        fast_min_confidence: float = 0.65,
    ):
        self.window = max(1, int(window))
        self.trigger_count = max(1, min(int(trigger_count), self.window))
        self.hold_frames = max(0, int(hold_frames))
        self.event_hold_frames = max(self.hold_frames, int(event_hold_frames))
        self.fast_window = max(1, int(fast_window))
        self.fast_trigger_count = max(1, min(int(fast_trigger_count), self.fast_window))
        self.fast_min_confidence = float(fast_min_confidence)
        self._votes: deque[bool] = deque(maxlen=self.window)
        self._fast_votes: deque[bool] = deque(maxlen=self.fast_window)
        self._hold_remaining = 0
        self._event_hold_remaining = 0
        self._event_active = False
        self._last_confirmed_reason = ""
        self._last_confirmed_source = ""

    def reset(self) -> None:
        self._votes.clear()
        self._fast_votes.clear()
        self._hold_remaining = 0
        self._event_hold_remaining = 0
        self._event_active = False
        self._last_confirmed_reason = ""
        self._last_confirmed_source = ""

    def update(self, ppe: dict[str, Any]) -> dict[str, Any]:
        candidate = bool(ppe.get("candidate", False))
        fast_candidate = self._is_fast_candidate(ppe)
        self._votes.append(candidate)
        self._fast_votes.append(fast_candidate)
        positives = sum(1 for value in self._votes if value)
        fast_positives = sum(1 for value in self._fast_votes if value)
        temporal_confirmed = len(self._votes) >= self.trigger_count and positives >= self.trigger_count
        fast_confirmed = (
            len(self._fast_votes) >= self.fast_trigger_count
            and fast_positives >= self.fast_trigger_count
        )
        confirmed = temporal_confirmed or fast_confirmed
        if confirmed:
            self._hold_remaining = self.hold_frames
            self._event_hold_remaining = self.event_hold_frames
            self._event_active = True
            event_reason = self._event_reason_from_ppe(ppe)
            if event_reason:
                self._last_confirmed_reason = event_reason
            confirmed_source = (
                "fast_head" if fast_confirmed else "temporal_window" if temporal_confirmed else ""
            )
            if confirmed_source:
                self._last_confirmed_source = confirmed_source
        elif self._hold_remaining > 0:
            self._hold_remaining -= 1
            if self._event_hold_remaining > 0:
                self._event_hold_remaining -= 1
        elif self._event_hold_remaining > 0:
            self._event_hold_remaining -= 1
        else:
            self._event_active = False
        warning = bool(confirmed or self._hold_remaining > 0)
        event_active = bool(self._event_active and (confirmed or self._event_hold_remaining > 0))
        confirmed_source = (
            "fast_head" if fast_confirmed else "temporal_window" if temporal_confirmed else ""
        )
        enriched = dict(ppe)
        enriched.update(
            {
                "warning": warning,
                "confirmed": bool(confirmed),
                "confirmed_source": confirmed_source,
                "event_active": event_active,
                "event_hold_remaining": int(self._event_hold_remaining),
                "event_last_reason": self._last_confirmed_reason,
                "event_last_confirmed_source": self._last_confirmed_source,
                "window_positive": int(positives),
                "window": int(self.window),
                "trigger_count": int(self.trigger_count),
                "fast_window_positive": int(fast_positives),
                "fast_window": int(self.fast_window),
                "fast_trigger_count": int(self.fast_trigger_count),
                "hold_remaining": int(self._hold_remaining),
            }
        )
        return enriched

    def _is_fast_candidate(self, ppe: dict[str, Any]) -> bool:
        if not bool(ppe.get("candidate", False)):
            return False
        if int(ppe.get("promoted_head_count", 0) or 0) > 0:
            return False
        if int(ppe.get("low_conf_temporal_head_count", 0) or 0) > 0:
            return False
        if int(ppe.get("head_count", 0) or 0) <= 0:
            return False
        if int(ppe.get("raw_head_count", 0) or 0) <= 0:
            return False
        if "missing_helmet_count" in ppe:
            if int(ppe.get("missing_helmet_count", 0) or 0) <= 0:
                return False
        elif int(ppe.get("helmet_count", 0) or 0) > 0:
            # Backwards compatibility for direct callers using the older
            # summary shape without target-level missing-helmet counts.
            return False
        confidence = _float(ppe.get("max_head_confidence"))
        return confidence >= self.fast_min_confidence

    @staticmethod
    def _event_reason_from_ppe(ppe: dict[str, Any]) -> str:
        if not (bool(ppe.get("candidate", False)) or int(ppe.get("missing_helmet_count", 0) or 0) > 0):
            return ""
        return str(ppe.get("reason") or "")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
