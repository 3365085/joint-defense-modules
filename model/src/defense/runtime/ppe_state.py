from __future__ import annotations

from collections import deque
from typing import Any


class SafetyHelmetState:
    """Temporal state machine for PPE warnings.

    Raw PPE results are noisy, especially on far targets. This class converts
    single-frame candidates into warning/confirmed states with a 3/5-style
    temporal window, matching Module A alarm semantics.
    """

    def __init__(self, window: int = 6, trigger_count: int = 3, hold_frames: int = 12):
        self.window = max(1, int(window))
        self.trigger_count = max(1, min(int(trigger_count), self.window))
        self.hold_frames = max(0, int(hold_frames))
        self._votes: deque[bool] = deque(maxlen=self.window)
        self._hold_remaining = 0

    def reset(self) -> None:
        self._votes.clear()
        self._hold_remaining = 0

    def update(self, ppe: dict[str, Any]) -> dict[str, Any]:
        candidate = bool(ppe.get("candidate", False))
        self._votes.append(candidate)
        positives = sum(1 for value in self._votes if value)
        confirmed = len(self._votes) >= self.trigger_count and positives >= self.trigger_count
        if confirmed:
            self._hold_remaining = self.hold_frames
        elif self._hold_remaining > 0:
            self._hold_remaining -= 1
        warning = bool(confirmed or self._hold_remaining > 0)
        enriched = dict(ppe)
        enriched.update(
            {
                "warning": warning,
                "confirmed": bool(confirmed),
                "window_positive": int(positives),
                "window": int(self.window),
                "trigger_count": int(self.trigger_count),
                "hold_remaining": int(self._hold_remaining),
            }
        )
        return enriched
