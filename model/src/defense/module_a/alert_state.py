from __future__ import annotations

from collections import deque


class AlertState:
    """3/5 style rolling confirmation state with an optional real-time window.

    Backwards compatibility contract:

    * ``update(suspicious)`` (single-argument form) keeps the exact legacy
      semantics used by the offline pipeline: queue/trigger_count/hold_remaining
      and nothing else. Offline regression outputs must stay bit-for-bit
      identical to the pre-time-aware behaviour, so this path never consults
      ``ts_queue``.
    * ``update(suspicious, frame_ts=<float>)`` additionally enforces a soft
      real-time constraint: even when ``sum(queue) >= trigger_count``, the
      confirmed alert is suppressed if the time span covered by the current
      window exceeds ``alert_window_seconds_tolerance`` seconds (i.e. the
      window is not "formed" in real time terms).

    Policy for mixing modes within a single stream:
      The offline (``frame_ts=None``) path **clears** ``ts_queue`` on every
      call so any previously recorded timestamps cannot leak into a subsequent
      time-aware call. This guarantees the offline path behaves as if the
      timestamp feature did not exist, independent of prior state.
    """

    def __init__(
        self,
        window: int = 5,
        trigger_count: int = 3,
        hold_frames: int = 4,
        alert_window_seconds_tolerance: float = 2.0,
    ):
        self.window = max(1, int(window))
        self.trigger_count = max(1, min(int(trigger_count), self.window))
        self.hold_frames = max(0, int(hold_frames))
        self.alert_window_seconds_tolerance = float(alert_window_seconds_tolerance)
        self.queue: deque[int] = deque(maxlen=self.window)
        self.ts_queue: deque[float] = deque(maxlen=self.window)
        self.hold_remaining = 0

    def update(
        self,
        suspicious: bool,
        frame_ts: float | None = None,
    ) -> tuple[bool, bool]:
        self.queue.append(1 if suspicious else 0)

        if frame_ts is None:
            # Offline / legacy path: drop any previously recorded timestamps
            # so the soft time constraint cannot silently carry over if the
            # caller later switches to the time-aware form.
            if self.ts_queue:
                self.ts_queue.clear()
            alert_confirmed = sum(self.queue) >= self.trigger_count
        else:
            self.ts_queue.append(float(frame_ts))
            count_ok = sum(self.queue) >= self.trigger_count
            if count_ok and len(self.ts_queue) >= self.trigger_count:
                span = self.ts_queue[-1] - self.ts_queue[0]
                alert_confirmed = span <= self.alert_window_seconds_tolerance
            else:
                alert_confirmed = False

        # Hold semantics (P0-A-3 fix 2026-05-13):
        # ``attack_state_active`` is evaluated BEFORE the hold counter is
        # decremented so that ``hold_frames=N`` means "N quiet frames stay
        # active after the last suspicious frame", not ``N-1``. The previous
        # implementation decremented first and then checked ``>0``, which
        # burned one frame of the hold budget. Callers that relied on the
        # old behaviour should set ``hold_frames`` one higher to match.
        active = suspicious or self.hold_remaining > 0
        if suspicious:
            self.hold_remaining = self.hold_frames
        elif self.hold_remaining > 0:
            self.hold_remaining -= 1
        return alert_confirmed, active

    def reset(self) -> None:
        self.queue.clear()
        self.ts_queue.clear()
        self.hold_remaining = 0
