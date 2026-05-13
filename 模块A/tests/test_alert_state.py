"""Alert state machine — 3/5 rolling confirmation + time-aware extension."""
from __future__ import annotations

from defense.module_a.alert_state import AlertState


def test_confirmed_after_three_of_five() -> None:
    alert = AlertState(window=5, trigger_count=3, hold_frames=0)
    for value in [True, False, True, False, True]:
        confirmed, _ = alert.update(value)
    assert confirmed is True


def test_not_confirmed_below_trigger_count() -> None:
    alert = AlertState(window=5, trigger_count=3, hold_frames=0)
    results = [alert.update(v)[0] for v in [True, False, True, False, False]]
    assert results[-1] is False


def test_hold_extends_state_active() -> None:
    alert = AlertState(window=3, trigger_count=1, hold_frames=3)
    # Post-fix semantic (P0-A-3): ``hold_frames=N`` keeps state active for
    # N additional quiet frames after the last suspicious frame.
    confirmed, active = alert.update(True)
    assert confirmed is True
    assert active is True
    # Three quiet frames should stay active due to the hold.
    for _ in range(3):
        _, active_q = alert.update(False)
        assert active_q is True
    # The fourth quiet frame drops state active back to False.
    _, after_hold = alert.update(False)
    assert after_hold is False


def test_time_aware_mode_rejects_too_wide_window() -> None:
    alert = AlertState(
        window=5, trigger_count=3, hold_frames=0, alert_window_seconds_tolerance=1.0
    )
    # Five suspicious frames spread across 2 seconds — wider than tolerance 1.0s.
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
    confirmed = None
    for ts in timestamps:
        confirmed, _ = alert.update(True, frame_ts=ts)
    assert confirmed is False


def test_time_aware_mode_accepts_tight_window() -> None:
    alert = AlertState(
        window=5, trigger_count=3, hold_frames=0, alert_window_seconds_tolerance=1.0
    )
    confirmed = None
    for ts in [0.0, 0.1, 0.2, 0.3, 0.4]:
        confirmed, _ = alert.update(True, frame_ts=ts)
    assert confirmed is True


def test_reset_clears_state() -> None:
    alert = AlertState(window=3, trigger_count=1, hold_frames=2)
    alert.update(True)
    alert.reset()
    confirmed, active = alert.update(False)
    assert confirmed is False
    assert active is False


def test_offline_path_does_not_leak_into_time_aware() -> None:
    """Mixing offline (frame_ts=None) and time-aware updates should not let
    stale timestamps force-suppress a later time-aware window."""
    alert = AlertState(
        window=3, trigger_count=2, hold_frames=0, alert_window_seconds_tolerance=1.0
    )
    alert.update(True, frame_ts=10.0)  # Time-aware history
    alert.update(False)  # Offline — should clear ts_queue
    confirmed_1, _ = alert.update(True, frame_ts=100.0)
    confirmed_2, _ = alert.update(True, frame_ts=100.1)
    # After the offline call, timestamps restart fresh, so the 100.0->100.1
    # window is well within 1 s tolerance.
    assert confirmed_2 is True
