"""A3 track consistency — cross-frame ROI stability."""
from __future__ import annotations

import pytest

from defense.module_a.features.track_consistency import TrackConsistencyAnalyzer
from defense.module_a.types import ROI


def _helmet(roi_id: str, bbox: tuple[int, int, int, int], conf: float = 0.8) -> ROI:
    return ROI(roi_id=roi_id, bbox=bbox, label="helmet", confidence=conf)


def test_fresh_state_has_zero_score() -> None:
    analyzer = TrackConsistencyAnalyzer()
    out = analyzer.compute([])
    assert out["track_score"] == 0.0
    assert out["missing_track_count"] == 0
    assert out["active_track_count"] == 0


def test_stable_track_across_frames() -> None:
    analyzer = TrackConsistencyAnalyzer()
    roi = _helmet("h1", (100, 100, 200, 200), conf=0.8)
    analyzer.compute([roi])
    out = analyzer.compute([roi])
    # Same ROI → matched, no missing or confidence-drop.
    assert out["missing_track_count"] == 0
    assert out["confidence_drop_count"] == 0
    assert out["matched_track_count"] == 1


def test_confidence_drop_triggers_score() -> None:
    analyzer = TrackConsistencyAnalyzer(confidence_drop_trigger=0.25)
    high = _helmet("h1", (100, 100, 200, 200), conf=0.9)
    low = _helmet("h1", (100, 100, 200, 200), conf=0.3)
    analyzer.compute([high])
    out = analyzer.compute([low])
    assert out["confidence_drop_count"] == 1
    assert out["confidence_drop_score"] == pytest.approx(1.0, abs=1e-3)


def test_missing_tracks_increment_missing_count() -> None:
    analyzer = TrackConsistencyAnalyzer(high_confidence=0.45)
    high = _helmet("h1", (100, 100, 200, 200), conf=0.7)
    analyzer.compute([high])  # Seed a confident track
    out = analyzer.compute([])  # Track disappears
    # A high-confidence track went missing → count should be 1.
    assert out["missing_track_count"] == 1
    assert out["track_drop_score"] > 0.0


def test_label_cap_limits_candidates() -> None:
    analyzer = TrackConsistencyAnalyzer(max_candidates_per_label=2)
    rois = [_helmet(f"h{i}", (10 * i, 0, 10 * i + 80, 80), conf=0.5 + 0.1 * i) for i in range(5)]
    out = analyzer.compute(rois)
    # candidate cap is 2 per label → at most 2 tracks created on first frame.
    assert out["candidate_roi_count"] <= 2
