from __future__ import annotations

from defense.web.helpers import enrich_status


def test_enrich_status_counts_active_ppe_event_before_finalized_record() -> None:
    status = enrich_status(
        {
            "ppe_event_active": True,
            "recent_ppe_events": [],
        }
    )

    assert status["ppe_event_count"] == 1


def test_enrich_status_counts_completed_and_active_ppe_events() -> None:
    status = enrich_status(
        {
            "ppe_event_active": True,
            "recent_ppe_events": [{"event_id": 1}],
        }
    )

    assert status["ppe_event_count"] == 2
