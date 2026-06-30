from __future__ import annotations

from defense.module_a.postprocess.ppe_tracking import canonical_label


def test_display_tracking_person_aliases_match_ppe_summary_aliases() -> None:
    for alias in ("person", "worker", "human", "pedestrian"):
        assert canonical_label(alias) == "person"

