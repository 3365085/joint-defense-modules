from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from defense.runtime.evidence import EvidenceSession


INDEX = Path(__file__).resolve().parents[1] / "src/defense/web/static/index.html"


def test_active_a3b_event_requires_final_confirmation_and_is_not_nested() -> None:
    source = INDEX.read_text(encoding="utf-8")
    block_start = source.index("const completedSourceEvents")
    block_end = source.index('$(\"sourceAlerts\")', block_start)
    block = source[block_start:block_end]

    active_decl = re.search(
        r"const\s+activeSourceEvent\s*=\s*(?P<predicate>[^?]+?)\s*\?\s*\{",
        block,
    )
    assert active_decl is not None
    confirmation_predicate = active_decl.group("predicate").strip()
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", confirmation_predicate):
        predicate_decl = re.search(
            rf"const\s+{re.escape(confirmation_predicate)}\s*="
            r"\s*(?P<expression>[^;]+);",
            block,
        )
        assert predicate_decl is not None
        confirmation_expression = predicate_decl.group("expression")
    else:
        confirmation_expression = confirmation_predicate
    assert (
        "a3b_confirmed" in confirmation_expression
        or (
            "status.a3b_state" in confirmation_expression
            and "confirmed" in confirmation_expression
        )
    )
    assert "status.a3b_triggered ? {" not in block
    assert "status.a3b_triggered ? [{" not in block
    assert "[activeSourceEvent, ...completedSourceEvents]" in block


@pytest.mark.parametrize(
    (
        "top_alert",
        "physical_confirmed",
        "a3b_triggered",
        "a3b_state",
        "expected_channels",
    ),
    [
        (True, True, False, "normal", {"module_a"}),
        (True, False, True, "confirmed", {"a3b"}),
    ],
    ids=["physical-only", "a3b-confirmed-only"],
)
def test_evidence_session_keeps_physical_and_a3b_channels_distinct(
    tmp_path: Path,
    top_alert: bool,
    physical_confirmed: bool,
    a3b_triggered: bool,
    a3b_state: str,
    expected_channels: set[str],
) -> None:
    session = EvidenceSession(
        source_type="file",
        source="contract.mp4",
        profile="test",
        run_id=7,
        source_epoch=1,
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
        max_frames_per_event=1,
    )
    session.update(
        frame_idx=30,
        frame=np.zeros((16, 16, 3), dtype=np.uint8),
        info={},
        ppe={},
        status={
            "run_id": 7,
            "source_epoch": 1,
            "source_time_s": 1.0,
            "alert_confirmed": top_alert,
            "physical_alert_confirmed": physical_confirmed,
            "module_a_fresh_confirmed": physical_confirmed,
            "attack_detected": physical_confirmed,
            "attack_state_active": physical_confirmed,
            "p_adv": 0.8 if physical_confirmed else 0.1,
            "reason": "physical" if physical_confirmed else "",
            "a3b_triggered": a3b_triggered,
            "a3b_state": a3b_state,
            "a3b_event_score": 0.82 if a3b_triggered else 0.0,
            "a3b_confirmed_score": (
                0.82 if a3b_state == "confirmed" else 0.0
            ),
            "a3b_triggered_source": (
                "rebuilt_media_confirmed"
                if a3b_state == "confirmed"
                else "none"
            ),
        },
    )

    completed = session.close()

    assert {event["channel"] for event in completed} == expected_channels
