from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from defense.runtime.a3b_soft_trigger import (
    A3BSoftTriggerConfig,
    A3BSoftTriggerState,
)
from defense.runtime.frame_processor import (
    FrameProcessor,
    _source_auth_media_suppression_active,
)


def _quality_payload(
    score: float = 0.70,
    *,
    result_seq: int | None = None,
    include_result_seq: bool = True,
    source_path: str = "D:/neutral/case.mp4",
    base_triggered: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_path": source_path,
        "p_media": score,
        "live_score": score,
        "score": score,
        "triggered": base_triggered,
        "p_media_candidate_count": 1,
        "p_media_bbox": [120, 120, 420, 420],
        "p_media_scores": {
            "edge": 0.30,
            "track": 0.80,
            "yolo_context": 0.20,
        },
    }
    if include_result_seq:
        payload["a3b_result_seq"] = result_seq
    return payload


def _high_score_payload(source_path: str) -> dict[str, object]:
    return {
        "source_path": source_path,
        "a3b_result_seq": 1,
        "p_media": 0.90,
        "live_score": 0.90,
        "p_media_bbox": [140, 140, 400, 400],
        "p_media_candidate_count": 0,
        "p_media_scores": {"edge": 0.0, "track": 0.80},
    }


def _rebuilt_quality_payload(
    *,
    result_seq: int,
    candidate_score: float = 0.80,
    camera_motion_suppressed: bool = False,
) -> dict[str, object]:
    return {
        "source_path": "D:/neutral/rebuilt.mp4",
        "result_contract_source": "rebuilt",
        "a3b_result_seq": result_seq,
        "a3b_result_fresh": True,
        "p_media": 0.72,
        "live_score": 0.72,
        "score": 0.72,
        "p_media_candidate_count": 1,
        "p_media_bbox": [140, 140, 420, 420],
        "media_candidate_allowed": True,
        "p_media_scores": {
            "candidate_score": candidate_score,
            "edge": 0.50,
            "border_contrast": 0.90,
            "track": 0.80,
        },
        "p_media_camera_motion_state": {
            "suppressed": camera_motion_suppressed,
        },
    }


def test_config_direct_and_mapping_defaults_are_identical() -> None:
    direct = A3BSoftTriggerConfig()
    mapped = A3BSoftTriggerConfig.from_mapping(None)

    assert {
        field.name: getattr(direct, field.name)
        for field in fields(A3BSoftTriggerConfig)
    } == {
        field.name: getattr(mapped, field.name)
        for field in fields(A3BSoftTriggerConfig)
    }


def test_duplicate_positive_result_seq_does_not_add_window_votes() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "allow_window_accumulated_trigger": True,
            "min_window_hits": 2,
            "min_consecutive_hits": 2,
            "trigger_threshold": 0.60,
        }
    )

    first = state.update(_quality_payload(result_seq=1))
    duplicate = state.update(_quality_payload(result_seq=1))
    duplicate_again = state.update(_quality_payload(result_seq=1))

    assert first["triggered"] is False
    assert duplicate["triggered"] is False
    assert duplicate_again["triggered"] is False
    assert duplicate_again["debug"]["quality_window_hits"] == 1
    assert duplicate_again["debug"]["stable_hits"] == 1
    assert duplicate_again["debug"]["result_seq_mode"] == "duplicate_seq"
    assert (
        duplicate_again["debug"]["independent_evidence_consumed"]
        is False
    )

    second_result = state.update(_quality_payload(result_seq=2))
    same_confirmed_result = state.update(_quality_payload(result_seq=2))

    assert second_result["triggered"] is True
    assert second_result["triggered_source"] == "window_accumulated"
    assert second_result["debug"]["quality_window_hits"] == 2
    assert same_confirmed_result["triggered"] is True
    assert same_confirmed_result["debug"]["quality_window_hits"] == 2
    assert same_confirmed_result["debug"]["result_seq_mode"] == "duplicate_seq"


def test_duplicate_result_seq_reuses_content_analysis_but_keeps_bbox_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = A3BSoftTriggerState(
        {
            "min_window_hits": 2,
            "trigger_threshold": 0.60,
        }
    )
    original_debug = state._debug
    debug_calls = 0

    def counted_debug(
        static_media: dict[str, object],
        observed_score: float,
    ) -> dict[str, object]:
        nonlocal debug_calls
        debug_calls += 1
        return original_debug(static_media, observed_score)

    monkeypatch.setattr(state, "_debug", counted_debug)

    first = state.update(_quality_payload(0.80, result_seq=21))
    duplicate = state.update(_quality_payload(0.80, result_seq=21))
    duplicate_again = state.update(_quality_payload(0.80, result_seq=21))
    second_result = state.update(_quality_payload(0.80, result_seq=22))

    assert debug_calls == 2
    assert first["effective_bbox"] == [120, 120, 420, 420]
    assert duplicate["effective_bbox"] == first["effective_bbox"]
    assert duplicate_again["effective_bbox"] == first["effective_bbox"]
    assert duplicate["observed_score"] == first["observed_score"]
    assert duplicate["state"] == first["state"]
    assert duplicate["debug"]["analysis_cache_hit"] is True
    assert duplicate_again["debug"]["analysis_cache_hit"] is True
    assert second_result["debug"]["analysis_cache_hit"] is False
    assert second_result["triggered"] is True


def test_duplicate_positive_result_seq_does_not_add_track_confirmation_votes() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "allow_window_accumulated_trigger": False,
            "min_consecutive_hits": 2,
            "trigger_threshold": 0.60,
        }
    )

    first = state.update(
        _quality_payload(result_seq=10, base_triggered=True)
    )
    duplicate = state.update(
        _quality_payload(result_seq=10, base_triggered=True)
    )
    second_result = state.update(
        _quality_payload(result_seq=11, base_triggered=True)
    )

    assert first["triggered"] is False
    assert duplicate["triggered"] is False
    assert duplicate["debug"]["quality_window_hits"] == 1
    assert second_result["triggered"] is True
    assert second_result["triggered_source"] == "confirmed_track"
    assert second_result["debug"]["quality_window_hits"] == 2


def test_confirmed_hold_bridges_transient_rebuilt_tighten_gate_failure() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_hold_frames": 3,
        }
    )

    state.update(_rebuilt_quality_payload(result_seq=1))
    confirmed = state.update(_rebuilt_quality_payload(result_seq=2))
    held = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
        )
    )

    assert confirmed["triggered"] is True
    assert held["triggered"] is True
    assert held["triggered_source"].endswith("_hold")
    assert held["debug"]["held_trigger"] is True
    assert "rebuilt_tighten_gate_failed" in held["debug"]["blocking_failed_gates"]
    assert held["debug"]["hold_blocking_failed_gates"] == []


def test_duplicate_result_seq_does_not_consume_trigger_hold() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_hold_frames": 3,
        }
    )

    state.update(_rebuilt_quality_payload(result_seq=1))
    confirmed = state.update(_rebuilt_quality_payload(result_seq=2))
    first_gap = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
        )
    )
    duplicate_gap = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
        )
    )
    duplicate_gap_again = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
        )
    )
    second_gap = state.update(
        _rebuilt_quality_payload(
            result_seq=4,
            candidate_score=0.60,
        )
    )

    assert confirmed["debug"]["trigger_hold_remaining"] == 3
    assert first_gap["triggered"] is True
    assert first_gap["debug"]["trigger_hold_remaining"] == 2
    assert first_gap["debug"]["hold_tick_consumed"] is True
    assert duplicate_gap["triggered"] is True
    assert duplicate_gap["debug"]["trigger_hold_remaining"] == 2
    assert duplicate_gap["debug"]["hold_tick_consumed"] is False
    assert duplicate_gap["debug"]["hold_retained_on_duplicate"] is True
    assert duplicate_gap_again["debug"]["trigger_hold_remaining"] == 2
    assert second_gap["debug"]["trigger_hold_remaining"] == 1
    assert second_gap["debug"]["hold_tick_consumed"] is True


def test_duplicate_result_seq_camera_suppression_terminates_hold() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_hold_frames": 3,
        }
    )

    state.update(_rebuilt_quality_payload(result_seq=1))
    state.update(_rebuilt_quality_payload(result_seq=2))
    held = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
        )
    )
    blocked_duplicate = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
            camera_motion_suppressed=True,
        )
    )

    assert held["triggered"] is True
    assert blocked_duplicate["debug"]["analysis_cache_hit"] is True
    assert blocked_duplicate["triggered"] is False
    assert blocked_duplicate["debug"]["trigger_hold_remaining"] == 0
    assert "camera_motion_suppressed" in (
        blocked_duplicate["debug"]["current_explicit_guard_failures"]
    )


def test_duplicate_result_seq_policy_suppression_terminates_hold() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_hold_frames": 3,
        }
    )

    state.update(_rebuilt_quality_payload(result_seq=1))
    state.update(_rebuilt_quality_payload(result_seq=2))
    held = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
        )
    )
    blocked_payload = _rebuilt_quality_payload(
        result_seq=3,
        candidate_score=0.60,
    )
    blocked_payload["policy"] = {
        "media_candidate_allowed": True,
        "suppressed": True,
        "reason": "camera_translation_edge",
    }
    blocked_duplicate = state.update(blocked_payload)

    assert held["triggered"] is True
    assert blocked_duplicate["debug"]["analysis_cache_hit"] is True
    assert blocked_duplicate["triggered"] is False
    assert blocked_duplicate["debug"]["trigger_hold_remaining"] == 0
    assert "rebuilt_policy_suppressed" in (
        blocked_duplicate["debug"]["current_explicit_guard_failures"]
    )


def test_duplicate_result_seq_staleness_terminates_hold() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_hold_frames": 3,
        }
    )

    state.update(_rebuilt_quality_payload(result_seq=1))
    state.update(_rebuilt_quality_payload(result_seq=2))
    held = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
        )
    )
    stale_payload = _rebuilt_quality_payload(
        result_seq=3,
        candidate_score=0.60,
    )
    stale_payload["a3b_result_fresh"] = False
    stale_duplicate = state.update(stale_payload)

    assert held["triggered"] is True
    assert stale_duplicate["debug"]["analysis_cache_hit"] is True
    assert stale_duplicate["triggered"] is False
    assert stale_duplicate["debug"]["trigger_hold_remaining"] == 0
    assert "rebuilt_result_stale" in (
        stale_duplicate["debug"]["current_explicit_guard_failures"]
    )


def test_confirmed_hold_does_not_bridge_camera_motion_suppression() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_hold_frames": 3,
        }
    )

    state.update(_rebuilt_quality_payload(result_seq=1))
    confirmed = state.update(_rebuilt_quality_payload(result_seq=2))
    blocked = state.update(
        _rebuilt_quality_payload(
            result_seq=3,
            candidate_score=0.60,
            camera_motion_suppressed=True,
        )
    )

    assert confirmed["triggered"] is True
    assert blocked["triggered"] is False
    assert "camera_motion_suppressed" in blocked["debug"]["hold_blocking_failed_gates"]


def test_missing_result_seq_keeps_explicit_legacy_per_call_compatibility() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_threshold": 0.60,
        }
    )

    first = state.update(
        _quality_payload(include_result_seq=False)
    )
    second = state.update(
        _quality_payload(include_result_seq=False)
    )

    assert first["triggered"] is False
    assert second["triggered"] is True
    assert second["debug"]["quality_window_hits"] == 2
    assert second["debug"]["result_seq_mode"] == "legacy_no_seq"
    assert second["debug"]["independent_evidence_consumed"] is True


def test_missing_result_seq_consumes_hold_once_per_legacy_call() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_hold_frames": 3,
        }
    )

    first_payload = _rebuilt_quality_payload(result_seq=1)
    first_payload.pop("a3b_result_seq")
    second_payload = _rebuilt_quality_payload(result_seq=2)
    second_payload.pop("a3b_result_seq")
    state.update(first_payload)
    confirmed = state.update(second_payload)

    gap_payload = _rebuilt_quality_payload(
        result_seq=3,
        candidate_score=0.60,
    )
    gap_payload.pop("a3b_result_seq")
    first_gap = state.update(gap_payload)
    second_gap = state.update(gap_payload)

    assert confirmed["triggered"] is True
    assert confirmed["debug"]["hold_clock_mode"] == "legacy_call"
    assert first_gap["triggered"] is True
    assert first_gap["debug"]["trigger_hold_remaining"] == 2
    assert first_gap["debug"]["hold_tick_consumed"] is True
    assert second_gap["triggered"] is True
    assert second_gap["debug"]["trigger_hold_remaining"] == 1
    assert second_gap["debug"]["hold_tick_consumed"] is True


@pytest.mark.parametrize("result_seq", [0, -1, "invalid"])
def test_non_positive_or_invalid_present_result_seq_never_adds_votes(
    result_seq: object,
) -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_threshold": 0.60,
        }
    )

    result = state.update(
        _quality_payload(result_seq=result_seq)  # type: ignore[arg-type]
    )

    assert result["triggered"] is False
    assert result["debug"]["quality_window_hits"] == 0
    assert result["debug"]["result_seq_mode"] == "invalid_non_positive"
    assert result["debug"]["independent_evidence_consumed"] is False


def test_stale_result_seq_cannot_add_votes_or_replace_trusted_bbox() -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "min_window_hits": 2,
            "trigger_threshold": 0.60,
        }
    )
    first = state.update(_quality_payload(result_seq=5))
    stale_payload = _quality_payload(result_seq=4)
    stale_payload["p_media_bbox"] = [10, 10, 100, 100]
    stale = state.update(stale_payload)

    assert first["debug"]["last_trusted_bbox"] == [120, 120, 420, 420]
    assert stale["triggered"] is False
    assert stale["debug"]["quality_window_hits"] == 1
    assert stale["debug"]["result_seq_mode"] == "stale_seq"
    assert stale["debug"]["independent_evidence_consumed"] is False
    assert stale["debug"]["bbox_evidence_eligible"] is False
    assert stale["debug"]["last_trusted_bbox"] == [120, 120, 420, 420]


def test_source_keyword_is_diagnostic_only_and_path_invariant() -> None:
    diagnostic_config = {
        "observed_only_source_keywords": ["视频中出现干扰视频"],
        "trigger_source_keywords": ["视频中出现干扰视频"],
    }
    attack_path_state = A3BSoftTriggerState(diagnostic_config)
    neutral_path_state = A3BSoftTriggerState(diagnostic_config)

    attack = attack_path_state.update(
        _high_score_payload(
            "D:/素材/视频中出现干扰视频/case.mp4"
        )
    )
    neutral = neutral_path_state.update(
        _high_score_payload("D:/renamed/neutral-case.mp4")
    )

    for key in (
        "triggered",
        "triggered_source",
        "state",
        "confirmed_score",
        "confidence",
        "effective_bbox",
    ):
        assert attack[key] == neutral[key]
    assert attack["triggered"] is True
    assert attack["triggered_source"] == "observed_strong"
    assert attack["state"] == "suspect"
    assert attack["confirmed_score"] == 0.0
    assert attack["debug"]["confirmation_basis"] == "observed_only"
    assert attack["debug"]["source_keyword_policy"] == "diagnostic_only"
    assert neutral["debug"]["source_keyword_policy"] == "diagnostic_only"
    assert attack["debug"]["trigger_source_allowed"] is True
    assert neutral["debug"]["trigger_source_allowed"] is True
    assert attack["debug"]["trigger_source_keyword_matched"] is True
    assert neutral["debug"]["trigger_source_keyword_matched"] is False


def _rebuilt_soft_payload(
    *,
    result_seq: int,
    candidate_score: float = 0.6995,
    edge: float = 0.56,
    border_contrast: float = 0.89,
    fresh: bool = True,
    candidate_allowed: bool = True,
    suppressed: bool = False,
    bbox: list[int] | None = None,
    authoritative_confirmed: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "result_contract_source": "rebuilt",
        "source_path": "D:/neutral/rebuilt-case.mp4",
        "a3b_result_seq": result_seq,
        "a3b_result_fresh": fresh,
        "p_media": 0.67,
        "live_score": 0.67,
        "p_media_candidate_count": 1,
        "p_media_bbox": bbox or [120, 120, 420, 420],
        "p_media_strong_evidence": True,
        "media_candidate_allowed": candidate_allowed,
        "policy": {
            "media_candidate_allowed": candidate_allowed,
            "suppressed": suppressed,
        },
        "suppression": {
            "media_candidate_allowed": candidate_allowed,
            "suppressed": suppressed,
        },
        "p_media_scores": {
            "candidate_score": candidate_score,
            "edge": edge,
            "border_contrast": border_contrast,
            "track": 0.55,
            "yolo_context": 0.80,
        },
    }
    if authoritative_confirmed:
        payload.update(
            {
                "media_confirmed": True,
                "confirmed": True,
                "triggered": True,
                "score": 0.67,
            }
        )
    return payload


def _edge_temporal_payload(
    *,
    result_seq: int,
    edge: float,
    source_frame_idx: int,
    source_timestamp: float,
    interval: int = 6,
    fresh: bool = True,
    candidate_allowed: bool = True,
    suppressed: bool = False,
    bbox: list[int] | None = None,
) -> dict[str, object]:
    payload = _rebuilt_soft_payload(
        result_seq=result_seq,
        candidate_score=0.72,
        edge=edge,
        border_contrast=0.90,
        fresh=fresh,
        candidate_allowed=candidate_allowed,
        suppressed=suppressed,
        bbox=bbox,
    )
    payload.update(
        {
            "a3b_source_frame_idx": source_frame_idx,
            "a3b_source_timestamp": source_timestamp,
            "a3b_source_fps": 30.0,
            "a3b_source_interval_frames": interval,
        }
    )
    return payload


def test_rebuilt_soft_trigger_uses_tighten_gate_with_numeric_tolerance() -> None:
    state = A3BSoftTriggerState()

    first = state.update(_rebuilt_soft_payload(result_seq=1))
    second = state.update(_rebuilt_soft_payload(result_seq=2))

    assert first["triggered"] is False
    assert second["triggered"] is True
    assert second["triggered_source"] == "confirmed_track"
    assert second["debug"]["rebuilt_tighten_gate_applied"] is True
    assert second["debug"]["rebuilt_tighten_gate_passed"] is True
    assert second["debug"]["rebuilt_candidate_pass"] is True


def test_edge_boundary_near_then_strict_confirms_with_two_results() -> None:
    state = A3BSoftTriggerState()

    near = state.update(
        _edge_temporal_payload(
            result_seq=4,
            edge=0.5994022869,
            source_frame_idx=24,
            source_timestamp=0.8,
        )
    )
    strict = state.update(
        _edge_temporal_payload(
            result_seq=5,
            edge=0.5712258009,
            source_frame_idx=30,
            source_timestamp=1.0,
        )
    )

    assert near["triggered"] is False
    assert near["debug"]["rebuilt_edge_near_upper"] is True
    assert near["debug"]["rebuilt_edge_upper_hysteresis"] == pytest.approx(
        (0.58 - 0.45) / 6.0
    )
    assert near["debug"]["near_quality_hit"] is True
    assert near["debug"]["quality_window_result_hits"] == 0
    assert near["debug"]["temporal_quality_window_result_hits"] == 1

    assert strict["triggered"] is True
    assert strict["state"] == "confirmed"
    assert strict["triggered_source"] == "window_accumulated"
    assert strict["debug"]["quality_window_result_hits"] == 1
    assert strict["debug"]["temporal_quality_window_result_hits"] == 2
    assert strict["debug"]["strict_quality_required_for_bridge"] is True


def test_edge_boundary_near_only_cannot_confirm() -> None:
    state = A3BSoftTriggerState()
    result = None

    for seq, edge in enumerate(
        [0.5994, 0.5871, 0.5939],
        start=1,
    ):
        source_frame_idx = seq * 6
        result = state.update(
            _edge_temporal_payload(
                result_seq=seq,
                edge=edge,
                source_frame_idx=source_frame_idx,
                source_timestamp=source_frame_idx / 30.0,
            )
        )

    assert result is not None
    assert result["triggered"] is False
    assert result["debug"]["quality_window_result_hits"] == 0
    assert result["debug"]["temporal_quality_window_result_hits"] >= 2
    assert result["debug"]["near_quality_window_result_hits"] >= 2


def test_extreme_edge_cannot_bridge_with_strict_quality() -> None:
    state = A3BSoftTriggerState()

    strict = state.update(
        _edge_temporal_payload(
            result_seq=1,
            edge=0.57,
            source_frame_idx=6,
            source_timestamp=0.2,
        )
    )
    extreme = state.update(
        _edge_temporal_payload(
            result_seq=2,
            edge=0.64,
            source_frame_idx=12,
            source_timestamp=0.4,
        )
    )

    assert strict["triggered"] is False
    assert extreme["triggered"] is False
    assert extreme["debug"]["rebuilt_edge_near_upper"] is False
    assert extreme["debug"]["near_quality_hit"] is False
    assert extreme["debug"]["temporal_quality_window_result_hits"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"fresh": False},
        {"candidate_allowed": False},
        {"suppressed": True},
        {"bbox": [20, 200, 620, 400]},
    ],
    ids=[
        "stale",
        "candidate-disallowed",
        "policy-suppressed",
        "aspect-invalid",
    ],
)
def test_edge_boundary_near_quality_never_bypasses_current_guards(
    mutation: dict[str, object],
) -> None:
    state = A3BSoftTriggerState()

    near = state.update(
        _edge_temporal_payload(
            result_seq=1,
            edge=0.5994,
            source_frame_idx=6,
            source_timestamp=0.2,
            **mutation,
        )
    )
    strict = state.update(
        _edge_temporal_payload(
            result_seq=2,
            edge=0.57,
            source_frame_idx=12,
            source_timestamp=0.4,
        )
    )

    assert near["triggered"] is False
    assert near["debug"]["near_quality_hit"] is False
    assert strict["triggered"] is False
    assert strict["debug"]["quality_window_result_hits"] == 1
    assert strict["debug"]["temporal_quality_window_result_hits"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        _rebuilt_soft_payload(result_seq=1, candidate_score=0.698),
        _rebuilt_soft_payload(result_seq=1, edge=0.60),
        _rebuilt_soft_payload(result_seq=1, border_contrast=0.79),
        _rebuilt_soft_payload(result_seq=1, fresh=False),
        _rebuilt_soft_payload(result_seq=1, candidate_allowed=False),
        _rebuilt_soft_payload(result_seq=1, suppressed=True),
        _rebuilt_soft_payload(
            result_seq=1,
            bbox=[20, 200, 620, 400],
        ),
        _rebuilt_soft_payload(
            result_seq=1,
            bbox=[200, 20, 400, 620],
        ),
    ],
)
def test_rebuilt_soft_trigger_rejects_candidates_that_fail_backend_tighten_gate(
    payload: dict[str, object],
) -> None:
    state = A3BSoftTriggerState()

    result = state.update(payload)

    assert result["triggered"] is False
    assert result["state"] == "normal"
    assert result["debug"]["rebuilt_tighten_gate_applied"] is True
    assert result["debug"]["rebuilt_tighten_gate_passed"] is False
    assert "rebuilt_tighten_gate_failed" in result["debug"]["failed_gates"]


def test_rebuilt_soft_gate_aspect_boundaries_are_inclusive() -> None:
    state = A3BSoftTriggerState()
    boundary_bbox = [120, 100, 240, 400]

    first = state.update(
        _rebuilt_soft_payload(
            result_seq=1,
            bbox=boundary_bbox,
        )
    )
    second = state.update(
        _rebuilt_soft_payload(
            result_seq=2,
            bbox=boundary_bbox,
        )
    )

    assert first["debug"]["rebuilt_aspect_ratio"] == pytest.approx(0.40)
    assert first["debug"]["rebuilt_aspect_pass"] is True
    assert second["triggered"] is True


@pytest.mark.parametrize(
    "bbox",
    [
        [20, 200, 620, 400],
        [200, 20, 400, 620],
    ],
    ids=["too-wide", "too-narrow"],
)
def test_authoritative_media_confirmation_cannot_bypass_aspect_guard(
    bbox: list[int],
) -> None:
    state = A3BSoftTriggerState()

    first = state.update(
        _rebuilt_soft_payload(
            result_seq=1,
            bbox=bbox,
            authoritative_confirmed=True,
        )
    )
    second = state.update(
        _rebuilt_soft_payload(
            result_seq=2,
            bbox=bbox,
            authoritative_confirmed=True,
        )
    )

    assert first["triggered"] is False
    assert second["triggered"] is False
    assert second["state"] == "normal"
    assert second["debug"]["rebuilt_authoritative_confirmed"] is True
    assert second["debug"]["rebuilt_aspect_pass"] is False
    assert second["debug"]["rebuilt_tighten_gate_passed"] is False


def test_authoritative_aspect_guard_remains_active_when_optional_tighten_is_off(
) -> None:
    state = A3BSoftTriggerState(
        {"rebuilt_tighten_gate_enabled": False}
    )
    bbox = [20, 200, 620, 400]

    state.update(
        _rebuilt_soft_payload(
            result_seq=1,
            bbox=bbox,
            authoritative_confirmed=True,
        )
    )
    result = state.update(
        _rebuilt_soft_payload(
            result_seq=2,
            bbox=bbox,
            authoritative_confirmed=True,
        )
    )

    assert result["triggered"] is False
    assert result["debug"]["rebuilt_tighten_gate_applied"] is False
    assert "rebuilt_authoritative_guard_failed" in (
        result["debug"]["failed_gates"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"fresh": False},
        {"candidate_allowed": False},
        {"suppressed": True},
    ],
    ids=["stale", "candidate-disallowed", "policy-suppressed"],
)
def test_authoritative_media_confirmation_cannot_bypass_current_suppression(
    mutation: dict[str, object],
) -> None:
    state = A3BSoftTriggerState()

    first = state.update(
        _rebuilt_soft_payload(
            result_seq=1,
            authoritative_confirmed=True,
            **mutation,
        )
    )
    second = state.update(
        _rebuilt_soft_payload(
            result_seq=2,
            authoritative_confirmed=True,
            **mutation,
        )
    )

    assert first["triggered"] is False
    assert second["triggered"] is False
    assert second["state"] == "normal"
    assert second["debug"]["rebuilt_authoritative_confirmed"] is True
    assert second["debug"]["current_explicit_guard_failures"]


def _rebuilt_suppression_payload() -> dict[str, object]:
    return {
        "result_contract_source": "rebuilt",
        "a3b_result_fresh": True,
        "p_media_bbox": [100, 100, 400, 400],
        "p_media": 0.80,
        "media_candidate_allowed": True,
        "suppressed_reason": "none",
        "a3b_state": "candidate",
        "policy": {
            "media_candidate_allowed": True,
            "suppressed": False,
            "suppressed_reason": "none",
        },
        "suppression": {
            "media_candidate_allowed": True,
            "suppressed": False,
            "reason": "none",
        },
    }


def test_unsuppressed_fresh_rebuilt_candidate_can_hide_ppe() -> None:
    payload = _rebuilt_suppression_payload()

    assert _source_auth_media_suppression_active(payload) is True


@pytest.mark.parametrize(
    "mutation",
    [
        {"media_candidate_allowed": False},
        {"suppressed_reason": "low_display_target_plane_prefers_A1_A2_A3"},
        {"a3b_state": "suppressed"},
        {"a3b_result_fresh": False},
    ],
)
def test_policy_suppressed_or_stale_rebuilt_candidate_cannot_hide_ppe(
    mutation: dict[str, object],
) -> None:
    payload = _rebuilt_suppression_payload()
    payload.update(mutation)

    assert (
        _source_auth_media_suppression_active(
            payload,
            runtime_triggered=True,
        )
        is False
    )


def test_nested_suppression_cannot_be_bypassed_by_runtime_trigger() -> None:
    payload = _rebuilt_suppression_payload()
    payload["suppression"] = {
        "media_candidate_allowed": True,
        "suppressed": True,
        "reason": "camera_translation_edge",
    }

    assert (
        _source_auth_media_suppression_active(
            payload,
            runtime_triggered=True,
        )
        is False
    )


def test_legacy_candidate_without_freshness_field_remains_compatible() -> None:
    payload = {
        "p_media_bbox": [100, 100, 400, 400],
        "p_media": 0.80,
    }

    assert _source_auth_media_suppression_active(payload) is True


def test_module_a_effective_config_exposes_runtime_cap_and_backend_health() -> None:
    detector = SimpleNamespace(
        flow_requested_device="cuda:1",
        flow_effective_device="cpu",
        flow_backend="dis_cpu",
        flow_fallback_reason="cuda_unavailable",
        a4_classifier_configured=True,
        a4_classifier_loaded=False,
        a4_classifier_error="classifier missing",
        a4_classifier_fallback_reason="rule_fallback",
    )
    pipeline = SimpleNamespace(
        detector=detector,
        detector_impl="rebuilt",
    )
    bundle = SimpleNamespace(
        pipeline=pipeline,
        config={
            "runtime": {
                "process_fps_cap": 8,
                "detector_process_fps_cap": 12,
            },
            "module_a": {
                "detector_impl": "rebuilt",
                "device": "cuda:1",
            },
        },
    )
    processor = object.__new__(FrameProcessor)
    processor.bundle = bundle
    processor.pipeline = pipeline
    processor.a3b_soft = A3BSoftTriggerState(
        {
            "observed_only_source_keywords": [],
            "trigger_source_keywords": [],
        }
    )

    effective = processor._module_a_effective_config()

    assert effective["detector_process_fps_cap"] == 12
    assert effective["a3b_source_keyword_policy"] == "diagnostic_only"
    assert effective["a3b_source_keyword_match_required"] is False
    assert effective["a3b_observed_only_source_keywords"] == []
    assert effective["a3b_trigger_source_keywords"] == []
    assert effective["flow_requested_device"] == "cuda:1"
    assert effective["flow_effective_device"] == "cpu"
    assert effective["flow_backend"] == "dis_cpu"
    assert effective["flow_fallback_reason"] == "cuda_unavailable"
    assert effective["a4_classifier_configured"] is True
    assert effective["a4_classifier_loaded"] is False
    assert effective["a4_classifier_error"] == "classifier missing"
    assert (
        effective["a4_classifier_fallback_reason"]
        == "rule_fallback"
    )


@pytest.mark.parametrize(
    ("sensitivity", "module_config", "a3b_config"),
    [
        (
            "balanced",
            {"static_image_interval": 4},
            {
                "observed_threshold": 0.42,
                "trigger_threshold": 0.62,
                "strong_single_frame_threshold": 0.78,
                "observed_only_warning_threshold": 0.50,
                "observed_only_track_threshold": 0.50,
            },
        ),
        (
            "high",
            {"static_image_interval": 2},
            {
                "observed_threshold": 0.34,
                "trigger_threshold": 0.54,
                "strong_single_frame_threshold": 0.70,
                "observed_only_warning_threshold": 0.42,
                "observed_only_track_threshold": 0.42,
                "min_window_hits": 2,
                "observed_only_min_window_hits": 2,
                "min_consecutive_hits": 2,
            },
        ),
    ],
)
def test_module_a_effective_config_infers_matching_a3b_sensitivity(
    sensitivity: str,
    module_config: dict[str, object],
    a3b_config: dict[str, object],
) -> None:
    detector = SimpleNamespace(
        _a3b_interval=module_config["static_image_interval"],
    )
    pipeline = SimpleNamespace(detector=detector, detector_impl="rebuilt")
    bundle = SimpleNamespace(
        pipeline=pipeline,
        config={
            "module_a": module_config,
            "a3b": a3b_config,
        },
    )
    processor = object.__new__(FrameProcessor)
    processor.bundle = bundle
    processor.pipeline = pipeline
    processor.a3b_soft = A3BSoftTriggerState(a3b_config)

    effective = processor._module_a_effective_config()

    assert effective["a3b_sensitivity"] == sensitivity


def test_module_a_effective_config_does_not_mislabel_custom_a3b_thresholds() -> None:
    a3b_config = {
        "observed_threshold": 0.42,
        "trigger_threshold": 0.61,
        "strong_single_frame_threshold": 0.78,
        "observed_only_warning_threshold": 0.50,
        "observed_only_track_threshold": 0.50,
    }
    detector = SimpleNamespace(_a3b_interval=4)
    pipeline = SimpleNamespace(detector=detector, detector_impl="rebuilt")
    bundle = SimpleNamespace(
        pipeline=pipeline,
        config={
            "module_a": {"static_image_interval": 4},
            "a3b": a3b_config,
        },
    )
    processor = object.__new__(FrameProcessor)
    processor.bundle = bundle
    processor.pipeline = pipeline
    processor.a3b_soft = A3BSoftTriggerState(a3b_config)

    effective = processor._module_a_effective_config()

    assert effective["a3b_sensitivity"] is None
