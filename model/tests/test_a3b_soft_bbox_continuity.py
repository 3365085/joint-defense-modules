from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState
from defense.runtime.frame_processor import FrameProcessor


ATTACK_SOURCE = "D:/security_project_d/素材/视频中出现干扰视频/case.mp4"
TRUSTED_BBOX = [214, 41, 505, 364]
JUMPED_BBOX = [499, 136, 517, 208]


def _static_media(
    score: float,
    bbox: list[int] | None,
    *,
    source: str = ATTACK_SOURCE,
    result_seq: int | None = None,
) -> dict:
    payload = {
        "live_score": score,
        "live_score_display": score,
        "score": score,
        "p_media": score,
        "p_media_candidate_count": 1,
        "p_media_scores": {"edge": 0.24, "track": 0.8},
        "p_media_border_state": {"suppressed": False},
        "p_media_camera_motion_state": {"suppressed": False},
        "p_media_physical_motion_state": {"suppressed": False},
        "source_path": source,
    }
    if bbox is not None:
        payload["p_media_bbox"] = list(bbox)
    if result_seq is not None:
        payload["a3b_result_seq"] = result_seq
    return payload


def test_window_and_hold_trigger_keep_last_trusted_bbox_on_low_or_jumped_evidence() -> None:
    state = A3BSoftTriggerState(
        {
            "window_size": 3,
            "min_window_hits": 2,
            "max_gap_frames": 2,
            "trigger_hold_frames": 5,
        }
    )

    state.update(_static_media(0.64, TRUSTED_BBOX))
    window_triggered = state.update(_static_media(0.65, TRUSTED_BBOX))
    assert window_triggered["triggered"] is True
    assert window_triggered["triggered_source"] == "window_accumulated"
    assert window_triggered["effective_bbox"] == TRUSTED_BBOX

    for bbox_age, frame_idx in enumerate(range(44, 49), start=1):
        payload = _static_media(0.45, JUMPED_BBOX, result_seq=frame_idx)
        payload["p_media_candidate_count"] = 0
        held = state.update(payload)
        assert held["triggered"] is True
        assert held["effective_bbox"] == TRUSTED_BBOX
        assert held["debug"]["bbox_large_jump"] is True
        assert held["debug"]["current_evidence_low"] is True
        assert held["debug"]["trusted_bbox_fallback"] is True
        assert held["debug"]["trusted_bbox_age_frames"] == bbox_age
        assert held["debug"]["trusted_bbox_expired"] is False


def test_inactive_bbox_expires_after_max_gap_and_new_location_is_directly_trusted() -> None:
    state = A3BSoftTriggerState(
        {
            "window_size": 3,
            "min_window_hits": 2,
            "max_gap_frames": 2,
            "trigger_hold_frames": 2,
        }
    )
    moved_bbox = [10, 200, 280, 600]

    state.update(_static_media(0.64, TRUSTED_BBOX, result_seq=1))
    state.update(_static_media(0.65, TRUSTED_BBOX, result_seq=2))

    inactive = None
    for result_seq in range(3, 7):
        inactive = state.update(_static_media(0.10, JUMPED_BBOX, result_seq=result_seq))
        if inactive["debug"]["trusted_bbox_expired"]:
            break

    assert inactive is not None
    assert inactive["triggered"] is False
    assert inactive["effective_bbox"] is None
    assert inactive["debug"]["trusted_bbox_age_frames"] == 3
    assert inactive["debug"]["trusted_bbox_expired"] is True
    assert "max_gap_frames" in inactive["debug"]["trusted_bbox_expired_reasons"]
    assert inactive["debug"]["last_trusted_bbox"] is None
    assert inactive["debug"]["pending_trusted_bbox"] is None

    moved = state.update(_static_media(0.80, moved_bbox, result_seq=20))
    assert moved["triggered"] is True
    assert moved["effective_bbox"] == moved_bbox
    assert moved["debug"]["last_trusted_bbox"] == moved_bbox
    assert moved["debug"]["trusted_bbox_updated"] is True
    assert moved["debug"]["trusted_bbox_fallback"] is False
    assert moved["debug"]["pending_trusted_bbox"] is None


def test_inactive_bbox_expires_after_independent_result_sequence_silence() -> None:
    state = A3BSoftTriggerState(
        {
            "window_size": 3,
            "min_window_hits": 2,
            "max_gap_frames": 2,
            "trigger_hold_frames": 2,
        }
    )

    state.update(_static_media(0.64, TRUSTED_BBOX, result_seq=1))
    state.update(_static_media(0.65, TRUSTED_BBOX, result_seq=2))

    inactive = None
    for result_seq in range(3, 8):
        inactive = state.update(_static_media(0.43, None, result_seq=result_seq))

    assert inactive is not None
    assert inactive["triggered"] is False
    assert inactive["debug"]["stable_hits"] > 0
    assert inactive["debug"]["trusted_bbox_age_result_seqs"] == 5
    assert inactive["debug"]["trusted_bbox_expired"] is True
    assert "result_seq_silence" in inactive["debug"]["trusted_bbox_expired_reasons"]
    assert inactive["debug"]["last_trusted_bbox"] is None


def test_effective_bbox_is_not_exposed_after_reset_or_across_sources() -> None:
    state = A3BSoftTriggerState()
    state.update(_static_media(0.64, TRUSTED_BBOX))
    triggered = state.update(_static_media(0.65, TRUSTED_BBOX))
    assert triggered["effective_bbox"] == TRUSTED_BBOX

    state.reset()
    after_reset = state.update(_static_media(0.10, JUMPED_BBOX))
    assert after_reset["triggered"] is False
    assert after_reset["effective_bbox"] is None
    assert after_reset["debug"]["last_trusted_bbox"] is None

    state.update(_static_media(0.64, TRUSTED_BBOX))
    state.update(_static_media(0.65, TRUSTED_BBOX))
    clean_source = "D:/security_project_d/素材/真实视频/clean.mp4"
    switched = state.update(_static_media(0.10, JUMPED_BBOX, source=clean_source))
    assert switched["triggered"] is False
    assert switched["effective_bbox"] is None
    assert switched["debug"]["last_trusted_bbox"] is None

    unknown = state.update(_static_media(0.95, JUMPED_BBOX, source=""))
    assert unknown["triggered"] is True
    assert unknown["effective_bbox"] == JUMPED_BBOX
    assert unknown["debug"]["last_trusted_bbox"] == JUMPED_BBOX
    assert unknown["debug"]["source_keyword_policy"] == "diagnostic_only"


def test_consistent_high_quality_bbox_transition_is_accepted_after_three_new_results() -> None:
    state = A3BSoftTriggerState()
    moved_bbox = [10, 200, 280, 600]
    moved_bbox_next = [12, 202, 282, 602]

    state.update(_static_media(0.64, TRUSTED_BBOX, result_seq=1))
    state.update(_static_media(0.65, TRUSTED_BBOX, result_seq=2))

    first_moved = state.update(_static_media(0.66, moved_bbox, result_seq=10))
    assert first_moved["effective_bbox"] == TRUSTED_BBOX
    assert first_moved["debug"]["pending_trusted_bbox"] == moved_bbox
    assert first_moved["debug"]["pending_trusted_bbox_hits"] == 1

    duplicate_result = state.update(_static_media(0.66, moved_bbox, result_seq=10))
    assert duplicate_result["effective_bbox"] == TRUSTED_BBOX
    assert duplicate_result["debug"]["pending_trusted_bbox_hits"] == 1

    low_thin = state.update(_static_media(0.293, JUMPED_BBOX, result_seq=11))
    assert low_thin["effective_bbox"] == TRUSTED_BBOX
    assert low_thin["debug"]["pending_trusted_bbox"] is None
    assert low_thin["debug"]["pending_trusted_bbox_hits"] == 0
    assert low_thin["debug"]["pending_trusted_bbox_expired"] is True
    assert low_thin["debug"]["pending_trusted_bbox_expired_reason"] == "evidence_gap"

    restarted = state.update(_static_media(0.66, moved_bbox, result_seq=12))
    assert restarted["effective_bbox"] == TRUSTED_BBOX
    assert restarted["debug"]["pending_trusted_bbox_hits"] == 1

    second_moved = state.update(_static_media(0.67, moved_bbox_next, result_seq=13))
    assert second_moved["effective_bbox"] == TRUSTED_BBOX
    assert second_moved["debug"]["pending_trusted_bbox_hits"] == 2
    assert second_moved["debug"]["pending_trusted_bbox_accepted"] is False

    moved_bbox_third = [14, 204, 284, 604]
    accepted = state.update(_static_media(0.68, moved_bbox_third, result_seq=14))
    assert accepted["effective_bbox"] == moved_bbox_third
    assert accepted["debug"]["last_trusted_bbox"] == moved_bbox_third
    assert accepted["debug"]["pending_trusted_bbox"] is None
    assert accepted["debug"]["pending_trusted_bbox_hits"] == 0
    assert accepted["debug"]["pending_trusted_bbox_accepted"] is True


def test_high_quality_aspect_ratio_jump_needs_three_independent_results() -> None:
    state = A3BSoftTriggerState()
    broad_bbox = [9, 84, 509, 236]

    state.update(_static_media(0.78, TRUSTED_BBOX, result_seq=1))
    state.update(_static_media(0.78, TRUSTED_BBOX, result_seq=2))

    first_broad = state.update(_static_media(0.60, broad_bbox, result_seq=10))
    assert first_broad["debug"]["bbox_large_jump"] is True
    assert first_broad["effective_bbox"] == TRUSTED_BBOX
    assert first_broad["debug"]["pending_trusted_bbox_hits"] == 1

    second_broad = state.update(_static_media(0.61, broad_bbox, result_seq=11))
    assert second_broad["effective_bbox"] == TRUSTED_BBOX
    assert second_broad["debug"]["pending_trusted_bbox_hits"] == 2
    assert second_broad["debug"]["pending_trusted_bbox_accepted"] is False


def test_containing_bbox_expansion_needs_three_results_but_contraction_is_immediate() -> None:
    state = A3BSoftTriggerState()
    expanded_bbox = [6, 42, 507, 392]
    expanded_bbox_next = [8, 43, 509, 393]
    expanded_bbox_third = [9, 44, 508, 393]

    state.update(_static_media(0.78, TRUSTED_BBOX, result_seq=1))
    state.update(_static_media(0.78, TRUSTED_BBOX, result_seq=2))

    first_expanded = state.update(
        _static_media(0.66, expanded_bbox, result_seq=10)
    )
    assert first_expanded["debug"]["bbox_large_jump"] is True
    assert first_expanded["effective_bbox"] == TRUSTED_BBOX
    assert first_expanded["debug"]["pending_trusted_bbox_hits"] == 1

    second_expanded = state.update(
        _static_media(0.67, expanded_bbox_next, result_seq=11)
    )
    assert second_expanded["effective_bbox"] == TRUSTED_BBOX
    assert second_expanded["debug"]["pending_trusted_bbox_hits"] == 2

    accepted = state.update(
        _static_media(0.68, expanded_bbox_third, result_seq=12)
    )
    assert accepted["effective_bbox"] == expanded_bbox_third
    assert accepted["debug"]["pending_trusted_bbox_accepted"] is True

    contracted = state.update(
        _static_media(0.69, TRUSTED_BBOX, result_seq=13)
    )
    assert contracted["debug"]["bbox_large_jump"] is False
    assert contracted["effective_bbox"] == TRUSTED_BBOX
    assert contracted["debug"]["trusted_bbox_updated"] is True


def test_contained_contraction_with_aspect_change_is_immediate_but_thin_strip_is_pending() -> None:
    state = A3BSoftTriggerState()
    inner_bbox = [100, 100, 200, 200]
    expanded_bbox = [0, 90, 300, 210]

    state.update(_static_media(0.78, inner_bbox, result_seq=1))
    state.update(_static_media(0.78, inner_bbox, result_seq=2))
    state.update(_static_media(0.66, expanded_bbox, result_seq=10))
    state.update(_static_media(0.67, expanded_bbox, result_seq=11))
    expanded = state.update(
        _static_media(0.68, expanded_bbox, result_seq=12)
    )
    assert expanded["effective_bbox"] == expanded_bbox
    assert expanded["debug"]["pending_trusted_bbox_accepted"] is True

    contracted = state.update(
        _static_media(0.69, inner_bbox, result_seq=13)
    )
    assert contracted["debug"]["bbox_large_jump"] is False
    assert contracted["effective_bbox"] == inner_bbox
    assert contracted["debug"]["trusted_bbox_updated"] is True

    thin_inner_strip = [140, 100, 155, 200]
    thin = state.update(
        _static_media(0.70, thin_inner_strip, result_seq=14)
    )
    assert thin["debug"]["bbox_large_jump"] is True
    assert thin["effective_bbox"] == inner_bbox
    assert thin["debug"]["pending_trusted_bbox"] == thin_inner_strip
    assert thin["debug"]["pending_trusted_bbox_hits"] == 1


def test_reset_and_source_switch_clear_pending_bbox_transition() -> None:
    state = A3BSoftTriggerState()
    moved_bbox = [10, 200, 280, 600]
    state.update(_static_media(0.64, TRUSTED_BBOX, result_seq=1))
    state.update(_static_media(0.65, TRUSTED_BBOX, result_seq=2))
    pending = state.update(_static_media(0.66, moved_bbox, result_seq=10))
    assert pending["debug"]["pending_trusted_bbox_hits"] == 1

    state.reset()
    after_reset = state.update(_static_media(0.10, moved_bbox, result_seq=11))
    assert after_reset["debug"]["pending_trusted_bbox"] is None
    assert after_reset["debug"]["pending_trusted_bbox_hits"] == 0

    state.update(_static_media(0.64, TRUSTED_BBOX, result_seq=20))
    state.update(_static_media(0.65, TRUSTED_BBOX, result_seq=21))
    state.update(_static_media(0.66, moved_bbox, result_seq=22))
    switched = state.update(
        _static_media(
            0.66,
            moved_bbox,
            source="D:/security_project_d/素材/真实视频/clean.mp4",
            result_seq=23,
        )
    )
    assert switched["debug"]["pending_trusted_bbox"] is None
    assert switched["debug"]["pending_trusted_bbox_hits"] == 0


class _Pipeline:
    def __init__(self, info: dict, detections: DetectionFrameResult) -> None:
        self.info = info
        self.detections = detections

    def process_runtime_frame(self, frame: np.ndarray, **_kwargs):
        return frame, self.detections, self.info

    def reset(self) -> None:
        return None


def test_frame_processor_updates_soft_state_once_before_ppe_and_reuses_effective_bbox() -> None:
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = DetectionFrameResult(
        image=frame,
        boxes=[[300, 100, 420, 330]],
        classes=[2],
        confidences=[0.9],
        names={0: "helmet", 1: "head", 2: "person"},
        backend="fake",
        artifact_path="fake://model",
        inference_ms=0.0,
        raw_result=None,
    )
    info = {
        "details": {
            "module_a_features": {
                "static_media": _static_media(0.293, JUMPED_BBOX),
            }
        },
        "latency_breakdown": {},
        "reason_codes": [],
    }
    bundle = SimpleNamespace(
        pipeline=_Pipeline(info, detections),
        backend="fake",
        model_family="fake",
        artifact_path="fake://model",
        config={"a3b": {"trigger_hold_frames": 3}},
    )
    processor = FrameProcessor(bundle)
    processor.a3b_soft.update(_static_media(0.64, TRUSTED_BBOX))
    processor.a3b_soft.update(_static_media(0.65, TRUSTED_BBOX))

    update_calls = 0
    original_update = processor.a3b_soft.update

    def counted_update(static_media: dict) -> dict:
        nonlocal update_calls
        update_calls += 1
        return original_update(static_media)

    processor.a3b_soft.update = counted_update  # type: ignore[method-assign]
    processed = processor.process(
        frame,
        frame_idx=44,
        source_type="file",
        source=ATTACK_SOURCE,
        profile="default",
        realtime=True,
        video_time_s=1.5,
        source_fps=30.0,
        dropped_frames=0,
        display_options={"show_boxes": False},
        feature_options={},
        custom_model={},
        target_frame_budget_ms=1000.0,
    )

    assert update_calls == 1
    assert processed.status["a3b_triggered"] is True
    assert processed.status["a3b_bbox"] == TRUSTED_BBOX
    assert processed.status["ppe_source_auth_media_bbox"] == TRUSTED_BBOX
    assert processed.status["ppe_source_auth_media_suppressed"] is True
    assert processed.status["ppe_source_auth_media_suppressed_person_count"] == 1
    assert processed.ppe_tracks == []
