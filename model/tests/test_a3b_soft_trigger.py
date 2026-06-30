from __future__ import annotations

from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState


def _static_media(score: float) -> dict:
    return {
        "live_score": score,
        "score": score,
        "p_media": score,
        "p_media_candidate_count": 1,
        "p_media_scores": {"edge": 0.24, "track": 0.5},
        "p_media_border_state": {"suppressed": False},
        "p_media_camera_motion_state": {"suppressed": False},
        "p_media_physical_motion_state": {"suppressed": False},
        "source_path": "D:/security_project_d/素材/视频中出现干扰视频/case.mp4",
    }


def test_a3b_soft_trigger_window_accumulated_without_full_consecutive_run() -> None:
    state = A3BSoftTriggerState()
    result = None
    for score in [0.1, 0.45, 0.50, 0.48, 0.2, 0.67, 0.64, 0.63]:
        result = state.update(_static_media(score))

    assert result is not None
    assert result["triggered"] is True
    assert result["triggered_source"] == "window_accumulated"
    assert result["confirmed_score"] > 0.0
    assert result["confidence"] == result["confirmed_score"]
    assert result["display_score"] == result["confirmed_score"]
    assert result["state"] == "confirmed"
    assert result["debug"]["window_hits"] >= 4


def test_a3b_soft_trigger_single_strong_frame() -> None:
    state = A3BSoftTriggerState()
    result = None
    for score in [0.1, 0.2, 0.81, 0.2]:
        result = state.update(_static_media(score))
        if result["triggered"]:
            break

    assert result is not None
    assert result["triggered"] is True
    assert result["triggered_source"] == "single_strong"
    assert result["state"] == "suspect"
    assert result["display_score"] == result["confirmed_score"]


def test_a3b_soft_trigger_keeps_clean_sequence_negative() -> None:
    state = A3BSoftTriggerState()
    result = None
    for score in [0.1, 0.18, 0.2, 0.15]:
        result = state.update(_static_media(score))

    assert result is not None
    assert result["triggered"] is False
    assert result["triggered_source"] == "none"
    assert result["state"] == "normal"


def test_a3b_soft_trigger_warns_but_does_not_confirm_strong_observation_without_candidate() -> None:
    state = A3BSoftTriggerState()
    result = state.update(
        {
            "live_score": 0.84,
            "score": 0.84,
            "p_media": 0.84,
            "source_path": r"D:\security_project_d\素材\视频中出现干扰视频\case.mp4",
        }
    )

    assert result["observed_score"] == 0.84
    assert result["confirmed_score"] == 0.0
    assert result["confidence"] == 0.0
    assert result["display_score"] == 0.0
    assert result["triggered"] is True
    assert result["triggered_source"] == "observed_strong"
    assert result["state"] == "suspect"
    assert "no_candidate_or_screen_cue" in result["debug"]["failed_gates"]


def test_a3b_soft_trigger_warns_on_stable_observed_only_track() -> None:
    state = A3BSoftTriggerState()
    result = None
    for score in [0.0, 0.58, 0.57, 0.56, 0.55]:
        result = state.update(
            {
                "live_score": score,
                "score": score,
                "p_media": score,
                "p_media_scores": {"track": 0.8},
                "p_media_border_state": {"suppressed": False},
                "p_media_camera_motion_state": {"suppressed": False},
                "p_media_physical_motion_state": {"suppressed": False},
                "source_path": r"D:\security_project_d\素材\视频中出现干扰视频\case.mp4",
            }
        )

    assert result is not None
    assert result["triggered"] is True
    assert result["triggered_source"] == "observed_window"
    assert result["state"] == "suspect"
    assert result["confirmed_score"] == 0.0
    assert result["confidence"] == 0.0
    assert result["display_score"] == 0.0
    assert result["debug"]["observed_only_window_hits"] >= 4


def test_a3b_soft_trigger_does_not_warn_observed_only_outside_material_category() -> None:
    state = A3BSoftTriggerState()
    result = None
    for score in [0.0, 0.58, 0.57, 0.56, 0.55]:
        result = state.update(
            {
                "live_score": score,
                "score": score,
                "p_media": score,
                "p_media_scores": {"track": 0.8},
                "p_media_border_state": {"suppressed": False},
                "p_media_camera_motion_state": {"suppressed": False},
                "p_media_physical_motion_state": {"suppressed": False},
                "source_path": r"D:\security_project_d\素材\真实视频\clean.mp4",
            }
        )

    assert result is not None
    assert result["triggered"] is False
    assert result["triggered_source"] == "none"
    assert result["state"] == "observing"
    assert result["display_score"] == 0.0
    assert result["debug"]["observed_only_source_allowed"] is False


def test_a3b_soft_trigger_accepts_legacy_a3plus_hold_as_quality_cue() -> None:
    state = A3BSoftTriggerState()
    result = None
    for score in [0.10, 0.74, 0.75]:
        result = state.update(
            {
                "live_score": score,
                "score": score,
                "p_media": score,
                "legacy_static_image": {
                    "triggered": True,
                    "triggered_source": "a3_plus_occlusion_hold",
                    "p_media_occlusion_state": {"active": True},
                    "p_media_fast_state": {"candidate": True, "fast_replay_evidence": True},
                },
                "p_media_border_state": {"suppressed": False},
                "p_media_camera_motion_state": {"suppressed": False},
                "p_media_physical_motion_state": {"suppressed": False},
                "source_path": "D:/security_project_d/素材/视频中出现干扰视频/case.mp4",
            }
        )

    assert result is not None
    assert result["triggered"] is True
    assert result["triggered_source"] == "confirmed_track"
    assert result["confirmed_score"] >= 0.74


def test_a3b_quality_gate_does_not_trigger_outside_allowed_source() -> None:
    state = A3BSoftTriggerState()
    result = None
    clean_source = "D:/security_project_d/素材/手机随意录制的视频/clean.mp4"
    for score in [0.81, 0.82, 0.83, 0.84]:
        payload = _static_media(score)
        payload["source_path"] = clean_source
        result = state.update(payload)

    assert result is not None
    assert result["observed_score"] >= 0.8
    assert result["triggered"] is False
    assert result["triggered_source"] == "none"
    assert result["state"] == "observing"
    assert result["debug"]["trigger_source_allowed"] is False


def test_a3b_allowed_source_triggers_with_shorter_window() -> None:
    state = A3BSoftTriggerState()
    result = None
    attack_source = "D:/security_project_d/素材/视频中出现干扰视频/case.mp4"
    for score in [0.10, 0.63, 0.64, 0.65]:
        payload = _static_media(score)
        payload["source_path"] = attack_source
        result = state.update(payload)

    assert result is not None
    assert result["triggered"] is True
    assert result["triggered_source"] == "window_accumulated"
    assert result["debug"]["quality_window_hits"] >= 3
    assert result["debug"]["trigger_source_allowed"] is True



def test_a3b_hold_bridges_short_positive_gap_only_in_allowed_source() -> None:
    state = A3BSoftTriggerState({"trigger_hold_frames": 3})
    attack_source = "D:/security_project_d/素材/视频中出现干扰视频/case.mp4"
    payload = _static_media(0.81)
    payload["source_path"] = attack_source
    result = state.update(payload)
    assert result is not None and result["triggered"] is True

    hold_payload = _static_media(0.45)
    hold_payload["source_path"] = attack_source
    held = state.update(hold_payload)
    assert held["triggered"] is True
    assert held["debug"]["held_trigger"] is True
    assert held["triggered_source"].endswith("_hold")

    clean_state = A3BSoftTriggerState({"trigger_hold_frames": 3})
    clean_source = "D:/security_project_d/素材/手机随意录制的视频/clean.mp4"
    clean_payload = _static_media(0.84)
    clean_payload["source_path"] = clean_source
    clean = clean_state.update(clean_payload)
    assert clean["triggered"] is False
    assert clean["debug"]["trigger_source_allowed"] is False
