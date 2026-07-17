from __future__ import annotations

from types import SimpleNamespace

import pytest

from defense.runtime.frame_processor import FrameProcessor, build_branch_cards
from defense.runtime.overlay_records import (
    build_overlay_record,
    preview_module_info_from_overlay,
)


class _Pipeline:
    def __init__(self, detector: object) -> None:
        self.detector = detector
        self.detector_impl = "rebuilt"

    def reset(self) -> None:
        pass


def _processor(detector: object | None = None) -> FrameProcessor:
    pipeline = _Pipeline(detector or SimpleNamespace())
    bundle = SimpleNamespace(
        pipeline=pipeline,
        config={"module_a": {}, "runtime": {}},
        backend="fake",
        model_family="fake",
        artifact_path="",
    )
    return FrameProcessor(bundle)


def _status(
    *,
    primary_channel: str,
    alert_held: bool,
    hold_remaining: int = 0,
    physical_confirmed: bool = True,
    physical_active: bool | None = None,
    p_adv: float = 0.8,
    p_blind: float = 0.0,
    blind_type: str = "none",
    a3b_triggered: bool = False,
    a3b_state: str = "normal",
    a3b_source: str = "none",
    a3b_observed_score: float = 0.0,
    a3b_confirmed_score: float = 0.0,
    processor: FrameProcessor | None = None,
    static_media: dict[str, object] | None = None,
    a3b_debug: dict[str, object] | None = None,
) -> dict[str, object]:
    if physical_active is None:
        physical_active = physical_confirmed
    processor = processor or _processor()
    return processor._build_status(
        source_type="file",
        source="normal.mp4",
        profile="desktop_rtx",
        realtime=False,
        frame_idx=12,
        video_time_s=0.4,
        source_fps=30.0,
        fps=25.0,
        dropped_frames=0,
        info={
            "p_adv": p_adv,
            "alert_confirmed": physical_confirmed,
            "attack_detected": physical_active,
            "attack_state_active": physical_active,
            "reason_codes": (
                [
                    "B_BLIND_MOTION_BLUR"
                    if primary_channel == "blind"
                    else "a3_flow_artifact"
                ]
                if physical_active
                else []
            ),
            "details": {
                "joint_decision": {
                    "primary_channel": primary_channel,
                    "confirm_window": {
                        "alert_held": alert_held,
                        "alert_hold_remaining": hold_remaining,
                    },
                },
                "blinding": {
                    "p_blind": p_blind,
                    "p_blind_triggered": primary_channel == "blind",
                    "blind_type": blind_type,
                },
            },
        },
        ppe={},
        ppe_tracks=[],
        display_options={},
        feature_options={},
        custom_model={},
        redetect_budget_ok=False,
        redetect_count=0,
        redetect_ms=0.0,
        processing_ms=5.0,
        target_frame_budget_ms=33.0,
        raw_boxes_count=0,
        static_media=static_media or {},
        a3b_soft={
            "triggered": a3b_triggered,
            "observed_score": a3b_observed_score,
            "confirmed_score": a3b_confirmed_score,
            "confidence": a3b_confirmed_score,
            "display_score": max(
                a3b_observed_score,
                a3b_confirmed_score,
            ),
            "state": a3b_state,
            "effective_bbox": (
                [80, 90, 420, 520] if a3b_triggered else None
            ),
            "triggered_source": a3b_source,
            "reason": "",
            "debug": a3b_debug or {},
        },
    )


@pytest.mark.parametrize(
    (
        "physical_confirmed",
        "physical_active",
        "a3b_triggered",
        "a3b_state",
        "a3b_source",
        "a3b_observed_score",
        "a3b_confirmed_score",
        "expected_top_alert",
        "expected_physical_card_state",
        "expected_a3b_card_state",
    ),
    [
        (
            True,
            True,
            False,
            "normal",
            "none",
            0.0,
            0.0,
            True,
            "确认告警",
            "OK",
        ),
        (
            False,
            False,
            True,
            "confirmed",
            "rebuilt_media_confirmed",
            0.84,
            0.84,
            True,
            "OK",
            "确认",
        ),
        (
            False,
            False,
            True,
            "suspect",
            "observed_window",
            0.71,
            0.0,
            False,
            "OK",
            "疑似",
        ),
        (
            False,
            False,
            False,
            "normal",
            "none",
            0.0,
            0.0,
            False,
            "OK",
            "OK",
        ),
    ],
    ids=[
        "physical-only",
        "a3b-confirmed-only",
        "a3b-suspect-only",
        "normal",
    ],
)
def test_top_level_alert_is_confirmed_union_without_cross_contaminating_cards(
    physical_confirmed: bool,
    physical_active: bool,
    a3b_triggered: bool,
    a3b_state: str,
    a3b_source: str,
    a3b_observed_score: float,
    a3b_confirmed_score: float,
    expected_top_alert: bool,
    expected_physical_card_state: str,
    expected_a3b_card_state: str,
) -> None:
    status = _status(
        primary_channel="adv" if physical_active else "none",
        alert_held=False,
        physical_confirmed=physical_confirmed,
        physical_active=physical_active,
        p_adv=0.76 if physical_active else 0.12,
        a3b_triggered=a3b_triggered,
        a3b_state=a3b_state,
        a3b_source=a3b_source,
        a3b_observed_score=a3b_observed_score,
        a3b_confirmed_score=a3b_confirmed_score,
    )

    # The legacy fields remain the physical-channel contract. The explicit
    # module_a_* field is the public umbrella state used by the top-level Web
    # alert.
    assert status["alert_confirmed"] is physical_confirmed
    assert status["physical_alert_confirmed"] is physical_confirmed
    assert status["module_a_alert_confirmed"] is expected_top_alert
    assert status["attack_detected"] is physical_active
    assert status["attack_state_active"] is physical_active
    physical_card, a3b_card = status["branch_cards"]
    assert physical_card["state"] == expected_physical_card_state
    assert a3b_card["state"] == expected_a3b_card_state
    if not physical_confirmed:
        assert physical_card["border_class"] != "card-confirmed"

    record = build_overlay_record(
        status=status,
        ppe_tracks=[],
        run_id=1,
        display_options={},
    )
    preview = preview_module_info_from_overlay(record)
    assert record["alert_confirmed"] is physical_confirmed
    assert record["physical_alert_confirmed"] is physical_confirmed
    assert record["module_a_alert_confirmed"] is expected_top_alert
    assert preview["alert_confirmed"] is expected_top_alert
    assert (
        preview["details"]["a3b"]["media_confirmed"]
        is (a3b_state == "confirmed")
    )


def test_media_primary_legacy_umbrella_is_exposed_only_as_a3b() -> None:
    status = _status(
        primary_channel="media",
        alert_held=False,
        physical_confirmed=True,
        physical_active=True,
        p_adv=0.81,
        a3b_triggered=True,
        a3b_state="confirmed",
        a3b_source="rebuilt_media_confirmed",
        a3b_observed_score=0.84,
        a3b_confirmed_score=0.84,
    )

    assert status["alert_confirmed"] is False
    assert status["physical_alert_confirmed"] is False
    assert status["attack_detected"] is False
    assert status["physical_attack_detected"] is False
    assert status["attack_state_active"] is False
    assert status["physical_attack_state_active"] is False
    assert status["module_a_alert_confirmed"] is True
    assert status["module_a_attack_detected"] is True
    assert status["module_a_attack_state_active"] is True
    assert status["module_a_primary_channel"] == "a3b"
    assert status["module_a_alert_channel"] == "a3b"

    physical_card, a3b_card = status["branch_cards"]
    assert physical_card["border_class"] != "card-confirmed"
    assert a3b_card["border_class"] == "card-warning"
    assert a3b_card["state"] != "OK"

    record = build_overlay_record(
        status=status,
        ppe_tracks=[],
        run_id=1,
        display_options={},
    )
    preview = preview_module_info_from_overlay(record)
    assert record["alert_confirmed"] is False
    assert record["physical_alert_confirmed"] is False
    assert record["module_a_alert_confirmed"] is True
    assert preview["alert_confirmed"] is True
    assert preview["details"]["a3b"]["media_confirmed"] is True


def test_blind_confirmed_status_and_card_use_blind_channel_score() -> None:
    status = _status(
        primary_channel="blind",
        alert_held=False,
        p_adv=0.13,
        p_blind=0.74,
        blind_type="motion_blur",
    )

    assert status["module_a_primary_channel"] == "blind"
    assert status["module_a_alert_held"] is False
    assert status["module_a_fresh_confirmed"] is True
    assert status["p_blind"] == 0.74
    assert status["blind_type"] == "motion_blur"
    card = status["branch_cards"][0]
    assert card["title"] == "致盲/去信号攻击（p_blind）"
    assert card["score_source"] == "p_blind"
    assert card["score"] == 0.74
    assert card["state"] == "确认告警"
    assert card["primary_channel"] == "blind"


def test_held_alert_is_visible_as_hold_in_card_and_preview_contract() -> None:
    status = _status(
        primary_channel="adv",
        alert_held=True,
        hold_remaining=7,
        p_adv=0.46,
    )

    card = status["branch_cards"][0]
    assert status["module_a_fresh_confirmed"] is False
    assert card["state"] == "告警保持"
    assert "剩余 7 帧" in card["state_detail"]
    record = build_overlay_record(
        status=status,
        ppe_tracks=[],
        run_id=1,
        display_options={},
    )
    preview = preview_module_info_from_overlay(record)
    assert record["module_a_alert_held"] is True
    assert record["module_a_alert_hold_remaining"] == 7
    assert preview["alert_display_held"] is True
    assert preview["module_a_alert_held"] is True
    assert preview["module_a_primary_channel"] == "adv"


def test_held_media_alert_remains_confirmed_in_public_a3b_contract() -> None:
    status = _status(
        primary_channel="media",
        alert_held=True,
        hold_remaining=44,
        physical_confirmed=True,
        a3b_triggered=True,
        a3b_state="suspect",
        a3b_source="single_strong",
        a3b_observed_score=0.72,
        a3b_confirmed_score=0.72,
    )

    assert status["physical_alert_confirmed"] is False
    assert status["module_a_alert_confirmed"] is True
    assert status["module_a_alert_channel"] == "a3b"
    assert status["a3b_confirmed_alert"] is True
    assert status["a3b_state"] == "confirmed"
    assert status["a3b_triggered_source"] == "rebuilt_media_hold"
    assert status["a3b_debug"]["rebuilt_backend_media_alert_held"] is True


def test_quality_reacquisition_after_authoritative_media_confirmation_is_confirmed() -> None:
    processor = _processor()
    first = _status(
        processor=processor,
        primary_channel="media",
        alert_held=False,
        physical_confirmed=True,
        a3b_triggered=True,
        a3b_state="suspect",
        a3b_source="single_strong",
        a3b_observed_score=0.78,
        a3b_confirmed_score=0.78,
        static_media={
            "result_contract_source": "rebuilt",
            "media_confirmed": True,
            "p_media_confirmed_score": 0.78,
            "p_media_bbox": [80, 90, 420, 520],
        },
        a3b_debug={
            "quality_gate_passed": True,
            "current_explicit_guard_failures": [],
        },
    )
    assert first["a3b_confirmed_alert"] is True

    reacquired = _status(
        processor=processor,
        primary_channel="none",
        alert_held=False,
        physical_confirmed=False,
        physical_active=False,
        a3b_triggered=True,
        a3b_state="suspect",
        a3b_source="single_strong",
        a3b_observed_score=0.74,
        a3b_confirmed_score=0.74,
        a3b_debug={
            "quality_gate_passed": True,
            "current_explicit_guard_failures": [],
        },
    )

    assert reacquired["module_a_alert_confirmed"] is True
    assert reacquired["a3b_confirmed_alert"] is True
    assert reacquired["a3b_state"] == "confirmed"
    assert reacquired["a3b_triggered_source"] == "rebuilt_media_reacquired"
    assert reacquired["a3b_debug"]["rebuilt_reacquired_after_authoritative"] is True

    held = _status(
        processor=processor,
        primary_channel="none",
        alert_held=False,
        physical_confirmed=False,
        physical_active=False,
        a3b_triggered=False,
        a3b_state="normal",
        a3b_observed_score=0.0,
        a3b_confirmed_score=0.0,
    )
    assert held["module_a_alert_confirmed"] is True
    assert held["module_a_alert_held"] is True
    assert held["module_a_alert_hold_remaining"] == 89
    assert held["a3b_state"] == "confirmed"
    assert held["a3b_triggered_source"] == "rebuilt_media_public_hold"
    assert held["a3b_debug"]["rebuilt_public_alert_hold_active"] is True


def test_effective_config_exposes_total_module_a_alert_policy() -> None:
    detector = SimpleNamespace(
        theta_adv=0.65,
        theta_blind=0.55,
        _alert_hold_frames=12,
        _alert_hold_refresh_on_padv=True,
        _adv_cand_bridge_frames=4,
        _sustained_adv_enabled=True,
        _sustained_adv_seconds=2.0,
        _sustained_adv_run_mult=1.6,
        _sustained_adv_require_target=False,
        _sustained_adv_require_physical_support=False,
        _blind_sustained_enabled=True,
        _blind_sustained_floor=12,
    )

    effective = _processor(detector)._module_a_effective_config()

    assert effective["rebuilt_theta_adv"] == 0.65
    assert effective["rebuilt_theta_blind"] == 0.55
    assert effective["rebuilt_alert_hold_frames"] == 12
    assert effective["rebuilt_alert_hold_refresh_on_padv"] is True
    assert effective["rebuilt_adv_candidate_bridge_frames"] == 4
    assert effective["rebuilt_sustained_adv_escalation"] is True
    assert effective["rebuilt_sustained_adv_seconds"] == 2.0
    assert effective["rebuilt_sustained_adv_run_mult"] == 1.6
    assert effective["rebuilt_sustained_adv_require_target"] is False
    assert (
        effective["rebuilt_sustained_adv_require_physical_support"]
        is False
    )
    assert effective["rebuilt_blind_sustained_escalation"] is True
    assert effective["rebuilt_blind_sustained_floor"] == 12


def test_build_branch_cards_preserves_adv_compatibility() -> None:
    card = build_branch_cards(
        {
            "p_adv": 0.72,
            "alert_confirmed": True,
            "attack_detected": True,
            "attack_state_active": True,
            "module_a_primary_channel": "adv",
            "module_a_alert_held": False,
        }
    )[0]

    assert card["branch"] == "p_adv"
    assert card["title"] == "物理对抗扰动（p_adv）"
    assert card["score_source"] == "p_adv"
    assert card["score"] == 0.72
    assert card["state"] == "确认告警"
