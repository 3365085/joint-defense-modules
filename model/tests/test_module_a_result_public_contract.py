from __future__ import annotations

from defense.module_a.types import ModuleAResult


def _result(
    *,
    single_frame_suspicious: bool = False,
    alert_confirmed: bool = False,
    attack_state_active: bool = False,
) -> ModuleAResult:
    return ModuleAResult(
        frame_idx=7,
        p_adv=0.0,
        single_frame_suspicious=single_frame_suspicious,
        alert_confirmed=alert_confirmed,
        attack_state_active=attack_state_active,
        reason_codes=[],
        features={},
    )


def test_confirmed_media_alert_is_not_exported_as_normal() -> None:
    info = _result(alert_confirmed=True, attack_state_active=True).to_info_dict()

    assert info["single_frame_suspicious"] is False
    assert info["alert_confirmed"] is True
    assert info["attack_detected"] is True
    assert info["is_attack"] is True
    assert info["layer_triggered"] == "MODULE_A_PHYSICAL"


def test_normal_result_keeps_public_attack_fields_clear() -> None:
    info = _result().to_info_dict()

    assert info["attack_detected"] is False
    assert info["is_attack"] is False
    assert info["layer_triggered"] == "NORMAL"
