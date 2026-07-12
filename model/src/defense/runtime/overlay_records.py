from __future__ import annotations

from typing import Any, Iterable


def _normalized_reason_codes(raw_reason_codes: Any) -> list[str]:
    if isinstance(raw_reason_codes, str):
        raw_reason_codes = [raw_reason_codes]
    values = raw_reason_codes if isinstance(raw_reason_codes, (list, tuple, set)) else []
    return [
        str(code)
        for code in values
        if str(code or "").strip() and str(code).lower() != "none"
    ]


def _overlay_reason_codes(overlay: dict[str, Any]) -> list[str]:
    reason_codes = _normalized_reason_codes(overlay.get("reason_codes"))
    if not reason_codes:
        reason = str(overlay.get("reason") or "").strip()
        if reason and reason.lower() != "none":
            reason_codes = [reason]
    a3b_reason = str(overlay.get("a3b_reason") or "").strip()
    if not reason_codes and a3b_reason and a3b_reason.lower() != "none":
        reason_codes = [a3b_reason]
    return reason_codes


def _last_active_reason_codes(history: Iterable[dict[str, Any]]) -> list[str]:
    for item in reversed(list(history)):
        alert_active = bool(item.get("alert_confirmed") or item.get("attack_state_active"))
        if not alert_active:
            break
        candidate = _overlay_reason_codes(item)
        if candidate:
            return candidate
    return []


def annotate_alert_display_context(
    overlay: dict[str, Any],
    history: Iterable[dict[str, Any]] = (),
    *,
    display_frame_idx: int | None = None,
    display_scene_cut_frame_idx: int | None = None,
) -> dict[str, Any]:
    out = dict(overlay)
    alert_active = bool(out.get("alert_confirmed") or out.get("attack_state_active"))
    current_reason_codes = _overlay_reason_codes(out)
    active_history_reason_codes = _last_active_reason_codes(history)
    state_frame_idx = out.get("frame_idx")
    timing_held = bool(
        out.get("alert_display_timing_held")
        or (
            alert_active
            and display_frame_idx is not None
            and display_scene_cut_frame_idx is not None
            and state_frame_idx is not None
            and int(state_frame_idx) < int(display_scene_cut_frame_idx)
            and int(display_scene_cut_frame_idx) <= int(display_frame_idx)
        )
    )
    out["alert_display_timing_held"] = timing_held
    reasonless_active_hold = bool(
        alert_active
        and bool(out.get("attack_detected"))
        and not current_reason_codes
        and active_history_reason_codes
    )
    alert_display_held = bool(
        out.get("alert_display_held")
        or (
            alert_active
            and (
                not bool(out.get("attack_detected"))
                or bool(out.get("held"))
                or timing_held
                or reasonless_active_hold
            )
        )
    )
    out["alert_display_held"] = alert_display_held
    if not alert_display_held:
        return out

    last_reason_codes = _normalized_reason_codes(out.get("alert_last_reason_codes"))
    if not last_reason_codes:
        if current_reason_codes:
            last_reason_codes = current_reason_codes
    if not last_reason_codes:
        last_reason_codes = active_history_reason_codes
    out["alert_last_reason_codes"] = last_reason_codes
    return out


def preview_module_info_from_overlay(overlay: dict[str, Any]) -> dict[str, Any]:
    context = annotate_alert_display_context(overlay)
    reason_codes = _overlay_reason_codes(context)
    last_reason_codes = _normalized_reason_codes(
        context.get("alert_last_reason_codes")
    )

    a3b_source = str(context.get("a3b_triggered_source") or "").strip()
    if a3b_source and a3b_source.lower() != "none":
        layer = a3b_source
    elif bool(
        context.get("alert_confirmed")
        or context.get("attack_detected")
        or context.get("attack_state_active")
    ):
        layer = "MODULE_A_PHYSICAL"
    else:
        layer = "NORMAL"

    return {
        "p_adv": context.get("p_adv"),
        "alert_confirmed": bool(context.get("alert_confirmed")),
        "attack_detected": bool(context.get("attack_detected")),
        "attack_state_active": bool(context.get("attack_state_active")),
        "alert_display_held": bool(context.get("alert_display_held")),
        "alert_last_reason_codes": last_reason_codes,
        "timing_ms": float(context.get("timing_ms") or 0.0),
        "layer_triggered": layer,
        "reason_codes": reason_codes,
    }


def build_overlay_record(
    *,
    status: dict[str, Any],
    ppe_tracks: list[dict[str, Any]],
    run_id: int,
    display_options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "running": bool(status.get("running", True)),
        "source_ended": bool(status.get("source_ended", False)),
        "detector_pipeline_mode": str(status.get("detector_pipeline_mode") or "backend_latest_only"),
        "source_epoch": int(status.get("source_epoch") or status.get("epoch") or 0),
        "frame_id": int(status.get("frame_id") or 0),
        "video_time_s": float(status.get("video_time_s") or 0.0),
        "source_time_s": float(status.get("source_time_s") or status.get("video_time_s") or 0.0),
        "wall_time_ms": float(status.get("wall_time_ms") or 0.0),
        "frame_idx": int(status.get("frame_idx") or 0),
        "p_adv": status.get("p_adv"),
        "p_adv_display": status.get("p_adv_display"),
        "p_adv_missing_reason": str(status.get("p_adv_missing_reason") or ""),
        "alert_confirmed": bool(status.get("alert_confirmed")),
        "attack_detected": bool(status.get("attack_detected")),
        "attack_state_active": bool(status.get("attack_state_active")),
        "reason": str(status.get("reason") or ""),
        "reason_codes": list(status.get("reason_codes") or []),
        "ppe_warning": bool(status.get("ppe_warning")),
        "ppe_candidate": bool(status.get("ppe_candidate")),
        "ppe_confirmed": bool(status.get("ppe_confirmed")),
        "ppe_confirmed_source": str(status.get("ppe_confirmed_source") or ""),
        "ppe_event_active": bool(status.get("ppe_event_active")),
        "ppe_event_hold_remaining": int(status.get("ppe_event_hold_remaining") or 0),
        "ppe_event_last_reason": str(status.get("ppe_event_last_reason") or ""),
        "ppe_event_last_confirmed_source": str(status.get("ppe_event_last_confirmed_source") or ""),
        "ppe_person_count": int(status.get("ppe_person_count") or 0),
        "ppe_raw_person_count": int(status.get("ppe_raw_person_count", status.get("ppe_person_count")) or 0),
        "ppe_inferred_person_count": int(status.get("ppe_inferred_person_count", status.get("ppe_person_count")) or 0),
        "ppe_person_context_count": int(status.get("ppe_person_context_count", status.get("ppe_person_count")) or 0),
        "ppe_weak_person_count": int(status.get("ppe_weak_person_count") or 0),
        "ppe_promoted_person_count": int(status.get("ppe_promoted_person_count") or 0),
        "ppe_effective_person_count": int(status.get("ppe_effective_person_count", status.get("ppe_person_count")) or 0),
        "ppe_helmet_count": int(status.get("ppe_helmet_count") or 0),
        "ppe_raw_helmet_count": int(status.get("ppe_raw_helmet_count", status.get("ppe_helmet_count")) or 0),
        "ppe_weak_helmet_count": int(status.get("ppe_weak_helmet_count") or 0),
        "ppe_promoted_helmet_count": int(status.get("ppe_promoted_helmet_count") or 0),
        "ppe_effective_helmet_count": int(status.get("ppe_effective_helmet_count", status.get("ppe_helmet_count")) or 0),
        "ppe_head_count": int(status.get("ppe_head_count") or 0),
        "ppe_raw_head_count": int(status.get("ppe_raw_head_count", status.get("ppe_head_count")) or 0),
        "ppe_weak_head_count": int(status.get("ppe_weak_head_count") or 0),
        "ppe_promoted_head_count": int(status.get("ppe_promoted_head_count") or 0),
        "ppe_effective_head_count": int(status.get("ppe_effective_head_count", status.get("ppe_head_count")) or 0),
        "ppe_missing_helmet_count": int(status.get("ppe_missing_helmet_count") or 0),
        "ppe_has_person_class": bool(status.get("ppe_has_person_class", False)),
        "ppe_evidence_mode": str(status.get("ppe_evidence_mode") or ""),
        "ppe_uncertain": bool(status.get("ppe_uncertain", False)),
        "ppe_reason": str(status.get("ppe_reason") or ""),
        "ppe_source_auth_media_suppressed": bool(status.get("ppe_source_auth_media_suppressed", False)),
        "ppe_source_auth_temporal_reset": bool(status.get("ppe_source_auth_temporal_reset", False)),
        "ppe_source_auth_media_bbox": status.get("ppe_source_auth_media_bbox"),
        "ppe_source_auth_media_suppressed_count": int(status.get("ppe_source_auth_media_suppressed_count") or 0),
        "ppe_source_auth_media_suppressed_person_count": int(
            status.get("ppe_source_auth_media_suppressed_person_count") or 0
        ),
        "ppe_source_auth_media_suppressed_head_count": int(
            status.get("ppe_source_auth_media_suppressed_head_count") or 0
        ),
        "ppe_source_auth_media_suppressed_helmet_count": int(
            status.get("ppe_source_auth_media_suppressed_helmet_count") or 0
        ),
        "ppe_source_auth_media_suppression_reason": str(
            status.get("ppe_source_auth_media_suppression_reason") or ""
        ),
        "ppe_tracks": [dict(track) for track in ppe_tracks],
        "timing_ms": float(status.get("timing_ms") or 0.0),
        "processing_ms": float(status.get("processing_ms") or 0.0),
        "detector_inference_ms": float(status.get("detector_inference_ms") or 0.0),
        "module_a_timing_ms": float(status.get("module_a_timing_ms") or 0.0),
        "display_options": dict(status.get("display_options") or display_options),
        "branch_cards": [dict(card) for card in status.get("branch_cards", []) or []],
        "a3b_score": float(status.get("a3b_score") or status.get("a3b_confidence") or 0.0),
        "a3b_confidence": float(status.get("a3b_confidence") or status.get("a3b_confirmed_score") or 0.0),
        "a3b_observed_score": float(status.get("a3b_observed_score") or 0.0),
        "a3b_confirmed_score": float(status.get("a3b_confirmed_score") or 0.0),
        "a3b_display_score": float(status.get("a3b_display_score") or status.get("a3b_confidence") or status.get("a3b_confirmed_score") or 0.0),
        "a3b_event_score": float(status.get("a3b_event_score") or status.get("a3b_confidence") or status.get("a3b_confirmed_score") or status.get("a3b_observed_score") or 0.0),
        "a3b_state": str(status.get("a3b_state") or "normal"),
        "a3b_triggered": bool(status.get("a3b_triggered")),
        "a3b_p_media": float(status.get("a3b_p_media") or 0.0),
        "a3b_bbox": status.get("a3b_bbox"),
        "a3b_triggered_source": str(status.get("a3b_triggered_source") or "none"),
        "a3b_reason": str(status.get("a3b_reason") or ""),
        "a3b_debug": dict(status.get("a3b_debug") or {}),
        "raw_boxes_count": int(status.get("raw_boxes_count") or 0),
        "ppe_boxes_count": int(status.get("ppe_boxes_count") or 0),
        "tracked_boxes_count": int(status.get("tracked_boxes_count") or 0),
        "render_boxes_count": int(status.get("render_boxes_count") or len(ppe_tracks)),
        "overlay_match_window_ms": float(status.get("overlay_match_window_ms") or 180.0),
        "overlay_hold_ms": float(status.get("overlay_hold_ms") or 550.0),
        "overlay_interpolate_ms": float(status.get("overlay_interpolate_ms") or 400.0),
        "overlay_max_age_ms": float(status.get("overlay_max_age_ms") or 950.0),
    }
