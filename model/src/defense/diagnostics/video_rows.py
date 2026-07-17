from __future__ import annotations

from pathlib import Path
from typing import Any

from defense.module_a.result_contract import adapt_a3b_result
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState

FIELDNAMES = [
    "video", "frame_idx", "alert_confirmed", "single_frame_suspicious", "attack_state_active",
    "p_adv", "p_adv_display", "p_media", "a3b_observed_score", "a3b_confirmed_score",
    "a3b_display_score", "a3b_triggered", "a3b_source", "a3b_reason", "overexposure_ratio",
    "is_glare", "temporal_change", "temporal_local_max", "motion_score", "flow_local_ratio",
    "blur_score", "track_score", "confidence_drop_score", "timing_ms", "reason_codes",
]


def compact_reason_counts(reason_counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])))


def alert_ranges(alert_frames: list[int]) -> list[dict[str, int]]:
    if not alert_frames:
        return []
    ranges: list[dict[str, int]] = []
    start = prev = int(alert_frames[0])
    for frame_idx in alert_frames[1:]:
        frame_idx = int(frame_idx)
        if frame_idx == prev + 1:
            prev = frame_idx
            continue
        ranges.append({"start": start, "end": prev, "count": prev - start + 1})
        start = prev = frame_idx
    ranges.append({"start": start, "end": prev, "count": prev - start + 1})
    return ranges


def frame_row(video_path: Path, frame_idx: int, info: dict[str, Any], a3b_state: A3BSoftTriggerState | None = None) -> dict[str, Any]:
    details = info.get("details", {})
    features = details.get("module_a_features", {})
    module_a = details.get("module_a", {})
    overexposure = features.get("overexposure", {})
    temporal = features.get("temporal", {})
    flow = features.get("flow", {})
    blur = features.get("blur", {})
    track = features.get("track", {})
    static_image = adapt_a3b_result(info)
    if not static_image:
        static_image = dict(features.get("static_image", {}))
    static_image["source_path"] = str(video_path)
    a3b_soft = a3b_state.update(static_image) if a3b_state is not None else None
    raw_p_media = float(static_image.get("p_media", 0.0) or 0.0)
    observed_score = float(
        (a3b_soft or {}).get(
            "observed_score",
            max(
                raw_p_media,
                float(static_image.get("live_score", 0.0) or 0.0),
                float(static_image.get("live_score_display", 0.0) or 0.0),
                float(static_image.get("classifier_score", 0.0) or 0.0),
            ),
        ) or 0.0
    )
    confirmed_score = float((a3b_soft or {}).get("confirmed_score", raw_p_media if static_image.get("triggered") else 0.0) or 0.0)
    display_score = float((a3b_soft or {}).get("display_score", max(raw_p_media, observed_score, confirmed_score * 0.8)) or 0.0)
    a3b_triggered = bool((a3b_soft or {}).get("triggered", static_image.get("triggered", False)))
    a3b_source = str((a3b_soft or {}).get("triggered_source", static_image.get("triggered_source", "none")))
    a3b_reason = str((a3b_soft or {}).get("reason", a3b_source if a3b_triggered else "none"))
    return {
        "video": video_path.name,
        "frame_idx": frame_idx,
        "alert_confirmed": bool(info.get("alert_confirmed", False)),
        "single_frame_suspicious": bool(module_a.get("single_frame_suspicious", False)),
        "attack_state_active": bool(info.get("attack_state_active", False)),
        "p_adv": float(info.get("p_adv", 0.0) or 0.0),
        "p_adv_display": float(info.get("p_adv_display", info.get("p_adv", 0.0)) or 0.0),
        "p_media": raw_p_media,
        "a3b_observed_score": observed_score,
        "a3b_confirmed_score": confirmed_score,
        "a3b_display_score": display_score,
        "a3b_triggered": a3b_triggered,
        "a3b_source": a3b_source,
        "a3b_reason": a3b_reason,
        "overexposure_ratio": float(overexposure.get("ratio", 0.0) or 0.0),
        "is_glare": bool(overexposure.get("is_glare", False)),
        "temporal_change": float(temporal.get("change_rate", 0.0) or 0.0),
        "temporal_local_max": float(temporal.get("local_max", 0.0) or 0.0),
        "motion_score": float(flow.get("motion_score", 0.0) or 0.0),
        "flow_local_ratio": float(flow.get("local_max_ratio", 0.0) or 0.0),
        "blur_score": float(blur.get("score", 0.0) or 0.0),
        "track_score": float(
            track.get(
                "score",
                static_image.get(
                    "track_score",
                    (static_image.get("p_media_scores") or {}).get("track", 0.0),
                ),
            )
            or 0.0
        ),
        "confidence_drop_score": float(track.get("confidence_drop_score", 0.0) or 0.0),
        "timing_ms": float(info.get("timing_ms", 0.0) or 0.0),
        "reason_codes": ";".join(str(code) for code in info.get("reason_codes", [])),
    }


def is_interesting(row: dict[str, Any]) -> bool:
    return bool(row["alert_confirmed"] or row["single_frame_suspicious"] or row["a3b_triggered"] or row["is_glare"] or row["reason_codes"])
