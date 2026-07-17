from __future__ import annotations

from typing import Any


_NO_SUPPRESSION_REASONS = {"", "none", "not_computed"}
_BORDER_SUPPRESSION_REASONS = {"border_or_letterbox"}
_CAMERA_SUPPRESSION_REASONS = {
    "background_window_camera_motion",
    "camera_translation_edge",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bbox(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return list(value)


def _suppression_state(
    *,
    active: bool,
    reason: str,
    score_cap: float,
    policy_score: float,
    candidate_allowed: bool,
) -> dict[str, Any]:
    return {
        "suppressed": bool(active),
        "reason": reason if active else "none",
        "score_cap": float(score_cap),
        "p_media_policy": float(policy_score),
        "media_candidate_allowed": bool(candidate_allowed),
    }


def adapt_a3b_result(info: dict[str, Any]) -> dict[str, Any]:
    """Return the shared legacy-shaped A3b result without losing backend fields.

    Legacy ``details.module_a_features.static_media`` is copied unchanged.
    Rebuilt ``details.a3b`` keeps every native field and gains only the aliases
    consumed by runtime status, soft confirmation, diagnostics, and bbox
    suppression.
    """

    details = info.get("details", {})
    if not isinstance(details, dict):
        return {}

    module_a_features = details.get("module_a_features", {})
    if isinstance(module_a_features, dict):
        legacy = module_a_features.get("static_media", {})
        if isinstance(legacy, dict) and legacy:
            return dict(legacy)

    rebuilt = details.get("a3b", {})
    if not isinstance(rebuilt, dict) or not rebuilt:
        return {}

    result = dict(rebuilt)
    result.setdefault("result_contract_source", "rebuilt")
    candidates = result.get("media_candidates")
    if not isinstance(candidates, (list, tuple)):
        candidates = []
        result["media_candidates"] = candidates

    scores = result.get("p_media_scores")
    if not isinstance(scores, dict):
        scores = {}
        result["p_media_scores"] = scores

    bbox = _bbox(result.get("p_media_bbox"))
    p_media = _float(result.get("p_media", result.get("p_media_policy")))
    policy_score = _float(result.get("p_media_policy"), p_media)
    confirmed_score = _float(result.get("p_media_confirmed_score"))
    display_score = _float(result.get("a3b_display_score"), p_media)
    policy_triggered = bool(result.get("p_media_triggered", False))
    confirmed = bool(result.get("media_confirmed", result.get("confirmed", False)))
    candidate_count = _int(
        result.get(
            "p_media_candidate_count",
            result.get("candidate_count", len(candidates)),
        )
    )
    strong_evidence = bool(
        result.get("p_media_strong_evidence", result.get("strong_evidence", False))
    )

    suppression = result.get("suppression")
    suppression = dict(suppression) if isinstance(suppression, dict) else {}
    suppressed_reason = str(
        result.get("suppressed_reason", suppression.get("reason", "none")) or "none"
    )
    score_cap = _float(result.get("score_cap", suppression.get("score_cap")), 1.0)
    candidate_allowed = bool(
        result.get(
            "media_candidate_allowed",
            suppression.get("media_candidate_allowed", policy_triggered),
        )
    )
    background_suppressed = bool(
        result.get(
            "p_media_background_static_suppressed",
            result.get(
                "background_static_suppressed",
                suppression.get("background_static_suppressed", False),
            ),
        )
    )
    policy_suppressed = bool(
        suppressed_reason.lower() not in _NO_SUPPRESSION_REASONS
        or background_suppressed
    )
    state = str(
        result.get("a3b_state")
        or result.get("state")
        or ("confirmed" if confirmed else "candidate" if policy_triggered else "normal")
    )
    if confirmed:
        state = "confirmed"

    result.setdefault("score", confirmed_score)
    result.setdefault("live_score", p_media)
    result.setdefault("live_score_display", display_score)
    result.setdefault("static_image_score", policy_score)
    result.setdefault("p_media", p_media)
    result.setdefault("p_media_policy", policy_score)
    result.setdefault("p_media_triggered", policy_triggered)
    result["media_confirmed"] = confirmed
    result["confirmed"] = confirmed
    result["triggered"] = confirmed
    result["static_image_triggered"] = confirmed
    result.setdefault("p_media_candidate_count", candidate_count)
    result.setdefault("candidate_count", candidate_count)
    result.setdefault("p_media_strong_evidence", strong_evidence)
    result.setdefault("strong_evidence", strong_evidence)
    result.setdefault("p_media_background_static_suppressed", background_suppressed)
    result.setdefault("background_static_suppressed", background_suppressed)
    result["a3b_state"] = state
    result["state"] = state
    result.setdefault("bbox", bbox)
    result.setdefault("candidate_bbox", bbox)
    result["p_media_bbox"] = bbox

    policy = result.get("policy")
    policy = dict(policy) if isinstance(policy, dict) else {}
    policy.setdefault("p_media_raw", _float(result.get("p_media_raw")))
    policy.setdefault("p_media_policy", policy_score)
    policy.setdefault("p_media_triggered", policy_triggered)
    policy.setdefault("media_candidate_allowed", candidate_allowed)
    policy.setdefault("suppressed", policy_suppressed)
    policy.setdefault("suppressed_reason", suppressed_reason)
    policy.setdefault("score_cap", score_cap)
    policy.setdefault("background_static_suppressed", background_suppressed)
    result["policy"] = policy

    suppression.setdefault("suppressed", policy_suppressed)
    suppression.setdefault("reason", suppressed_reason)
    suppression.setdefault("score_cap", score_cap)
    suppression.setdefault("p_media_policy", policy_score)
    suppression.setdefault("media_candidate_allowed", candidate_allowed)
    suppression.setdefault("background_static_suppressed", background_suppressed)
    result["suppression"] = suppression
    result.setdefault(
        "p_media_policy_state",
        _suppression_state(
            active=policy_suppressed,
            reason=suppressed_reason,
            score_cap=score_cap,
            policy_score=policy_score,
            candidate_allowed=candidate_allowed,
        ),
    )

    reason_key = suppressed_reason.lower()
    if reason_key in _BORDER_SUPPRESSION_REASONS:
        result.setdefault(
            "p_media_border_state",
            _suppression_state(
                active=True,
                reason=suppressed_reason,
                score_cap=score_cap,
                policy_score=policy_score,
                candidate_allowed=candidate_allowed,
            ),
        )
    if reason_key in _CAMERA_SUPPRESSION_REASONS:
        result.setdefault(
            "p_media_camera_motion_state",
            _suppression_state(
                active=True,
                reason=suppressed_reason,
                score_cap=score_cap,
                policy_score=policy_score,
                candidate_allowed=candidate_allowed,
            ),
        )
    if (
        "target_attached" in reason_key
        or "glare_or_texture" in reason_key
        or "adv_explained" in reason_key
    ):
        result.setdefault(
            "p_media_physical_motion_state",
            _suppression_state(
                active=True,
                reason=suppressed_reason,
                score_cap=score_cap,
                policy_score=policy_score,
                candidate_allowed=candidate_allowed,
            ),
        )

    return result
