from __future__ import annotations

import time
from typing import Any

import cv2

from .classifier_features import (
    build_classifier_features,
    build_static_media_classifier_features,
)
from .types import ModuleAInput, ModuleAResult


def _select_fusion_p_adv(
    fusion_backend: str,
    *,
    rule_p_adv: float,
    classifier_result: dict[str, Any] | None,
) -> float:
    classifier_p_adv = float((classifier_result or {}).get("classifier_p_adv", 0.0))
    if fusion_backend == "classifier":
        return classifier_p_adv
    if fusion_backend == "rule_or_classifier":
        return max(float(rule_p_adv), classifier_p_adv)
    return float(rule_p_adv)


def _select_fusion_suspicious(
    fusion_backend: str,
    *,
    rule_suspicious: bool,
    classifier_result: dict[str, Any] | None,
) -> bool:
    classifier_suspicious = bool((classifier_result or {}).get("classifier_triggered", False))
    if fusion_backend == "classifier":
        return classifier_suspicious
    if fusion_backend == "rule_or_classifier":
        return bool(rule_suspicious or classifier_suspicious)
    return bool(rule_suspicious)


def process_module_a(detector: Any, item: ModuleAInput) -> ModuleAResult:
    """Run one Module A frame through A1, A2, A3, A3b, and A4.

    The logic is intentionally behavior-preserving relative to the original
    detector, but it now lives outside the detector class so the class is an
    orchestrator shell rather than a God Object.
    """
    started = time.perf_counter()
    frame = item.frame
    if frame.shape[0] != detector.frame_size or frame.shape[1] != detector.frame_size:
        frame = cv2.resize(frame, (detector.frame_size, detector.frame_size))

    rois = detector._prepare_rois(item.rois, frame.shape[1], frame.shape[0])
    gray = detector._frame_to_gray_tensor(frame)

    # --- Per-feature timing instrumentation (Requirements 5.1/5.2/5.3/5.6) ---
    # Each block wraps exactly one logical feature using ``time.perf_counter``.
    # We do **not** call ``torch.cuda.synchronize`` between blocks: kernels
    # are enqueued asynchronously on the CUDA stream, so these numbers are
    # "host-side launch window" approximations. Accepting this approximation
    # is a deliberate latency/realtime trade-off (design §3 / tasks §3.1).
    a1_overexposure_ms = 0.0
    a2_temporal_ms = 0.0
    a3_motion_ms = 0.0
    a3b_static_media_ms = 0.0
    a4_fusion_ms = 0.0

    # A1 — overexposure
    _t0 = time.perf_counter()
    overexposure = detector.overexposure.compute(gray)
    if (
        bool(overexposure.get("is_glare", False))
        or float(overexposure.get("ratio", 0.0)) >= detector.glare_ratio_threshold
    ):
        detector.a3b_glare_suppress_remaining = detector.a3b_glare_suppress_frames
        detector.glare_hold_remaining = detector.glare_hold_frames
    elif detector.a3b_glare_suppress_remaining > 0:
        detector.a3b_glare_suppress_remaining -= 1
    elif detector.glare_hold_remaining > 0:
        detector.glare_hold_remaining -= 1
    if detector.a3b_physical_suppress_remaining > 0:
        detector.a3b_physical_suppress_remaining -= 1
    detector._sync_if_profile()
    a1_overexposure_ms = (time.perf_counter() - _t0) * 1000.0

    # A2 — LBP texture + temporal texture
    _t0 = time.perf_counter()
    lbp = detector.texture.compute_lbp(gray)
    texture = detector.texture.summarize(lbp, rois)
    temporal = detector.temporal.compute(detector.prev_lbp, lbp, rois, radius=detector.texture.radius)
    detector._sync_if_profile()
    a2_temporal_ms = (time.perf_counter() - _t0) * 1000.0

    # A3 — motion artifact + blur + track + light-flow (+ merge)
    _t0 = time.perf_counter()
    motion = detector.motion.compute(detector.prev_gray, gray, rois)
    blur = detector.blur.compute(gray, rois)
    track = detector.track.compute(rois)
    light_flow = detector.light_flow.compute(
        detector.prev_gray,
        gray,
        rois,
        run=detector._should_run_light_flow(item.frame_idx, temporal),
    )
    # Hold-last-value for light_flow: when skipped, carry forward the
    # last computed score so p_adv / display doesn't flicker.
    if light_flow.get("light_flow_available", False):
        detector._last_light_flow_score = float(light_flow.get("light_flow_score", 0.0))
        detector._last_light_flow_ratio = float(
            light_flow.get("light_flow_local_anomaly_ratio", 0.0)
        )
    else:
        light_flow["light_flow_score"] = getattr(detector, "_last_light_flow_score", 0.0)
        light_flow["light_flow_local_anomaly_ratio"] = getattr(
            detector, "_last_light_flow_ratio", 0.0
        )
    motion = detector._merge_light_flow(motion, light_flow)
    detector._sync_if_profile()
    a3_motion_ms = (time.perf_counter() - _t0) * 1000.0

    # A3b — static media spoof
    # Gating logic: per-ROI patch comparison is expensive, so the
    # ``_should_run_static_image`` gate decides whether to include
    # the ROI pass. Passing ``rois=None`` still lets
    # ``GPUStaticMediaSpoofDetector.compute`` return the empty
    # defaults via ``_empty()`` while the A3+ candidate path
    # (L0→L1→L2) continues to run independently.
    #
    # Display continuity (2026-05-13): on non-run frames we now
    # carry forward the LAST computed static_image_score and
    # static_image_triggered state so the front-end sees a smooth
    # confidence curve instead of "0 → score → 0 → score" flicker.
    run_roi_pass = False
    if detector.static_image_enabled and detector.prev_gray is not None:
        _t0 = time.perf_counter()
        # Dynamic interval: when hold_score > high_score_threshold, force every frame
        # (2026-06-11 架构修复)
        _effective_interval = (
            1 if (detector.static_image_dynamic_interval_enabled
                  and detector.static_image_hold_score >= detector.static_image_high_score_threshold)
            else detector.static_image_interval
        )
        run_roi_pass = detector._should_run_static_image(
            item.frame_idx, temporal, effective_interval=_effective_interval
        )
        static_image = detector.static_image.compute(
            detector.prev_gray,
            gray,
            rois if run_roi_pass else None,
            context_rois=rois,
        )
        # Save raw p_media before suppression chain caps it (2026-06-11 修复)
        _raw_p_media = float(static_image.get("p_media", 0.0))
        legacy_static_triggered = bool(static_image.get("static_image_triggered", False))
        if detector.a3b_glare_suppress_remaining > 0 and (
            bool(static_image.get("static_image_triggered", False))
            or float(static_image.get("p_media", 0.0)) >= 0.42
        ):
            static_image["p_media_triggered"] = False
            static_image["static_image_triggered"] = False
            static_image["p_media"] = min(float(static_image.get("p_media", 0.0)), 0.40)
            static_image["static_image_score"] = min(
                float(static_image.get("static_image_score", 0.0)), 0.40
            )
            static_image["static_image_triggered_source"] = "physical_glare_suppressed"
        # Hold-last-value: when the ROI pass didn't run, the detector
        # returns zeros. Replace with the held full state so downstream
        # A3BSoftTrigger quality_gate / screen_cue / strong_media
        # survive interval skips (2026-06-11 架构修复).
        if not run_roi_pass and detector.static_image_hold_carryover_enabled and detector.static_image_hold_state:
            # Carry forward ALL fields that A3BSoftTrigger.quality_gate depends on
            for _key in ("p_media", "p_media_triggered", "p_media_scores",
                         "p_media_candidate_count", "p_media_bbox",
                         "p_media_strong_evidence", "p_media_replay_state",
                         "p_media_fast_state", "p_media_occlusion_state",
                         "line_score", "screen_or_paper_like",
                         "static_image_score", "static_image_triggered",
                         "static_image_triggered_source", "classifier_score",
                         "classifier_triggered"):
                if _key in detector.static_image_hold_state:
                    static_image[_key] = detector.static_image_hold_state[_key]
            # Replace at-rest score with the held score so downstream
            # fusion / display stays continuous.
            if "static_image_score" not in detector.static_image_hold_state:
                static_image["static_image_score"] = detector.static_image_hold_score
        elif run_roi_pass:
            # Update the held state from the fresh computation.
            detector.static_image_hold_score = float(
                static_image.get("static_image_score", 0.0)
            )
            # Save the full carryover snapshot
            if detector.static_image_hold_carryover_enabled:
                carry_keys = ("p_media", "p_media_triggered", "p_media_scores",
                              "p_media_candidate_count", "p_media_bbox",
                              "p_media_strong_evidence", "p_media_replay_state",
                              "p_media_fast_state", "p_media_occlusion_state",
                              "line_score", "screen_or_paper_like",
                              "static_image_score", "static_image_triggered",
                              "static_image_triggered_source", "classifier_score",
                              "classifier_triggered")
                detector.static_image_hold_state.clear()
                for _key in carry_keys:
                    if _key in static_image:
                        detector.static_image_hold_state[_key] = static_image[_key]
        if legacy_static_triggered and not detector.static_media_legacy_direct_alert_enabled:
            static_image["static_image_triggered"] = False
            static_image["p_media_triggered"] = False
            static_image["static_image_triggered_source"] = "legacy_observed"
        fast_state = detector._update_static_media_fast_state(static_image)
        replay_state = detector._update_static_media_replay_state(
            static_image, temporal, blur
        )
        border_state = detector._static_media_border_suppression(static_image)
        static_image["p_media_border_state"] = border_state
        if border_state["suppressed"]:
            replay_state["triggered"] = False
            replay_state["candidate"] = False
            replay_state["suppressed_by_border"] = True
            fast_state["triggered"] = False
            fast_state["candidate"] = False
            fast_state["suppressed_by_border"] = True
            static_image["p_media_triggered"] = False
            static_image["static_image_triggered"] = False
            static_image["static_image_score"] = min(
                float(static_image.get("static_image_score", 0.0)),
                float(border_state["score_cap"]),
            )
            static_image["static_image_triggered_source"] = "border_or_letterbox_suppressed"
        camera_motion_state = detector._static_media_camera_motion_suppression(
            static_image,
            motion,
        )
        static_image["p_media_camera_motion_state"] = camera_motion_state
        if camera_motion_state["suppressed"]:
            replay_state["triggered"] = False
            replay_state["candidate"] = False
            replay_state["suppressed_by_camera_motion"] = True
            fast_state["triggered"] = False
            fast_state["candidate"] = False
            fast_state["suppressed_by_camera_motion"] = True
            static_image["p_media_triggered"] = False
            static_image["static_image_triggered"] = False
            static_image["p_media"] = min(
                float(static_image.get("p_media", 0.0)),
                float(camera_motion_state["score_cap"]),
            )
            static_image["static_image_score"] = min(
                float(static_image.get("static_image_score", 0.0)),
                float(camera_motion_state["score_cap"]),
            )
            static_image["static_image_triggered_source"] = "camera_motion_suppressed"
        physical_media_state = detector._static_media_physical_motion_suppression(
            static_image,
            motion,
            temporal,
            overexposure,
        )
        static_image["p_media_physical_motion_state"] = physical_media_state
        if physical_media_state["suppressed"]:
            detector.a3b_physical_suppress_remaining = max(
                detector.a3b_physical_suppress_remaining,
                detector.a3b_physical_suppress_frames,
            )
        suppress_by_physical_hold = detector.a3b_physical_suppress_remaining > 0
        if physical_media_state["suppressed"] or suppress_by_physical_hold:
            replay_state["triggered"] = False
            replay_state["candidate"] = False
            replay_state["suppressed_by_physical_motion"] = True
            fast_state["triggered"] = False
            fast_state["candidate"] = False
            fast_state["suppressed_by_physical_motion"] = True
            static_image["p_media_triggered"] = False
            static_image["static_image_triggered"] = False
            static_image["p_media"] = min(
                float(static_image.get("p_media", 0.0)),
                float(physical_media_state["score_cap"]),
            )
            static_image["static_image_score"] = min(
                float(static_image.get("static_image_score", 0.0)),
                float(physical_media_state["score_cap"]),
            )
            static_image["static_image_triggered_source"] = (
                "physical_motion_suppressed"
                if physical_media_state["suppressed"]
                else "physical_motion_hold_suppressed"
            )
            if physical_media_state["suppressed"] and bool(
                physical_media_state.get("target_related", False)
            ):
                motion["physical_media_motion_triggered"] = True
                motion["physical_media_motion_score"] = float(physical_media_state["p_adv"])
        suppressed_by_static_media_policy = bool(
            border_state["suppressed"]
            or camera_motion_state["suppressed"]
            or physical_media_state["suppressed"]
            or suppress_by_physical_hold
        )
        live_score = max(
            float(static_image.get("static_image_score", 0.0)),
            float(static_image.get("p_media", 0.0)),
            float(replay_state.get("p_media", 0.0)),
        )
        detector.static_media_display_score = detector._ema(
            detector.static_media_display_score,
            live_score,
            detector.static_media_display_alpha,
        )
        static_image["static_image_live_score_raw"] = live_score
        static_image["static_image_live_score_display"] = float(detector.static_media_display_score)
        static_image["static_image_live_score"] = float(detector.static_media_display_score)
        static_image["p_media_replay_state"] = replay_state
        static_image["p_media_fast_state"] = fast_state
        if fast_state["triggered"]:
            static_image["static_image_triggered"] = True
            static_image["static_image_score"] = max(
                float(static_image.get("static_image_score", 0.0)),
                float(fast_state["p_media"]),
            )
            static_image["static_image_triggered_source"] = "a3_plus_fast"
        if replay_state["triggered"]:
            static_image["static_image_triggered"] = True
            static_image["static_image_score"] = max(
                float(static_image.get("static_image_score", 0.0)),
                float(replay_state["p_media"]),
            )
            static_image["static_image_triggered_source"] = "a3_plus_replay"
        occlusion_state = detector._update_static_media_occlusion_state(
            static_image, replay_state, fast_state
        )
        if suppressed_by_static_media_policy:
            occlusion_state["active"] = False
            occlusion_state["suppressed_by_policy"] = True
        static_image["p_media_occlusion_state"] = occlusion_state
        if occlusion_state["active"] and not (
            fast_state["triggered"] or replay_state["triggered"]
        ):
            static_image["static_image_triggered"] = True
            static_image["static_image_score"] = max(
                float(static_image.get("static_image_score", 0.0)),
                float(occlusion_state["score"]),
            )
            static_image["static_image_triggered_source"] = "a3_plus_occlusion_hold"
        # High-score bypass: if raw p_media is high enough, trust it immediately
        # without waiting for replay/fast/occlusion state machines. This is the
        # single biggest latency reducer for A3b (2026-06-11).
        # NOTE: uses _raw_p_media saved BEFORE suppression caps it.
        if (
            not suppressed_by_static_media_policy
            and not static_image.get("static_image_triggered", False)
            and _raw_p_media >= detector.a3b_high_score_bypass_threshold
        ):
            static_image["static_image_triggered"] = True
            static_image["static_image_score"] = max(
                float(static_image.get("static_image_score", 0.0)), _raw_p_media
            )
            static_image["static_image_triggered_source"] = "high_score_bypass"
        motion = detector._merge_static_image(motion, static_image)
        detector._sync_if_profile()
        a3b_static_media_ms = (time.perf_counter() - _t0) * 1000.0

    # A3b-classifier — Static_Media_Classifier scoring (Task 5.4 / Req 7.3+7.5)
    # Runs every frame when the artifact is configured so that
    # ``classifier_score`` is always available for event evidence / offline
    # replay, regardless of whether the rollout gate is open. The OR
    # combination with the existing heuristic only kicks in when
    # ``static_media_classifier_enabled`` is True (Req 6.3 hard rule).
    # Timing is folded into the A3b bucket so ``module_a_breakdown`` stays
    # in the 6-field contract established by Task 3.1.
    if detector.static_media_classifier is not None:
        _t0 = time.perf_counter()
        classifier_features = build_static_media_classifier_features(motion)
        sm_classifier = detector.static_media_classifier.compute(classifier_features)

        classifier_p_adv = float(sm_classifier["classifier_p_adv"])
        classifier_triggered = bool(sm_classifier["classifier_triggered"])
        motion["static_image_classifier_score"] = classifier_p_adv
        motion["static_image_classifier_triggered"] = classifier_triggered
        motion["static_image_classifier_threshold"] = float(
            sm_classifier["classifier_threshold"]
        )
        motion["static_image_classifier_artifact"] = str(sm_classifier["classifier_artifact"])
        motion["static_image_classifier_kind"] = str(sm_classifier["classifier_kind"])
        motion["static_image_classifier_enabled"] = detector.static_media_classifier_enabled

        # OR semantics — only when the gate is open does the classifier
        # actually push ``static_image_triggered`` / ``static_image_score``.
        if detector.static_media_classifier_enabled and classifier_triggered:
            motion["static_image_triggered"] = True
            # Lift the rule-fusion-visible score to at least the classifier
            # probability so downstream ``static_image_score_trigger``
            # interpretation remains monotonic with the combined signal.
            motion["static_image_score"] = max(
                float(motion.get("static_image_score", 0.0)),
                classifier_p_adv,
            )
            # Remember that the trigger came at least partially from the
            # classifier so event evidence can attribute it correctly.
            motion["static_image_classifier_forced_trigger"] = True
        else:
            motion["static_image_classifier_forced_trigger"] = False

        detector._sync_if_profile()
        a3b_static_media_ms += (time.perf_counter() - _t0) * 1000.0

        # Re-update carryover after classifier scoring modifies motion fields
        # so non-exec frames see the classifier-enhanced state (2026-06-11).
        if run_roi_pass and detector.static_image_hold_carryover_enabled:
            for _key in ("static_image_classifier_score", "static_image_classifier_triggered",
                         "static_image_classifier_forced_trigger"):
                if _key in motion:
                    detector.static_image_hold_state[_key] = motion[_key]

    source_auth: dict[str, Any] = {}

    # A4 — Target-anchored suspicious判定 + rule fusion + classifier
    # ================================================================
    # 2026-05-13 重写：suspicious 判定从"全图统计量驱动"改为
    # "目标锚点驱动"。参考 doc/A3_target_anchored_false_positive_suppression.txt
    _a4_t0 = time.perf_counter()
    fusion = detector.fusion.compute(texture, temporal, motion, overexposure, blur, track)

    # A4 classifier score is always retained for diagnostics. The configured
    # fusion backend selects the authoritative p_adv/display signal here and
    # the authoritative suspicious signal after the rule-side hold logic.
    classifier_result = None
    rule_p_adv = float(fusion.get("p_adv", 0.0))
    if detector.classifier_fusion is not None:
        classifier_features = build_classifier_features(
            overexposure=overexposure,
            texture=texture,
            temporal=temporal,
            motion=motion,
            blur=blur,
            track=track,
            fusion=fusion,
            roi_count=len(rois),
        )
        classifier_result = detector.classifier_fusion.compute(classifier_features)
        fusion.update(classifier_result)
        fusion["p_adv_raw"] = rule_p_adv
    selected_p_adv = _select_fusion_p_adv(
        detector.fusion_backend,
        rule_p_adv=rule_p_adv,
        classifier_result=classifier_result,
    )
    fusion["fusion_backend"] = detector.fusion_backend
    fusion["rule_p_adv"] = rule_p_adv
    fusion["selected_p_adv"] = selected_p_adv
    fusion["p_adv"] = selected_p_adv
    detector.p_adv_display_score = detector._ema(
        detector.p_adv_display_score,
        selected_p_adv,
        detector.p_adv_display_alpha,
    )
    detector._p_adv_display_hold.append(float(detector.p_adv_display_score))
    p_adv_display_smoothed = float(sorted(detector._p_adv_display_hold)[len(detector._p_adv_display_hold) // 2])
    fusion["p_adv_display"] = p_adv_display_smoothed

    # --- Target-anchored 判定（核心改动）---
    # 构建 static_image 信息供 target_anchored 使用
    static_image_info = {
        "triggered": bool(motion.get("static_image_triggered", False)),
        "score": float(motion.get("static_image_score", 0.0)),
    }
    anchored = detector.target_anchored.evaluate(
        rois=rois,
        overexposure=overexposure,
        blur=blur,
        track=track,
        temporal=temporal,
        motion=motion,
        static_image=static_image_info,
        classifier_result=classifier_result,
    )
    suspicious = bool(anchored["suspicious"])
    # 合并 reason codes：target_anchored 的 + fusion 里已有的信息性 codes
    reason_codes = list(anchored["reason_codes"])
    # 保留 fusion 里的信息性 codes（不触发 suspicious 但有记录价值）
    for code in fusion.get("reason_codes", []):
        if code not in reason_codes:
            reason_codes.append(code)
    if "natural_exposure_suppressed" in reason_codes:
        # Keep diagnostics honest but avoid presenting normal auto-exposure
        # as an active attack reason in the Web card/event copy.
        suppressed_codes = {
            "overexposure",
            "overexposure_hold",
            "local_blur_degradation",
            "paired_temporal_blur_degradation",
        }
        reason_codes = [code for code in reason_codes if code not in suppressed_codes]
    if bool(motion.get("physical_media_motion_triggered", False)):
        suspicious = True
        rule_p_adv = max(
            rule_p_adv,
            float(motion.get("physical_media_motion_score", detector.physical_media_motion_min_p_adv)),
        )
        fusion["rule_p_adv"] = rule_p_adv
        fusion["selected_p_adv"] = _select_fusion_p_adv(
            detector.fusion_backend,
            rule_p_adv=rule_p_adv,
            classifier_result=classifier_result,
        )
        fusion["p_adv"] = fusion["selected_p_adv"]
        if "physical_patch_motion" not in reason_codes:
            reason_codes.append("physical_patch_motion")
    fusion["reason_codes"] = reason_codes
    fusion["target_anchored"] = anchored

    # static_image hold 逻辑保留（A3b 触发后保持几帧，限制为 target-related 触发以防止背景误报 hold 泄露）
    target_anchored_static_triggered = static_image_info["triggered"] and bool(motion.get("target_related", False))
    if target_anchored_static_triggered:
        detector.static_image_hold_remaining = detector.static_image_hold_frames
        detector.static_image_hold_score = static_image_info["score"]
    elif detector.static_image_hold_remaining > 0:
        detector.static_image_hold_remaining -= 1

    if detector.static_image_hold_remaining > 0 and not suspicious:
        # A3b 之前触发过，hold 期间保持 suspicious
        suspicious = True
        if "static_image_spoof_hold" not in reason_codes:
            reason_codes.append("static_image_spoof_hold")
        fusion["reason_codes"] = reason_codes

    if detector.glare_hold_remaining > 0 and not suspicious:
        if bool(fusion.get("overexposure_triggered", False)) or bool(
            fusion.get("glare_triggered", False)
        ):
            suspicious = True
            if "overexposure_hold" not in reason_codes:
                reason_codes.append("overexposure_hold")
            fusion["reason_codes"] = reason_codes

    blur_score = float(blur.get("blur_score", 0.0))
    temporal_local = float(temporal.get("local_max", 0.0))
    if (
        detector.blur_hold_frames > 0
        and suspicious
        and blur_score >= detector.blur_hold_score_threshold
        and temporal_local >= detector.blur_hold_temporal_threshold
    ):
        detector.blur_hold_remaining = detector.blur_hold_frames
    elif detector.blur_hold_remaining > 0:
        detector.blur_hold_remaining -= 1

    if detector.blur_hold_remaining > 0 and not suspicious:
        suspicious = True
        if "blur_hold" not in reason_codes:
            reason_codes.append("blur_hold")
        fusion["reason_codes"] = reason_codes

    detector._a3b_display_hold.append(1.0 if bool(static_image_info["triggered"]) else 0.0)
    a3b_recent_trigger_ratio = sum(detector._a3b_display_hold) / max(1, len(detector._a3b_display_hold))
    detector.a3b_display_score = detector._ema(
        detector.a3b_display_score,
        max(float(static_image_info["score"]), a3b_recent_trigger_ratio),
        detector.a3b_display_alpha,
    )

    strong_attack_reason = "none"
    if detector.strong_evidence_hold_frames > 0:
        motion_score = float(motion.get("motion_score", 0.0))
        track_score = float(track.get("track_score", 0.0))
        confidence_drop = float(track.get("confidence_drop_score", 0.0))
        p_adv_for_hold = float(fusion.get("p_adv", 0.0))
        strong_track_seed = bool(
            track_score >= detector.strong_evidence_min_track_score
            or confidence_drop >= detector.strong_evidence_min_conf_drop
            or p_adv_for_hold >= detector.strong_evidence_min_p_adv
        )
        has_target_track_drop = "target_track_consistency_drop" in reason_codes
        has_target_motion = "target_motion_temporal_anomaly" in reason_codes
        has_motion_artifact = "motion_artifact" in reason_codes
        has_temporal_texture = "local_temporal_texture_change" in reason_codes
        strong_attack_triggered = bool(
            (
                has_target_track_drop
                and has_target_motion
                and (motion_score >= 0.5 or temporal_local >= detector.strong_temporal_trigger)
                and strong_track_seed
            )
            or (
                has_motion_artifact
                and has_temporal_texture
                and temporal_local >= detector.strong_local_temporal_trigger
                and p_adv_for_hold >= detector.strong_evidence_min_p_adv
            )
        )
        if strong_attack_triggered:
            detector.strong_evidence_hold_remaining = detector.strong_evidence_hold_frames
            detector.strong_evidence_hold_score = max(
                detector.strong_evidence_hold_score,
                float(fusion.get("p_adv", 0.0)),
            )
            if has_target_track_drop and has_target_motion:
                strong_attack_reason = "target_track_motion_hold"
            elif has_motion_artifact and has_temporal_texture:
                strong_attack_reason = "temporal_motion_hold"
            else:
                strong_attack_reason = "strong_evidence_hold"
        elif detector.strong_evidence_hold_remaining > 0:
            detector.strong_evidence_hold_remaining -= 1

        if detector.strong_evidence_hold_remaining > 0 and not suspicious:
            suspicious = True
            if strong_attack_reason == "none":
                strong_attack_reason = "strong_evidence_hold"
            if strong_attack_reason not in reason_codes:
                reason_codes.append(strong_attack_reason)
            fusion["reason_codes"] = reason_codes

    rule_suspicious = suspicious
    classifier_suspicious = bool(
        (classifier_result or {}).get("classifier_triggered", False)
    )
    suspicious = _select_fusion_suspicious(
        detector.fusion_backend,
        rule_suspicious=rule_suspicious,
        classifier_result=classifier_result,
    )
    if (
        classifier_suspicious
        and detector.fusion_backend in {"classifier", "rule_or_classifier"}
        and "classifier_fusion" not in reason_codes
    ):
        reason_codes.append("classifier_fusion")
    fusion["reason_codes"] = reason_codes
    fusion["rule_suspicious"] = rule_suspicious
    fusion["classifier_suspicious"] = classifier_suspicious
    fusion["is_suspicious"] = suspicious
    detector._sync_if_profile()
    a4_fusion_ms = (time.perf_counter() - _a4_t0) * 1000.0

    # Compute the p_adv alert state *before* suppressing Source_Authenticity:
    # suppression must be keyed on the confirmed/holdover state machine output
    # (Requirement 1.4/1.5), not on the raw per-frame ``suspicious`` flag.
    #
    # Streaming path: when the caller (VideoDefensePipeline.process_envelope)
    # propagates ``ModuleAInput.timestamp`` from ``FrameEnvelope.source_ts``,
    # we feed it to ``AlertState.update`` so the 3/5 window honors a real
    # time-span tolerance (Requirement 2.6).  Offline MP4 path keeps
    # ``timestamp == 0.0`` -> ``frame_ts=None`` so legacy bit-for-bit
    # equivalence is preserved (Requirement 11 / offline regression).
    alert_frame_ts = item.timestamp if item.timestamp > 0.0 else None
    alert_confirmed, attack_state_active = detector.alert_state.update(
        suspicious,
        frame_ts=alert_frame_ts,
        intensity=float(fusion.get("p_adv", 0.0)),
    )
    detector.prev_gray = gray.detach()
    detector.prev_lbp = lbp.detach()
    detector.frame_idx = item.frame_idx + 1

    timing_ms = (time.perf_counter() - started) * 1000.0
    roi_results = detector._merge_roi_results(texture, temporal, motion)
    features = {
        "delta_h": float(texture["delta_h"]),
        "texture_local_max": float(texture["local_max"]),
        "change_t": float(temporal["change_t"]),
        "local_change_max": float(temporal["local_max"]),
        "motion_score": float(motion["motion_score"]),
        "flow_region_count": int(motion["region_count"]),
        "flow_max_magnitude": float(motion["max_magnitude"]),
        "flow_local_ratio": float(motion["local_max_ratio"]),
        "light_flow_score": float(motion.get("light_flow_score", 0.0)),
        "light_flow_local_anomaly_ratio": float(
            motion.get("light_flow_local_anomaly_ratio", 0.0)
        ),
        "light_flow_region_count": int(motion.get("light_flow_region_count", 0)),
        "static_image_score": float(motion.get("static_image_score", 0.0)),
        "static_image_trigger_count": int(motion.get("static_image_trigger_count", 0)),
        "static_image_patch_similarity": float(
            motion.get("static_image_patch_similarity", 0.0)
        ),
        "static_image_center_motion": float(motion.get("static_image_center_motion", 0.0)),
        "blur_score": float(blur.get("blur_score", 0.0)),
        "blur_roi_energy_ratio": float(blur.get("blur_roi_energy_ratio", 1.0)),
        "blur_low_energy_ratio": float(blur.get("blur_low_energy_ratio", 0.0)),
        "track_score": float(track.get("track_score", 0.0)),
        "track_drop_score": float(track.get("track_drop_score", 0.0)),
        "track_missing_count": int(track.get("missing_track_count", 0)),
        "confidence_drop_score": float(track.get("confidence_drop_score", 0.0)),
        "overexposure_ratio": float(overexposure["ratio"]),
    }
    details = detector._build_details(
        overexposure=overexposure,
        texture=texture,
        temporal=temporal,
        motion=motion,
        blur=blur,
        track=track,
        fusion=fusion,
        source_auth=source_auth,
        rois=rois,
        roi_results=roi_results,
        frame_idx=item.frame_idx,
    )
    # --- Expose per-feature timing breakdown (Requirements 5.1/5.6) ---
    # Landed on two paths so downstream consumers don't have to care about
    # the layout: the canonical ``module_a_breakdown`` top-level and a
    # convenience copy nested under ``module_a_features`` for parity with
    # the other feature blocks. ``VideoDefensePipeline._run_detection``
    # reads from the top-level copy and forwards it to
    # ``info["latency_breakdown"]["module_a_breakdown"]``.
    module_a_breakdown = {
        "a1_overexposure_ms": float(a1_overexposure_ms),
        "a2_temporal_ms": float(a2_temporal_ms),
        "a3_motion_ms": float(a3_motion_ms),
        "a3b_static_media_ms": float(a3b_static_media_ms),
        "a4_fusion_ms": float(a4_fusion_ms),
        "source_auth_ms": 0.0,
    }
    details["module_a_breakdown"] = module_a_breakdown
    details.setdefault("module_a_features", {})["module_a_breakdown"] = dict(module_a_breakdown)
    return ModuleAResult(
        frame_idx=item.frame_idx,
        p_adv=float(fusion["p_adv"]),
        single_frame_suspicious=suspicious,
        alert_confirmed=alert_confirmed,
        attack_state_active=attack_state_active,
        reason_codes=list(fusion["reason_codes"]),
        features=features,
        roi_results=roi_results,
        attack_mask=None,
        timing_ms=timing_ms,
        details=details,
    )
