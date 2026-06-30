# rebuilt_demo/src/module_a/target_anchored.py
from __future__ import annotations

from collections import deque
from typing import Any


class TargetAnchoredAnalyzer:
    """以目标检测框为锚点的异常判定器。

    设计原则：
      1. 没有目标框 → 不报警（除非全图过曝/全目标突然消失，或者高置信度分类器/局部LBP簇触发兜底）
      2. 只分析目标框内/附近的异常
      3. 可疑区域必须与目标空间相关
      4. 背景静态物（广告牌、柱子、门框）抑制
      5. 连续帧投票由外层 AlertState 负责
    """

    def __init__(
        self,
        # ROI 内异常阈值
        roi_blur_threshold: float = 0.60,
        roi_overexposure_threshold: float = 0.15,
        roi_confidence_drop_threshold: float = 0.25,
        roi_texture_anomaly_threshold: float = 0.12,
        # 轨迹异常阈值
        track_drop_threshold: float = 0.40,
        track_confidence_drop_threshold: float = 0.20,
        motion_score_threshold: float = 0.35,
        light_flow_score_threshold: float = 0.45,
        paired_temporal_motion_threshold: float = 0.18,
        # 背景抑制
        static_position_frames: int = 30,
        static_iou_threshold: float = 0.85,
        # 全图兜底（仅限极端情况）
        allow_global_fallback: bool = True,
        global_fallback_min_prev_targets: int = 2,
        global_fallback_overexposure_threshold: float = 0.20,
        # Flow-local anomaly threshold
        flow_local_anomaly_threshold: float = 0.68,
        # 自然曝光抑制
        natural_exposure_suppression: bool = True,
        natural_exposure_max_ratio: float = 0.18,
        natural_exposure_max_light_flow: float = 0.35,
        natural_exposure_max_motion_score: float = 1.01,
        detector: Any = None,
    ):
        self.roi_blur_threshold = float(roi_blur_threshold)
        self.roi_overexposure_threshold = float(roi_overexposure_threshold)
        self.roi_confidence_drop_threshold = float(roi_confidence_drop_threshold)
        self.roi_texture_anomaly_threshold = float(roi_texture_anomaly_threshold)
        self.track_drop_threshold = float(track_drop_threshold)
        self.track_confidence_drop_threshold = float(track_confidence_drop_threshold)
        self.motion_score_threshold = float(motion_score_threshold)
        self.light_flow_score_threshold = float(light_flow_score_threshold)
        self.paired_temporal_motion_threshold = float(paired_temporal_motion_threshold)
        self.static_position_frames = int(static_position_frames)
        self.static_iou_threshold = float(static_iou_threshold)
        self.allow_global_fallback = bool(allow_global_fallback)
        self.global_fallback_min_prev_targets = int(global_fallback_min_prev_targets)
        self.global_fallback_overexposure_threshold = float(
            global_fallback_overexposure_threshold
        )
        self.natural_exposure_suppression = bool(natural_exposure_suppression)
        self.natural_exposure_max_ratio = float(natural_exposure_max_ratio)
        self.natural_exposure_max_light_flow = float(natural_exposure_max_light_flow)
        self.natural_exposure_max_motion_score = float(natural_exposure_max_motion_score)
        self.flow_local_anomaly_threshold = float(flow_local_anomaly_threshold)
        self.detector = detector
        self._recent_target_counts: deque[int] = deque(maxlen=10)
        self._recent_track_supports: deque[bool] = deque(maxlen=5)
        self._recent_static_triggers: deque[bool] = deque(maxlen=5)
        self._recent_no_target_suspicious: deque[int] = deque(maxlen=5)
        self.glare_active = False
        # 相对基线 EMA（仅在非攻击帧更新），用于区分"自然高亮背景"与"攻击性强光"
        self._ratio_ema: float = 0.0
        self._ratio_ema_ready: bool = False

    def reset(self) -> None:
        self._recent_target_counts.clear()
        self._recent_track_supports.clear()
        self._recent_static_triggers.clear()
        self._recent_no_target_suspicious.clear()
        self.glare_active = False
        self._ratio_ema = 0.0
        self._ratio_ema_ready = False

    def evaluate(
        self,
        rois: list[Any],
        overexposure: dict[str, Any],
        blur: dict[str, Any],
        track: dict[str, Any],
        temporal: dict[str, Any],
        motion: dict[str, Any],
        static_image: dict[str, Any],
        classifier_result: dict[str, Any] | None = None,
        texture: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """每帧调用一次，返回 target-anchored 判定结果。"""
        real_rois = [r for r in rois if getattr(r, "label", "") != "grid"]
        n_targets = len(real_rois)
        self._recent_target_counts.append(n_targets)
        has_targets = n_targets > 0

        reason_codes: list[str] = []
        roi_anomaly_count = 0
        suspicious = False
        global_fallback_fired = False
        classifier_bonus = False

        # --- 强光状态机更新 (Physical Glare Active State Machine) ---
        ratio = float(overexposure.get("ratio", 0.0) or 0.0)
        temporal_flash = bool(overexposure.get("temporal_flash", False))
        overexposure_threshold = float(overexposure.get("threshold", 0.06))
        temporal_local_max = float(temporal.get("local_max", 0.0) if temporal else 0.0)

        # 算法修复：用相对基线（EMA）代替绝对阈值，区分"自然高亮背景"和"攻击性强光"
        # 基线仅在非攻击状态下更新，避免攻击帧污染
        if not self.glare_active:
            alpha = 0.50 if not self._ratio_ema_ready else 0.05
            self._ratio_ema = alpha * ratio + (1.0 - alpha) * self._ratio_ema
            self._ratio_ema_ready = True
        glare_baseline = self._ratio_ema if self._ratio_ema_ready else 0.0

        # 激活条件 1：亮度突然瞬变（闪烁）且有过曝面积
        flash_trigger = temporal_flash and ratio >= overexposure_threshold
        # 激活条件 2：显著高于基线的持续强光（ratio 比基线高 0.20 且绝对值 >= 0.25 且纹理波动明显）
        continuous_glare_trigger = (
            ratio >= glare_baseline + 0.20
            and ratio >= 0.25
            and temporal_local_max >= 0.10
        )

        if (flash_trigger or continuous_glare_trigger) and not self.glare_active:
            self.glare_active = True

        # 维持条件：仍然显著高于基线，不能靠绝对低阈值永久锁死
        if self.glare_active:
            if ratio >= glare_baseline + 0.12 and ratio >= 0.18:
                if "overexposure" not in reason_codes:
                    reason_codes.append("overexposure")
                return {
                    "suspicious": True,
                    "reason_codes": reason_codes,
                    "roi_anomaly_count": n_targets,
                    "has_targets": has_targets,
                    "global_fallback_fired": False,
                    "classifier_bonus": False,
                }
            else:
                # 强光退去，退出激活
                self.glare_active = False

        # Try to retrieve texture from the caller's local variables (process_module_a) if None
        if texture is None:
            try:
                import sys
                frame = sys._getframe(1)
                texture = frame.f_locals.get("texture")
            except Exception:
                pass

        # ============================================================
        # 条件 1：必须有目标锚点 (With Fallback Guard for Blind-out attacks)
        # ============================================================
        if not has_targets:
            raw_suspicious = False
            raw_reason = ""

            # 1. 兜底：前几帧有稳定目标但突然全部消失 + 全图强异常
            if self.allow_global_fallback and self._targets_suddenly_disappeared():
                if self._global_extreme_anomaly(overexposure):
                    raw_suspicious = True
                    raw_reason = "targets_disappeared_with_global_anomaly"

            # 2. No-Target Fallback Guard (YOLO boxes are empty, but classifier or local LBP is high)
            if not raw_suspicious:
                classifier_p_adv = 0.0
                if classifier_result is not None:
                    classifier_p_adv = float(classifier_result.get("classifier_p_adv", 0.0))

                local_lbp_anomaly = 0.0
                global_lbp_anomaly = 0.0
                if texture is not None:
                    local_lbp_anomaly = float(texture.get("local_max", 0.0))
                    global_lbp_anomaly = float(texture.get("delta_h", 0.0))
                elif temporal is not None:
                    local_lbp_anomaly = float(temporal.get("local_max", 0.0))
                    global_lbp_anomaly = float(temporal.get("change_t", 0.0))

                lbp_anomaly_ratio = local_lbp_anomaly / max(global_lbp_anomaly, 1e-6)
                has_lbp_anomaly_cluster = bool(local_lbp_anomaly >= 0.35 and lbp_anomaly_ratio >= 3.5)

                if classifier_p_adv >= 0.92:
                    raw_suspicious = True
                    raw_reason = "no_target_classifier_fallback"
                elif has_lbp_anomaly_cluster:
                    raw_suspicious = True
                    raw_reason = "no_target_lbp_cluster_fallback"

            self._recent_no_target_suspicious.append(1 if raw_suspicious else 0)
            self._recent_track_supports.clear()
            self._recent_static_triggers.clear()

            # 清除有目标时的 hold，因为既然当前已经没有目标，不应该保持目标相关的 hold
            if self.detector is not None:
                self.detector.static_image_hold_remaining = 0
                self.detector.blur_hold_remaining = 0
                self.detector.strong_evidence_hold_remaining = 0

            # 必须在最近 5 帧中至少 3 帧触发，才认为是真实的无目标攻击
            if sum(self._recent_no_target_suspicious) >= 3:
                suspicious = True
                if raw_reason and raw_reason not in reason_codes:
                    reason_codes.append(raw_reason)
                if raw_reason == "targets_disappeared_with_global_anomaly":
                    global_fallback_fired = True

            return {
                "suspicious": suspicious,
                "reason_codes": reason_codes,
                "roi_anomaly_count": 0,
                "has_targets": False,
                "global_fallback_fired": global_fallback_fired,
                "classifier_bonus": False,
            }

        # ============================================================
        # 条件 2-5：逐 ROI 判断库内异常
        # ============================================================
        self._recent_no_target_suspicious.clear()

        temporal_flash = bool(overexposure.get("temporal_flash", False))
        if bool(overexposure.get("is_glare", False)):
            if temporal_flash and has_targets:
                suspicious = True
                roi_anomaly_count = n_targets
                reason_codes.append("overexposure_flash")
            else:
                natural_state = self._natural_exposure_state(
                    overexposure=overexposure,
                    blur=blur,
                    track=track,
                    temporal=temporal,
                    motion=motion,
                )
                if natural_state["suppressed"]:
                    reason_codes.append("natural_exposure_suppressed")
                elif (
                    natural_state["extreme"]
                    or (
                        natural_state["ratio"] >= self.roi_overexposure_threshold
                        and natural_state["track_support"]
                        and natural_state["local_attack_support"]
                    )
                ):
                    suspicious = True
                    roi_anomaly_count = n_targets
                    reason_codes.append("overexposure")
                else:
                    reason_codes.append("weak_overexposure_suppressed")

        # Static Media Spoof 时序平滑：
        static_triggered_raw = bool(static_image.get("triggered", False)) and bool(motion.get("target_related", False))
        self._recent_static_triggers.append(static_triggered_raw)
        static_image_spoof_filtered = sum(self._recent_static_triggers) >= 3

        if static_image_spoof_filtered:
            suspicious = True
            roi_anomaly_count += 1
            reason_codes.append("static_image_spoof")
        else:
            # 强制清零 detector 上的 hold_remaining，防止单帧抖动产生的 hold over 漏报/误报
            if self.detector is not None:
                self.detector.static_image_hold_remaining = 0

        blur_score = float(blur.get("blur_score", 0.0))
        track_score = float(track.get("track_score", 0.0))
        confidence_drop = float(track.get("confidence_drop_score", 0.0))
        motion_score = float(motion.get("motion_score", 0.0))
        light_flow_score = float(motion.get("light_flow_score", 0.0))
        temporal_local = float(temporal.get("local_max", 0.0))

        # Track Support 时序平滑：
        track_support_raw = (
            track_score >= self.track_drop_threshold
            or confidence_drop >= self.track_confidence_drop_threshold
        )
        self._recent_track_supports.append(track_support_raw)
        track_support = sum(self._recent_track_supports) >= 3

        # 同样地，如果连续帧内未发生 track 异常，重置 strong_evidence_hold 以减少自然误报
        if not track_support and self.detector is not None:
            self.detector.strong_evidence_hold_remaining = 0

        strong_temporal = temporal_local >= max(0.50, self.paired_temporal_motion_threshold)

        evidence_count = 0
        if blur_score >= self.roi_blur_threshold:
            evidence_count += 1
        if track_support:
            evidence_count += 1
        if motion_score >= self.motion_score_threshold:
            evidence_count += 1
        if light_flow_score >= self.light_flow_score_threshold:
            evidence_count += 1

        evidence_without_motion_flow = (
            (1 if blur_score >= self.roi_blur_threshold else 0)
            + (1 if track_support else 0)
        )
        normal_person_activity = (
            evidence_without_motion_flow < 1
            and motion_score >= self.motion_score_threshold
            and temporal_local >= self.paired_temporal_motion_threshold
            and not track_support
        )
        if normal_person_activity:
            if motion_score >= 0.92 or temporal_local >= 0.92:
                reason_codes.append("extreme_motion_temporal_override")
            else:
                evidence_count = 0

        if evidence_count >= 2 and strong_temporal and (track_support or blur_score >= self.roi_blur_threshold):
            suspicious = True
            roi_anomaly_count += 1
            if track_support and motion_score >= self.motion_score_threshold:
                reason_codes.append("target_track_consistency_drop")
            elif blur_score >= self.roi_blur_threshold and track_support:
                reason_codes.append("target_blur_temporal_anomaly")
            else:
                reason_codes.append("target_motion_temporal_anomaly")
        elif (
            light_flow_score >= self.light_flow_score_threshold
            and temporal_local >= max(0.55, self.paired_temporal_motion_threshold)
            and track_support
        ):
            suspicious = True
            roi_anomaly_count += 1
            reason_codes.append("target_light_flow_anomaly")

        # Flow-local specific path
        local_ratio = float(motion.get("local_max_ratio", 0.0))
        flow_threshold = self.flow_local_anomaly_threshold
        if (
            not suspicious
            and has_targets
            and not normal_person_activity
            and motion_score >= 0.75
            and local_ratio >= flow_threshold
            and temporal_local >= 0.20
        ):
            suspicious = True
            roi_anomaly_count += 1
            reason_codes.append("flow_local_anomaly")

        # --- 物理对抗贴纸检测规则 (Physical Patch Sharpness Anomaly Detection) ---
        if not suspicious and has_targets:
            has_patch_anomaly = False
            roi_blurs = blur.get("roi_results", [])
            for r_blur in roi_blurs:
                roi_obj = r_blur.get("roi", {})
                label = roi_obj.get("label", "") if isinstance(roi_obj, dict) else getattr(roi_obj, "label", "")
                if label == "grid":
                    continue
                
                sharpness_ratio = float(r_blur.get("sharpness_ratio", 0.0))
                # 物理贴纸的 sharpness_ratio 至少为 1.9，正常人最高为 0.70。
                # 我们将阈值设为 1.45，提供充足的安全毛利，并且需要伴随微弱的局部纹理变化 temporal_local，排除静止背景物的误报
                if sharpness_ratio >= 1.45 and temporal_local >= 0.12:
                    has_patch_anomaly = True
                    break

            if has_patch_anomaly:
                suspicious = True
                roi_anomaly_count = max(roi_anomaly_count, 1)
                if "physical_patch_motion" not in reason_codes:
                    reason_codes.append("physical_patch_motion")

        # Classifier Integration
        if classifier_result is not None:
            classifier_triggered = bool(
                classifier_result.get("classifier_triggered", False)
            )
            classifier_p_adv = float(
                classifier_result.get("classifier_p_adv", 0.0)
            )
            if classifier_triggered and suspicious:
                classifier_bonus = True
                if "classifier_adv_bonus" not in reason_codes:
                    reason_codes.append("classifier_adv_bonus")
            elif classifier_p_adv >= 0.90 and has_targets and roi_anomaly_count > 0:
                suspicious = True
                classifier_bonus = True
                if "classifier_adv_high_confidence" not in reason_codes:
                    reason_codes.append("classifier_adv_high_confidence")

        return {
            "suspicious": suspicious,
            "reason_codes": reason_codes,
            "roi_anomaly_count": roi_anomaly_count,
            "has_targets": True,
            "global_fallback_fired": False,
            "classifier_bonus": classifier_bonus,
        }

    def _targets_suddenly_disappeared(self) -> bool:
        if len(self._recent_target_counts) < 5:
            return False
        recent = list(self._recent_target_counts)
        prev_counts = recent[-5:-1]
        current = recent[-1]
        return (
            current == 0
            and all(c >= self.global_fallback_min_prev_targets for c in prev_counts)
        )

    def _natural_exposure_state(
        self,
        *,
        overexposure: dict[str, Any],
        blur: dict[str, Any],
        track: dict[str, Any],
        temporal: dict[str, Any],
        motion: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.natural_exposure_suppression:
            return {"suppressed": False, "reason": "disabled"}
        ratio = float(overexposure.get("ratio", 0.0) or 0.0)
        under = float(overexposure.get("underexposed_ratio", 0.0) or 0.0)
        blur_score = float(blur.get("blur_score", 0.0) or 0.0)
        track_score = float(track.get("track_score", 0.0) or 0.0)
        confidence_drop = float(track.get("confidence_drop_score", 0.0) or 0.0)
        temporal_local = float(temporal.get("local_max", 0.0) or 0.0)
        motion_score = float(motion.get("motion_score", 0.0) or 0.0)
        light_flow_score = float(motion.get("light_flow_score", 0.0) or 0.0)
        light_flow_ratio = float(motion.get("light_flow_local_anomaly_ratio", motion.get("local_max_ratio", 0.0)) or 0.0)
        extreme = self._global_extreme_anomaly(overexposure)
        track_support = (
            track_score >= self.track_drop_threshold
            or confidence_drop >= self.track_confidence_drop_threshold
        )
        local_attack_support = bool(
            track_support
            and (
                light_flow_score >= self.light_flow_score_threshold
                or light_flow_ratio >= self.natural_exposure_max_light_flow
                or (temporal_local >= 0.55 and motion_score >= self.motion_score_threshold)
            )
        )
        moderate_camera_exposure = (
            ratio <= self.natural_exposure_max_ratio
            and under < 0.80
            and not extreme
        )
        suppressed = bool(
            moderate_camera_exposure
            and not track_support
            and not local_attack_support
        )
        return {
            "suppressed": suppressed,
            "reason": "moderate_global_exposure_without_target_anchor" if suppressed else "not_suppressed",
            "ratio": ratio,
            "underexposed_ratio": under,
            "blur_score": blur_score,
            "track_score": track_score,
            "confidence_drop_score": confidence_drop,
            "temporal_local": temporal_local,
            "motion_score": motion_score,
            "light_flow_score": light_flow_score,
            "light_flow_ratio": light_flow_ratio,
            "track_support": bool(track_support),
            "local_attack_support": bool(local_attack_support),
            "extreme": bool(extreme),
        }

    def _global_extreme_anomaly(self, overexposure: dict[str, Any]) -> bool:
        ratio = float(overexposure.get("ratio", 0.0))
        under = float(overexposure.get("underexposed_ratio", 0.0))
        return (
            ratio >= self.global_fallback_overexposure_threshold
            or under >= 0.80
        )
