"""Target-anchored suspicious判定器。

核心原则（参考 doc/A3_target_anchored_false_positive_suppression.txt）：

  只有当异常发生在目标框内或目标框附近，并且异常区域与目标存在稳定关系时，
  才进入报警判断。全图统计量不直接触发报警。

输入：
  - YOLO 检测框（rois）
  - per-ROI 特征（blur、temporal、motion、overexposure）
  - 轨迹状态（track consistency）
  - 全图级信号（overexposure、A4 classifier）

输出：
  - target_anchored_suspicious: bool
  - reason_codes: list[str]
  - roi_anomaly_details: list[dict]
"""

from __future__ import annotations

from collections import deque
from typing import Any


class TargetAnchoredAnalyzer:
    """以目标检测框为锚点的异常判定器。

    设计原则：
      1. 没有目标框 → 不报警（除非全图过曝/全目标突然消失）
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
        # 自然曝光/手机自动曝光抑制：真实视频从室内到室外、阳光区域、
        # 自动曝光锁定失败时，经常会出现中等比例过曝 + 全局模糊，
        # 但没有目标轨迹置信下降/局部贴片运动证据。该类情况不应独立报警。
        natural_exposure_suppression: bool = True,
        natural_exposure_max_ratio: float = 0.18,
        natural_exposure_max_light_flow: float = 0.35,
        natural_exposure_max_motion_score: float = 1.01,
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
        # 历史目标数量（用于"目标突然消失"兜底）
        self._recent_target_counts: deque[int] = deque(maxlen=10)

    def reset(self) -> None:
        self._recent_target_counts.clear()

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
    ) -> dict[str, Any]:
        """每帧调用一次，返回 target-anchored 判定结果。

        Returns:
            {
                "suspicious": bool,
                "reason_codes": list[str],
                "roi_anomaly_count": int,
                "has_targets": bool,
                "global_fallback_fired": bool,
                "classifier_bonus": bool,
            }
        """
        n_targets = len(rois)
        self._recent_target_counts.append(n_targets)
        has_targets = n_targets > 0

        reason_codes: list[str] = []
        roi_anomaly_count = 0
        suspicious = False
        global_fallback_fired = False
        classifier_bonus = False

        # ============================================================
        # 条件 1：必须有目标锚点
        # ============================================================
        if not has_targets:
            # 兜底：前几帧有稳定目标但突然全部消失 + 全图强异常
            if self.allow_global_fallback and self._targets_suddenly_disappeared():
                if self._global_extreme_anomaly(overexposure):
                    suspicious = True
                    global_fallback_fired = True
                    reason_codes.append("targets_disappeared_with_global_anomaly")
            return {
                "suspicious": suspicious,
                "reason_codes": reason_codes,
                "roi_anomaly_count": 0,
                "has_targets": False,
                "global_fallback_fired": global_fallback_fired,
                "classifier_bonus": False,
            }

        # ============================================================
        # 条件 2-5：逐 ROI 判断框内异常
        # ============================================================
        # 核心原则：只有当异常**跟随目标运动**或**直接作用于目标框**时才触发。
        # 静态背景物（柱子、门框、广告牌）即使与人框短暂重叠也不算。

        # --- 全图过曝（物理攻击，影响所有 ROI）---
        # 原逻辑只要 is_glare=True 就立刻进入 suspicious，这会把手机/监控
        # 常见的自然曝光变化（室内到室外、阳光区域、自动曝光漂移）误判成
        # 物理扰动。现在过曝必须满足以下任一条件才报警：
        #   1) 过曝比例极端；
        #   2) 伴随目标锚定证据（轨迹掉置信/局部光流/强局部时序）；
        #   3) 未被自然曝光抑制器判定为 benign。
        if bool(overexposure.get("is_glare", False)):
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

        # --- A3b 翻拍/假目标（基于 YOLO ROI 的 patch-track）---
        # patch-track 本身就要求：同一 ROI 内容连续 6 帧高度相似 +
        # 有运动证据（center_motion / context_motion）。
        # 这天然满足"跟随目标 + 连续帧命中"的要求。
        if bool(static_image.get("triggered", False)):
            suspicious = True
            roi_anomaly_count += 1
            reason_codes.append("static_image_spoof")

        # --- A3 physical perturbation anchored by existing YOLO targets ---
        blur_score = float(blur.get("blur_score", 0.0))
        track_score = float(track.get("track_score", 0.0))
        confidence_drop = float(track.get("confidence_drop_score", 0.0))
        motion_score = float(motion.get("motion_score", 0.0))
        light_flow_score = float(motion.get("light_flow_score", 0.0))
        temporal_local = float(temporal.get("local_max", 0.0))
        track_support = (
            track_score >= self.track_drop_threshold
            or confidence_drop >= self.track_confidence_drop_threshold
        )
        blur_support = blur_score >= self.roi_blur_threshold

        if (
            blur_score >= self.roi_blur_threshold
            and temporal_local >= max(0.50, self.paired_temporal_motion_threshold)
            and track_support
        ):
            suspicious = True
            roi_anomaly_count += 1
            reason_codes.append("target_blur_temporal_anomaly")
        track_context_support = (
            motion_score >= self.motion_score_threshold
            or temporal_local >= self.paired_temporal_motion_threshold
        )
        strong_temporal_context = temporal_local >= max(
            0.50, self.paired_temporal_motion_threshold
        )

        if track_support and track_context_support and strong_temporal_context:
            suspicious = True
            roi_anomaly_count += 1
            reason_codes.append("target_track_consistency_drop")
        if (
            motion_score >= self.motion_score_threshold
            and strong_temporal_context
            and track_support
        ):
            suspicious = True
            roi_anomaly_count += 1
            reason_codes.append("target_motion_temporal_anomaly")
        if (
            light_flow_score >= self.light_flow_score_threshold
            and temporal_local >= max(0.55, self.paired_temporal_motion_threshold)
            and track_support
        ):
            suspicious = True
            roi_anomaly_count += 1
            reason_codes.append("target_light_flow_anomaly")

        # ============================================================
        # A4 classifier 加分逻辑
        # ============================================================
        # 规则：
        # - 如果 ROI 异常已成立 + classifier 也触发 → 加强置信（bonus）
        # - 如果 ROI 异常未成立但 classifier 强触发（p_adv >= 0.85）
        #   且有目标框存在 → 允许 classifier 独立触发
        #   （覆盖 occlusion/visibility 等 A4 擅长但 ROI 信号弱的攻击）
        # - 其他情况 → 只记录不触发
        if classifier_result is not None:
            classifier_triggered = bool(
                classifier_result.get("classifier_triggered", False)
            )
            classifier_p_adv = float(
                classifier_result.get("classifier_p_adv", 0.0)
            )
            if classifier_triggered and suspicious:
                # ROI 异常已成立 + classifier 确认 → bonus
                classifier_bonus = True
                if "classifier_adv_bonus" not in reason_codes:
                    reason_codes.append("classifier_adv_bonus")
            elif classifier_p_adv >= 0.90 and has_targets and roi_anomaly_count > 0:
                # A4 非常确信（>= 0.90）+ 有目标框 → 允许独立触发
                # 这覆盖了 occlusion/visibility 等 ROI 信号弱但 classifier
                # 能从 46 维特征组合中识别的攻击类型。
                # 真实视频的 classifier_p_adv 通常 < 0.01，不会误触发。
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
        """前几帧有稳定目标但当前帧突然全部消失。"""
        if len(self._recent_target_counts) < 5:
            return False
        # 最近 5 帧中前 4 帧都有目标，当前帧没有
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
        """Return whether glare looks like a benign camera exposure change.

        Suppression is deliberately conservative: it only applies to moderate
        glare ratios and only when there is no target-anchored evidence such as
        track confidence drop, high light-flow anomaly, or strong local temporal
        burst.  This preserves obvious laser/flash/patch attacks while removing
        the common false positive where a normal phone video moves from indoor
        shade to sunlight.
        """
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
        # Large normal subject/camera motion often creates high optical-flow
        # and temporal scores together with auto-exposure drift. Treat that as
        # attack support only when the target track itself is unstable.
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
        """全图级极端异常（强光/全黑）。"""
        ratio = float(overexposure.get("ratio", 0.0))
        under = float(overexposure.get("underexposed_ratio", 0.0))
        return (
            ratio >= self.global_fallback_overexposure_threshold
            or under >= 0.80
        )
