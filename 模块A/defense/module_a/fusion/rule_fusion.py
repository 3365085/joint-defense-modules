from __future__ import annotations

from typing import Any

import torch


class GPURuleFusion:
    def __init__(
        self,
        device: str,
        weights: list[float] | tuple[float, ...] = (0.20, 0.30, 0.20, 0.10, 0.20),
        threshold: float = 0.55,
        temporal_trigger: float = 0.03,
        local_temporal_trigger: float = 0.045,
        local_flow_ratio_trigger: float = 0.42,
        strong_temporal_trigger: float = 0.10,
        strong_local_temporal_trigger: float = 0.50,
        paired_local_temporal_trigger: float = 0.50,
        paired_local_flow_trigger: float = 0.45,
        light_flow_anomaly_trigger: float = 0.22,
        light_flow_score_trigger: float = 0.35,
        paired_light_flow_temporal_trigger: float = 0.35,
        blur_score_trigger: float = 0.45,
        paired_blur_temporal_trigger: float = 0.18,
        track_score_trigger: float = 0.50,
        paired_track_temporal_trigger: float = 0.18,
        paired_track_blur_trigger: float = 0.25,
        static_image_score_trigger: float = 0.62,
    ):
        self.device = device
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
        if self.weights.numel() != 5:
            raise ValueError("Module A rule fusion expects five weights")
        self.threshold = float(threshold)
        self.temporal_trigger = float(temporal_trigger)
        self.local_temporal_trigger = float(local_temporal_trigger)
        self.local_flow_ratio_trigger = float(local_flow_ratio_trigger)
        self.strong_temporal_trigger = float(strong_temporal_trigger)
        self.strong_local_temporal_trigger = float(strong_local_temporal_trigger)
        self.paired_local_temporal_trigger = float(paired_local_temporal_trigger)
        self.paired_local_flow_trigger = float(paired_local_flow_trigger)
        self.light_flow_anomaly_trigger = float(light_flow_anomaly_trigger)
        self.light_flow_score_trigger = float(light_flow_score_trigger)
        self.paired_light_flow_temporal_trigger = float(paired_light_flow_temporal_trigger)
        self.blur_score_trigger = float(blur_score_trigger)
        self.paired_blur_temporal_trigger = float(paired_blur_temporal_trigger)
        self.track_score_trigger = float(track_score_trigger)
        self.paired_track_temporal_trigger = float(paired_track_temporal_trigger)
        self.paired_track_blur_trigger = float(paired_track_blur_trigger)
        self.static_image_score_trigger = float(static_image_score_trigger)

    def compute(
        self,
        texture: dict[str, Any],
        temporal: dict[str, Any],
        motion: dict[str, Any],
        overexposure: dict[str, Any],
        blur: dict[str, Any] | None = None,
        track: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        blur = blur or {}
        track = track or {}
        feature_tensor = torch.tensor(
            [
                float(texture.get("delta_h", 0.0)),
                float(temporal.get("change_t", 0.0)),
                float(motion.get("motion_score", 0.0)),
                float(overexposure.get("ratio", 0.0)),
                float(motion.get("static_image_score", 0.0)),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        p_adv = float(torch.clamp(torch.dot(feature_tensor, self.weights), 0.0, 1.0).item())

        temporal_triggered = float(temporal.get("change_t", 0.0)) >= self.temporal_trigger
        local_temporal_triggered = (
            float(temporal.get("local_max", 0.0)) >= self.local_temporal_trigger
        )
        local_flow_triggered = (
            float(motion.get("local_max_ratio", 0.0)) >= self.local_flow_ratio_trigger
        )
        strong_temporal_triggered = (
            float(temporal.get("change_t", 0.0)) >= self.strong_temporal_trigger
            and float(temporal.get("local_max", 0.0)) >= self.strong_local_temporal_trigger
        )
        paired_temporal_flow_triggered = (
            float(temporal.get("local_max", 0.0)) >= self.paired_local_temporal_trigger
            and float(motion.get("local_max_ratio", 0.0)) >= self.paired_local_flow_trigger
        )
        light_flow_triggered = bool(motion.get("light_flow_available", False)) and (
            float(motion.get("light_flow_local_anomaly_ratio", 0.0))
            >= self.light_flow_anomaly_trigger
            or float(motion.get("light_flow_score", 0.0)) >= self.light_flow_score_trigger
        )
        paired_temporal_light_flow_triggered = (
            float(temporal.get("local_max", 0.0)) >= self.paired_light_flow_temporal_trigger
            and light_flow_triggered
        )
        blur_triggered = float(blur.get("blur_score", 0.0)) >= self.blur_score_trigger
        paired_temporal_blur_triggered = (
            float(temporal.get("local_max", 0.0)) >= self.paired_blur_temporal_trigger
            and blur_triggered
        )
        track_triggered = float(track.get("track_score", 0.0)) >= self.track_score_trigger
        paired_track_triggered = (
            track_triggered
            and float(blur.get("blur_score", 0.0)) >= self.paired_track_blur_trigger
            and float(temporal.get("local_max", 0.0)) >= self.paired_track_temporal_trigger
        )
        static_image_triggered = bool(motion.get("static_image_triggered", False))
        overexposure_triggered = bool(overexposure.get("is_glare", False))
        p_adv_triggered = p_adv >= self.threshold
        roi_temporal_triggered = any(
            r.get("triggered", False) for r in temporal.get("roi_results", [])
        )

        reason_codes: list[str] = []
        if overexposure_triggered:
            reason_codes.append("overexposure")
        if temporal_triggered:
            reason_codes.append("temporal_texture_change")
        if local_temporal_triggered:
            reason_codes.append("local_temporal_texture_change")
        if local_flow_triggered:
            reason_codes.append("motion_artifact")
        if roi_temporal_triggered:
            reason_codes.append("roi_temporal_texture_change")
        if strong_temporal_triggered:
            reason_codes.append("strong_temporal_texture_change")
        if paired_temporal_flow_triggered:
            reason_codes.append("paired_temporal_flow_change")
        if light_flow_triggered:
            reason_codes.append("light_optical_flow_artifact")
        if paired_temporal_light_flow_triggered:
            reason_codes.append("paired_temporal_light_flow_change")
        if blur_triggered:
            reason_codes.append("local_blur_degradation")
        if paired_temporal_blur_triggered:
            reason_codes.append("paired_temporal_blur_degradation")
        if track_triggered:
            reason_codes.append("track_consistency_drop")
        if paired_track_triggered:
            reason_codes.append("paired_track_consistency_drop")
        if static_image_triggered:
            reason_codes.append("static_image_spoof")
            # A3+ PR6: distinguish new path trigger from legacy
            if motion.get("static_image_triggered_source") in {"a3_plus", "a3_plus_fast", "a3_plus_replay", "a3_plus_occlusion_hold"}:
                reason_codes.append("static_media_spoof")
            if motion.get("static_image_triggered_source") == "a3_plus_fast":
                reason_codes.append("static_media_fast_confirmed")
            if motion.get("static_image_triggered_source") == "a3_plus_replay":
                reason_codes.append("static_media_replay_persistent")
            if motion.get("static_image_triggered_source") == "a3_plus_occlusion_hold":
                reason_codes.append("static_media_occlusion_hold")
        if p_adv_triggered:
            reason_codes.append("p_adv")

        suspicious = (
            # --- ROI-anchored signals (框级异常) ---
            # 过曝：全图级但物理攻击明确
            overexposure_triggered
            # A4 classifier：已经融合了 ROI 级特征（46 维包含 roi_count、
            # track_score、blur_roi_energy 等），它的判定本身就是框相关的
            or p_adv_triggered
            # 轨迹丢失 + 模糊：YOLO 框突然消失且画面模糊 = 遮挡攻击
            or paired_track_triggered
            # A3b 翻拍：基于 YOLO ROI 的 patch-track 判定
            or static_image_triggered
            # 模糊 + 时域变化 paired：ROI 内模糊（blur_score 基于 ROI）
            or paired_temporal_blur_triggered
        )

        # 注意：以下全图统计量不再直接触发 suspicious，只记录 reason_code：
        # - temporal_triggered (全图 LBP 变化)
        # - local_flow_triggered (全图帧间差分)
        # - strong_temporal_triggered
        # - paired_temporal_flow_triggered
        # - paired_temporal_light_flow_triggered
        # 这些在正常手持/行人场景下太容易触发，不适合作为报警依据。
        # 它们仍然作为 A4 classifier 的输入特征参与融合判断。

        return {
            "p_adv": p_adv,
            "threshold": self.threshold,
            "is_suspicious": bool(suspicious),
            "reason_codes": reason_codes,
            "temporal_triggered": temporal_triggered,
            "local_temporal_triggered": local_temporal_triggered,
            "local_flow_triggered": local_flow_triggered,
            "roi_temporal_triggered": roi_temporal_triggered,
            "strong_temporal_triggered": strong_temporal_triggered,
            "paired_temporal_flow_triggered": paired_temporal_flow_triggered,
            "light_flow_triggered": light_flow_triggered,
            "paired_temporal_light_flow_triggered": paired_temporal_light_flow_triggered,
            "blur_triggered": blur_triggered,
            "paired_temporal_blur_triggered": paired_temporal_blur_triggered,
            "track_triggered": track_triggered,
            "paired_track_triggered": paired_track_triggered,
            "static_image_triggered": static_image_triggered,
            "overexposure_triggered": overexposure_triggered,
            "p_adv_triggered": p_adv_triggered,
        }
