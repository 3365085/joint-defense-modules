from __future__ import annotations

from typing import Any


class StaticMediaPolicyMixin:
    """Stateful A3b/static-media replay, fast-trigger, occlusion, and suppression policies."""

    def _static_media_physical_motion_suppression(
        self,
        static_image: dict[str, Any],
        motion: dict[str, Any],
        temporal: dict[str, Any],
        overexposure: dict[str, Any],
    ) -> dict[str, Any]:
        """Split moving target-attached patches away from A3b static-media alerts."""
        p_media = float(static_image.get("p_media", 0.0))
        target_related = bool(static_image.get("p_media_target_related", False))
        motion_score = float(motion.get("motion_score", 0.0))
        flow_ratio = float(motion.get("local_max_ratio", 0.0))
        temporal_local = float(temporal.get("local_max", 0.0))
        exposure_ratio = float(overexposure.get("ratio", 0.0) or 0.0)
        glare = bool(overexposure.get("is_glare", False)) or (
            float(overexposure.get("ratio", 0.0)) >= self.glare_ratio_threshold
        )
        strong_target_motion = target_related
        exposure_motion = bool(
            getattr(self, "static_media_exposure_motion_suppress_enabled", True)
            and not target_related
            and p_media >= 0.42
            and motion_score >= 0.75
            and temporal_local >= getattr(
                self, "static_media_exposure_motion_min_temporal", 0.20
            )
            and exposure_ratio >= getattr(
                self, "static_media_exposure_motion_min_ratio", 0.008
            )
        )
        suppressed = bool(p_media >= 0.42 and (strong_target_motion or exposure_motion))
        reason = "none"
        if suppressed:
            reason = "exposure_motion"
            if strong_target_motion:
                reason = "target_attached_patch"
            if strong_target_motion and glare:
                reason = "target_attached_glare"
        return {
            "suppressed": bool(suppressed),
            "reason": reason,
            "p_media": float(p_media),
            "target_related": bool(target_related),
            "exposure_motion": bool(exposure_motion),
            "exposure_ratio": float(exposure_ratio),
            "motion_score": float(motion_score),
            "flow_ratio": float(flow_ratio),
            "temporal_local": float(temporal_local),
            "glare": bool(glare),
            "score_cap": float(self.physical_media_motion_score_cap),
            "p_adv": float(self.physical_media_motion_min_p_adv),
        }

    def _static_media_camera_motion_suppression(
        self,
        static_image: dict[str, Any],
        motion: dict[str, Any],
    ) -> dict[str, Any]:
        """Suppress edge-only A3b candidates caused by real camera translation."""
        bbox = static_image.get("p_media_bbox")
        scores = static_image.get("p_media_scores", {})
        p_media = float(static_image.get("p_media", 0.0) or 0.0)
        target_related = bool(static_image.get("p_media_target_related", False))
        edge_score = float(scores.get("edge", 0.0) or 0.0)
        yolo_context = float(scores.get("yolo_context", 0.0) or 0.0)
        motion_score = float(motion.get("motion_score", 0.0) or 0.0)
        light_valid_ratio = float(motion.get("light_flow_valid_ratio", 0.0) or 0.0)
        flow_ratio = float(motion.get("local_max_ratio", 0.0) or 0.0)
        touches_horizontal_edge = False
        bbox_area = 0.0
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            touches_horizontal_edge = x1 <= 20.0 or x2 >= 620.0
        edge_only_candidate = bool(
            not target_related
            and yolo_context < self.static_media_camera_motion_max_yolo_context
            and (touches_horizontal_edge or edge_score >= self.static_media_fast_min_edge_score)
        )
        moving_camera = bool(
            motion_score >= 0.8
            and (
                light_valid_ratio >= self.static_media_camera_motion_min_valid_ratio
                or flow_ratio <= getattr(
                    self, "static_media_camera_motion_max_flow_ratio", 0.42
                )
            )
        )
        suppressed = bool(
            self.static_media_camera_motion_suppress_enabled
            and p_media >= self.static_media_camera_motion_min_p_media
            and edge_only_candidate
            and moving_camera
        )
        return {
            "enabled": bool(self.static_media_camera_motion_suppress_enabled),
            "suppressed": bool(suppressed),
            "reason": "camera_translation_edge" if suppressed else "none",
            "p_media": float(p_media),
            "target_related": bool(target_related),
            "edge_score": float(edge_score),
            "yolo_context": float(yolo_context),
            "motion_score": float(motion_score),
            "light_flow_valid_ratio": float(light_valid_ratio),
            "flow_ratio": float(flow_ratio),
            "touches_horizontal_edge": bool(touches_horizontal_edge),
            "bbox_area": float(bbox_area),
            "score_cap": float(self.static_media_camera_motion_score_cap),
        }

    def _static_media_border_suppression(self, static_image: dict[str, Any]) -> dict[str, Any]:
        bbox = static_image.get("p_media_bbox")
        suppressed = False
        reason = "none"
        touches_edge = False
        area_ratio = 0.0
        if self.static_media_border_suppress_enabled and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            touches_edge = (
                x1 <= self.static_media_border_margin
                or y1 <= self.static_media_border_margin
                or x2 >= 640.0 - self.static_media_border_margin
                or y2 >= 640.0 - self.static_media_border_margin
            )
            area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (640.0 * 640.0)
            target_related = bool(static_image.get("p_media_target_related", False))
            p_media = float(static_image.get("p_media", 0.0))
            if touches_edge and not target_related and p_media >= 0.55:
                suppressed = True
                reason = "frame_border"
            if touches_edge and area_ratio >= self.static_media_border_max_area_ratio and not target_related:
                suppressed = True
                reason = "letterbox_or_large_border"
        return {
            "enabled": bool(self.static_media_border_suppress_enabled),
            "suppressed": bool(suppressed),
            "reason": reason,
            "touches_edge": bool(touches_edge),
            "area_ratio": float(area_ratio),
            "score_cap": 0.45,
        }

    def _update_static_media_replay_state(
        self,
        static_image: dict[str, Any],
        temporal: dict[str, Any],
        blur: dict[str, Any],
    ) -> dict[str, Any]:
        scores = static_image.get("p_media_scores", {})
        bbox = static_image.get("p_media_bbox")
        bbox_area = 0.0
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            touches_horizontal_edge = x1 <= 20.0 or x2 >= 620.0
        else:
            touches_horizontal_edge = False

        p_media = float(static_image.get("p_media", 0.0))
        target_related = bool(static_image.get("p_media_target_related", False))
        temporal_change = float(temporal.get("change_t", 0.0))
        local_temporal = float(temporal.get("local_max", 0.0))
        blur_score = float(blur.get("blur_score", 0.0))
        edge_score = float(scores.get("edge", 0.0))
        yolo_context = float(scores.get("yolo_context", 0.0))
        warp_residual = float(scores.get("warp_residual", 0.0))
        flow_gap = float(scores.get("flow_gap", 0.0))
        local_evidence = (
            blur_score >= self.static_media_replay_min_blur
            or temporal_change >= self.static_media_replay_min_temporal
            or local_temporal >= self.static_media_replay_min_temporal
        )
        replay_evidence = (
            warp_residual >= self.static_media_replay_min_warp_residual
            or flow_gap >= self.static_media_replay_min_flow_gap
        )
        support_evidence = bool(
            target_related
            or edge_score >= self.static_media_fast_min_edge_score
            or yolo_context >= self.static_media_fast_min_yolo_context
            or flow_gap >= self.static_media_replay_min_support_flow_gap
        )
        evidence = (
            p_media >= self.static_media_replay_min_p_media
            and not target_related
            and 0.0 < bbox_area <= self.static_media_replay_max_bbox_area
            and (
                touches_horizontal_edge
                or bbox_area <= self.static_media_free_candidate_max_bbox_area
            )
            and local_evidence
            and replay_evidence
            and support_evidence
        )
        bbox_state = None
        if evidence and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox_state = (
                (x1 + x2) * 0.5,
                (y1 + y2) * 0.5,
                max(1.0, bbox_area),
            )
        self.static_media_replay_votes.append(1 if evidence else 0)
        self.static_media_replay_bboxes.append(bbox_state)
        votes = sum(self.static_media_replay_votes)
        trigger_count = max(4, min(self.static_media_replay_window, int(self.static_media_replay_window * 0.25 + 0.999)))
        tracked_bboxes = [state for state in self.static_media_replay_bboxes if state is not None]
        center_span_x = 0.0
        center_span_y = 0.0
        area_ratio = 1.0
        stable_candidate_track = False
        if len(tracked_bboxes) >= trigger_count:
            center_span_x = max(v[0] for v in tracked_bboxes) - min(v[0] for v in tracked_bboxes)
            center_span_y = max(v[1] for v in tracked_bboxes) - min(v[1] for v in tracked_bboxes)
            min_area = max(1.0, min(v[2] for v in tracked_bboxes))
            area_ratio = max(v[2] for v in tracked_bboxes) / min_area
            stable_candidate_track = (
                center_span_x <= self.static_media_replay_max_center_span
                and center_span_y <= self.static_media_replay_max_center_span
                and area_ratio <= self.static_media_replay_max_area_ratio
            )
        triggered = (
            evidence
            and
            len(self.static_media_replay_votes) >= trigger_count
            and votes >= trigger_count
            and stable_candidate_track
        )
        return {
            "candidate": bool(evidence),
            "triggered": bool(triggered),
            "votes": int(votes),
            "window": int(self.static_media_replay_window),
            "trigger_count": int(trigger_count),
            "bbox_area": float(bbox_area),
            "p_media": float(p_media),
            "target_related": bool(target_related),
            "target_iou": float(scores.get("target_iou", 0.0)),
            "target_proximity": float(scores.get("target_proximity", 0.0)),
            "target_area_ratio": float(scores.get("target_area_ratio", 0.0)),
            "warp_residual": float(warp_residual),
            "flow_gap": float(flow_gap),
            "edge_score": float(edge_score),
            "yolo_context": float(yolo_context),
            "support_evidence": bool(support_evidence),
            "local_evidence": bool(local_evidence),
            "replay_evidence": bool(replay_evidence),
            "stable_candidate_track": bool(stable_candidate_track),
            "center_span_x": float(center_span_x),
            "center_span_y": float(center_span_y),
            "area_ratio": float(area_ratio),
        }

    def _update_static_media_occlusion_state(
        self,
        static_image: dict[str, Any],
        replay_state: dict[str, Any],
        fast_state: dict[str, Any],
    ) -> dict[str, Any]:
        p_media = float(static_image.get("p_media", 0.0) or 0.0)
        live_score = float(static_image.get("static_image_live_score", p_media) or 0.0)
        bbox_area = max(
            float(replay_state.get("bbox_area", 0.0) or 0.0),
            float(fast_state.get("bbox_area", 0.0) or 0.0),
        )
        suppressed_by_border = bool(
            replay_state.get("suppressed_by_border", False)
            or fast_state.get("suppressed_by_border", False)
            or static_image.get("static_image_triggered_source") == "border_or_letterbox_suppressed"
        )
        if suppressed_by_border:
            self.static_media_occlusion_last_reason = "border_suppressed"
            return {
                "active": False,
                "reason": "border_suppressed",
                "remaining": 0,
                "hold_frames": int(self.static_media_occlusion_hold_frames),
                "score": 0.0,
                "p_media": float(p_media),
                "live_score": float(live_score),
                "bbox_area": float(bbox_area),
                "edge_or_large": True,
                "reacquired_irregular": False,
                "suppressed_by_border": True,
            }
        confirmed_media_lock = bool(
            replay_state.get("triggered", False) or fast_state.get("triggered", False)
        )
        triggered_now = bool(
            confirmed_media_lock
            and (
                static_image.get("static_image_triggered", False)
                or replay_state.get("triggered", False)
                or fast_state.get("triggered", False)
            )
        )
        edge_or_large = bool(
            fast_state.get("touches_horizontal_edge", False)
            or fast_state.get("touches_vertical_edge", False)
            or bbox_area >= self.static_media_replay_max_bbox_area * 0.55
        )
        reacquired_irregular = bool(
            self.static_media_occlusion_hold_remaining > 0
            and p_media >= self.static_media_occlusion_reacquire_min_p_media
            and (edge_or_large or not fast_state.get("stable_fast_track", False))
        )

        reason = "none"
        if triggered_now:
            self.static_media_occlusion_hold_remaining = self.static_media_occlusion_hold_frames
            self.static_media_occlusion_hold_score = max(
                self.static_media_occlusion_min_score,
                live_score,
                p_media,
                float(static_image.get("static_image_score", 0.0) or 0.0),
            )
            reason = "confirmed_media_lock"
        elif reacquired_irregular:
            self.static_media_occlusion_hold_remaining = self.static_media_occlusion_hold_frames
            self.static_media_occlusion_hold_score = max(
                self.static_media_occlusion_min_score,
                self.static_media_occlusion_hold_score,
                p_media,
            )
            reason = "irregular_edge_reacquired"
        elif self.static_media_occlusion_hold_remaining > 0:
            self.static_media_occlusion_hold_remaining -= 1
            reason = "occluded_media_hold"

        active = self.static_media_occlusion_hold_remaining > 0
        if not active:
            self.static_media_occlusion_hold_score = 0.0
            reason = "none"
        self.static_media_occlusion_last_reason = reason
        return {
            "active": bool(active),
            "reason": reason,
            "remaining": int(self.static_media_occlusion_hold_remaining),
            "hold_frames": int(self.static_media_occlusion_hold_frames),
            "score": float(self.static_media_occlusion_hold_score),
            "p_media": float(p_media),
            "live_score": float(live_score),
            "bbox_area": float(bbox_area),
            "edge_or_large": bool(edge_or_large),
            "reacquired_irregular": bool(reacquired_irregular),
            "suppressed_by_border": False,
        }

    def _update_static_media_fast_state(
        self, static_image: dict[str, Any]
    ) -> dict[str, Any]:
        scores = static_image.get("p_media_scores", {})
        bbox = static_image.get("p_media_bbox")
        bbox_area = 0.0
        bbox_state = None
        touches_vertical_edge = False
        touches_horizontal_edge = False
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            bbox_state = ((x1 + x2) * 0.5, (y1 + y2) * 0.5, max(1.0, bbox_area))
            touches_vertical_edge = y1 <= self.static_media_fast_edge_margin or y2 >= 640.0 - self.static_media_fast_edge_margin
            touches_horizontal_edge = x1 <= 20.0 or x2 >= 620.0
        p_media = float(static_image.get("p_media", 0.0))
        target_related = bool(static_image.get("p_media_target_related", False))
        strong_media_evidence = bool(static_image.get("p_media_strong_evidence", False))
        edge_score = float(scores.get("edge", 0.0))
        yolo_context = float(scores.get("yolo_context", 0.0))
        warp_residual = float(scores.get("warp_residual", 0.0))
        flow_gap = float(scores.get("flow_gap", 0.0))
        replay_signal = max(warp_residual, flow_gap)
        fast_support_evidence = bool(
            target_related
            or strong_media_evidence
            or edge_score >= self.static_media_fast_min_edge_score
            or yolo_context >= self.static_media_fast_min_yolo_context
        )
        primary_replay_evidence = (
            p_media >= self.static_media_fast_min_p_media
            and (
                warp_residual >= self.static_media_fast_min_replay_signal
                or flow_gap >= self.static_media_fast_alt_min_flow_gap
                or strong_media_evidence
            )
        )
        alternate_replay_evidence = (
            p_media >= self.static_media_fast_alt_min_p_media
            and (
                target_related
                or strong_media_evidence
                or (
                    touches_horizontal_edge
                    and replay_signal >= self.static_media_fast_edge_min_replay_signal
                )
            )
        )
        fast_replay_evidence = primary_replay_evidence or alternate_replay_evidence
        evidence = (
            0.0 < bbox_area <= self.static_media_replay_max_bbox_area
            and (
                target_related
                or touches_horizontal_edge
                or bbox_area <= self.static_media_free_candidate_max_bbox_area
            )
            and not touches_vertical_edge
            and fast_support_evidence
            and (
                (target_related and p_media >= self.static_media_fast_min_p_media)
                or fast_replay_evidence
            )
        )
        self.static_media_fast_votes.append(1 if evidence else 0)
        self.static_media_fast_bboxes.append(bbox_state if evidence else None)
        votes = sum(self.static_media_fast_votes)
        trigger_count = min(self.static_media_fast_trigger_count, self.static_media_fast_window)
        tracked_bboxes = [state for state in self.static_media_fast_bboxes if state is not None]
        center_span_x = 0.0
        center_span_y = 0.0
        area_ratio = 1.0
        stable_fast_track = False
        if len(tracked_bboxes) >= trigger_count:
            center_span_x = max(v[0] for v in tracked_bboxes) - min(v[0] for v in tracked_bboxes)
            center_span_y = max(v[1] for v in tracked_bboxes) - min(v[1] for v in tracked_bboxes)
            min_area = max(1.0, min(v[2] for v in tracked_bboxes))
            area_ratio = max(v[2] for v in tracked_bboxes) / min_area
            stable_fast_track = (
                center_span_x <= self.static_media_replay_max_center_span
                and center_span_y <= self.static_media_replay_max_center_span
                and area_ratio <= self.static_media_replay_max_area_ratio
            )
        triggered = (
            evidence
            and
            len(self.static_media_fast_votes) >= trigger_count
            and votes >= trigger_count
            and stable_fast_track
        )
        return {
            "candidate": bool(evidence),
            "triggered": bool(triggered),
            "votes": int(votes),
            "window": int(self.static_media_fast_window),
            "trigger_count": int(trigger_count),
            "p_media": float(p_media),
            "target_related": bool(target_related),
            "strong_media_evidence": bool(strong_media_evidence),
            "edge_score": float(edge_score),
            "yolo_context": float(yolo_context),
            "fast_support_evidence": bool(fast_support_evidence),
            "warp_residual": float(warp_residual),
            "flow_gap": float(flow_gap),
            "replay_signal": float(replay_signal),
            "fast_replay_evidence": bool(fast_replay_evidence),
            "primary_replay_evidence": bool(primary_replay_evidence),
            "alternate_replay_evidence": bool(alternate_replay_evidence),
            "stable_fast_track": bool(stable_fast_track),
            "center_span_x": float(center_span_x),
            "center_span_y": float(center_span_y),
            "area_ratio": float(area_ratio),
            "bbox_area": float(bbox_area),
            "touches_vertical_edge": bool(touches_vertical_edge),
            "touches_horizontal_edge": bool(touches_horizontal_edge),
        }

