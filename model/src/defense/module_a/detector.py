from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from .artifacts import _resolve_artifact_path
from .classifier_features import build_classifier_features, build_static_media_classifier_features
from .detail_builders import build_details, build_p_media_extras
from .detector_setup import initialize_detector, reset_detector_state
from .process_pipeline import process_module_a
from .static_media_policy import StaticMediaPolicyMixin
from .types import ROI, ModuleAInput, ModuleAResult




class ModuleADetector(StaticMediaPolicyMixin):
    """General physical perturbation detector for video streams."""

    def __init__(self, config: dict[str, Any] | None = None):
        initialize_detector(self, config)

    def reset(self) -> None:
        reset_detector_state(self)

    def _sync_if_profile(self) -> None:
        if self.profile_cuda_sync and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(self.device))

    def process(self, item: ModuleAInput) -> ModuleAResult:
        return process_module_a(self, item)

    @staticmethod
    def _ema(previous: float, current: float, alpha: float) -> float:
        """Fast-rise / slow-fall confidence smoothing for display scores."""
        alpha = max(0.0, min(1.0, float(alpha)))
        previous = float(previous) if previous is not None else 0.0
        current = float(current)
        if current >= previous:
            return current
        return previous + alpha * (current - previous)

    def _update_roi_temporal_burst(self, triggered: bool) -> bool:
        self.roi_temporal_history.append(1 if triggered else 0)
        return sum(self.roi_temporal_history) >= self.roi_temporal_burst_trigger_count

    def _is_benign_global_motion(
        self,
        motion: dict[str, Any],
        overexposure: dict[str, Any],
    ) -> bool:
        if not self.benign_global_motion_filter_enabled:
            return False
        if bool(overexposure.get("is_glare", False)):
            return False
        if float(overexposure.get("ratio", 0.0)) > self.benign_global_motion_overexposure_max:
            return False
        return int(motion.get("region_count", 0)) >= self.benign_global_motion_region_count_min





    def _build_classifier_features(
        self,
        overexposure: dict[str, Any],
        texture: dict[str, Any],
        temporal: dict[str, Any],
        motion: dict[str, Any],
        blur: dict[str, Any],
        track: dict[str, Any],
        fusion: dict[str, Any],
        roi_count: int,
    ) -> dict[str, float]:
        return build_classifier_features(
            overexposure=overexposure,
            texture=texture,
            temporal=temporal,
            motion=motion,
            blur=blur,
            track=track,
            fusion=fusion,
            roi_count=roi_count,
        )

    def _build_static_media_classifier_features(self, motion: dict[str, Any]) -> dict[str, float]:
        return build_static_media_classifier_features(motion)

    def _build_details(
        self,
        overexposure: dict[str, Any],
        texture: dict[str, Any],
        temporal: dict[str, Any],
        motion: dict[str, Any],
        blur: dict[str, Any],
        track: dict[str, Any],
        fusion: dict[str, Any],
        source_auth: dict[str, Any],
        rois: list[ROI],
        roi_results: list[dict[str, Any]],
        frame_idx: int,
    ) -> dict[str, Any]:
        return build_details(
            self,
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
            frame_idx=frame_idx,
        )

    def _build_p_media_extras(self, motion: dict[str, Any]) -> dict[str, Any]:
        return build_p_media_extras(motion)

    def _frame_to_gray_tensor(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame).to(self.device, non_blocking=True).float()
        # Input frame is BGR. Keep color conversion on GPU.
        gray = tensor[..., 0] * 0.114 + tensor[..., 1] * 0.587 + tensor[..., 2] * 0.299
        return gray.view(1, 1, gray.shape[0], gray.shape[1]).contiguous()

    def _prepare_rois(self, rois: list[ROI] | None, width: int, height: int) -> list[ROI]:
        prepared: list[ROI] = []
        if rois:
            for roi in rois:
                clipped = roi.clipped(width, height, min_size=8)
                if clipped is not None:
                    prepared.append(clipped)
        if prepared or not self.use_grid_when_no_roi:
            return prepared
        return self._make_grid_rois(width, height)

    def _make_grid_rois(self, width: int, height: int) -> list[ROI]:
        rois: list[ROI] = []
        n = max(1, self.grid_roi_count)
        cell_w = width // n
        cell_h = height // n
        for gy in range(n):
            for gx in range(n):
                x1 = gx * cell_w
                y1 = gy * cell_h
                x2 = width if gx == n - 1 else (gx + 1) * cell_w
                y2 = height if gy == n - 1 else (gy + 1) * cell_h
                rois.append(ROI(f"grid_{gy}_{gx}", (x1, y1, x2, y2), label="grid"))
        return rois

    def _merge_roi_results(
        self,
        texture: dict[str, Any],
        temporal: dict[str, Any],
        motion: dict[str, Any],
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for source_name, source in (
            ("texture", texture.get("roi_results", [])),
            ("temporal", temporal.get("roi_results", [])),
            ("motion", motion.get("roi_results", [])),
            ("static_image", motion.get("static_image_roi_results", [])),
        ):
            for item in source:
                roi = item.get("roi", {})
                roi_id = str(roi.get("roi_id", "unknown"))
                target = by_id.setdefault(roi_id, {"roi": roi})
                target[source_name] = {k: v for k, v in item.items() if k != "roi"}
        return list(by_id.values())

    def _should_run_light_flow(self, frame_idx: int, temporal: dict[str, Any]) -> bool:
        if not self.light_flow_enabled:
            return False
        if self.prev_gray is None:
            return False
        if frame_idx % self.light_flow_interval == 0:
            return True
        return (
            float(temporal.get("change_t", 0.0)) >= self.light_flow_temporal_candidate
            or float(temporal.get("local_max", 0.0)) >= self.light_flow_local_temporal_candidate
        )

    def _should_run_static_image(self, frame_idx: int, temporal: dict[str, Any],
                                  effective_interval: int | None = None) -> bool:
        if not self.static_image_enabled:
            return False
        if self.prev_gray is None:
            return False
        interval = effective_interval if effective_interval is not None else self.static_image_interval
        if frame_idx % interval == 0:
            return True
        return (
            float(temporal.get("change_t", 0.0)) >= self.static_image_temporal_candidate
            or float(temporal.get("local_max", 0.0)) >= self.static_image_local_temporal_candidate
        )

    def _merge_light_flow(
        self,
        motion: dict[str, Any],
        light_flow: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(motion)
        merged.update(light_flow)
        if light_flow.get("light_flow_available", False):
            merged["backend"] = "gpu_frame_diff+gpu_lk_lite"
        return merged

    def _merge_static_image(
        self,
        motion: dict[str, Any],
        static_image: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(motion)
        merged.update(static_image)
        if static_image.get("static_image_triggered", False):
            merged["backend"] = f"{merged.get('backend', 'gpu_frame_diff')}+gpu_static_media_spoof"
        return merged






