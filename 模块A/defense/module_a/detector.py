from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .alert_state import AlertState
from .features.a3.static_media import GPUStaticMediaSpoofDetector
from .features.blur_degradation import GPUBlurDegradationDetector
from .features.lbp_texture import GPULBPTextureAnalyzer
from .features.light_flow import GPULightOpticalFlowDetector
from .features.motion_artifact import GPUMotionArtifactDetector
from .features.overexposure import GPUOverexposureDetector
from .features.temporal_texture import GPUTemporalTextureAnalyzer
from .features.track_consistency import TrackConsistencyAnalyzer
from .fusion.classifier_fusion import TorchLogisticFusion
from .fusion.rule_fusion import GPURuleFusion
from .fusion.target_anchored import TargetAnchoredAnalyzer
from .scheduler import ModuleAScheduler
from .source_authenticity import SourceAuthenticityDetector
from .types import ROI, ModuleAInput, ModuleAResult


def _resolve_artifact_path(raw_path: str) -> Path:
    """Resolve an artifact path relative to the module A package root.

    Search order (first existing wins):

    1. ``$MODULE_A_ROOT / raw_path`` — explicit override used when the module
       is embedded inside a larger workspace (联合防御模块 root does not map
       to ``parents[N]`` anymore). Documented in 架构说明.md §八.
    2. ``parents[2] / raw_path`` — package root when the file lives at
       ``defense/module_a/detector.py`` inside the delivery package.
    3. ``parents[3] / raw_path`` — legacy location from the original
       security_project_c layout. Kept for backward-compat with artifact
       JSONs that already hardcode that level.
    4. ``Path.cwd() / raw_path`` — last-resort fallback for scripts that
       explicitly ``cd`` into a working directory before launching.

    Returns the first resolvable path. Absolute paths are returned as-is.
    Raises nothing; if no candidate exists we still return the best-guess
    ``parents[2]`` path so downstream ``open()`` produces a clear error
    message naming the expected location.
    """
    import os

    resolved = Path(raw_path)
    if resolved.is_absolute():
        return resolved

    here = Path(__file__).resolve()
    candidates: list[Path] = []
    module_root_env = os.environ.get("MODULE_A_ROOT")
    if module_root_env:
        candidates.append(Path(module_root_env).expanduser() / resolved)
    candidates.extend(
        [
            here.parents[2] / resolved,  # <pkg>/
            here.parents[3] / resolved,  # legacy: one level up
            Path.cwd() / resolved,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # None found — return the canonical package-root candidate so the eventual
    # FileNotFoundError message is informative rather than pointing at cwd.
    return here.parents[2] / resolved


class ModuleADetector:
    """General physical perturbation detector for video streams."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        module_config = config.get("module_a", config)
        self.require_gpu = bool(module_config.get("require_gpu", True))
        if self.require_gpu and not torch.cuda.is_available():
            raise RuntimeError("Module A GPU mode is required, but CUDA is not available")
        explicit_device = module_config.get("device")
        if explicit_device:
            self.device = str(explicit_device)
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            raise RuntimeError(
                "No CUDA device available and no explicit device configured. "
                "Module A requires a CUDA device for inference."
            )
        if self.require_gpu and not self.device.startswith("cuda"):
            raise RuntimeError(f"Module A GPU mode requires a CUDA device, got {self.device}")

        self.frame_size = int(module_config.get("frame_size", 640))
        self.grid_roi_count = int(module_config.get("grid_roi_count", 4))
        self.use_grid_when_no_roi = bool(module_config.get("use_grid_when_no_roi", True))
        self.emit_roi_details = bool(module_config.get("emit_roi_details", False))
        self.emit_roi_texture_details = bool(module_config.get("emit_roi_texture_details", False))
        self.emit_roi_temporal_details = bool(
            module_config.get("emit_roi_temporal_details", self.emit_roi_details)
        )
        self.emit_roi_motion_details = bool(module_config.get("emit_roi_motion_details", False))
        self.roi_temporal_burst_window = max(
            1, int(module_config.get("roi_temporal_burst_window", 5))
        )
        self.roi_temporal_burst_trigger_count = max(
            1,
            min(
                int(module_config.get("roi_temporal_burst_trigger_count", 2)),
                self.roi_temporal_burst_window,
            ),
        )
        self.roi_temporal_history: deque[int] = deque(maxlen=self.roi_temporal_burst_window)
        self.benign_global_motion_filter_enabled = bool(
            module_config.get("benign_global_motion_filter_enabled", True)
        )
        self.benign_global_motion_region_count_min = int(
            module_config.get("benign_global_motion_region_count_min", 500)
        )
        self.benign_global_motion_overexposure_max = float(
            module_config.get("benign_global_motion_overexposure_max", 0.02)
        )
        self.light_flow_enabled = bool(module_config.get("light_flow_enabled", True))
        self.light_flow_interval = max(1, int(module_config.get("light_flow_interval", 3)))
        self.light_flow_temporal_candidate = float(
            module_config.get("light_flow_temporal_candidate", 0.025)
        )
        self.light_flow_local_temporal_candidate = float(
            module_config.get("light_flow_local_temporal_candidate", 0.18)
        )
        self.static_image_enabled = bool(module_config.get("static_image_enabled", True))
        self.static_image_interval = max(
            1,
            int(
                module_config.get(
                    "static_image_interval", 10
                )
            ),
        )
        self.static_image_temporal_candidate = float(
            module_config.get("static_image_temporal_candidate", 0.015)
        )
        self.static_image_local_temporal_candidate = float(
            module_config.get("static_image_local_temporal_candidate", 0.10)
        )
        self.static_image_hold_frames = max(
            0, int(module_config.get("static_image_hold_frames", 5))
        )
        self.static_image_hold_remaining = 0
        self.static_image_hold_score = 0.0
        self.a3b_display_alpha = float(module_config.get("a3b_display_alpha", 0.35))
        self.a3b_display_score = 0.0
        self.p_adv_display_alpha = float(module_config.get("p_adv_display_alpha", 0.35))
        self.p_adv_display_score = 0.0
        self.static_media_replay_window = max(
            1, int(module_config.get("static_media_replay_window", 30))
        )
        self.static_media_replay_min_p_media = float(
            module_config.get("static_media_replay_min_p_media", 0.70)
        )
        self.static_media_replay_max_bbox_area = float(
            module_config.get("static_media_replay_max_bbox_area", 12000.0)
        )
        self.static_media_replay_min_temporal = float(
            module_config.get("static_media_replay_min_temporal", 0.03)
        )
        self.static_media_replay_min_blur = float(
            module_config.get("static_media_replay_min_blur", 0.45)
        )
        self.static_media_replay_min_warp_residual = float(
            module_config.get("static_media_replay_min_warp_residual", 0.18)
        )
        self.static_media_replay_min_flow_gap = float(
            module_config.get("static_media_replay_min_flow_gap", 0.35)
        )
        self.static_media_replay_max_center_span = float(
            module_config.get("static_media_replay_max_center_span", 80.0)
        )
        self.static_media_replay_max_area_ratio = float(
            module_config.get("static_media_replay_max_area_ratio", 3.0)
        )
        self.static_media_replay_votes: deque[int] = deque(
            maxlen=self.static_media_replay_window
        )
        self.static_media_replay_bboxes: deque[tuple[float, float, float] | None] = deque(
            maxlen=self.static_media_replay_window
        )
        self.static_media_fast_window = max(
            1, int(module_config.get("static_media_fast_window", 6))
        )
        self.static_media_fast_trigger_count = max(
            1, int(module_config.get("static_media_fast_trigger_count", 3))
        )
        self.static_media_fast_min_p_media = float(
            module_config.get("static_media_fast_min_p_media", 0.80)
        )
        self.static_media_fast_min_replay_signal = float(
            module_config.get("static_media_fast_min_replay_signal", 0.30)
        )
        self.static_media_fast_alt_min_p_media = float(
            module_config.get("static_media_fast_alt_min_p_media", 0.72)
        )
        self.static_media_fast_alt_min_warp_residual = float(
            module_config.get("static_media_fast_alt_min_warp_residual", 0.18)
        )
        self.static_media_fast_alt_min_flow_gap = float(
            module_config.get("static_media_fast_alt_min_flow_gap", 0.80)
        )
        self.static_media_fast_edge_margin = float(
            module_config.get("static_media_fast_edge_margin", 3.0)
        )
        self.static_media_fast_votes: deque[int] = deque(
            maxlen=self.static_media_fast_window
        )
        self.static_media_fast_bboxes: deque[tuple[float, float, float] | None] = deque(
            maxlen=self.static_media_fast_window
        )
        self.static_media_occlusion_hold_frames = max(
            0, int(module_config.get("static_media_occlusion_hold_frames", 1500))
        )
        self.static_media_occlusion_min_score = float(
            module_config.get("static_media_occlusion_min_score", 0.76)
        )
        self.static_media_occlusion_reacquire_min_p_media = float(
            module_config.get("static_media_occlusion_reacquire_min_p_media", 0.55)
        )
        self.static_media_occlusion_hold_remaining = 0
        self.static_media_occlusion_hold_score = 0.0
        self.static_media_occlusion_last_reason = "none"
        self.static_media_display_alpha = float(module_config.get("static_media_display_alpha", 0.35))
        self.a3b_display_score = 0.0
        self.p_adv_display_score = 0.0
        self.static_media_display_score = 0.0
        self.source_authenticity_enabled = bool(
            module_config.get("source_authenticity_enabled", False)
        )
        # --- Synth_Classifier (A6 sub-feature, Task 6.4 / Req 9.1+9.2+9.6) ---
        # Independent of the main A4 ``classifier_fusion`` AND of the
        # Static_Media_Classifier: this artifact scores the clip-level
        # 15-dim aggregate of Source_Authenticity features and, when the
        # rollout gate is open, takes over ``p_synth`` via its
        # ``classifier_p_adv`` output. Not configured -> not constructed
        # -> current hand-weighted ``p_synth`` formula is preserved
        # bit-for-bit (Req 6.3 spirit: no silent activation).
        self.synth_classifier_enabled = bool(module_config.get("synth_classifier_enabled", False))
        self.synth_classifier_window = max(1, int(module_config.get("synth_classifier_window", 60)))
        self.synth_classifier: TorchLogisticFusion | None = None
        synth_classifier_artifact = module_config.get("synth_classifier_artifact")
        if synth_classifier_artifact:
            synth_artifact_path = _resolve_artifact_path(str(synth_classifier_artifact))
            self.synth_classifier = TorchLogisticFusion(
                synth_artifact_path,
                self.device,
                calibration_model=module_config.get("synth_classifier_calibration_model"),
                threshold_override=module_config.get("synth_classifier_threshold_override"),
            )

        self.fusion_backend = str(module_config.get("fusion_backend", "rule")).lower()
        if self.fusion_backend not in {"rule", "classifier", "rule_or_classifier"}:
            raise ValueError(f"Unsupported Module A fusion backend: {self.fusion_backend}")

        self.scheduler = ModuleAScheduler(module_config.get("keyframe_interval", 3))
        self.alert_state = AlertState(
            window=module_config.get("alert_window", 5),
            trigger_count=module_config.get("alert_trigger_count", 3),
            hold_frames=module_config.get("attack_state_hold_frames", 4),
        )
        self.glare_ratio_threshold = float(module_config.get("glare_ratio_threshold", 0.06))
        self.a3b_glare_suppress_frames = max(
            0, int(module_config.get("a3b_glare_suppress_frames", 30))
        )
        self.a3b_glare_suppress_remaining = 0
        self.overexposure = GPUOverexposureDetector(
            self.glare_ratio_threshold
        )
        self.texture = GPULBPTextureAnalyzer(
            radius=module_config.get("lbp_radius", 3),
            grid_size=module_config.get("texture_grid_size", 16),
            emit_roi_details=self.emit_roi_texture_details,
        )
        self.temporal = GPUTemporalTextureAnalyzer(
            threshold=module_config.get("lbp_temporal_change_threshold", 0.25),
            grid_size=module_config.get("texture_grid_size", 16),
            emit_roi_details=self.emit_roi_temporal_details,
            persistence_frames=int(module_config.get("temporal_persistence_frames", 1)),
            adaptive_baseline=bool(module_config.get("temporal_adaptive_baseline", True)),
            adaptive_ema_alpha=float(module_config.get("temporal_adaptive_ema_alpha", 0.02)),
            adaptive_multiplier=float(module_config.get("temporal_adaptive_multiplier", 2.0)),
            adaptive_floor=float(module_config.get("temporal_adaptive_floor", 0.015)),
        )
        self.motion = GPUMotionArtifactDetector(
            diff_threshold=module_config.get("motion_diff_threshold", 25.0),
            grid_size=module_config.get("motion_grid_size", 16),
            emit_roi_details=self.emit_roi_motion_details,
        )
        self.blur = GPUBlurDegradationDetector(
            roi_energy_ratio_trigger=module_config.get("blur_roi_energy_ratio_trigger", 0.55),
            low_energy_ratio_trigger=module_config.get("blur_low_energy_ratio_trigger", 0.55),
            low_energy_global_factor=module_config.get("blur_low_energy_global_factor", 0.35),
            min_roi_area=module_config.get("blur_min_roi_area", 900),
            emit_roi_details=self.emit_roi_motion_details,
        )
        self.track = TrackConsistencyAnalyzer(
            labels=tuple(module_config.get("track_labels", ("person", "helmet", "head"))),
            iou_threshold=module_config.get("track_iou_threshold", 0.12),
            center_distance_ratio=module_config.get("track_center_distance_ratio", 0.55),
            high_confidence=module_config.get("track_high_confidence", 0.45),
            confidence_drop_trigger=module_config.get("track_confidence_drop_trigger", 0.25),
            max_missing=module_config.get("track_max_missing", 4),
            score_normalizer=module_config.get("track_score_normalizer", 2.0),
            max_candidates_per_label=module_config.get("track_max_candidates_per_label", 4),
            max_tracks_per_label=module_config.get("track_max_tracks_per_label", 8),
        )
        self.light_flow = GPULightOpticalFlowDetector(
            flow_size=module_config.get("light_flow_size", 160),
            window_size=module_config.get("light_flow_window_size", 9),
            grid_size=module_config.get(
                "light_flow_grid_size", module_config.get("motion_grid_size", 16)
            ),
            residual_threshold=module_config.get("light_flow_residual_threshold", 0.75),
            min_magnitude=module_config.get("light_flow_min_magnitude", 0.25),
            cell_ratio_threshold=module_config.get("light_flow_cell_ratio_threshold", 0.20),
            score_region_normalizer=module_config.get("light_flow_score_region_normalizer", 8.0),
            emit_roi_details=self.emit_roi_motion_details,
        )
        self.static_image = GPUStaticMediaSpoofDetector(
            target_labels=tuple(module_config.get("static_image_target_labels", ("person",))),
            screen_labels=tuple(
                module_config.get("static_image_screen_labels", ("helmet", "head"))
            ),
            patch_size=module_config.get("static_image_patch_size", 64),
            min_similarity=module_config.get("static_image_min_similarity", 0.94),
            trigger_stable_count=module_config.get("static_image_trigger_stable_count", 2),
            min_edge_mean=module_config.get("static_image_min_edge_mean", 0.038),
            screen_min_edge_mean=module_config.get("static_image_screen_min_edge_mean", 0.018),
            min_center_motion=module_config.get("static_image_min_center_motion", 0.0012),
            context_motion_threshold=module_config.get(
                "static_image_context_motion_threshold", 0.010
            ),
            context_contrast_threshold=module_config.get(
                "static_image_context_contrast_threshold", 1.6
            ),
            min_roi_area=module_config.get("static_image_min_roi_area", 1200),
            screen_min_roi_area=module_config.get("static_image_screen_min_roi_area", 450),
            screen_max_roi_area=module_config.get("static_image_screen_max_roi_area", 8000),
            screen_context_expand_ratio=module_config.get(
                "static_image_screen_context_expand_ratio", 2.4
            ),
            screen_min_context_edge_mean=module_config.get(
                "static_image_screen_min_context_edge_mean", 0.004
            ),
            screen_min_context_std=module_config.get("static_image_screen_min_context_std", 0.22),
            screen_min_line_score=module_config.get("static_image_screen_min_line_score", 0.10),
            screen_max_roi_context_area_ratio=module_config.get(
                "static_image_screen_max_roi_context_area_ratio", 0.42
            ),
            screen_person_containment_threshold=module_config.get(
                "static_image_screen_person_containment_threshold", 0.72
            ),
            min_roi_confidence=module_config.get("static_image_min_roi_confidence", 0.50),
            score_trigger=module_config.get("static_image_score_trigger", 0.80),
            expand_ratio=module_config.get("static_image_expand_ratio", 0.35),
            edge_margin_px=module_config.get("static_image_edge_margin_px", 6),
            min_same_label_count=module_config.get("static_image_min_same_label_count", 2),
            max_person_area_ratio=module_config.get("static_image_max_person_area_ratio", 0.65),
            max_context_iou=module_config.get("static_image_max_context_iou", 0.20),
            max_tracks=module_config.get("static_image_max_tracks", 64),
            emit_roi_details=self.emit_roi_motion_details,
            multiscale_fallback_enabled=module_config.get(
                "static_image_multiscale_fallback_enabled", True
            ),
            multiscale_trigger_count=module_config.get(
                "static_image_multiscale_trigger_count", 1
            ),
            backend=module_config.get("static_image_backend", "legacy"),
        )
        self.source_authenticity = SourceAuthenticityDetector(
            enabled=self.source_authenticity_enabled,
            interval=module_config.get(
                "source_authenticity_interval", module_config.get("keyframe_interval", 3)
            ),
            window=module_config.get("source_authenticity_window", 30),
            min_window=module_config.get("source_authenticity_min_window", 8),
            threshold=module_config.get("source_authenticity_threshold", 0.78),
            warning_window=module_config.get("source_authenticity_warning_window", 4),
            warning_trigger_count=module_config.get("source_authenticity_warning_trigger_count", 2),
            hold_frames=module_config.get("source_authenticity_hold_frames", 10),
            repeated_diff_threshold=module_config.get(
                "source_authenticity_repeated_diff_threshold", 0.0035
            ),
            low_motion_threshold=module_config.get(
                "source_authenticity_low_motion_threshold", 0.010
            ),
            low_edge_threshold=module_config.get("source_authenticity_low_edge_threshold", 0.020),
            flicker_threshold=module_config.get("source_authenticity_flicker_threshold", 0.018),
            roi_jitter_threshold=module_config.get(
                "source_authenticity_roi_jitter_threshold", 0.020
            ),
            clip_classifier=self.synth_classifier,
            classifier_enabled=self.synth_classifier_enabled,
            classifier_window=self.synth_classifier_window,
        )
        self.fusion = GPURuleFusion(
            device=self.device,
            weights=module_config.get("module_a_fusion_weights", (0.20, 0.30, 0.20, 0.10, 0.20)),
            threshold=module_config.get("p_adv_threshold", 0.55),
            temporal_trigger=module_config.get("temporal_trigger", 0.03),
            local_temporal_trigger=module_config.get("local_temporal_trigger", 0.045),
            local_flow_ratio_trigger=module_config.get("local_flow_ratio_trigger", 0.42),
            strong_temporal_trigger=module_config.get("strong_temporal_trigger", 0.10),
            strong_local_temporal_trigger=module_config.get("strong_local_temporal_trigger", 0.50),
            paired_local_temporal_trigger=module_config.get("paired_local_temporal_trigger", 0.50),
            paired_local_flow_trigger=module_config.get("paired_local_flow_trigger", 0.45),
            light_flow_anomaly_trigger=module_config.get("light_flow_anomaly_trigger", 0.22),
            light_flow_score_trigger=module_config.get("light_flow_score_trigger", 0.35),
            paired_light_flow_temporal_trigger=module_config.get(
                "paired_light_flow_temporal_trigger", 0.35
            ),
            blur_score_trigger=module_config.get("blur_score_trigger", 0.45),
            paired_blur_temporal_trigger=module_config.get("paired_blur_temporal_trigger", 0.18),
            track_score_trigger=module_config.get("track_score_trigger", 0.50),
            paired_track_temporal_trigger=module_config.get("paired_track_temporal_trigger", 0.18),
            paired_track_blur_trigger=module_config.get("paired_track_blur_trigger", 0.25),
            static_image_score_trigger=module_config.get("static_image_score_trigger", 0.76),
        )
        self.classifier_fusion: TorchLogisticFusion | None = None
        classifier_artifact = module_config.get("classifier_artifact")
        if self.fusion_backend in {"classifier", "rule_or_classifier"}:
            if not classifier_artifact:
                raise ValueError(
                    "classifier_artifact is required when fusion_backend uses classifier"
                )
            artifact_path = _resolve_artifact_path(str(classifier_artifact))
            self.classifier_fusion = TorchLogisticFusion(
                artifact_path,
                self.device,
                calibration_model=module_config.get("classifier_calibration_model"),
                threshold_override=module_config.get("classifier_threshold_override"),
            )

        # --- Target-anchored analyzer (2026-05-13) ---
        self.target_anchored = TargetAnchoredAnalyzer(
            roi_blur_threshold=float(module_config.get("target_anchored_blur_threshold", 0.40)),
            roi_overexposure_threshold=float(
                module_config.get("target_anchored_overexposure_threshold", 0.15)
            ),
            roi_confidence_drop_threshold=float(
                module_config.get("target_anchored_confidence_drop_threshold", 0.25)
            ),
            roi_texture_anomaly_threshold=float(
                module_config.get("target_anchored_texture_anomaly_threshold", 0.12)
            ),
            track_drop_threshold=float(
                module_config.get("target_anchored_track_drop_threshold", 0.40)
            ),
            track_confidence_drop_threshold=float(
                module_config.get("target_anchored_track_confidence_drop_threshold", 0.20)
            ),
        )

        # --- Static_Media_Classifier (A3b sub-feature, Task 5.4 / Req 7.3+7.5) ---
        # Independent of the main A4 ``classifier_fusion``: this artifact scores
        # only the per-frame ``static_media`` feature block and feeds its output
        # back via ``classifier_score`` / ``classifier_triggered`` under
        # ``module_a_features.static_media``. Not configured -> not constructed
        # -> current heuristic behaviour is preserved bit-for-bit.
        #
        # The ``static_media_classifier_enabled`` switch (default False) is the
        # rollout gate: training may land before real hand-held LAN footage is
        # available, in which case the artifact is loaded for shadow scoring
        # but must NOT push ``static_image_triggered`` (Req 6.3 hard rule).
        self.static_media_classifier_enabled = bool(
            module_config.get("static_media_classifier_enabled", False)
        )
        self.static_media_classifier: TorchLogisticFusion | None = None
        static_media_classifier_artifact = module_config.get("static_media_classifier_artifact")
        if static_media_classifier_artifact:
            sm_artifact_path = _resolve_artifact_path(str(static_media_classifier_artifact))
            self.static_media_classifier = TorchLogisticFusion(
                sm_artifact_path,
                self.device,
                calibration_model=module_config.get("static_media_classifier_calibration_model"),
                threshold_override=module_config.get(
                    "static_media_classifier_threshold_override"
                ),
            )

        self.prev_gray: torch.Tensor | None = None
        self.prev_lbp: torch.Tensor | None = None
        self.frame_idx = 0
        self._last_light_flow_score: float = 0.0
        self._last_light_flow_ratio: float = 0.0

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_lbp = None
        self.frame_idx = 0
        self._last_light_flow_score = 0.0
        self._last_light_flow_ratio = 0.0
        self.roi_temporal_history.clear()
        self.alert_state.reset()
        self.track.reset()
        self.static_image.reset()
        self.source_authenticity.reset()
        self.temporal.reset()
        self.target_anchored.reset()
        self.static_image_hold_remaining = 0
        self.static_image_hold_score = 0.0
        self.static_media_replay_votes.clear()
        self.static_media_replay_bboxes.clear()
        self.static_media_fast_votes.clear()
        self.static_media_fast_bboxes.clear()
        self.static_media_occlusion_hold_remaining = 0
        self.static_media_occlusion_hold_score = 0.0
        self.static_media_occlusion_last_reason = "none"
        self.a3b_display_score = 0.0
        self.p_adv_display_score = 0.0
        self.static_media_display_score = 0.0
        self.a3b_glare_suppress_remaining = 0

    def process(self, item: ModuleAInput) -> ModuleAResult:
        started = time.perf_counter()
        frame = item.frame
        if frame.shape[0] != self.frame_size or frame.shape[1] != self.frame_size:
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))

        rois = self._prepare_rois(item.rois, frame.shape[1], frame.shape[0])
        gray = self._frame_to_gray_tensor(frame)

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
        source_auth_ms = 0.0

        # A1 — overexposure
        _t0 = time.perf_counter()
        overexposure = self.overexposure.compute(gray)
        if (
            bool(overexposure.get("is_glare", False))
            or float(overexposure.get("ratio", 0.0)) >= self.glare_ratio_threshold
        ):
            self.a3b_glare_suppress_remaining = self.a3b_glare_suppress_frames
        elif self.a3b_glare_suppress_remaining > 0:
            self.a3b_glare_suppress_remaining -= 1
        a1_overexposure_ms = (time.perf_counter() - _t0) * 1000.0

        # A2 — LBP texture + temporal texture
        _t0 = time.perf_counter()
        lbp = self.texture.compute_lbp(gray)
        texture = self.texture.summarize(lbp, rois)
        temporal = self.temporal.compute(self.prev_lbp, lbp, rois, radius=self.texture.radius)
        a2_temporal_ms = (time.perf_counter() - _t0) * 1000.0

        # A3 — motion artifact + blur + track + light-flow (+ merge)
        _t0 = time.perf_counter()
        motion = self.motion.compute(self.prev_gray, gray, rois)
        blur = self.blur.compute(gray, rois)
        track = self.track.compute(rois)
        light_flow = self.light_flow.compute(
            self.prev_gray,
            gray,
            rois,
            run=self._should_run_light_flow(item.frame_idx, temporal),
        )
        # Hold-last-value for light_flow: when skipped, carry forward the
        # last computed score so p_adv / display doesn't flicker.
        if light_flow.get("light_flow_available", False):
            self._last_light_flow_score = float(light_flow.get("light_flow_score", 0.0))
            self._last_light_flow_ratio = float(
                light_flow.get("light_flow_local_anomaly_ratio", 0.0)
            )
        else:
            light_flow["light_flow_score"] = getattr(self, "_last_light_flow_score", 0.0)
            light_flow["light_flow_local_anomaly_ratio"] = getattr(
                self, "_last_light_flow_ratio", 0.0
            )
        motion = self._merge_light_flow(motion, light_flow)
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
        if self.static_image_enabled and self.prev_gray is not None:
            _t0 = time.perf_counter()
            run_roi_pass = self._should_run_static_image(item.frame_idx, temporal)
            static_image = self.static_image.compute(
                self.prev_gray,
                gray,
                rois if run_roi_pass else None,
                context_rois=rois,
            )
            if (
                self.a3b_glare_suppress_remaining > 0
            ) and bool(static_image.get("static_image_triggered", False)) and not bool(
                static_image.get("p_media_triggered", False)
            ):
                static_image["static_image_triggered"] = False
                static_image["static_image_score"] = min(
                    float(static_image.get("static_image_score", 0.0)), 0.40
                )
                static_image["static_image_triggered_source"] = "physical_glare_suppressed"
            # Hold-last-value: when the ROI pass didn't run, the detector
            # returns zeros. Replace with the held score so downstream
            # fusion / display stays continuous.
            if not run_roi_pass and self.static_image_hold_score > 0.0:
                static_image["static_image_score"] = self.static_image_hold_score
            elif run_roi_pass:
                # Update the held score from the fresh computation.
                self.static_image_hold_score = float(
                    static_image.get("static_image_score", 0.0)
                )
            replay_state = self._update_static_media_replay_state(
                static_image, temporal, blur
            )
            fast_state = self._update_static_media_fast_state(static_image, replay_state)
            live_score = max(
                float(static_image.get("static_image_score", 0.0)),
                float(static_image.get("p_media", 0.0)),
                float(replay_state.get("p_media", 0.0)),
            )
            self.static_media_display_score = self._ema(
                self.static_media_display_score,
                live_score,
                self.static_media_display_alpha,
            )
            static_image["static_image_live_score_raw"] = live_score
            static_image["static_image_live_score_display"] = float(self.static_media_display_score)
            static_image["static_image_live_score"] = float(self.static_media_display_score)
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
            occlusion_state = self._update_static_media_occlusion_state(
                static_image, replay_state, fast_state
            )
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
            motion = self._merge_static_image(motion, static_image)
            a3b_static_media_ms = (time.perf_counter() - _t0) * 1000.0

        # A3b-classifier — Static_Media_Classifier scoring (Task 5.4 / Req 7.3+7.5)
        # Runs every frame when the artifact is configured so that
        # ``classifier_score`` is always available for event evidence / offline
        # replay, regardless of whether the rollout gate is open. The OR
        # combination with the existing heuristic only kicks in when
        # ``static_media_classifier_enabled`` is True (Req 6.3 hard rule).
        # Timing is folded into the A3b bucket so ``module_a_breakdown`` stays
        # in the 6-field contract established by Task 3.1.
        if self.static_media_classifier is not None:
            _t0 = time.perf_counter()
            classifier_features = self._build_static_media_classifier_features(motion)
            sm_classifier = self.static_media_classifier.compute(classifier_features)

            classifier_p_adv = float(sm_classifier["classifier_p_adv"])
            classifier_triggered = bool(sm_classifier["classifier_triggered"])
            motion["static_image_classifier_score"] = classifier_p_adv
            motion["static_image_classifier_triggered"] = classifier_triggered
            motion["static_image_classifier_threshold"] = float(
                sm_classifier["classifier_threshold"]
            )
            motion["static_image_classifier_artifact"] = str(sm_classifier["classifier_artifact"])
            motion["static_image_classifier_kind"] = str(sm_classifier["classifier_kind"])
            motion["static_image_classifier_enabled"] = self.static_media_classifier_enabled

            # OR semantics — only when the gate is open does the classifier
            # actually push ``static_image_triggered`` / ``static_image_score``.
            if self.static_media_classifier_enabled and classifier_triggered:
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

            a3b_static_media_ms += (time.perf_counter() - _t0) * 1000.0

        # Source_Authenticity (p_synth)
        _t0 = time.perf_counter()
        source_auth = self.source_authenticity.compute(
            self.prev_gray,
            gray,
            rois,
            item.frame_idx,
            temporal=temporal,
            motion=motion,
        )
        source_auth_ms = (time.perf_counter() - _t0) * 1000.0

        # A4 — Target-anchored suspicious判定 + rule fusion + classifier
        # ================================================================
        # 2026-05-13 重写：suspicious 判定从"全图统计量驱动"改为
        # "目标锚点驱动"。参考 doc/A3_target_anchored_false_positive_suppression.txt
        _a4_t0 = time.perf_counter()
        fusion = self.fusion.compute(texture, temporal, motion, overexposure, blur, track)

        # A4 classifier 仍然运行（计算 p_adv），但不独立触发 suspicious
        classifier_result = None
        if self.classifier_fusion is not None:
            classifier_features = self._build_classifier_features(
                overexposure=overexposure,
                texture=texture,
                temporal=temporal,
                motion=motion,
                blur=blur,
                track=track,
                fusion=fusion,
                roi_count=len(rois),
            )
            classifier_result = self.classifier_fusion.compute(classifier_features)
            fusion.update(classifier_result)
            # p_adv 取 rule 和 classifier 的 max（用于显示/记录）
            fusion["p_adv"] = max(
                float(fusion.get("p_adv", 0.0)),
                float(classifier_result.get("classifier_p_adv", 0.0)),
            )
        self.p_adv_display_score = self._ema(
            self.p_adv_display_score,
            float(fusion.get("p_adv", 0.0)),
            self.p_adv_display_alpha,
        )
        fusion["p_adv_display"] = float(self.p_adv_display_score)

        # --- Target-anchored 判定（核心改动）---
        # 构建 static_image 信息供 target_anchored 使用
        static_image_info = {
            "triggered": bool(motion.get("static_image_triggered", False)),
            "score": float(motion.get("static_image_score", 0.0)),
        }
        anchored = self.target_anchored.evaluate(
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
        fusion["reason_codes"] = reason_codes
        fusion["target_anchored"] = anchored

        # static_image hold 逻辑保留（A3b 触发后保持几帧）
        if static_image_info["triggered"]:
            self.static_image_hold_remaining = self.static_image_hold_frames
            self.static_image_hold_score = static_image_info["score"]
        elif self.static_image_hold_remaining > 0:
            self.static_image_hold_remaining -= 1

        if self.static_image_hold_remaining > 0 and not suspicious:
            # A3b 之前触发过，hold 期间保持 suspicious
            suspicious = True
            if "static_image_spoof_hold" not in reason_codes:
                reason_codes.append("static_image_spoof_hold")
            fusion["reason_codes"] = reason_codes

        fusion["is_suspicious"] = suspicious
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
        alert_confirmed, attack_state_active = self.alert_state.update(
            suspicious,
            frame_ts=alert_frame_ts,
        )
        source_auth = self._suppress_source_auth_for_physical_event(
            source_auth,
            alert_confirmed,
            attack_state_active,
        )

        self.prev_gray = gray.detach()
        self.prev_lbp = lbp.detach()
        self.frame_idx = item.frame_idx + 1

        timing_ms = (time.perf_counter() - started) * 1000.0
        roi_results = self._merge_roi_results(texture, temporal, motion)
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
            "p_synth": float(source_auth.get("p_synth", 0.0)),
            "source_authenticity_repeated_ratio": float(
                source_auth.get("source_authenticity_repeated_ratio", 0.0)
            ),
            "source_authenticity_highfreq_std": float(
                source_auth.get("source_authenticity_highfreq_std", 0.0)
            ),
            "source_authenticity_roi_jitter": float(
                source_auth.get("source_authenticity_roi_jitter", 0.0)
            ),
            "blur_score": float(blur.get("blur_score", 0.0)),
            "blur_roi_energy_ratio": float(blur.get("blur_roi_energy_ratio", 1.0)),
            "blur_low_energy_ratio": float(blur.get("blur_low_energy_ratio", 0.0)),
            "track_score": float(track.get("track_score", 0.0)),
            "track_drop_score": float(track.get("track_drop_score", 0.0)),
            "track_missing_count": int(track.get("missing_track_count", 0)),
            "confidence_drop_score": float(track.get("confidence_drop_score", 0.0)),
            "overexposure_ratio": float(overexposure["ratio"]),
        }
        details = self._build_details(
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
            "source_auth_ms": float(source_auth_ms),
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
        flow_region_count = float(motion.get("region_count", 0.0))
        flow_max = float(motion.get("max_magnitude", 0.0))
        return {
            "overexposure_ratio": float(overexposure.get("ratio", 0.0)),
            "underexposed_ratio": float(overexposure.get("underexposed_ratio", 0.0)),
            "texture_score": float(texture.get("delta_h", 0.0)),
            "local_texture_score": float(texture.get("local_max", 0.0)),
            "texture_global_mean": float(texture.get("global_mean", 0.0)),
            "texture_global_std": float(texture.get("global_std", 0.0)),
            "temporal_change": float(temporal.get("change_t", 0.0)),
            "local_temporal_change": float(temporal.get("local_max", 0.0)),
            "flow_region_ratio": min(1.0, flow_region_count / 6400.0),
            "flow_max_magnitude_norm": min(1.0, flow_max / 255.0),
            "local_flow_ratio": float(motion.get("local_max_ratio", 0.0)),
            "motion_score": float(motion.get("motion_score", 0.0)),
            "light_flow_score": float(motion.get("light_flow_score", 0.0)),
            "light_flow_local_anomaly_ratio": float(
                motion.get("light_flow_local_anomaly_ratio", 0.0)
            ),
            "light_flow_max_residual_norm": min(
                1.0, float(motion.get("light_flow_max_residual", 0.0)) / 16.0
            ),
            "static_image_score": float(motion.get("static_image_score", 0.0)),
            "static_image_patch_similarity": float(
                motion.get("static_image_patch_similarity", 0.0)
            ),
            "static_image_center_motion": float(motion.get("static_image_center_motion", 0.0)),
            "static_image_screen_like": 1.0
            if motion.get("static_image_screen_like", False)
            else 0.0,
            "blur_score": float(blur.get("blur_score", 0.0)),
            "blur_roi_energy_ratio": min(4.0, float(blur.get("blur_roi_energy_ratio", 1.0))) / 4.0,
            "blur_low_energy_ratio": float(blur.get("blur_low_energy_ratio", 0.0)),
            "track_score": float(track.get("track_score", 0.0)),
            "track_drop_score": float(track.get("track_drop_score", 0.0)),
            "confidence_drop_score": float(track.get("confidence_drop_score", 0.0)),
            "track_missing_count_norm": min(1.0, float(track.get("missing_track_count", 0)) / 5.0),
            "track_confidence_drop_count_norm": min(
                1.0, float(track.get("confidence_drop_count", 0)) / 5.0
            ),
            "roi_count_norm": min(1.0, float(roi_count) / 20.0),
            "rule_p_adv": float(fusion.get("p_adv", 0.0)),
            "rule_temporal_triggered": 1.0 if fusion.get("temporal_triggered", False) else 0.0,
            "rule_local_temporal_triggered": 1.0
            if fusion.get("local_temporal_triggered", False)
            else 0.0,
            "rule_roi_temporal_triggered": 1.0
            if fusion.get("roi_temporal_triggered", False)
            else 0.0,
            "rule_sustained_roi_temporal_triggered": 1.0
            if fusion.get("sustained_roi_temporal_triggered", False)
            else 0.0,
            "rule_local_flow_triggered": 1.0 if fusion.get("local_flow_triggered", False) else 0.0,
            "rule_strong_temporal_triggered": 1.0
            if fusion.get("strong_temporal_triggered", False)
            else 0.0,
            "rule_paired_temporal_flow_triggered": 1.0
            if fusion.get("paired_temporal_flow_triggered", False)
            else 0.0,
            "rule_light_flow_triggered": 1.0 if fusion.get("light_flow_triggered", False) else 0.0,
            "rule_paired_temporal_light_flow_triggered": 1.0
            if fusion.get("paired_temporal_light_flow_triggered", False)
            else 0.0,
            "rule_blur_triggered": 1.0 if fusion.get("blur_triggered", False) else 0.0,
            "rule_paired_temporal_blur_triggered": 1.0
            if fusion.get("paired_temporal_blur_triggered", False)
            else 0.0,
            "rule_track_triggered": 1.0 if fusion.get("track_triggered", False) else 0.0,
            "rule_paired_track_triggered": 1.0
            if fusion.get("paired_track_triggered", False)
            else 0.0,
            "rule_static_image_triggered": 1.0
            if fusion.get("static_image_triggered", False)
            else 0.0,
            "rule_overexposure_triggered": 1.0
            if fusion.get("overexposure_triggered", False)
            else 0.0,
            "rule_benign_global_motion_suppressed": 1.0
            if fusion.get("benign_global_motion_suppressed", False)
            else 0.0,
        }

    def _build_static_media_classifier_features(
        self,
        motion: dict[str, Any],
    ) -> dict[str, float]:
        """Build the 16-dim feature vector consumed by Static_Media_Classifier.

        Mirrors :func:`tools.train_static_media_classifier.extract_classifier_features`
        so the production inference path and the offline training path score
        the same feature definitions. ``roi_count_norm`` is intentionally fixed
        at 0.5 to match the training extractor (the per-frame ``static_media``
        block has no equivalent count; the training dataset uses the same
        placeholder). ``best_stable_count_norm`` and ``trigger_count_norm``
        share the training-side divisor of 5.0.

        All lookups default to ``0.0``/``False`` so that frames on which
        ``_should_run_static_image`` skipped the A3b block (and therefore did
        not populate ``motion["static_image_*"]``) still produce a well-defined
        16-dim vector; those frames score out to the classifier's "no signal"
        region and do not spuriously trigger.
        """
        stable_count = float(motion.get("static_image_stable_count", 0))
        trigger_count = float(motion.get("static_image_trigger_count", 0))
        return {
            "best_static_image_score": float(motion.get("static_image_score", 0.0)),
            "best_patch_similarity": float(motion.get("static_image_patch_similarity", 0.0)),
            "best_stable_count_norm": min(1.0, stable_count / 5.0),
            "best_center_motion": float(motion.get("static_image_center_motion", 0.0)),
            "best_roi_motion": float(motion.get("static_image_roi_motion", 0.0)),
            "best_context_motion": float(motion.get("static_image_context_motion", 0.0)),
            "best_motion_contrast": float(motion.get("static_image_motion_contrast", 0.0)),
            "best_edge_mean": float(motion.get("static_image_edge_mean", 0.0)),
            "best_screen_like": 1.0 if motion.get("static_image_screen_like", False) else 0.0,
            "best_screen_context_edge_mean": float(
                motion.get("static_image_screen_context_edge_mean", 0.0)
            ),
            "best_screen_context_std": float(motion.get("static_image_screen_context_std", 0.0)),
            "best_screen_line_score": float(motion.get("static_image_screen_line_score", 0.0)),
            "best_screen_roi_context_area_ratio": float(
                motion.get("static_image_screen_roi_context_area_ratio", 0.0)
            ),
            "best_screen_inside_person": 1.0
            if motion.get("static_image_screen_inside_person", False)
            else 0.0,
            "trigger_count_norm": min(1.0, trigger_count / 5.0),
            # Fixed placeholder matching tools/train_static_media_classifier.py
            # (_DEFAULT_ROI_COUNT_NORM). Kept in sync with the training
            # extractor so production/training scoring stays aligned.
            "roi_count_norm": 0.5,
        }

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

    def _should_run_static_image(self, frame_idx: int, temporal: dict[str, Any]) -> bool:
        if not self.static_image_enabled:
            return False
        if self.prev_gray is None:
            return False
        if frame_idx % self.static_image_interval == 0:
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

        p_media = float(static_image.get("p_media", 0.0))
        target_related = bool(static_image.get("p_media_target_related", False))
        temporal_change = float(temporal.get("change_t", 0.0))
        local_temporal = float(temporal.get("local_max", 0.0))
        blur_score = float(blur.get("blur_score", 0.0))
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
        evidence = (
            p_media >= self.static_media_replay_min_p_media
            and not target_related
            and 0.0 < bbox_area <= self.static_media_replay_max_bbox_area
            and local_evidence
            and replay_evidence
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
        trigger_count = max(8, min(self.static_media_replay_window, int(self.static_media_replay_window * 2 / 3 + 0.999)))
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
        bbox_area = max(
            float(replay_state.get("bbox_area", 0.0) or 0.0),
            float(fast_state.get("bbox_area", 0.0) or 0.0),
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
        }

    def _update_static_media_fast_state(
        self, static_image: dict[str, Any], replay_state: dict[str, Any]
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
        warp_residual = float(scores.get("warp_residual", replay_state.get("warp_residual", 0.0)))
        flow_gap = float(scores.get("flow_gap", replay_state.get("flow_gap", 0.0)))
        replay_signal = max(warp_residual, flow_gap)
        primary_replay_evidence = (
            p_media >= self.static_media_fast_min_p_media
            and warp_residual >= self.static_media_fast_min_replay_signal
        )
        alternate_replay_evidence = False
        fast_replay_evidence = primary_replay_evidence
        evidence = (
            0.0 < bbox_area <= self.static_media_replay_max_bbox_area
            and not touches_vertical_edge
            and (
                (target_related and p_media >= self.static_media_fast_min_p_media)
                or (fast_replay_evidence and touches_horizontal_edge)
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

    @staticmethod
    def _suppress_source_auth_for_physical_event(
        source_auth: dict[str, Any],
        alert_confirmed: bool,
        attack_state_active: bool,
    ) -> dict[str, Any]:
        """Suppress the ``p_synth`` warning while a p_adv alert is active.

        Semantics (Requirements 1.4, 1.5, 9.6):

        * When the Source_Authenticity branch did not raise a warning there is
          nothing to suppress; the two explicit suppression flags are stamped
          ``False`` so downstream consumers always see an unambiguous state.
        * Suppression fires when ``alert_confirmed`` is ``True`` (3/5 or 4/5
          physical-perturbation state machine has confirmed the alert) OR when
          ``attack_state_active`` is ``True``; the latter covers the holdover
          window (``AlertState.hold_remaining > 0``) as well as the single-frame
          suspicious frames that seed that window.
        * ``p_synth`` is preserved bit-for-bit for offline replay; only
          ``source_authenticity_warning`` and ``source_authenticity_confirmed``
          are forced to ``False``.
        * ``source_authenticity_suppressed_by_p_adv`` (design.md) and the
          legacy ``source_authenticity_suppressed_by_physical`` key (consumed
          by ``_build_details`` / ``tools/run_experiment.py``) are set to the
          **same** bool so existing downstream consumers do not break.
        """
        warning = bool(source_auth.get("source_authenticity_warning", False))
        should_suppress = warning and (alert_confirmed or attack_state_active)

        suppressed = dict(source_auth)
        if not should_suppress:
            # Always stamp the two explicit flags so the contract is visible on
            # every frame, not only inside suppression windows.
            suppressed["source_authenticity_suppressed_by_p_adv"] = False
            suppressed["source_authenticity_suppressed_by_physical"] = False
            return suppressed

        suppressed["source_authenticity_warning"] = False
        suppressed["source_authenticity_confirmed"] = False
        suppressed["source_authenticity_suppressed_by_p_adv"] = True
        suppressed["source_authenticity_suppressed_by_physical"] = True
        reason = str(suppressed.get("source_authenticity_reason", ""))
        suppressed["source_authenticity_reason"] = (
            f"{reason}；已被物理扰动通道抑制" if reason else "已被物理扰动通道抑制"
        )
        return suppressed

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
        feature_details = {
            "overexposure": {
                "ratio": float(overexposure["ratio"]),
                "underexposed_ratio": float(overexposure["underexposed_ratio"]),
                "is_glare": bool(overexposure["is_glare"]),
                "threshold": float(overexposure["threshold"]),
            },
            "lbp": {
                "anomaly_ratio": float(texture["delta_h"]),
                "local_max": float(texture["local_max"]),
                "global_mean": float(texture["global_mean"]),
                "global_std": float(texture["global_std"]),
                "backend": "gpu_lbp",
            },
            "temporal": {
                "change_rate": float(temporal["change_t"]),
                "local_max": float(temporal["local_max"]),
                "threshold": float(temporal["threshold"]),
                "backend": "gpu_lbp_temporal",
            },
            "flow": {
                "region_count": int(motion["region_count"]),
                "max_magnitude": float(motion["max_magnitude"]),
                "local_max_ratio": float(motion["local_max_ratio"]),
                "motion_score": float(motion["motion_score"]),
                "backend": motion["backend"],
                "light_flow_available": bool(motion.get("light_flow_available", False)),
                "light_flow_backend": str(motion.get("light_flow_backend", "disabled")),
                "light_flow_region_count": int(motion.get("light_flow_region_count", 0)),
                "light_flow_max_magnitude": float(motion.get("light_flow_max_magnitude", 0.0)),
                "light_flow_mean_magnitude": float(motion.get("light_flow_mean_magnitude", 0.0)),
                "light_flow_dominant_magnitude": float(
                    motion.get("light_flow_dominant_magnitude", 0.0)
                ),
                "light_flow_max_residual": float(motion.get("light_flow_max_residual", 0.0)),
                "light_flow_local_anomaly_ratio": float(
                    motion.get("light_flow_local_anomaly_ratio", 0.0)
                ),
                "light_flow_score": float(motion.get("light_flow_score", 0.0)),
                "light_flow_valid_ratio": float(motion.get("light_flow_valid_ratio", 0.0)),
            },
            "static_image": {
                "score": float(motion.get("static_image_score", 0.0)),
                "live_score": float(motion.get("static_image_live_score_raw", motion.get("static_image_score", 0.0))),
                "live_score_display": float(motion.get("static_image_live_score_display", motion.get("static_image_live_score", motion.get("static_image_score", 0.0)))),
                "triggered": bool(motion.get("static_image_triggered", False)),
                "trigger_count": int(motion.get("static_image_trigger_count", 0)),
                "patch_similarity": float(motion.get("static_image_patch_similarity", 0.0)),
                "stable_count": int(motion.get("static_image_stable_count", 0)),
                "center_motion": float(motion.get("static_image_center_motion", 0.0)),
                "roi_motion": float(motion.get("static_image_roi_motion", 0.0)),
                "context_motion": float(motion.get("static_image_context_motion", 0.0)),
                "motion_contrast": float(motion.get("static_image_motion_contrast", 0.0)),
                "edge_mean": float(motion.get("static_image_edge_mean", 0.0)),
                "screen_like": bool(motion.get("static_image_screen_like", False)),
                "screen_context_edge_mean": float(
                    motion.get("static_image_screen_context_edge_mean", 0.0)
                ),
                "screen_context_std": float(motion.get("static_image_screen_context_std", 0.0)),
                "screen_line_score": float(motion.get("static_image_screen_line_score", 0.0)),
                "screen_roi_context_area_ratio": float(
                    motion.get("static_image_screen_roi_context_area_ratio", 0.0)
                ),
                "screen_inside_person": bool(
                    motion.get("static_image_screen_inside_person", False)
                ),
                "backend": str(motion.get("static_image_backend", "gpu_static_media_spoof")),
                "p_media": float(motion.get("p_media", 0.0)),
                "p_media_triggered": bool(motion.get("p_media_triggered", False)),
                "p_media_type": str(motion.get("p_media_type", "normal")),
                "p_media_bbox": motion.get("p_media_bbox"),
                "p_media_target_related": bool(
                    motion.get("p_media_target_related", False)
                ),
                "p_media_strong_evidence": bool(
                    motion.get("p_media_strong_evidence", False)
                ),
                "p_media_background_static_suppressed": bool(
                    motion.get("p_media_background_static_suppressed", False)
                ),
                "p_media_scores": dict(motion.get("p_media_scores", {})),
                "p_media_replay_state": dict(motion.get("p_media_replay_state", {})),
                "p_media_fast_state": dict(motion.get("p_media_fast_state", {})),
                "p_media_occlusion_state": dict(motion.get("p_media_occlusion_state", {})),
                "triggered_source": str(
                    motion.get("static_image_triggered_source", "none")
                ),
                # Task 5.4 / Req 7.3 — expose Static_Media_Classifier shadow
                # scores on every frame so offline replay and event evidence
                # can reason about classifier vs. heuristic attribution even
                # when the rollout gate (``classifier_enabled``) is closed.
                "classifier_score": float(motion.get("static_image_classifier_score", 0.0)),
                "classifier_triggered": bool(
                    motion.get("static_image_classifier_triggered", False)
                ),
                "classifier_threshold": float(motion.get("static_image_classifier_threshold", 0.0)),
                "classifier_artifact": str(motion.get("static_image_classifier_artifact", "")),
                "classifier_enabled": bool(self.static_media_classifier_enabled),
                "classifier_forced_trigger": bool(
                    motion.get("static_image_classifier_forced_trigger", False)
                ),
            },
            "source_authenticity": {
                "enabled": bool(source_auth.get("source_authenticity_enabled", False)),
                "evaluated": bool(source_auth.get("source_authenticity_evaluated", False)),
                "available": bool(source_auth.get("source_authenticity_available", False)),
                "p_synth": float(source_auth.get("p_synth", 0.0)),
                "warning": bool(source_auth.get("source_authenticity_warning", False)),
                "confirmed": bool(source_auth.get("source_authenticity_confirmed", False)),
                "suppressed_by_physical": bool(
                    source_auth.get("source_authenticity_suppressed_by_physical", False)
                ),
                "suppressed_by_p_adv": bool(
                    source_auth.get("source_authenticity_suppressed_by_p_adv", False)
                ),
                "clip_suspicious": bool(
                    source_auth.get("source_authenticity_clip_suspicious", False)
                ),
                "hold_remaining": int(source_auth.get("source_authenticity_hold_remaining", 0)),
                "reason": str(source_auth.get("source_authenticity_reason", "")),
                "window_size": int(source_auth.get("source_authenticity_window_size", 0)),
                "repeated_ratio": float(source_auth.get("source_authenticity_repeated_ratio", 0.0)),
                "low_motion_ratio": float(
                    source_auth.get("source_authenticity_low_motion_ratio", 0.0)
                ),
                "frame_delta_mean": float(
                    source_auth.get("source_authenticity_frame_delta_mean", 0.0)
                ),
                "edge_mean": float(source_auth.get("source_authenticity_edge_mean", 0.0)),
                "highfreq_mean": float(source_auth.get("source_authenticity_highfreq_mean", 0.0)),
                "highfreq_std": float(source_auth.get("source_authenticity_highfreq_std", 0.0)),
                "diff_std_mean": float(source_auth.get("source_authenticity_diff_std_mean", 0.0)),
                "roi_jitter": float(source_auth.get("source_authenticity_roi_jitter", 0.0)),
                "frame_delta": float(source_auth.get("source_authenticity_frame_delta", 0.0)),
                "scores": source_auth.get("source_authenticity_scores", {}),
                # Task 6.4 — Synth_Classifier shadow / attribution fields.
                # ``handcrafted_p_synth`` is always present (the
                # pre-Task-6.4 formula output), ``classifier_p_synth`` is
                # meaningful only when ``classifier_available`` is True,
                # and ``classifier_active`` tells replay readers whether
                # ``p_synth`` was overwritten by the classifier on this
                # frame. The ``classifier_enabled`` flag mirrors the
                # config gate so downstream code can distinguish "artifact
                # loaded but gate closed" from "artifact never loaded".
                "handcrafted_p_synth": float(
                    source_auth.get("source_authenticity_handcrafted_p_synth", 0.0)
                ),
                "classifier_p_synth": float(
                    source_auth.get("source_authenticity_classifier_p_synth", 0.0) or 0.0
                ),
                "classifier_enabled": bool(
                    source_auth.get("source_authenticity_classifier_enabled", False)
                ),
                "classifier_available": bool(
                    source_auth.get("source_authenticity_classifier_available", False)
                ),
                "classifier_active": bool(
                    source_auth.get("source_authenticity_classifier_active", False)
                ),
                "classifier_window": int(
                    source_auth.get("source_authenticity_classifier_window", 0)
                ),
                "classifier_buffer_size": int(
                    source_auth.get("source_authenticity_classifier_buffer_size", 0)
                ),
                "classifier_artifact": str(
                    source_auth.get("source_authenticity_classifier_artifact", "")
                ),
                "classifier_kind": str(source_auth.get("source_authenticity_classifier_kind", "")),
                "classifier_threshold": float(
                    source_auth.get("source_authenticity_classifier_threshold", 0.0)
                ),
                "backend": str(source_auth.get("source_authenticity_backend", "gpu_clip_stats")),
            },
            "blur": {
                "score": float(blur.get("blur_score", 0.0)),
                "roi_energy_ratio": float(blur.get("blur_roi_energy_ratio", 1.0)),
                "low_energy_ratio": float(blur.get("blur_low_energy_ratio", 0.0)),
                "global_mean": float(blur.get("blur_global_mean", 0.0)),
                "global_max": float(blur.get("blur_global_max", 0.0)),
                "best_roi_is_grid": bool(blur.get("blur_best_roi_is_grid", False)),
                "backend": blur.get("backend", "gpu_laplacian_blur"),
            },
            "track": {
                "score": float(track.get("track_score", 0.0)),
                "drop_score": float(track.get("track_drop_score", 0.0)),
                "confidence_drop_score": float(track.get("confidence_drop_score", 0.0)),
                "missing_track_count": int(track.get("missing_track_count", 0)),
                "confidence_drop_count": int(track.get("confidence_drop_count", 0)),
                "matched_track_count": int(track.get("matched_track_count", 0)),
                "active_track_count": int(track.get("active_track_count", 0)),
                "candidate_roi_count": int(track.get("candidate_roi_count", 0)),
                "backend": track.get("backend", "roi_track_consistency"),
            },
            "fusion": {
                "p_adv": float(fusion["p_adv"]),
                "p_adv_display": float(fusion.get("p_adv_display", fusion["p_adv"])),
                "p_adv_raw": float(fusion["p_adv"]),
                "threshold": float(fusion["threshold"]),
                "backend": self.fusion_backend,
                "reason": ",".join(fusion["reason_codes"]),
                "classifier_p_adv": float(fusion.get("classifier_p_adv", 0.0)),
                "classifier_threshold": float(fusion.get("classifier_threshold", 0.0)),
                "classifier_triggered": bool(fusion.get("classifier_triggered", False)),
                "classifier_kind": str(fusion.get("classifier_kind", "")),
                "classifier_transform_mode": str(fusion.get("classifier_transform_mode", "")),
                "classifier_calibration_model": str(fusion.get("classifier_calibration_model", "")),
                "overexposure_triggered": bool(fusion.get("overexposure_triggered", False)),
                "temporal_triggered": bool(fusion["temporal_triggered"]),
                "flow_triggered": bool(fusion["local_flow_triggered"]),
                "roi_temporal_triggered": bool(fusion.get("roi_temporal_triggered", False)),
                "sustained_roi_temporal_triggered": bool(
                    fusion.get("sustained_roi_temporal_triggered", False)
                ),
                "benign_global_motion_suppressed": bool(
                    fusion.get("benign_global_motion_suppressed", False)
                ),
                "texture_high_triggered": False,
                "texture_low_triggered": False,
                "texture_very_low_triggered": False,
                "local_texture_triggered": False,
                "local_temporal_triggered": bool(fusion["local_temporal_triggered"]),
                "local_flow_triggered": bool(fusion["local_flow_triggered"]),
                "p_adv_triggered": bool(fusion["p_adv_triggered"]),
                "strong_temporal_triggered": bool(fusion["strong_temporal_triggered"]),
                "paired_temporal_flow_triggered": bool(fusion["paired_temporal_flow_triggered"]),
                "light_flow_triggered": bool(fusion.get("light_flow_triggered", False)),
                "paired_temporal_light_flow_triggered": bool(
                    fusion.get("paired_temporal_light_flow_triggered", False)
                ),
                "blur_triggered": bool(fusion.get("blur_triggered", False)),
                "paired_temporal_blur_triggered": bool(
                    fusion.get("paired_temporal_blur_triggered", False)
                ),
                "track_triggered": bool(fusion.get("track_triggered", False)),
                "paired_track_triggered": bool(fusion.get("paired_track_triggered", False)),
                "static_image_triggered": bool(fusion.get("static_image_triggered", False)),
                "static_image_hold_active": bool(fusion.get("static_image_hold_active", False)),
                "strong_flow_triggered": False,
                "fft_triggered": False,
            },
            "scheduler": {
                "frame_idx": int(frame_idx),
                "is_keyframe": self.scheduler.is_keyframe(frame_idx),
                "slow_path_requested": self.scheduler.should_run_slow_path(
                    frame_idx,
                    bool(fusion["is_suspicious"]),
                ),
            },
            "rois": [roi.to_dict() for roi in rois],
            "roi_results": roi_results,
            "emit_roi_details": self.emit_roi_details,
            "roi_detail_modes": {
                "texture": self.emit_roi_texture_details,
                "temporal": self.emit_roi_temporal_details,
                "motion": self.emit_roi_motion_details,
                "static_image": self.emit_roi_motion_details,
            },
            "gpu": {
                "device": str(self.device),
                "required": bool(self.require_gpu),
            },
        }
        static_image_details = feature_details.get("static_image", {})
        feature_details["static_media"] = {
            "score": float(static_image_details.get("score", 0.0)),
            "live_score": float(
                max(
                    static_image_details.get("live_score", 0.0),
                    static_image_details.get("score", 0.0),
                    static_image_details.get("p_media", 0.0),
                    static_image_details.get("classifier_score", 0.0),
                )
            ),
            "triggered": bool(static_image_details.get("triggered", False)),
            "trigger_count": int(static_image_details.get("trigger_count", 0)),
            "media_type": "screen_or_paper"
            if static_image_details.get("screen_like", False)
            else "flat_media_candidate",
            "inner_target_locked_to_plane": bool(static_image_details.get("triggered", False)),
            "screen_or_paper_like": bool(static_image_details.get("screen_like", False)),
            "patch_similarity": float(static_image_details.get("patch_similarity", 0.0)),
            "stable_count": int(static_image_details.get("stable_count", 0)),
            "center_motion": float(static_image_details.get("center_motion", 0.0)),
            "context_motion": float(static_image_details.get("context_motion", 0.0)),
            "line_score": float(static_image_details.get("screen_line_score", 0.0)),
            "context_std": float(static_image_details.get("screen_context_std", 0.0)),
            "legacy_static_image": static_image_details,
            "backend": str(static_image_details.get("backend", "gpu_static_media_spoof")),
            "p_media": float(static_image_details.get("p_media", 0.0)),
            "p_media_triggered": bool(static_image_details.get("p_media_triggered", False)),
            "p_media_type": str(static_image_details.get("p_media_type", "normal")),
            "p_media_bbox": static_image_details.get("p_media_bbox"),
            "p_media_target_related": bool(
                static_image_details.get("p_media_target_related", False)
            ),
            "p_media_strong_evidence": bool(
                static_image_details.get("p_media_strong_evidence", False)
            ),
            "p_media_background_static_suppressed": bool(
                static_image_details.get("p_media_background_static_suppressed", False)
            ),
            "p_media_scores": dict(static_image_details.get("p_media_scores", {})),
            "p_media_replay_state": dict(
                static_image_details.get("p_media_replay_state", {})
            ),
            "p_media_fast_state": dict(
                static_image_details.get("p_media_fast_state", {})
            ),
            "p_media_occlusion_state": dict(
                static_image_details.get("p_media_occlusion_state", {})
            ),
            "triggered_source": str(static_image_details.get("triggered_source", "none")),
            # Task 5.4 / Req 7.3 — mirror the classifier fields from
            # ``static_image`` so ``module_a_features.static_media`` is the
            # single source of truth for Static_Media_Classifier attribution.
            "classifier_score": float(static_image_details.get("classifier_score", 0.0)),
            "classifier_triggered": bool(static_image_details.get("classifier_triggered", False)),
            "classifier_threshold": float(static_image_details.get("classifier_threshold", 0.0)),
            "classifier_artifact": str(static_image_details.get("classifier_artifact", "")),
            "classifier_enabled": bool(static_image_details.get("classifier_enabled", False)),
            "classifier_forced_trigger": bool(
                static_image_details.get("classifier_forced_trigger", False)
            ),
            "static_image_triggered": bool(static_image_details.get("triggered", False)),
        }
        flow_details = feature_details.get("flow", {})
        source_details = feature_details.get("source_authenticity", {})
        feature_details["a3"] = {
            "motion_artifact": {
                "score": float(flow_details.get("motion_score", 0.0)),
                "region_count": int(flow_details.get("region_count", 0)),
                "local_flow_ratio": float(flow_details.get("local_max_ratio", 0.0)),
                "light_flow_available": bool(flow_details.get("light_flow_available", False)),
                "light_flow_score": float(flow_details.get("light_flow_score", 0.0)),
                "light_flow_local_anomaly_ratio": float(
                    flow_details.get("light_flow_local_anomaly_ratio", 0.0)
                ),
            },
            "static_media": feature_details["static_media"],
            "source_authenticity": {
                "enabled": bool(source_details.get("enabled", False)),
                "available": bool(source_details.get("available", False)),
                "p_synth": float(source_details.get("p_synth", 0.0)),
                "warning": bool(source_details.get("warning", False)),
                "confirmed": bool(source_details.get("confirmed", False)),
                "suppressed_by_physical": bool(source_details.get("suppressed_by_physical", False)),
                "reason": str(source_details.get("reason", "")),
                "backend": str(source_details.get("backend", "gpu_clip_stats")),
            },
            "physical_channel": {
                "p_adv": float(fusion["p_adv"]),
                "p_adv_display": float(fusion.get("p_adv_display", fusion["p_adv"])),
                "p_adv_raw": float(fusion["p_adv"]),
                "suspicious": bool(fusion["is_suspicious"]),
                "reason_codes": list(fusion["reason_codes"]),
            },
            "source_channel": {
                "p_synth": float(source_details.get("p_synth", 0.0)),
                "warning": bool(source_details.get("warning", False)),
                "confirmed": bool(source_details.get("confirmed", False)),
            },
        }
        return {
            "module_a_features": feature_details,
            "module_a": {
                "p_adv": float(fusion["p_adv"]),
                "p_adv_display": float(fusion.get("p_adv_display", fusion["p_adv"])),
                "p_adv_raw": float(fusion["p_adv"]),
                "reason_codes": list(fusion["reason_codes"]),
                "single_frame_suspicious": bool(fusion["is_suspicious"]),
            },
            # A3+ PR6: embed p_media diagnostics in extras for Web display
            "extras": self._build_p_media_extras(motion),
        }

    def _build_p_media_extras(self, motion: dict[str, Any]) -> dict[str, Any]:
        """A3+ PR6: Build p_media diagnostics for the extras field.

        Embeds p_media_scores into the details extras so Web display can
        show A3+ diagnostics without adding a new card. Only populates
        when p_media > 0.0 to avoid noise on normal frames.
        """
        extras: dict[str, Any] = {}
        p_media = float(motion.get("p_media", 0.0))
        if p_media > 0.0:
            extras["p_media"] = p_media
            extras["p_media_type"] = str(motion.get("p_media_type", "normal"))
            extras["p_media_scores"] = motion.get("p_media_scores", {})
        return extras
