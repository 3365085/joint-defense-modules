"""Configuration dataclass for GPUStaticMediaSpoofDetector.

Extracts the 30+ parameter declarations and type-coercion/clamping logic
from the detector's __init__ into a single, typed configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StaticMediaConfig:
    """Typed configuration for GPUStaticMediaSpoofDetector.

    All parameters have the same defaults as the original __init__ signature.
    Type coercion and clamping are applied in __post_init__ to match the
    original constructor behaviour exactly.
    """

    # Target / screen label sets
    target_labels: tuple[str, ...] = ("person",)
    screen_labels: tuple[str, ...] = ("helmet", "head")

    # Patch tracking parameters
    patch_size: int = 64
    min_similarity: float = 0.94
    trigger_stable_count: int = 2

    # Edge thresholds
    min_edge_mean: float = 0.038
    screen_min_edge_mean: float = 0.018

    # Motion thresholds
    min_center_motion: float = 0.0012
    context_motion_threshold: float = 0.010
    context_contrast_threshold: float = 1.6

    # ROI area constraints
    min_roi_area: int = 1200
    screen_min_roi_area: int = 450
    screen_max_roi_area: int = 8000

    # Screen context parameters
    screen_context_expand_ratio: float = 2.4
    screen_min_context_edge_mean: float = 0.004
    screen_min_context_std: float = 0.22
    screen_min_line_score: float = 0.10
    screen_max_roi_context_area_ratio: float = 0.42
    screen_person_containment_threshold: float = 0.72

    # Scoring parameters
    min_roi_confidence: float = 0.50
    score_trigger: float = 0.80
    expand_ratio: float = 0.35

    # Geometry / filtering parameters
    edge_margin_px: int = 6
    min_same_label_count: int = 2
    max_person_area_ratio: float = 0.65
    max_context_iou: float = 0.20
    max_tracks: int = 64

    # Output control
    emit_roi_details: bool = False

    # L0 multi-scale fallback (P1-A-6 optimised 2026-05-13).
    #
    # When True and the main full-frame L0 pass finds fewer than
    # ``multiscale_trigger_count`` active candidates, the detector re-runs
    # the extractor on a set of sub-regions. The pre-2026-05-13 code
    # resized each 320×320 quadrant crop back up to 640×640 with
    # ``F.interpolate`` *before* passing to the extractor; this added ~10
    # ms (GPU→CPU roundtrips × 5 + two resize passes each) and dominated
    # A3b latency on nearly every frame. We preserve the 4-quadrant
    # behaviour by default because removing it regressed adv_patch
    # detection, but the extractor now receives crops directly so the
    # interpolate round-trip cost is gone.
    #
    # Set to False on streams that don't need small-patch coverage to
    # unlock the pure full-frame fast path (~2 ms A3b).
    multiscale_fallback_enabled: bool = True
    # When the main pass returns fewer than this many active candidates,
    # the fallback sweep runs. Default 1 = "only fire when the main pass
    # saw nothing".
    multiscale_trigger_count: int = 1

    # Backend selector (P1-A-edge 2026-05-13):
    #
    #   "legacy"           — full pipeline: Legacy YOLO-ROI loop + A3+
    #                        candidate cascade (L0→L1→L2→L3). Retains
    #                        cv2.Canny / findContours / findHomography
    #                        calls, NOT edge-NPU friendly.
    #
    #   "legacy_yolo_only" — skip the entire A3+ cascade. Only the
    #                        Legacy YOLO-ROI patch-track loop runs, which
    #                        is pure torch and therefore NPU-friendly.
    #                        The p_media_* fields still populate but stay
    #                        at 0 / "normal". ``static_image_triggered``
    #                        still fires from the Legacy path.
    #
    #   "target_anchored_a3plus" — full OpenCV A3+ cascade runs, but L3
    #                        can only trigger on candidates spatially tied
    #                        to YOLO targets. This is the production path
    #                        for recovering screen/patch recall while
    #                        suppressing background rectangles.
    #
    #   "torch_native"     — full A3+ cascade runs, but L0 candidate
    #                        extraction uses a pure-torch implementation
    #                        (Sobel + density grid + vectorised rectangle
    #                        enumeration) instead of cv2.Canny + findContours.
    #                        L2 Homography verification is still cv2-based
    #                        in this release (on roadmap: torch replacement).
    #                        Works on any NPU that supports conv/pool/
    #                        element-wise ops.
    #
    # Default remains "legacy" for bit-for-bit behaviour preservation;
    # flip to "torch_native" for edge-NPU targets (RKNN, ONNX-Runtime-Mobile).
    backend: str = "legacy"

    def __post_init__(self) -> None:
        """Apply type coercion and clamping identical to the original __init__."""
        # Set conversions for label tuples
        object.__setattr__(
            self,
            "_target_labels_set",
            {str(label) for label in self.target_labels},
        )
        object.__setattr__(
            self,
            "_screen_labels_set",
            {str(label) for label in self.screen_labels},
        )

        # Type coercion with clamping
        self.patch_size = max(16, int(self.patch_size))
        self.min_similarity = float(self.min_similarity)
        self.trigger_stable_count = max(1, int(self.trigger_stable_count))
        self.min_edge_mean = float(self.min_edge_mean)
        self.screen_min_edge_mean = float(self.screen_min_edge_mean)
        self.min_center_motion = float(self.min_center_motion)
        self.context_motion_threshold = float(self.context_motion_threshold)
        self.context_contrast_threshold = float(self.context_contrast_threshold)
        self.min_roi_area = int(self.min_roi_area)
        self.screen_min_roi_area = int(self.screen_min_roi_area)
        self.screen_max_roi_area = int(self.screen_max_roi_area)
        self.screen_context_expand_ratio = float(self.screen_context_expand_ratio)
        self.screen_min_context_edge_mean = float(self.screen_min_context_edge_mean)
        self.screen_min_context_std = float(self.screen_min_context_std)
        self.screen_min_line_score = float(self.screen_min_line_score)
        self.screen_max_roi_context_area_ratio = float(self.screen_max_roi_context_area_ratio)
        self.screen_person_containment_threshold = float(self.screen_person_containment_threshold)
        self.min_roi_confidence = float(self.min_roi_confidence)
        self.score_trigger = float(self.score_trigger)
        self.expand_ratio = float(self.expand_ratio)
        self.edge_margin_px = max(0, int(self.edge_margin_px))
        self.min_same_label_count = max(1, int(self.min_same_label_count))
        self.max_person_area_ratio = float(self.max_person_area_ratio)
        self.max_context_iou = float(self.max_context_iou)
        self.max_tracks = max(1, int(self.max_tracks))
        self.emit_roi_details = bool(self.emit_roi_details)
        self.multiscale_fallback_enabled = bool(self.multiscale_fallback_enabled)
        self.multiscale_trigger_count = max(1, int(self.multiscale_trigger_count))
        backend = str(self.backend).lower()
        if backend not in {
            "legacy",
            "legacy_yolo_only",
            "target_anchored_a3plus",
            "torch_native",
        }:
            raise ValueError(
                f"Unsupported static_image backend: {backend!r}; "
                "expected one of: legacy, legacy_yolo_only, "
                "target_anchored_a3plus, torch_native"
            )
        self.backend = backend

    @property
    def target_labels_set(self) -> set[str]:
        """Pre-computed set of target labels for O(1) membership tests."""
        return self._target_labels_set  # type: ignore[attr-defined]

    @property
    def screen_labels_set(self) -> set[str]:
        """Pre-computed set of screen labels for O(1) membership tests."""
        return self._screen_labels_set  # type: ignore[attr-defined]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StaticMediaConfig:
        """Construct a StaticMediaConfig from a plain dict.

        Unknown keys are silently ignored for forward-compatibility.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)
