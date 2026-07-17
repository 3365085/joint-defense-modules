from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


_TRUSTED_BBOX_TRANSITION_HITS = 3
_TRANSIENT_TRIGGER_HOLD_GATES = frozenset({"rebuilt_tighten_gate_failed"})
_TRUSTED_BBOX_EXPANSION_AREA_MULTIPLIER = 1.45
_TRUSTED_BBOX_EXPANSION_PREVIOUS_COVERAGE_MIN = 0.85
_TRUSTED_BBOX_CONTRACTION_SIDE_RATIO_MIN = 0.30
_EDGE_BOUNDARY_HYSTERESIS_FRACTION = 1.0 / 6.0


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


def _positive_result_seq(value: Any) -> int | None:
    """Normalize a published A3b result sequence without accepting sentinels."""

    if isinstance(value, bool):
        return None
    try:
        result_seq = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return result_seq if result_seq > 0 else None


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_list(value: tuple[float, float, float, float] | None) -> list[float | int] | None:
    if value is None:
        return None
    return [int(item) if item.is_integer() else float(item) for item in value]


def _bbox_large_jump(
    previous: tuple[float, float, float, float] | None,
    current: tuple[float, float, float, float] | None,
) -> bool:
    """Reject a discontinuous candidate without tuning A3b detection thresholds."""

    if previous is None or current is None:
        return False
    px1, py1, px2, py2 = previous
    cx1, cy1, cx2, cy2 = current
    previous_area = (px2 - px1) * (py2 - py1)
    current_area = (cx2 - cx1) * (cy2 - cy1)
    if previous_area <= 0.0 or current_area <= 0.0:
        return True

    ix1 = max(px1, cx1)
    iy1 = max(py1, cy1)
    ix2 = min(px2, cx2)
    iy2 = min(py2, cy2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = previous_area + current_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    area_ratio = min(previous_area, current_area) / max(previous_area, current_area)
    previous_coverage = intersection / previous_area
    current_coverage = intersection / current_area
    width_ratio = (cx2 - cx1) / (px2 - px1)
    height_ratio = (cy2 - cy1) / (py2 - py1)
    containing_expansion = bool(
        current_area
        >= previous_area * _TRUSTED_BBOX_EXPANSION_AREA_MULTIPLIER
        and previous_coverage >= _TRUSTED_BBOX_EXPANSION_PREVIOUS_COVERAGE_MIN
    )
    contained_contraction = bool(
        previous_area
        >= current_area * _TRUSTED_BBOX_EXPANSION_AREA_MULTIPLIER
        and current_coverage >= _TRUSTED_BBOX_EXPANSION_PREVIOUS_COVERAGE_MIN
        and width_ratio >= _TRUSTED_BBOX_CONTRACTION_SIDE_RATIO_MIN
        and height_ratio >= _TRUSTED_BBOX_CONTRACTION_SIDE_RATIO_MIN
    )

    previous_center_x = (px1 + px2) * 0.5
    previous_center_y = (py1 + py2) * 0.5
    current_center_x = (cx1 + cx2) * 0.5
    current_center_y = (cy1 + cy2) * 0.5
    center_distance = (
        (current_center_x - previous_center_x) ** 2
        + (current_center_y - previous_center_y) ** 2
    ) ** 0.5
    previous_diagonal = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
    center_shift_ratio = center_distance / previous_diagonal if previous_diagonal > 0.0 else 1.0
    previous_aspect = (px2 - px1) / max(1e-6, py2 - py1)
    current_aspect = (cx2 - cx1) / max(1e-6, cy2 - cy1)
    aspect_ratio = min(previous_aspect, current_aspect) / max(
        previous_aspect,
        current_aspect,
    )

    # A high-quality, reasonably sized inner rectangle is a safer and more
    # precise suppression region than an already-trusted outer rectangle.
    # Accept that contraction immediately even when the outer/inner aspect
    # ratios differ.  The per-side floor keeps thin internal strips on the
    # normal three-result pending path.
    if contained_contraction:
        return False

    return bool(
        (
            iou < 0.10
            and (area_ratio < 0.35 or center_shift_ratio > 0.50)
        )
        or (iou < 0.50 and aspect_ratio < 0.50)
        # A transient edge candidate can nearly contain the trusted phone
        # screen while expanding far into the surrounding wall/background.
        # Treat only the outward expansion as a pending transition; a later
        # contraction back to a precise inner screen remains immediately
        # acceptable.
        or containing_expansion
    )


def _current_explicit_guard_failures(
    static_media: dict[str, Any],
) -> list[str]:
    """Read mutable suppression/freshness guards on every runtime call."""

    failed: list[str] = []
    border = (
        static_media.get("p_media_border_state")
        if isinstance(static_media.get("p_media_border_state"), dict)
        else {}
    )
    camera = (
        static_media.get("p_media_camera_motion_state")
        if isinstance(static_media.get("p_media_camera_motion_state"), dict)
        else {}
    )
    physical = (
        static_media.get("p_media_physical_motion_state")
        if isinstance(static_media.get("p_media_physical_motion_state"), dict)
        else {}
    )
    if border.get("suppressed"):
        failed.append("border_suppressed")
    if camera.get("suppressed"):
        failed.append("camera_motion_suppressed")
    if physical.get("suppressed"):
        failed.append("physical_motion_suppressed")

    if static_media.get("result_contract_source") != "rebuilt":
        return failed

    policy = (
        static_media.get("policy")
        if isinstance(static_media.get("policy"), dict)
        else {}
    )
    suppression = (
        static_media.get("suppression")
        if isinstance(static_media.get("suppression"), dict)
        else {}
    )
    policy_state = (
        static_media.get("p_media_policy_state")
        if isinstance(static_media.get("p_media_policy_state"), dict)
        else {}
    )
    if not bool(static_media.get("a3b_result_fresh", False)):
        failed.append("rebuilt_result_stale")

    candidate_allowed_values = [
        static_media.get("media_candidate_allowed")
        if "media_candidate_allowed" in static_media
        else None,
        policy.get("media_candidate_allowed")
        if "media_candidate_allowed" in policy
        else None,
        suppression.get("media_candidate_allowed")
        if "media_candidate_allowed" in suppression
        else None,
        policy_state.get("media_candidate_allowed")
        if "media_candidate_allowed" in policy_state
        else None,
    ]
    if any(
        value is not None and not bool(value)
        for value in candidate_allowed_values
    ):
        failed.append("rebuilt_candidate_disallowed")

    no_suppression_reasons = {"", "none", "normal", "not_suppressed"}
    suppression_reasons = {
        str(static_media.get("suppressed_reason") or "").strip().lower(),
        str(policy.get("suppressed_reason") or "").strip().lower(),
        str(policy.get("reason") or "").strip().lower(),
        str(suppression.get("suppressed_reason") or "").strip().lower(),
        str(suppression.get("reason") or "").strip().lower(),
        str(policy_state.get("suppressed_reason") or "").strip().lower(),
        str(policy_state.get("reason") or "").strip().lower(),
    }
    explicitly_suppressed = bool(
        str(static_media.get("a3b_state") or "").strip().lower()
        == "suppressed"
        or any(
            bool(container.get("suppressed", False))
            for container in (policy, suppression, policy_state)
        )
        or any(
            reason not in no_suppression_reasons
            for reason in suppression_reasons
        )
    )
    if explicitly_suppressed:
        failed.append("rebuilt_policy_suppressed")
    return failed


@dataclass(slots=True)
class A3BSoftTriggerConfig:
    enabled: bool = True
    observed_threshold: float = 0.42
    trigger_threshold: float = 0.62
    strong_single_frame_threshold: float = 0.78
    observed_only_warning_threshold: float = 0.50
    observed_only_track_threshold: float = 0.50
    observed_only_source_keywords: tuple[str, ...] = field(default_factory=tuple)
    trigger_source_keywords: tuple[str, ...] = field(default_factory=tuple)
    window_size: int = 12
    min_window_hits: int = 3
    observed_only_min_window_hits: int = 3
    min_consecutive_hits: int = 2
    decay: float = 0.88
    max_gap_frames: int = 5
    trigger_hold_frames: int = 5
    allow_soft_trigger: bool = True
    allow_single_strong_trigger: bool = True
    allow_window_accumulated_trigger: bool = True
    allow_observed_only_warning: bool = True
    rebuilt_tighten_gate_enabled: bool = True
    rebuilt_gate_candidate_min: float = 0.70
    rebuilt_gate_edge_min: float = 0.45
    rebuilt_gate_edge_max: float = 0.58
    rebuilt_gate_border_contrast_min: float = 0.80
    rebuilt_gate_candidate_tolerance: float = 0.001
    rebuilt_gate_aspect_ratio_min: float = 0.40
    rebuilt_gate_aspect_ratio_max: float = 2.50

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "A3BSoftTriggerConfig":
        data = data or {}
        defaults = cls()
        source_keywords = data.get(
            "observed_only_source_keywords",
            defaults.observed_only_source_keywords,
        )
        if isinstance(source_keywords, str):
            source_keywords = [item.strip() for item in source_keywords.split(",")]
        if not isinstance(source_keywords, (list, tuple)):
            source_keywords = defaults.observed_only_source_keywords
        trigger_keywords = data.get("trigger_source_keywords", source_keywords)
        if isinstance(trigger_keywords, str):
            trigger_keywords = [item.strip() for item in trigger_keywords.split(",")]
        if not isinstance(trigger_keywords, (list, tuple)):
            trigger_keywords = tuple(source_keywords)
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            observed_threshold=_float(
                data.get("observed_threshold"),
                defaults.observed_threshold,
            ),
            trigger_threshold=_float(
                data.get("trigger_threshold"),
                defaults.trigger_threshold,
            ),
            strong_single_frame_threshold=_float(
                data.get("strong_single_frame_threshold"),
                defaults.strong_single_frame_threshold,
            ),
            observed_only_warning_threshold=_float(
                data.get("observed_only_warning_threshold"),
                defaults.observed_only_warning_threshold,
            ),
            observed_only_track_threshold=_float(
                data.get("observed_only_track_threshold"),
                defaults.observed_only_track_threshold,
            ),
            observed_only_source_keywords=tuple(str(item) for item in source_keywords if str(item).strip()),
            trigger_source_keywords=tuple(str(item) for item in trigger_keywords if str(item).strip()),
            window_size=max(
                1,
                _int(data.get("window_size"), defaults.window_size),
            ),
            min_window_hits=max(
                1,
                _int(data.get("min_window_hits"), defaults.min_window_hits),
            ),
            observed_only_min_window_hits=max(
                1,
                _int(
                    data.get("observed_only_min_window_hits"),
                    defaults.observed_only_min_window_hits,
                ),
            ),
            min_consecutive_hits=max(
                1,
                _int(
                    data.get("min_consecutive_hits"),
                    defaults.min_consecutive_hits,
                ),
            ),
            decay=max(
                0.0,
                min(1.0, _float(data.get("decay"), defaults.decay)),
            ),
            max_gap_frames=max(
                0,
                _int(data.get("max_gap_frames"), defaults.max_gap_frames),
            ),
            trigger_hold_frames=max(
                0,
                _int(
                    data.get("trigger_hold_frames"),
                    defaults.trigger_hold_frames,
                ),
            ),
            allow_soft_trigger=bool(
                data.get("allow_soft_trigger", defaults.allow_soft_trigger)
            ),
            allow_single_strong_trigger=bool(
                data.get(
                    "allow_single_strong_trigger",
                    defaults.allow_single_strong_trigger,
                )
            ),
            allow_window_accumulated_trigger=bool(
                data.get(
                    "allow_window_accumulated_trigger",
                    defaults.allow_window_accumulated_trigger,
                )
            ),
            allow_observed_only_warning=bool(
                data.get(
                    "allow_observed_only_warning",
                    defaults.allow_observed_only_warning,
                )
            ),
            rebuilt_tighten_gate_enabled=bool(
                data.get(
                    "rebuilt_tighten_gate_enabled",
                    defaults.rebuilt_tighten_gate_enabled,
                )
            ),
            rebuilt_gate_candidate_min=_float(
                data.get("rebuilt_gate_candidate_min"),
                defaults.rebuilt_gate_candidate_min,
            ),
            rebuilt_gate_edge_min=_float(
                data.get("rebuilt_gate_edge_min"),
                defaults.rebuilt_gate_edge_min,
            ),
            rebuilt_gate_edge_max=_float(
                data.get("rebuilt_gate_edge_max"),
                defaults.rebuilt_gate_edge_max,
            ),
            rebuilt_gate_border_contrast_min=_float(
                data.get("rebuilt_gate_border_contrast_min"),
                defaults.rebuilt_gate_border_contrast_min,
            ),
            rebuilt_gate_candidate_tolerance=max(
                0.0,
                _float(
                    data.get("rebuilt_gate_candidate_tolerance"),
                    defaults.rebuilt_gate_candidate_tolerance,
                ),
            ),
            rebuilt_gate_aspect_ratio_min=max(
                0.0,
                _float(
                    data.get("rebuilt_gate_aspect_ratio_min"),
                    defaults.rebuilt_gate_aspect_ratio_min,
                ),
            ),
            rebuilt_gate_aspect_ratio_max=max(
                0.0,
                _float(
                    data.get("rebuilt_gate_aspect_ratio_max"),
                    defaults.rebuilt_gate_aspect_ratio_max,
                ),
            ),
        )


class A3BSoftTriggerState:
    """Soft confirmation state for A3b without weakening negative guards."""

    def __init__(self, config: dict[str, Any] | A3BSoftTriggerConfig | None = None) -> None:
        self.config = config if isinstance(config, A3BSoftTriggerConfig) else A3BSoftTriggerConfig.from_mapping(config)
        self.window: deque[dict[str, Any]] = deque(maxlen=self.config.window_size)
        self.consecutive_hits = 0
        self.consecutive_result_hits = 0
        self.gap_frames = 0
        self.decayed_score = 0.0
        self.trigger_hold_remaining = 0
        self.last_trigger_source = "none"
        self.last_confirmed_score = 0.0
        self.source_key = ""
        self.last_counted_result_seq: int | None = None
        self.last_counted_source_frame_idx: int | None = None
        self.last_counted_source_timestamp: float | None = None
        self.independent_evidence_counter = 0
        self.last_analyzed_result_seq: int | None = None
        self.last_analyzed_debug: dict[str, Any] | None = None
        self.last_trusted_bbox: tuple[float, float, float, float] | None = None
        self.last_trusted_bbox_age_frames = 0
        self.last_trusted_bbox_age_result_seqs = 0
        self.last_trusted_bbox_result_seq: Any = None
        self.pending_trusted_bbox: tuple[float, float, float, float] | None = None
        self.pending_trusted_bbox_hits = 0
        self.pending_trusted_bbox_result_seq: Any = None

    def reset(self) -> None:
        self.window.clear()
        self.consecutive_hits = 0
        self.consecutive_result_hits = 0
        self.gap_frames = 0
        self.decayed_score = 0.0
        self.trigger_hold_remaining = 0
        self.last_trigger_source = "none"
        self.last_confirmed_score = 0.0
        self.source_key = ""
        self.last_counted_result_seq = None
        self.last_counted_source_frame_idx = None
        self.last_counted_source_timestamp = None
        self.independent_evidence_counter = 0
        self.last_analyzed_result_seq = None
        self.last_analyzed_debug = None
        self.last_trusted_bbox = None
        self.last_trusted_bbox_age_frames = 0
        self.last_trusted_bbox_age_result_seqs = 0
        self.last_trusted_bbox_result_seq = None
        self.pending_trusted_bbox = None
        self.pending_trusted_bbox_hits = 0
        self.pending_trusted_bbox_result_seq = None

    def _source_frame_units(
        self,
        static_media: dict[str, Any],
        *,
        independent_evidence: bool,
    ) -> int:
        """Return source-frame-equivalent coverage for one new result."""

        if not independent_evidence:
            return 0

        interval_units = max(
            1,
            _int(static_media.get("a3b_source_interval_frames"), 1),
        )
        source_frame_idx: int | None = None
        raw_frame_idx = static_media.get("a3b_source_frame_idx")
        if raw_frame_idx is not None and not isinstance(raw_frame_idx, bool):
            try:
                source_frame_idx = int(raw_frame_idx)
            except (TypeError, ValueError):
                source_frame_idx = None

        source_timestamp: float | None = None
        raw_timestamp = static_media.get("a3b_source_timestamp")
        if raw_timestamp is not None:
            try:
                candidate_timestamp = float(raw_timestamp)
                if math.isfinite(candidate_timestamp):
                    source_timestamp = candidate_timestamp
            except (TypeError, ValueError):
                source_timestamp = None

        units = interval_units
        previous_frame_idx = self.last_counted_source_frame_idx
        if (
            source_frame_idx is not None
            and previous_frame_idx is not None
            and source_frame_idx > previous_frame_idx
        ):
            units = max(units, source_frame_idx - previous_frame_idx)

        source_fps = _float(static_media.get("a3b_source_fps"))
        previous_timestamp = self.last_counted_source_timestamp
        if (
            source_timestamp is not None
            and previous_timestamp is not None
            and source_timestamp > previous_timestamp
            and math.isfinite(source_fps)
            and source_fps > 0.0
        ):
            timestamp_units = int(
                round((source_timestamp - previous_timestamp) * source_fps)
            )
            units = max(units, timestamp_units)

        self.last_counted_source_frame_idx = source_frame_idx
        self.last_counted_source_timestamp = source_timestamp
        return max(1, int(units))

    def update(self, static_media: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        source_text = str(static_media.get("source_path") or static_media.get("source") or "")
        if self.source_key and source_text != self.source_key:
            self.reset()
        self.source_key = source_text
        legacy_static = static_media.get("legacy_static_image") if isinstance(static_media.get("legacy_static_image"), dict) else {}
        result_seq_present = (
            "a3b_result_seq" in static_media
            or "a3b_result_seq" in legacy_static
        )
        raw_result_seq = static_media.get(
            "a3b_result_seq",
            legacy_static.get("a3b_result_seq"),
        )
        result_seq = _positive_result_seq(raw_result_seq)
        analysis_cache_hit = bool(
            result_seq is not None
            and result_seq == self.last_analyzed_result_seq
            and self.last_analyzed_debug is not None
        )

        observed_score = max(
            _float(static_media.get("live_score")),
            _float(static_media.get("live_score_display")),
            _float(static_media.get("p_media")),
            _float(static_media.get("score")),
            _float(static_media.get("classifier_score")),
        )
        legacy_triggered = bool(legacy_static.get("triggered") or legacy_static.get("p_media_triggered"))
        base_triggered = bool(
            static_media.get("triggered")
            or static_media.get("p_media_triggered")
            or static_media.get("classifier_triggered")
            or legacy_triggered
        )
        base_confirmed = max(
            _float(static_media.get("score")) if base_triggered else 0.0,
            _float(static_media.get("p_media")) if bool(static_media.get("p_media_triggered")) else 0.0,
            _float(static_media.get("classifier_score")) if bool(static_media.get("classifier_triggered")) else 0.0,
            _float(legacy_static.get("score")) if legacy_triggered else 0.0,
            _float(legacy_static.get("p_media")) if legacy_triggered else 0.0,
        )

        if analysis_cache_hit:
            debug = dict(self.last_analyzed_debug or {})
            debug["failed_gates"] = list(debug.get("failed_gates") or [])
        else:
            debug = self._debug(static_media, observed_score)
            if (
                result_seq is not None
                and (
                    self.last_analyzed_result_seq is None
                    or result_seq > self.last_analyzed_result_seq
                )
            ):
                self.last_analyzed_result_seq = result_seq
                self.last_analyzed_debug = dict(debug)
                self.last_analyzed_debug["failed_gates"] = list(
                    debug.get("failed_gates") or []
                )
        current_guard_failures = _current_explicit_guard_failures(
            static_media
        )
        for gate in current_guard_failures:
            if gate not in debug["failed_gates"]:
                debug["failed_gates"].append(gate)
        quality_gate = bool(
            debug["quality_gate_passed"]
            and not current_guard_failures
        )
        debug["quality_gate_passed"] = quality_gate
        edge_gate_range = max(
            0.0,
            cfg.rebuilt_gate_edge_max - cfg.rebuilt_gate_edge_min,
        )
        edge_upper_hysteresis = (
            edge_gate_range * _EDGE_BOUNDARY_HYSTERESIS_FRACTION
        )
        rebuilt_edge_score = _float(
            debug.get("rebuilt_edge_score")
        )
        rebuilt_edge_near_upper = bool(
            debug.get("rebuilt_tighten_gate_applied", False)
            and rebuilt_edge_score > cfg.rebuilt_gate_edge_max
            and rebuilt_edge_score
            <= cfg.rebuilt_gate_edge_max + edge_upper_hysteresis
        )
        base_media_cue = bool(
            _int(debug.get("candidate_count")) > 0
            or debug.get("screen_cue", False)
        )
        near_quality_gate = bool(
            static_media.get("result_contract_source") == "rebuilt"
            and not debug.get("rebuilt_authoritative_confirmed", False)
            and debug.get("rebuilt_result_fresh", False)
            and debug.get("rebuilt_candidate_allowed", False)
            and not debug.get("rebuilt_policy_suppressed", False)
            and debug.get("rebuilt_candidate_pass", False)
            and rebuilt_edge_near_upper
            and debug.get("rebuilt_border_pass", False)
            and debug.get("rebuilt_aspect_pass", False)
            and base_media_cue
            and not current_guard_failures
        )
        failed_gates = debug.get("failed_gates") if isinstance(debug.get("failed_gates"), list) else []
        blocking_failed_gates = [gate for gate in failed_gates if gate != "no_candidate_or_screen_cue"]
        hold_blocking_failed_gates = [
            gate
            for gate in blocking_failed_gates
            if gate not in _TRANSIENT_TRIGGER_HOLD_GATES
        ]
        observed_only_source_keyword_matched = any(
            keyword and keyword in source_text
            for keyword in cfg.observed_only_source_keywords
        )
        trigger_source_keyword_matched = any(
            keyword and keyword in source_text
            for keyword in cfg.trigger_source_keywords
        )
        observed_allowed = bool(not blocking_failed_gates)
        observed_hit = bool(observed_score >= cfg.observed_threshold and observed_allowed)
        quality_hit = bool(observed_hit and quality_gate)
        near_quality_hit = bool(
            observed_score >= cfg.observed_threshold
            and near_quality_gate
        )
        temporal_quality_hit = bool(
            quality_hit or near_quality_hit
        )
        observed_only_hit = bool(observed_hit and not quality_gate)
        legacy_bbox = legacy_static.get("p_media_bbox") or legacy_static.get("bbox")
        current_bbox = _bbox(
            static_media.get("p_media_bbox")
            or static_media.get("bbox")
            or static_media.get("candidate_bbox")
            or legacy_bbox
        )
        trusted_bbox_updated = False
        trusted_bbox_expired = False
        trusted_bbox_expired_reasons: list[str] = []
        pending_bbox_accepted = False
        pending_bbox_expired = False
        pending_bbox_expired_reason = "none"
        if not result_seq_present:
            result_seq_mode = "legacy_no_seq"
            independent_evidence = True
        elif result_seq is None:
            result_seq_mode = "invalid_non_positive"
            independent_evidence = False
        elif (
            self.last_counted_result_seq is None
            or result_seq > self.last_counted_result_seq
        ):
            result_seq_mode = "new_positive_seq"
            independent_evidence = True
            self.last_counted_result_seq = result_seq
        elif result_seq == self.last_counted_result_seq:
            result_seq_mode = "duplicate_seq"
            independent_evidence = False
        else:
            result_seq_mode = "stale_seq"
            independent_evidence = False
        hold_clock_mode = (
            "legacy_call"
            if result_seq_mode == "legacy_no_seq"
            else "result_seq"
            if result_seq is not None
            else "invalid_result_seq"
        )
        source_frame_units = self._source_frame_units(
            static_media,
            independent_evidence=independent_evidence,
        )
        bbox_evidence_eligible = bool(
            independent_evidence or result_seq_mode == "duplicate_seq"
        )
        if self.last_trusted_bbox is not None and independent_evidence:
            self.last_trusted_bbox_age_frames += source_frame_units
            if (
                result_seq is not None
                and result_seq != self.last_trusted_bbox_result_seq
            ):
                self.last_trusted_bbox_age_result_seqs += 1
                self.last_trusted_bbox_result_seq = result_seq
        bbox_large_jump = _bbox_large_jump(self.last_trusted_bbox, current_bbox)
        if (
            bbox_evidence_eligible
            and
            quality_hit
            and current_bbox is not None
        ):
            if not bbox_large_jump:
                self.last_trusted_bbox = current_bbox
                self.last_trusted_bbox_age_frames = 0
                self.last_trusted_bbox_age_result_seqs = 0
                self.last_trusted_bbox_result_seq = result_seq
                trusted_bbox_updated = True
                self.pending_trusted_bbox = None
                self.pending_trusted_bbox_hits = 0
                self.pending_trusted_bbox_result_seq = None
            else:
                pending_consistent = bool(
                    self.pending_trusted_bbox is not None
                    and not _bbox_large_jump(self.pending_trusted_bbox, current_bbox)
                )
                if not pending_consistent:
                    self.pending_trusted_bbox = current_bbox
                    self.pending_trusted_bbox_hits = 1
                    self.pending_trusted_bbox_result_seq = result_seq
                elif independent_evidence:
                    self.pending_trusted_bbox = current_bbox
                    self.pending_trusted_bbox_hits += 1
                    self.pending_trusted_bbox_result_seq = result_seq
                if self.pending_trusted_bbox_hits >= _TRUSTED_BBOX_TRANSITION_HITS:
                    self.last_trusted_bbox = current_bbox
                    self.last_trusted_bbox_age_frames = 0
                    self.last_trusted_bbox_age_result_seqs = 0
                    self.last_trusted_bbox_result_seq = result_seq
                    trusted_bbox_updated = True
                    pending_bbox_accepted = True
                    self.pending_trusted_bbox = None
                    self.pending_trusted_bbox_hits = 0
                    self.pending_trusted_bbox_result_seq = None
        else:
            pending_bbox_expired = self.pending_trusted_bbox is not None
            if pending_bbox_expired:
                pending_bbox_expired_reason = "evidence_gap"
            self.pending_trusted_bbox = None
            self.pending_trusted_bbox_hits = 0
            self.pending_trusted_bbox_result_seq = None
        if independent_evidence:
            self.independent_evidence_counter += 1
            evidence_id = self.independent_evidence_counter
            if observed_hit:
                self.consecutive_hits += source_frame_units
                self.consecutive_result_hits += 1
                self.gap_frames = 0
            else:
                self.gap_frames += source_frame_units
                if self.gap_frames > cfg.max_gap_frames:
                    self.consecutive_hits = 0
                    self.consecutive_result_hits = 0

            window_units = min(cfg.window_size, source_frame_units)
            for _ in range(window_units):
                self.window.append(
                    {
                        "observed": float(observed_score),
                        "hit": bool(observed_hit),
                        "quality": bool(quality_gate),
                        "quality_hit": bool(quality_hit),
                        "near_quality_hit": bool(near_quality_hit),
                        "temporal_quality_hit": bool(
                            temporal_quality_hit
                        ),
                        "observed_only_hit": bool(observed_only_hit),
                        "result_seq": result_seq,
                        "result_seq_mode": result_seq_mode,
                        "evidence_id": evidence_id,
                    }
                )
        window_hits = sum(1 for item in self.window if item["hit"])
        quality_window_hits = sum(1 for item in self.window if item.get("quality_hit"))
        near_quality_window_hits = sum(
            1
            for item in self.window
            if item.get("near_quality_hit")
        )
        temporal_quality_window_hits = sum(
            1
            for item in self.window
            if item.get("temporal_quality_hit")
        )
        observed_only_window_hits = sum(1 for item in self.window if item.get("observed_only_hit"))
        quality_window_result_hits = len(
            {
                item.get("evidence_id")
                for item in self.window
                if item.get("quality_hit")
            }
        )
        near_quality_window_result_hits = len(
            {
                item.get("evidence_id")
                for item in self.window
                if item.get("near_quality_hit")
            }
        )
        temporal_quality_window_result_hits = len(
            {
                item.get("evidence_id")
                for item in self.window
                if item.get("temporal_quality_hit")
            }
        )
        observed_only_window_result_hits = len(
            {
                item.get("evidence_id")
                for item in self.window
                if item.get("observed_only_hit")
            }
        )
        window_score = max((_float(item["observed"]) for item in self.window if item["hit"]), default=0.0)
        quality_window_score = max((_float(item["observed"]) for item in self.window if item.get("quality_hit")), default=0.0)
        temporal_quality_window_score = max(
            (
                _float(item["observed"])
                for item in self.window
                if item.get("temporal_quality_hit")
            ),
            default=0.0,
        )
        observed_only_window_score = max(
            (_float(item["observed"]) for item in self.window if item.get("observed_only_hit")),
            default=0.0,
        )
        if independent_evidence:
            self.decayed_score = max(
                observed_score,
                self.decayed_score * (cfg.decay ** source_frame_units),
            )

        triggered = False
        source = "none"
        confirmed_score = base_confirmed if base_triggered and quality_gate else 0.0
        strong_observed_only = bool(
            cfg.allow_single_strong_trigger
            and not quality_gate
            and not blocking_failed_gates
            and observed_score >= cfg.strong_single_frame_threshold
        )
        observed_only_window = bool(
            cfg.allow_soft_trigger
            and cfg.allow_observed_only_warning
            and not quality_gate
            and not blocking_failed_gates
            and observed_only_window_hits >= cfg.observed_only_min_window_hits
            and observed_only_window_result_hits
            >= min(2, cfg.observed_only_min_window_hits)
            and observed_only_window_score >= cfg.observed_only_warning_threshold
            and _float(debug.get("track_score")) >= cfg.observed_only_track_threshold
        )
        if not cfg.enabled:
            debug["failed_gates"].append("disabled")
        elif (
            cfg.allow_single_strong_trigger
            and quality_gate
            and observed_score >= cfg.strong_single_frame_threshold
        ):
            triggered = True
            source = "single_strong"
            confirmed_score = max(confirmed_score, observed_score)
        elif strong_observed_only:
            triggered = True
            source = "observed_strong"
            confirmed_score = 0.0
        elif (
            cfg.allow_soft_trigger
            and cfg.allow_window_accumulated_trigger
            and temporal_quality_window_hits >= cfg.min_window_hits
            and temporal_quality_window_result_hits
            >= min(2, cfg.min_window_hits)
            and quality_window_result_hits >= 1
            and quality_window_score >= cfg.trigger_threshold
        ):
            triggered = True
            source = "window_accumulated"
            confirmed_score = max(confirmed_score, quality_window_score)
        elif observed_only_window:
            triggered = True
            source = "observed_window"
            confirmed_score = 0.0
        elif (
            cfg.allow_soft_trigger
            and temporal_quality_window_hits >= cfg.min_consecutive_hits
            and temporal_quality_window_result_hits
            >= min(2, cfg.min_consecutive_hits)
            and quality_window_result_hits >= 1
            and quality_window_score >= cfg.trigger_threshold
            and max(base_confirmed, observed_score)
            >= cfg.trigger_threshold
            and (quality_gate or near_quality_gate)
        ):
            triggered = True
            source = "confirmed_track"
            confirmed_score = max(
                confirmed_score,
                base_confirmed,
                quality_window_score,
            )
        elif (
            base_triggered
            and (quality_gate or near_quality_gate)
            and temporal_quality_window_hits >= cfg.min_consecutive_hits
            and temporal_quality_window_result_hits
            >= min(2, cfg.min_consecutive_hits)
            and quality_window_result_hits >= 1
        ):
            triggered = True
            source = "confirmed_track"
            confirmed_score = max(
                confirmed_score,
                base_confirmed,
                quality_window_score,
            )

        if not (quality_gate or near_quality_gate) and source not in {
            "observed_strong",
            "observed_window",
        }:
            triggered = False
            source = "none"
            confirmed_score = 0.0

        held_trigger = False
        hold_tick_consumed = False
        trigger_hold_before_update = int(self.trigger_hold_remaining)
        if triggered:
            self.trigger_hold_remaining = cfg.trigger_hold_frames
            self.last_trigger_source = source
            self.last_confirmed_score = float(confirmed_score)
        elif (
            cfg.trigger_hold_frames > 0
            and self.trigger_hold_remaining > 0
            and not hold_blocking_failed_gates
            and (observed_score >= cfg.observed_threshold or self.decayed_score >= cfg.observed_only_warning_threshold)
            and result_seq_mode
            in {"legacy_no_seq", "new_positive_seq", "duplicate_seq"}
        ):
            if independent_evidence:
                self.trigger_hold_remaining = max(
                    0,
                    self.trigger_hold_remaining - source_frame_units,
                )
                hold_tick_consumed = True
            triggered = True
            held_trigger = True
            source = f"{self.last_trigger_source}_hold" if self.last_trigger_source != "none" else "a3b_hold"
            confirmed_score = max(confirmed_score, min(1.0, self.last_confirmed_score * 0.85))
        else:
            self.trigger_hold_remaining = 0

        trusted_bbox_age_frames = int(self.last_trusted_bbox_age_frames)
        trusted_bbox_age_result_seqs = int(self.last_trusted_bbox_age_result_seqs)
        if self.last_trusted_bbox is not None and not triggered:
            if self.last_trusted_bbox_age_frames > cfg.max_gap_frames:
                trusted_bbox_expired_reasons.append("max_gap_frames")
            if self.last_trusted_bbox_age_result_seqs > cfg.max_gap_frames:
                trusted_bbox_expired_reasons.append("result_seq_silence")
            if trusted_bbox_expired_reasons:
                trusted_bbox_expired = True
                if self.pending_trusted_bbox is not None:
                    pending_bbox_expired = True
                    pending_bbox_expired_reason = "trusted_bbox_expired"
                self.last_trusted_bbox = None
                self.last_trusted_bbox_age_frames = 0
                self.last_trusted_bbox_age_result_seqs = 0
                self.last_trusted_bbox_result_seq = None
                self.pending_trusted_bbox = None
                self.pending_trusted_bbox_hits = 0
                self.pending_trusted_bbox_result_seq = None

        current_evidence_low = not quality_hit
        effective_bbox: tuple[float, float, float, float] | None = None
        trusted_bbox_fallback = False
        if triggered:
            if self.last_trusted_bbox is not None and (
                current_bbox is None or current_evidence_low or bbox_large_jump
            ):
                effective_bbox = self.last_trusted_bbox
                trusted_bbox_fallback = True
            elif current_bbox is not None:
                effective_bbox = current_bbox
            elif self.last_trusted_bbox is not None:
                effective_bbox = self.last_trusted_bbox
                trusted_bbox_fallback = True

        # Score contract:
        # - observed_score: frame/window evidence only.
        # - confirmed_score / confidence: confirmation confidence after temporal/quality gates.
        # - display_score: UI main score, intentionally equal to confirmed confidence.
        # Suspect observed-only alerts remain visible through state/source and a3b_event_score.
        confirmed_score = float(max(0.0, min(1.0, confirmed_score)))
        confidence = confirmed_score
        display_score = confidence
        if triggered:
            state_source = source[:-5] if held_trigger and source.endswith("_hold") else source
            state = "suspect" if state_source in {"single_strong", "observed_strong", "observed_window"} else "confirmed"
        elif strong_observed_only or observed_only_window:
            state = "suspect"
        elif observed_score >= cfg.observed_threshold and observed_allowed:
            state = "observing"
        else:
            state = "normal"
        if held_trigger:
            confirmation_basis = "hold"
        elif state == "confirmed":
            confirmation_basis = "quality_temporal"
        elif source == "single_strong":
            confirmation_basis = "quality_single_frame"
        elif triggered:
            confirmation_basis = "observed_only"
        elif blocking_failed_gates:
            confirmation_basis = "blocked"
        else:
            confirmation_basis = "none"
        debug.update(
            {
                "track_score": float(debug.get("track_score", 0.0)),
                "stable_hits": int(self.consecutive_hits),
                "stable_result_hits": int(
                    self.consecutive_result_hits
                ),
                "window_hits": int(window_hits),
                "window_score": float(window_score),
                "quality_window_hits": int(quality_window_hits),
                "quality_window_result_hits": int(
                    quality_window_result_hits
                ),
                "quality_window_score": float(quality_window_score),
                "rebuilt_edge_gate_range": float(edge_gate_range),
                "rebuilt_edge_upper_hysteresis": float(
                    edge_upper_hysteresis
                ),
                "rebuilt_edge_near_upper": bool(
                    rebuilt_edge_near_upper
                ),
                "near_quality_gate_passed": bool(near_quality_gate),
                "near_quality_hit": bool(near_quality_hit),
                "near_quality_window_hits": int(
                    near_quality_window_hits
                ),
                "near_quality_window_result_hits": int(
                    near_quality_window_result_hits
                ),
                "temporal_quality_window_hits": int(
                    temporal_quality_window_hits
                ),
                "temporal_quality_window_result_hits": int(
                    temporal_quality_window_result_hits
                ),
                "temporal_quality_window_score": float(
                    temporal_quality_window_score
                ),
                "strict_quality_required_for_bridge": True,
                "observed_only_window_hits": int(observed_only_window_hits),
                "observed_only_window_result_hits": int(
                    observed_only_window_result_hits
                ),
                "observed_only_window_score": float(observed_only_window_score),
                "decayed_score": float(self.decayed_score),
                "strong_observed_only": bool(strong_observed_only),
                "observed_only_window": bool(observed_only_window),
                "source_keyword_policy": "diagnostic_only",
                "observed_only_source_allowed": True,
                "trigger_source_allowed": True,
                "observed_only_source_keyword_matched": bool(
                    observed_only_source_keyword_matched
                ),
                "trigger_source_keyword_matched": bool(
                    trigger_source_keyword_matched
                ),
                "result_seq_present": bool(result_seq_present),
                "result_seq": result_seq,
                "result_seq_mode": result_seq_mode,
                "analysis_cache_hit": bool(analysis_cache_hit),
                "independent_evidence_consumed": bool(independent_evidence),
                "source_frame_units": int(source_frame_units),
                "source_frame_idx": static_media.get(
                    "a3b_source_frame_idx"
                ),
                "source_timestamp": static_media.get(
                    "a3b_source_timestamp"
                ),
                "source_fps": _float(
                    static_media.get("a3b_source_fps")
                ),
                "source_interval_frames": max(
                    1,
                    _int(
                        static_media.get(
                            "a3b_source_interval_frames"
                        ),
                        1,
                    ),
                ),
                "bbox_evidence_eligible": bool(bbox_evidence_eligible),
                "last_counted_result_seq": self.last_counted_result_seq,
                "held_trigger": bool(held_trigger),
                "hold_clock_mode": hold_clock_mode,
                "hold_tick_consumed": bool(hold_tick_consumed),
                "hold_retained_on_duplicate": bool(
                    held_trigger
                    and result_seq_mode == "duplicate_seq"
                    and not hold_tick_consumed
                ),
                "trigger_hold_before_update": trigger_hold_before_update,
                "confirmation_basis": confirmation_basis,
                "trigger_hold_remaining": int(self.trigger_hold_remaining),
                "current_explicit_guard_failures": list(
                    current_guard_failures
                ),
                "blocking_failed_gates": list(blocking_failed_gates),
                "hold_blocking_failed_gates": list(
                    hold_blocking_failed_gates
                ),
                "current_bbox": _bbox_list(current_bbox),
                "last_trusted_bbox": _bbox_list(self.last_trusted_bbox),
                "trusted_bbox_age_frames": trusted_bbox_age_frames,
                "trusted_bbox_age_result_seqs": trusted_bbox_age_result_seqs,
                "trusted_bbox_expired": bool(trusted_bbox_expired),
                "trusted_bbox_expired_reasons": list(trusted_bbox_expired_reasons),
                "bbox_large_jump": bool(bbox_large_jump),
                "current_evidence_low": bool(current_evidence_low),
                "trusted_bbox_updated": bool(trusted_bbox_updated),
                "trusted_bbox_fallback": bool(trusted_bbox_fallback),
                "pending_trusted_bbox": _bbox_list(self.pending_trusted_bbox),
                "pending_trusted_bbox_hits": int(self.pending_trusted_bbox_hits),
                "pending_trusted_bbox_required_hits": int(
                    _TRUSTED_BBOX_TRANSITION_HITS
                ),
                "pending_trusted_bbox_result_seq": self.pending_trusted_bbox_result_seq,
                "pending_trusted_bbox_accepted": bool(pending_bbox_accepted),
                "pending_trusted_bbox_expired": bool(pending_bbox_expired),
                "pending_trusted_bbox_expired_reason": pending_bbox_expired_reason,
                "thresholds": {
                    "observed_threshold": float(cfg.observed_threshold),
                    "trigger_threshold": float(cfg.trigger_threshold),
                    "strong_single_frame_threshold": float(cfg.strong_single_frame_threshold),
                    "observed_only_warning_threshold": float(cfg.observed_only_warning_threshold),
                    "observed_only_track_threshold": float(cfg.observed_only_track_threshold),
                },
                "state": state,
            }
        )
        reason = source if source != "none" else (";".join(debug["failed_gates"]) if debug["failed_gates"] else "none")
        return {
            "observed_score": float(max(0.0, min(1.0, observed_score))),
            "confirmed_score": confidence,
            "confidence": confidence,
            "display_score": display_score,
            "state": state,
            "triggered": bool(triggered),
            "triggered_source": source,
            "reason": reason,
            "effective_bbox": _bbox_list(effective_bbox),
            "debug": debug,
        }

    def _debug(self, static_media: dict[str, Any], observed_score: float) -> dict[str, Any]:
        scores = static_media.get("p_media_scores") if isinstance(static_media.get("p_media_scores"), dict) else {}
        border = static_media.get("p_media_border_state") if isinstance(static_media.get("p_media_border_state"), dict) else {}
        camera = static_media.get("p_media_camera_motion_state") if isinstance(static_media.get("p_media_camera_motion_state"), dict) else {}
        physical = static_media.get("p_media_physical_motion_state") if isinstance(static_media.get("p_media_physical_motion_state"), dict) else {}
        legacy_static = static_media.get("legacy_static_image") if isinstance(static_media.get("legacy_static_image"), dict) else {}
        replay_state = legacy_static.get("p_media_replay_state") if isinstance(legacy_static.get("p_media_replay_state"), dict) else {}
        fast_state = legacy_static.get("p_media_fast_state") if isinstance(legacy_static.get("p_media_fast_state"), dict) else {}
        occlusion_state = legacy_static.get("p_media_occlusion_state") if isinstance(legacy_static.get("p_media_occlusion_state"), dict) else {}
        failed: list[str] = []
        if border.get("suppressed"):
            failed.append("border_suppressed")
        if camera.get("suppressed"):
            failed.append("camera_motion_suppressed")
        if physical.get("suppressed"):
            failed.append("physical_motion_suppressed")

        candidate_count = _int(static_media.get("p_media_candidate_count"))
        screen_like = bool(static_media.get("screen_or_paper_like") or legacy_static.get("screen_like"))
        bbox = static_media.get("p_media_bbox") or legacy_static.get("p_media_bbox") or ()
        near_frame_edge = False
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = (_float(value) for value in bbox[:4])
            near_frame_edge = bool(x1 <= 32 or y1 <= 32 or x2 >= 608 or y2 >= 608)
        replay_candidate = bool(replay_state.get("candidate"))
        fast_candidate = bool(fast_state.get("candidate"))
        edge_score = _float(scores.get("edge"))
        yolo_context = _float(scores.get("yolo_context"))
        edge_screen_cue = bool(edge_score >= 0.18 and (near_frame_edge or replay_candidate or fast_candidate or yolo_context >= 0.10))
        legacy_source = str(legacy_static.get("triggered_source") or static_media.get("triggered_source") or "")
        legacy_a3plus_cue = bool(
            legacy_source.startswith("a3_plus")
            and (
                legacy_static.get("triggered")
                or replay_candidate
                or fast_candidate
                or occlusion_state.get("active")
                or replay_state.get("replay_evidence")
                or fast_state.get("fast_replay_evidence")
            )
        )
        strong_media = bool(static_media.get("p_media_strong_evidence") or static_media.get("classifier_triggered") or legacy_a3plus_cue)
        screen_cue = bool(
            screen_like
            or edge_screen_cue
            or _float(static_media.get("line_score")) >= 0.10
            or _float(static_media.get("classifier_score")) >= self.config.trigger_threshold
        )
        quality_gate = bool(candidate_count > 0 or screen_cue or strong_media)
        rebuilt_tighten_gate_applied = bool(
            static_media.get("result_contract_source") == "rebuilt"
            and self.config.rebuilt_tighten_gate_enabled
        )
        rebuilt_candidate_score = _float(scores.get("candidate_score"))
        rebuilt_border_contrast = _float(scores.get("border_contrast"))
        rebuilt_candidate_pass = bool(
            rebuilt_candidate_score
            + self.config.rebuilt_gate_candidate_tolerance
            >= self.config.rebuilt_gate_candidate_min
        )
        rebuilt_edge_pass = bool(
            self.config.rebuilt_gate_edge_min
            <= edge_score
            <= self.config.rebuilt_gate_edge_max
        )
        rebuilt_border_pass = bool(
            rebuilt_border_contrast
            >= self.config.rebuilt_gate_border_contrast_min
        )
        rebuilt_bbox = _bbox(
            static_media.get("p_media_bbox")
            or static_media.get("bbox")
            or static_media.get("candidate_bbox")
        )
        rebuilt_aspect_ratio = 0.0
        if rebuilt_bbox is not None:
            x1, y1, x2, y2 = rebuilt_bbox
            rebuilt_aspect_ratio = (x2 - x1) / max(1e-6, y2 - y1)
        rebuilt_aspect_pass = bool(
            rebuilt_bbox is not None
            and self.config.rebuilt_gate_aspect_ratio_min
            <= rebuilt_aspect_ratio
            <= self.config.rebuilt_gate_aspect_ratio_max
        )
        policy = (
            static_media.get("policy", {})
            if isinstance(static_media.get("policy"), dict)
            else {}
        )
        suppression = (
            static_media.get("suppression", {})
            if isinstance(static_media.get("suppression"), dict)
            else {}
        )
        rebuilt_result_fresh = bool(
            static_media.get("a3b_result_fresh", False)
        )
        rebuilt_candidate_allowed = bool(
            static_media.get(
                "media_candidate_allowed",
                policy.get(
                    "media_candidate_allowed",
                    suppression.get("media_candidate_allowed", False),
                ),
            )
        )
        rebuilt_policy_suppressed = bool(
            policy.get("suppressed", False)
            or suppression.get("suppressed", False)
        )
        rebuilt_authoritative_confirmed = bool(
            static_media.get("media_confirmed", False)
            or static_media.get("confirmed", False)
        )
        rebuilt_authoritative_guard_passed = bool(
            rebuilt_result_fresh
            and rebuilt_candidate_allowed
            and not rebuilt_policy_suppressed
            and rebuilt_aspect_pass
        )
        if rebuilt_authoritative_confirmed:
            rebuilt_tighten_gate_passed = (
                rebuilt_authoritative_guard_passed
            )
        else:
            rebuilt_tighten_gate_passed = bool(
                rebuilt_result_fresh
                and rebuilt_candidate_allowed
                and not rebuilt_policy_suppressed
                and rebuilt_candidate_pass
                and rebuilt_edge_pass
                and rebuilt_border_pass
                and rebuilt_aspect_pass
            )
        if (
            rebuilt_authoritative_confirmed
            and not rebuilt_authoritative_guard_passed
        ):
            failed.append("rebuilt_authoritative_guard_failed")
            quality_gate = False
        elif (
            rebuilt_tighten_gate_applied
            and not rebuilt_tighten_gate_passed
        ):
            failed.append("rebuilt_tighten_gate_failed")
            quality_gate = False
        if observed_score >= self.config.observed_threshold and not quality_gate:
            failed.append("no_candidate_or_screen_cue")
        return {
            "candidate_count": int(candidate_count),
            "best_candidate_score": float(max(_float(scores.get("edge")), _float(static_media.get("p_media")), observed_score)),
            "track_score": float(_float(scores.get("track"))),
            "quality_gate_passed": bool(quality_gate and not failed),
            "screen_cue": bool(screen_cue),
            "edge_screen_cue": bool(edge_screen_cue),
            "near_frame_edge": bool(near_frame_edge),
            "replay_candidate": bool(replay_candidate),
            "fast_candidate": bool(fast_candidate),
            "legacy_a3plus_cue": bool(legacy_a3plus_cue),
            "legacy_triggered_source": legacy_source,
            "yolo_context": float(yolo_context),
            "rebuilt_tighten_gate_applied": rebuilt_tighten_gate_applied,
            "rebuilt_tighten_gate_passed": rebuilt_tighten_gate_passed,
            "rebuilt_result_fresh": rebuilt_result_fresh,
            "rebuilt_candidate_allowed": rebuilt_candidate_allowed,
            "rebuilt_policy_suppressed": rebuilt_policy_suppressed,
            "rebuilt_authoritative_confirmed": rebuilt_authoritative_confirmed,
            "rebuilt_authoritative_guard_passed": (
                rebuilt_authoritative_guard_passed
            ),
            "rebuilt_candidate_score": rebuilt_candidate_score,
            "rebuilt_candidate_pass": rebuilt_candidate_pass,
            "rebuilt_edge_score": edge_score,
            "rebuilt_edge_pass": rebuilt_edge_pass,
            "rebuilt_border_contrast": rebuilt_border_contrast,
            "rebuilt_border_pass": rebuilt_border_pass,
            "rebuilt_aspect_ratio": rebuilt_aspect_ratio,
            "rebuilt_aspect_pass": rebuilt_aspect_pass,
            "failed_gates": failed,
        }
