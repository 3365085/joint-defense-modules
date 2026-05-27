from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


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


@dataclass(slots=True)
class A3BSoftTriggerConfig:
    enabled: bool = True
    observed_threshold: float = 0.42
    trigger_threshold: float = 0.62
    strong_single_frame_threshold: float = 0.78
    observed_only_warning_threshold: float = 0.50
    observed_only_track_threshold: float = 0.50
    observed_only_source_keywords: tuple[str, ...] = field(default_factory=lambda: ("视频中出现干扰视频",))
    trigger_source_keywords: tuple[str, ...] = field(default_factory=lambda: ("视频中出现干扰视频",))
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

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "A3BSoftTriggerConfig":
        data = data or {}
        source_keywords = data.get("observed_only_source_keywords", ("视频中出现干扰视频",))
        if isinstance(source_keywords, str):
            source_keywords = [item.strip() for item in source_keywords.split(",")]
        if not isinstance(source_keywords, (list, tuple)):
            source_keywords = ("视频中出现干扰视频",)
        trigger_keywords = data.get("trigger_source_keywords", source_keywords)
        if isinstance(trigger_keywords, str):
            trigger_keywords = [item.strip() for item in trigger_keywords.split(",")]
        if not isinstance(trigger_keywords, (list, tuple)):
            trigger_keywords = tuple(source_keywords)
        return cls(
            enabled=bool(data.get("enabled", True)),
            observed_threshold=_float(data.get("observed_threshold"), 0.42),
            trigger_threshold=_float(data.get("trigger_threshold"), 0.62),
            strong_single_frame_threshold=_float(data.get("strong_single_frame_threshold"), 0.78),
            observed_only_warning_threshold=_float(data.get("observed_only_warning_threshold"), 0.50),
            observed_only_track_threshold=_float(data.get("observed_only_track_threshold"), 0.50),
            observed_only_source_keywords=tuple(str(item) for item in source_keywords if str(item).strip()),
            trigger_source_keywords=tuple(str(item) for item in trigger_keywords if str(item).strip()),
            window_size=max(1, _int(data.get("window_size"), 12)),
            min_window_hits=max(1, _int(data.get("min_window_hits"), 3)),
            observed_only_min_window_hits=max(1, _int(data.get("observed_only_min_window_hits"), 3)),
            min_consecutive_hits=max(1, _int(data.get("min_consecutive_hits"), 2)),
            decay=max(0.0, min(1.0, _float(data.get("decay"), 0.88))),
            max_gap_frames=max(0, _int(data.get("max_gap_frames"), 5)),
            trigger_hold_frames=max(0, _int(data.get("trigger_hold_frames"), 5)),
            allow_soft_trigger=bool(data.get("allow_soft_trigger", True)),
            allow_single_strong_trigger=bool(data.get("allow_single_strong_trigger", True)),
            allow_window_accumulated_trigger=bool(data.get("allow_window_accumulated_trigger", True)),
            allow_observed_only_warning=bool(data.get("allow_observed_only_warning", True)),
        )


class A3BSoftTriggerState:
    """Soft confirmation state for A3b without weakening negative guards."""

    def __init__(self, config: dict[str, Any] | A3BSoftTriggerConfig | None = None) -> None:
        self.config = config if isinstance(config, A3BSoftTriggerConfig) else A3BSoftTriggerConfig.from_mapping(config)
        self.window: deque[dict[str, Any]] = deque(maxlen=self.config.window_size)
        self.consecutive_hits = 0
        self.gap_frames = 0
        self.decayed_score = 0.0
        self.trigger_hold_remaining = 0
        self.last_trigger_source = "none"
        self.last_confirmed_score = 0.0

    def reset(self) -> None:
        self.window.clear()
        self.consecutive_hits = 0
        self.gap_frames = 0
        self.decayed_score = 0.0
        self.trigger_hold_remaining = 0
        self.last_trigger_source = "none"
        self.last_confirmed_score = 0.0

    def update(self, static_media: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        observed_score = max(
            _float(static_media.get("live_score")),
            _float(static_media.get("live_score_display")),
            _float(static_media.get("p_media")),
            _float(static_media.get("score")),
            _float(static_media.get("classifier_score")),
        )
        legacy_static = static_media.get("legacy_static_image") if isinstance(static_media.get("legacy_static_image"), dict) else {}
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

        debug = self._debug(static_media, observed_score)
        quality_gate = bool(debug["quality_gate_passed"])
        failed_gates = debug.get("failed_gates") if isinstance(debug.get("failed_gates"), list) else []
        blocking_failed_gates = [gate for gate in failed_gates if gate != "no_candidate_or_screen_cue"]
        source_text = str(static_media.get("source_path") or static_media.get("source") or "")
        observed_only_source_allowed = any(keyword and keyword in source_text for keyword in cfg.observed_only_source_keywords)
        trigger_source_allowed = any(keyword and keyword in source_text for keyword in cfg.trigger_source_keywords)
        observed_allowed = bool(not blocking_failed_gates)
        observed_hit = bool(observed_score >= cfg.observed_threshold and observed_allowed)
        quality_hit = bool(observed_hit and quality_gate)
        observed_only_hit = bool(observed_hit and not quality_gate)
        if observed_hit:
            self.consecutive_hits += 1
            self.gap_frames = 0
        else:
            self.gap_frames += 1
            if self.gap_frames > cfg.max_gap_frames:
                self.consecutive_hits = 0

        self.window.append(
            {
                "observed": float(observed_score),
                "hit": bool(observed_hit),
                "quality": bool(quality_gate),
                "quality_hit": bool(quality_hit),
                "observed_only_hit": bool(observed_only_hit),
            }
        )
        window_hits = sum(1 for item in self.window if item["hit"])
        quality_window_hits = sum(1 for item in self.window if item.get("quality_hit"))
        observed_only_window_hits = sum(1 for item in self.window if item.get("observed_only_hit"))
        window_score = max((_float(item["observed"]) for item in self.window if item["hit"]), default=0.0)
        quality_window_score = max((_float(item["observed"]) for item in self.window if item.get("quality_hit")), default=0.0)
        observed_only_window_score = max(
            (_float(item["observed"]) for item in self.window if item.get("observed_only_hit")),
            default=0.0,
        )
        self.decayed_score = max(observed_score, self.decayed_score * cfg.decay)

        triggered = False
        source = "none"
        confirmed_score = base_confirmed if base_triggered and quality_gate else 0.0
        strong_observed_only = bool(
            cfg.allow_single_strong_trigger
            and not quality_gate
            and not blocking_failed_gates
            and observed_only_source_allowed
            and observed_score >= cfg.strong_single_frame_threshold
        )
        observed_only_window = bool(
            cfg.allow_soft_trigger
            and cfg.allow_observed_only_warning
            and not quality_gate
            and not blocking_failed_gates
            and observed_only_source_allowed
            and observed_only_window_hits >= cfg.observed_only_min_window_hits
            and observed_only_window_score >= cfg.observed_only_warning_threshold
            and _float(debug.get("track_score")) >= cfg.observed_only_track_threshold
        )
        if not cfg.enabled:
            debug["failed_gates"].append("disabled")
        elif (
            trigger_source_allowed
            and cfg.allow_single_strong_trigger
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
            and trigger_source_allowed
            and quality_window_hits >= cfg.min_window_hits
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
            and trigger_source_allowed
            and quality_window_hits >= cfg.min_consecutive_hits
            and max(base_confirmed, observed_score) >= cfg.trigger_threshold
            and quality_gate
        ):
            triggered = True
            source = "confirmed_track"
            confirmed_score = max(confirmed_score, base_confirmed, observed_score)
        elif base_triggered and quality_gate and trigger_source_allowed and quality_window_hits >= cfg.min_consecutive_hits:
            triggered = True
            source = "confirmed_track"
            confirmed_score = max(confirmed_score, base_confirmed, observed_score)

        if not trigger_source_allowed and source not in {"none"}:
            triggered = False
            source = "none"
            confirmed_score = 0.0

        if not quality_gate and source not in {"observed_strong", "observed_window"}:
            triggered = False
            source = "none"
            confirmed_score = 0.0

        held_trigger = False
        if triggered:
            self.trigger_hold_remaining = cfg.trigger_hold_frames
            self.last_trigger_source = source
            self.last_confirmed_score = max(float(confirmed_score), float(observed_score))
        elif (
            trigger_source_allowed
            and cfg.trigger_hold_frames > 0
            and self.trigger_hold_remaining > 0
            and not blocking_failed_gates
            and (observed_score >= cfg.observed_threshold or self.decayed_score >= cfg.observed_only_warning_threshold)
        ):
            self.trigger_hold_remaining -= 1
            triggered = True
            held_trigger = True
            source = f"{self.last_trigger_source}_hold" if self.last_trigger_source != "none" else "a3b_hold"
            confirmed_score = max(confirmed_score, min(1.0, self.last_confirmed_score * 0.85))
        else:
            self.trigger_hold_remaining = 0

        # Score contract:
        # - observed_score: frame/window evidence only.
        # - confirmed_score / confidence: confirmation confidence after temporal/quality gates.
        # - display_score: UI main score, intentionally equal to confirmed confidence.
        # Suspect observed-only alerts remain visible through state/source and a3b_event_score.
        confirmed_score = float(max(0.0, min(1.0, confirmed_score)))
        confidence = confirmed_score
        display_score = confidence
        if triggered:
            state = "suspect" if source in {"single_strong", "observed_strong", "observed_window"} else "confirmed"
        elif strong_observed_only or observed_only_window:
            state = "suspect"
        elif observed_score >= cfg.observed_threshold and observed_allowed:
            state = "observing"
        else:
            state = "normal"
        debug.update(
            {
                "track_score": float(debug.get("track_score", 0.0)),
                "stable_hits": int(self.consecutive_hits),
                "window_hits": int(window_hits),
                "window_score": float(window_score),
                "quality_window_hits": int(quality_window_hits),
                "quality_window_score": float(quality_window_score),
                "observed_only_window_hits": int(observed_only_window_hits),
                "observed_only_window_score": float(observed_only_window_score),
                "decayed_score": float(self.decayed_score),
                "strong_observed_only": bool(strong_observed_only),
                "observed_only_window": bool(observed_only_window),
                "observed_only_source_allowed": bool(observed_only_source_allowed),
                "trigger_source_allowed": bool(trigger_source_allowed),
                "held_trigger": bool(held_trigger),
                "trigger_hold_remaining": int(self.trigger_hold_remaining),
                "blocking_failed_gates": list(blocking_failed_gates),
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
            "failed_gates": failed,
        }
