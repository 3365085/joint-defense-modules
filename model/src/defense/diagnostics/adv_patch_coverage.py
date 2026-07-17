"""Streaming coverage diagnostics for authoritative ``adv_patch`` frame JSONL.

This module is deliberately read-only with respect to the production detector.
It reports observability metrics only and never turns them into an acceptance
decision.
"""

from __future__ import annotations

from collections import Counter
from math import floor, isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping
import json


DEFAULT_P_ADV_THRESHOLD = 0.65
_TIME_FIELDS = (
    "source_time_s",
    "source_time",
    "video_time_s",
    "video_time",
    "timestamp_s",
    "timestamp",
)
_FPS_FIELDS = ("source_fps", "video_fps", "fps")
_GATE_FIELDS = (
    "gate_scene_baseline",
    "gate_normal_motion",
    "normal_target_motion_exclusion",
    "normal_roi_flow_target_motion",
    "normal_articulated_target_motion",
    "normal_high_contrast_target_texture_motion",
    "gate_low_motion_bg",
    "gate_scene_spike",
)
_DECISION_FIELDS = (
    "adv_candidate_allowed",
    "adv_physical_support",
    "alert",
    "adv_confirmed",
)


def _rate(count: int, denominator: int) -> float | None:
    return float(count) / float(denominator) if denominator else None


def _finite_float(value: Any, *, field: str, line_number: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"line {line_number}: {field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number}: {field} must be a finite number"
        ) from exc
    if not isfinite(number):
        raise ValueError(f"line {line_number}: {field} must be a finite number")
    return number


def _optional_bool(row: Mapping[str, Any], field: str, *, line_number: int) -> bool | None:
    if field not in row or row[field] is None:
        return None
    value = row[field]
    if not isinstance(value, bool):
        raise ValueError(f"line {line_number}: {field} must be boolean when present")
    return value


def _optional_count(row: Mapping[str, Any], field: str, *, line_number: int) -> int | None:
    if field not in row or row[field] is None:
        return None
    value = row[field]
    if isinstance(value, bool):
        raise ValueError(f"line {line_number}: {field} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number}: {field} must be a non-negative integer"
        ) from exc
    if number < 0 or float(value) != float(number):
        raise ValueError(f"line {line_number}: {field} must be a non-negative integer")
    return number


def _reason(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    text = "" if value is None else str(value).strip()
    return text or "missing"


def _row_time(
    row: Mapping[str, Any],
    *,
    frame: int,
    fallback_fps: float | None,
    line_number: int,
) -> tuple[float | None, str | None, float | None]:
    for field in _TIME_FIELDS:
        if field in row and row[field] is not None:
            value = _finite_float(row[field], field=field, line_number=line_number)
            if value < 0:
                raise ValueError(f"line {line_number}: {field} must be >= 0")
            return value, field, fallback_fps

    row_fps = fallback_fps
    for field in _FPS_FIELDS:
        if field in row and row[field] is not None:
            row_fps = _finite_float(row[field], field=field, line_number=line_number)
            if row_fps <= 0:
                raise ValueError(f"line {line_number}: {field} must be > 0")
            break
    if row_fps is not None:
        return float(frame) / row_fps, "frame+fps", row_fps
    return None, None, row_fps


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q
    lower = floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _segment_payload(segment: Mapping[str, Any], *, fps: float | None) -> dict[str, Any]:
    start_time = segment.get("start_time_s")
    end_time = segment.get("end_time_s")
    duration = None
    if start_time is not None and end_time is not None:
        if fps is not None and segment.get("time_source") == "frame+fps":
            duration = max(0.0, float(end_time) - float(start_time)) + 1.0 / fps
        else:
            duration = max(0.0, float(end_time) - float(start_time))
    return {
        "start_frame": int(segment["start_frame"]),
        "end_frame": int(segment["end_frame"]),
        "frame_count": int(segment["frame_count"]),
        "start_time_s": start_time,
        "end_time_s": end_time,
        "duration_s": duration,
        "time_source": segment.get("time_source"),
    }


def analyze_adv_patch_coverage(
    jsonl_path: str | Path,
    *,
    p_adv_threshold: float = DEFAULT_P_ADV_THRESHOLD,
    fps: float | None = None,
) -> dict[str, Any]:
    """Read frame rows one by one and return coverage diagnostics.

    Malformed JSON, non-object rows, invalid required values and non-monotonic
    frame numbers raise ``ValueError``. Optional observability fields keep their
    own denominators so missing fields cannot be mistaken for negative values.
    """

    path = Path(jsonl_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"input JSONL does not exist or is not a file: {path}")
    threshold = _finite_float(p_adv_threshold, field="p_adv_threshold", line_number=0)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("p_adv_threshold must be in [0, 1]")
    if fps is not None:
        fps = _finite_float(fps, field="fps", line_number=0)
        if fps <= 0:
            raise ValueError("fps must be > 0")

    total_frames = 0
    previous_frame: int | None = None
    p_adv_values: list[float] = []
    above_threshold = 0
    decision_true: Counter[str] = Counter()
    decision_available: Counter[str] = Counter()
    gate_true: Counter[str] = Counter()
    gate_available: Counter[str] = Counter()
    anchor_true: Counter[str] = Counter()
    anchor_available: Counter[str] = Counter()
    blocked_above_threshold = 0
    explicit_reasons: Counter[str] = Counter()
    joint_reasons: Counter[str] = Counter()
    coverage_active_frames = 0
    coverage_available_frames = 0
    first_alarm: dict[str, Any] | None = None
    segments: list[dict[str, Any]] = []
    active_segment: dict[str, Any] | None = None
    last_effective_fps = fps

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"line {line_number}: blank lines are not allowed")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: JSON row must be an object")

            if "frame" not in row:
                raise ValueError(f"line {line_number}: missing required field frame")
            frame = _optional_count(row, "frame", line_number=line_number)
            assert frame is not None
            if previous_frame is not None and frame <= previous_frame:
                raise ValueError(
                    f"line {line_number}: frame must be strictly increasing "
                    f"({frame} <= {previous_frame})"
                )
            frame_is_contiguous = previous_frame is None or frame == previous_frame + 1
            previous_frame = frame

            if "p_adv" not in row:
                raise ValueError(f"line {line_number}: missing required field p_adv")
            p_adv = _finite_float(row["p_adv"], field="p_adv", line_number=line_number)
            if not 0.0 <= p_adv <= 1.0:
                raise ValueError(f"line {line_number}: p_adv must be in [0, 1]")
            p_adv_values.append(p_adv)
            total_frames += 1
            is_above = p_adv >= threshold
            above_threshold += int(is_above)

            flags: dict[str, bool | None] = {}
            for field in _DECISION_FIELDS:
                value = _optional_bool(row, field, line_number=line_number)
                flags[field] = value
                if value is not None:
                    decision_available[field] += 1
                    decision_true[field] += int(value)

            candidate_allowed = flags["adv_candidate_allowed"]
            if is_above and candidate_allowed is not True:
                blocked_above_threshold += 1
                explicit_reasons[_reason(row, "adv_explicit_suppression_reason")] += 1
                joint_reasons[_reason(row, "joint_suppressed_reason")] += 1

            for field in _GATE_FIELDS:
                value = _optional_bool(row, field, line_number=line_number)
                if value is not None:
                    gate_available[field] += 1
                    gate_true[field] += int(value)

            anchor_counts = {
                field: _optional_count(row, field, line_number=line_number)
                for field in ("raw_person", "raw_head", "raw_helmet")
            }
            for field, count in anchor_counts.items():
                if count is not None:
                    anchor_available[field] += 1
                    anchor_true[field] += int(count > 0)
            available_anchor_values = [value for value in anchor_counts.values() if value is not None]
            if available_anchor_values:
                anchor_available["any_anchor"] += 1
                anchor_true["any_anchor"] += int(any(value > 0 for value in available_anchor_values))
            head_helmet = [anchor_counts["raw_head"], anchor_counts["raw_helmet"]]
            if any(value is not None for value in head_helmet):
                anchor_available["head_or_helmet"] += 1
                anchor_true["head_or_helmet"] += int(
                    any(value is not None and value > 0 for value in head_helmet)
                )

            alert = flags["alert"]
            confirmed = flags["adv_confirmed"]
            coverage_available = alert is not None or confirmed is not None
            coverage_active = bool(alert) or bool(confirmed)
            if coverage_available:
                coverage_available_frames += 1
                coverage_active_frames += int(coverage_active)

            time_s, time_source, effective_fps = _row_time(
                row,
                frame=frame,
                fallback_fps=last_effective_fps,
                line_number=line_number,
            )
            last_effective_fps = effective_fps
            if coverage_active:
                if first_alarm is None:
                    first_alarm = {
                        "frame": frame,
                        "time_s": time_s,
                        "time_source": time_source,
                    }
                if active_segment is None or not frame_is_contiguous:
                    if active_segment is not None:
                        segments.append(_segment_payload(active_segment, fps=last_effective_fps))
                    active_segment = {
                        "start_frame": frame,
                        "end_frame": frame,
                        "frame_count": 1,
                        "start_time_s": time_s,
                        "end_time_s": time_s,
                        "time_source": time_source,
                    }
                else:
                    active_segment["end_frame"] = frame
                    active_segment["frame_count"] = int(active_segment["frame_count"]) + 1
                    active_segment["end_time_s"] = time_s
                    if active_segment.get("time_source") != time_source:
                        active_segment["time_source"] = "mixed"
            elif active_segment is not None:
                segments.append(_segment_payload(active_segment, fps=last_effective_fps))
                active_segment = None

    if total_frames == 0:
        raise ValueError("input JSONL contains no frame rows")
    if active_segment is not None:
        segments.append(_segment_payload(active_segment, fps=last_effective_fps))

    sorted_scores = sorted(p_adv_values)
    longest_segment = max(
        segments,
        key=lambda item: (int(item["frame_count"]), -int(item["start_frame"])),
        default=None,
    )
    coverage_evaluable = coverage_available_frames == total_frames
    coverage_reasons: list[str] = []
    if not coverage_evaluable:
        coverage_reasons.append(
            "alert_or_adv_confirmed_not_available_for_every_frame"
        )

    def _boolean_metrics(fields: Iterable[str], true_counts: Counter[str], available: Counter[str]) -> dict[str, Any]:
        return {
            field: {
                "true_frames": int(true_counts[field]),
                "available_frames": int(available[field]),
                "rate": _rate(int(true_counts[field]), int(available[field])),
            }
            for field in fields
        }

    return {
        "schema_version": 1,
        "input": str(path),
        "total_frames": total_frames,
        "coverage_evaluable": coverage_evaluable,
        "coverage_evaluation_reasons": coverage_reasons,
        "p_adv": {
            "threshold": threshold,
            "above_threshold_frames": above_threshold,
            "above_threshold_rate": _rate(above_threshold, total_frames),
            "max": max(p_adv_values),
            "quantiles": {
                "p50": _quantile(sorted_scores, 0.50),
                "p90": _quantile(sorted_scores, 0.90),
                "p95": _quantile(sorted_scores, 0.95),
                "p99": _quantile(sorted_scores, 0.99),
            },
        },
        "decisions": {
            **_boolean_metrics(_DECISION_FIELDS, decision_true, decision_available),
            "alert_or_adv_confirmed": {
                "true_frames": coverage_active_frames,
                "available_frames": coverage_available_frames,
                "rate": _rate(coverage_active_frames, coverage_available_frames),
            },
        },
        "first_alarm": first_alarm,
        "alarm_segments": {
            "count": len(segments),
            "longest": longest_segment,
            "segments": segments,
        },
        "above_threshold_blocked": {
            "frames": blocked_above_threshold,
            "rate_of_above_threshold": _rate(blocked_above_threshold, above_threshold),
            "adv_explicit_suppression_reasons": dict(explicit_reasons.most_common()),
            "joint_suppressed_reasons": dict(joint_reasons.most_common()),
        },
        "gates": _boolean_metrics(_GATE_FIELDS, gate_true, gate_available),
        "yolo_anchor_presence": _boolean_metrics(
            ("raw_person", "raw_head", "raw_helmet", "head_or_helmet", "any_anchor"),
            anchor_true,
            anchor_available,
        ),
    }


__all__ = [
    "DEFAULT_P_ADV_THRESHOLD",
    "analyze_adv_patch_coverage",
]
