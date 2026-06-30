from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .ppe_overlay_video import scale_ppe_tracks
from .ppe_overlay_summary import load_ppe_overlay_records

_HELD_SOURCES = {"held_static", "held_extrapolated_one_frame"}
_PPE_LABELS = ("person", "head", "helmet")


@dataclass(frozen=True)
class VisualRiskThresholds:
    abrupt_count_delta: int = 2
    held_warn_frames: int = 2
    held_fail_frames: int = 4
    label_switch_window: int = 12
    box_growth_ratio: float = 1.85
    box_growth_window: int = 30
    large_head_height_ratio: float = 0.16
    edge_margin_ratio: float = 0.012
    max_review_frames: int = 80
    motion_scale_width: int = 960
    motion_diff_threshold: int = 28
    motion_min_area_ratio: float = 0.0025
    motion_min_height_ratio: float = 0.16
    motion_min_aspect: float = 1.45
    motion_overlay_min_coverage: float = 0.015


def build_visual_risk_report(
    *,
    video_path: str | Path,
    overlay_json: str | Path | None = None,
    thresholds: VisualRiskThresholds | None = None,
    sample_video_colors: bool = True,
) -> dict[str, Any]:
    video = Path(video_path)
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(f"video does not exist: {video}")
    limits = thresholds or VisualRiskThresholds()
    video_info = _scan_video(video, sample_video_colors=sample_video_colors, thresholds=limits)

    records: list[dict[str, Any]] = []
    if overlay_json is not None:
        records = sorted(
            load_ppe_overlay_records(overlay_json),
            key=lambda item: int(item.get("frame_idx") or 0),
        )

    risks: list[dict[str, Any]] = []
    if records:
        risks.extend(_overlay_risks(records, video_shape=(video_info["height"], video_info["width"]), thresholds=limits))
        risks.extend(_motion_missing_target_risks(video_info, records, thresholds=limits))
    risks.extend(_video_risks(video_info, records))

    risks = sorted(
        risks,
        key=lambda item: (
            int(item.get("local_frame_index", item.get("range_start", 0)) or 0),
            str(item.get("risk_type") or ""),
        ),
    )
    risk_segments = _risk_segments(risks)
    summary = _risk_summary(risks, thresholds=limits)
    verdict = _verdict(summary)
    return {
        "schema_version": "visual_risk_scan_v1",
        "video_path": str(video),
        "overlay_json": str(Path(overlay_json)) if overlay_json is not None else None,
        "video": video_info,
        "overlay": _overlay_summary(records),
        "thresholds": limits.__dict__,
        "risk_summary": summary,
        "risks": risks,
        "risk_segments": risk_segments,
        "review_frames": _review_frames(risks, risk_segments=risk_segments, max_frames=limits.max_review_frames),
        "verdict": verdict,
        "next_action": _next_action(verdict),
    }


def write_visual_risk_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path,
    export_risk_frames: bool = False,
    risk_frame_padding: int = 2,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "visual_risk_report.json"
    md_path = out_dir / "visual_risk_report.md"
    csv_path = out_dir / "risk_segments.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_report_markdown(report), encoding="utf-8")
    _write_risk_csv(csv_path, report.get("risk_segments", []))
    written = {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "csv_path": str(csv_path),
    }
    if export_risk_frames:
        frames_dir = out_dir / "risk_frames"
        exported = export_review_frames_from_video(
            video_path=report["video_path"],
            frame_indices=report.get("review_frames", {}).get("local_frame_indices", []),
            output_dir=frames_dir,
            padding=risk_frame_padding,
        )
        written["risk_frames_dir"] = str(frames_dir)
        written["risk_frames_manifest"] = str(exported["manifest_path"])
    return written


def export_review_frames_from_video(
    *,
    video_path: str | Path,
    frame_indices: Iterable[int],
    output_dir: str | Path,
    padding: int = 2,
) -> dict[str, Any]:
    frames = sorted({max(0, int(frame) + delta) for frame in frame_indices for delta in range(-padding, padding + 1)})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames > 0:
        frames = [frame for frame in frames if frame < total_frames]
    exported: list[dict[str, Any]] = []
    try:
        for frame_idx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            path = out_dir / f"risk_frame_{frame_idx:06d}.png"
            ok, encoded = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            if not ok:
                raise RuntimeError(f"failed to encode risk frame: {path}")
            path.write_bytes(encoded.tobytes())
            exported.append(
                {
                    "frame_idx": int(frame_idx),
                    "time_s": float(frame_idx) / fps if fps > 0 else None,
                    "path": str(path),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "format": "png",
                    "temporary_review_artifact": True,
                }
            )
    finally:
        cap.release()
    manifest = {
        "video_path": str(Path(video_path)),
        "output_dir": str(out_dir),
        "padding": int(padding),
        "requested_center_frames": sorted({int(frame) for frame in frame_indices}),
        "exported_frame_count": len(exported),
        "frames": exported,
        "artifact_policy": {
            "retention_class": "temporary_visual_risk_review",
            "valid_for_final_acceptance": False,
            "cleanup_required": True,
            "cleanup_owner": "agent_or_operator",
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _scan_video(
    video_path: Path,
    *,
    sample_video_colors: bool,
    thresholds: VisualRiskThresholds,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames_read = 0
    samples: list[dict[str, Any]] = []
    motion_gray_frames: list[Any] = []
    sample_stride = max(1, total_frames // 24) if total_frames > 0 else 15
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if sample_video_colors and frame_idx % sample_stride == 0:
                samples.append(
                    {
                        "frame_idx": int(frame_idx),
                        "color_components": _color_component_counts(frame),
                    }
                )
            if width > 0 and height > 0:
                scale = min(1.0, float(thresholds.motion_scale_width) / float(width))
                if scale < 1.0:
                    small = cv2.resize(
                        frame,
                        (int(round(width * scale)), int(round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = frame
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                motion_gray_frames.append(cv2.GaussianBlur(gray, (5, 5), 0))
            frames_read += 1
            frame_idx += 1
    finally:
        cap.release()
    return {
        "fps": fps,
        "frame_count": total_frames,
        "frames_read": frames_read,
        "width": width,
        "height": height,
        "motion_scale_width": int(motion_gray_frames[0].shape[1]) if motion_gray_frames else None,
        "motion_scale_height": int(motion_gray_frames[0].shape[0]) if motion_gray_frames else None,
        "motion_candidates": _motion_candidates(motion_gray_frames, thresholds=thresholds),
        "color_sample_stride": sample_stride if sample_video_colors else None,
        "color_samples": samples,
    }


def _motion_candidates(frames: list[Any], *, thresholds: VisualRiskThresholds) -> list[list[dict[str, Any]]]:
    if len(frames) < 3:
        return [[] for _ in frames]
    stack = np.stack(frames, axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    height, width = frames[0].shape[:2]
    min_area = max(80, int(float(width * height) * thresholds.motion_min_area_ratio))
    min_height = max(24, int(float(height) * thresholds.motion_min_height_ratio))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    all_candidates: list[list[dict[str, Any]]] = []
    for index, gray in enumerate(frames):
        diff = cv2.absdiff(gray, background)
        temporal_window = min(6, max(1, len(frames) // 4))
        prev_index = max(0, index - temporal_window)
        next_index = min(len(frames) - 1, index + temporal_window)
        if prev_index != index:
            diff = cv2.max(diff, cv2.absdiff(gray, frames[prev_index]))
        if next_index != index:
            diff = cv2.max(diff, cv2.absdiff(gray, frames[next_index]))
        _, mask = cv2.threshold(diff, int(thresholds.motion_diff_threshold), 255, cv2.THRESH_BINARY)
        mask[: int(height * 0.08), :] = 0
        mask[int(height * 0.93) :, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_candidates: list[dict[str, Any]] = []
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or h < min_height or w < 18:
                continue
            aspect = float(h) / max(1.0, float(w))
            if aspect < thresholds.motion_min_aspect:
                continue
            if y < int(height * 0.2):
                continue
            if w > int(width * 0.45):
                continue
            frame_candidates.append(
                {
                    "local_frame_index": int(index),
                    "box": [x, y, x + w, y + h],
                    "area": area,
                    "aspect": round(aspect, 3),
                    "centroid": [round(float(centroids[label][0]), 3), round(float(centroids[label][1]), 3)],
                    "scale_width": int(width),
                    "scale_height": int(height),
                }
            )
        all_candidates.append(frame_candidates)
    return all_candidates


def _color_component_counts(frame: Any) -> dict[str, int]:
    colors = {
        "helmet_green": np.array([0, 220, 80], dtype=np.int16),
        "head_orange": np.array([0, 150, 255], dtype=np.int16),
        "person_yellow": np.array([255, 210, 80], dtype=np.int16),
    }
    counts: dict[str, int] = {}
    image = frame.astype(np.int16)
    for name, bgr in colors.items():
        diff = np.max(np.abs(image - bgr), axis=2)
        mask = (diff <= 52).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        count = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area >= 18 and max(w, h) >= 6:
                count += 1
        counts[name] = count
    return counts


def _overlay_risks(
    records: list[dict[str, Any]],
    *,
    video_shape: tuple[int, int],
    thresholds: VisualRiskThresholds,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    prev_counts: Counter[str] | None = None
    track_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    held_items: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for local_index, record in enumerate(records):
        scaled_tracks = scale_ppe_tracks(
            record.get("ppe_tracks") or [],
            target_shape=video_shape,
            source_shape=_record_box_source_shape(record),
        )
        counts = Counter(str(track.get("label") or "") for track in scaled_tracks)
        if prev_counts is not None:
            delta = sum(abs(int(counts.get(label, 0)) - int(prev_counts.get(label, 0))) for label in _PPE_LABELS)
            helmet_head_delta = (
                abs(int(counts.get("head", 0)) - int(prev_counts.get("head", 0)))
                + abs(int(counts.get("helmet", 0)) - int(prev_counts.get("helmet", 0)))
            )
            if delta >= thresholds.abrupt_count_delta or helmet_head_delta > 0:
                risks.append(
                    _risk(
                        "count_change",
                        "review",
                        local_index,
                        record,
                        message="visible PPE label count changed between adjacent records",
                        details={
                            "previous_counts": dict(prev_counts),
                            "current_counts": dict(counts),
                            "delta": int(delta),
                        },
                    )
                )
        prev_counts = counts

        for track in scaled_tracks:
            tid = _track_id(track)
            if tid is None:
                continue
            label = str(track.get("label") or "")
            box = _box(track.get("box"))
            if box is None:
                continue
            source = str(track.get("display_box_source") or "")
            misses = int(track.get("misses") or 0)
            fresh = bool(track.get("fresh_detection", True))
            item = {
                "local_frame_index": local_index,
                "source_frame_idx": _int(record.get("frame_idx")),
                "track_id": tid,
                "label": label,
                "box": box,
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "area": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]),
                "display_box_source": source,
                "misses": misses,
                "fresh_detection": fresh,
                "confidence": _float(track.get("confidence")),
                "temporal_promoted": bool(track.get("temporal_promoted", False)),
                "promoted_label": str(track.get("promoted_label") or ""),
            }
            history = track_history[tid]
            if history:
                previous = history[-1]
                if label != previous["label"] and {label, previous["label"]} <= {"head", "helmet"}:
                    risks.append(
                        _risk(
                            "label_switch",
                            "fail",
                            local_index,
                            record,
                            track_id=tid,
                            label=label,
                            box=box,
                            message="same track switched between head and helmet",
                            details={
                                "previous_label": previous["label"],
                                "previous_local_frame_index": previous["local_frame_index"],
                                "previous_source_frame_idx": previous["source_frame_idx"],
                                "window": int(local_index - previous["local_frame_index"]),
                            },
                        )
                    )
                _append_growth_risk(risks, item, history, record=record, thresholds=thresholds)
            _append_large_or_edge_risks(risks, item, record=record, video_shape=video_shape, thresholds=thresholds)
            history.append(item)

            if source in _HELD_SOURCES or misses > 0 or not fresh:
                held_items[tid].append(item)

    risks.extend(_held_segment_risks(held_items, records=records, thresholds=thresholds))
    risks.extend(_instability_risks(track_history, records=records, thresholds=thresholds))
    return risks


def _motion_missing_target_risks(
    video_info: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    thresholds: VisualRiskThresholds,
) -> list[dict[str, Any]]:
    candidates_by_frame = video_info.get("motion_candidates") or []
    motion_w = int(video_info.get("motion_scale_width") or 0)
    motion_h = int(video_info.get("motion_scale_height") or 0)
    video_w = int(video_info.get("width") or 0)
    video_h = int(video_info.get("height") or 0)
    if not candidates_by_frame or motion_w <= 0 or motion_h <= 0 or video_w <= 0 or video_h <= 0:
        return []

    risks: list[dict[str, Any]] = []
    for local_index, candidates in enumerate(candidates_by_frame[: len(records)]):
        if not candidates:
            continue
        record = records[local_index]
        tracks = scale_ppe_tracks(
            record.get("ppe_tracks") or [],
            target_shape=(motion_h, motion_w),
            source_shape=_record_box_source_shape(record),
        )
        track_boxes = [
            _box(track.get("box"))
            for track in tracks
            if str(track.get("label") or "") in {"head", "helmet", "person"}
        ]
        track_boxes = [box for box in track_boxes if box is not None]
        for candidate in candidates:
            candidate_box = [float(value) for value in candidate["box"]]
            coverage = _max_intersection_over_candidate(candidate_box, track_boxes)
            if coverage >= thresholds.motion_overlay_min_coverage:
                continue
            source_box = _scale_box(candidate_box, from_shape=(motion_h, motion_w), to_shape=(video_h, video_w))
            risks.append(
                _risk(
                    "missing_visible_target",
                    "fail",
                    local_index,
                    record,
                    box=source_box,
                    message="large moving foreground candidate has no PPE/person track coverage",
                    details={
                        "motion_box": [round(float(value), 3) for value in candidate_box],
                        "coverage": round(float(coverage), 5),
                        "area": int(candidate.get("area") or 0),
                        "aspect": candidate.get("aspect"),
                        "centroid": candidate.get("centroid"),
                        "motion_scale": [motion_w, motion_h],
                    },
                )
            )
    return _merge_missing_target_risks(risks)


def _merge_missing_target_risks(risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not risks:
        return []
    merged: list[dict[str, Any]] = []
    active = dict(risks[0])
    details = dict(active.get("details") or {})
    details["frame_count"] = 1
    details["max_coverage"] = float(details.get("coverage") or 0.0)
    active["details"] = details
    for risk in risks[1:]:
        active_end = int(active.get("range_end", active.get("local_frame_index")) or 0)
        active_box = active.get("box")
        risk_box = risk.get("box")
        same_segment = (
            int(risk.get("local_frame_index") or 0) <= active_end + 1
            and isinstance(active_box, list)
            and isinstance(risk_box, list)
            and _iou(active_box, risk_box) >= 0.12
        )
        if same_segment:
            active["range_start"] = int(active.get("range_start", active.get("local_frame_index")) or 0)
            active["range_end"] = int(risk.get("local_frame_index") or 0)
            active["source_frame_start"] = int(active.get("source_frame_start", active.get("source_frame_idx")) or 0)
            active["source_frame_end"] = int(risk.get("source_frame_idx") or active.get("source_frame_start") or 0)
            details = dict(active.get("details") or {})
            details["frame_count"] = int(details.get("frame_count") or 1) + 1
            details["max_coverage"] = max(float(details.get("max_coverage") or 0.0), float((risk.get("details") or {}).get("coverage") or 0.0))
            active["details"] = details
            continue
        merged.append(active)
        active = dict(risk)
        details = dict(active.get("details") or {})
        details["frame_count"] = 1
        details["max_coverage"] = float(details.get("coverage") or 0.0)
        active["details"] = details
    merged.append(active)
    return merged


def _append_growth_risk(
    risks: list[dict[str, Any]],
    item: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    record: dict[str, Any],
    thresholds: VisualRiskThresholds,
) -> None:
    recent = [
        prev
        for prev in history
        if 0 <= item["local_frame_index"] - prev["local_frame_index"] <= thresholds.box_growth_window
        and prev["area"] > 0
    ]
    if not recent:
        return
    min_area = min(float(prev["area"]) for prev in recent)
    min_width = min(float(prev["width"]) for prev in recent if float(prev["width"]) > 0)
    min_height = min(float(prev["height"]) for prev in recent if float(prev["height"]) > 0)
    area_ratio = float(item["area"]) / max(1.0, min_area)
    width_ratio = float(item["width"]) / max(1.0, min_width)
    height_ratio = float(item["height"]) / max(1.0, min_height)
    if area_ratio >= thresholds.box_growth_ratio and max(width_ratio, height_ratio) >= math.sqrt(thresholds.box_growth_ratio):
        risks.append(
            _risk(
                "box_growth",
                "fail" if item["label"] == "head" else "review",
                item["local_frame_index"],
                record,
                track_id=item["track_id"],
                label=item["label"],
                box=item["box"],
                message="track box grew sharply within the review window",
                details={
                    "area_ratio": round(area_ratio, 3),
                    "width_ratio": round(width_ratio, 3),
                    "height_ratio": round(height_ratio, 3),
                    "window": thresholds.box_growth_window,
                },
            )
        )


def _append_large_or_edge_risks(
    risks: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    record: dict[str, Any],
    video_shape: tuple[int, int],
    thresholds: VisualRiskThresholds,
) -> None:
    h, w = video_shape
    box = item["box"]
    margin_x = max(1.0, float(w) * thresholds.edge_margin_ratio)
    margin_y = max(1.0, float(h) * thresholds.edge_margin_ratio)
    touches_edge = box[0] <= margin_x or box[1] <= margin_y or box[2] >= float(w) - margin_x or box[3] >= float(h) - margin_y
    if touches_edge and item["label"] in {"head", "helmet"}:
        risks.append(
            _risk(
                "edge_touch",
                "review",
                item["local_frame_index"],
                record,
                track_id=item["track_id"],
                label=item["label"],
                box=box,
                message="PPE box touches or nearly touches a video boundary",
                details={"video_width": int(w), "video_height": int(h)},
            )
        )
    if item["label"] == "head" and float(item["height"]) >= float(h) * thresholds.large_head_height_ratio:
        risks.append(
            _risk(
                "large_head_box",
                "review",
                item["local_frame_index"],
                record,
                track_id=item["track_id"],
                label=item["label"],
                box=box,
                message="head box is large relative to the video frame",
                details={
                    "height_ratio": round(float(item["height"]) / max(1.0, float(h)), 3),
                    "threshold": thresholds.large_head_height_ratio,
                },
            )
        )


def _held_segment_risks(
    held_items: dict[int, list[dict[str, Any]]],
    *,
    records: list[dict[str, Any]],
    thresholds: VisualRiskThresholds,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for tid, items in held_items.items():
        for segment in _consecutive_segments(items):
            length = len(segment)
            if length < thresholds.held_warn_frames:
                continue
            first = segment[0]
            last = segment[-1]
            severity = "fail" if length >= thresholds.held_fail_frames else "review"
            risks.append(
                _risk(
                    "held_track",
                    severity,
                    first["local_frame_index"],
                    records[first["local_frame_index"]],
                    track_id=tid,
                    label=first["label"],
                    box=first["box"],
                    message="track uses held boxes for consecutive frames",
                    range_start=first["local_frame_index"],
                    range_end=last["local_frame_index"],
                    source_frame_start=first["source_frame_idx"],
                    source_frame_end=last["source_frame_idx"],
                    details={
                        "frames": length,
                        "sources": sorted({str(item["display_box_source"]) for item in segment}),
                        "max_misses": max(int(item["misses"]) for item in segment),
                    },
                )
            )
    return risks


def _instability_risks(
    track_history: dict[int, list[dict[str, Any]]],
    *,
    records: list[dict[str, Any]],
    thresholds: VisualRiskThresholds,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for tid, history in track_history.items():
        if len(history) < 2:
            continue
        labels = [str(item["label"]) for item in history if str(item["label"]) in {"head", "helmet"}]
        if not {"head", "helmet"} <= set(labels):
            continue
        first = min(
            (item for item in history if str(item["label"]) in {"head", "helmet"}),
            key=lambda item: item["local_frame_index"],
        )
        last = max(
            (item for item in history if str(item["label"]) in {"head", "helmet"}),
            key=lambda item: item["local_frame_index"],
        )
        if int(last["local_frame_index"]) - int(first["local_frame_index"]) <= thresholds.label_switch_window * 4:
            risks.append(
                _risk(
                    "track_label_instability",
                    "fail",
                    first["local_frame_index"],
                    records[first["local_frame_index"]],
                    track_id=tid,
                    label=first["label"],
                    box=first["box"],
                    message="track contains both head and helmet labels in a short span",
                    range_start=first["local_frame_index"],
                    range_end=last["local_frame_index"],
                    source_frame_start=first["source_frame_idx"],
                    source_frame_end=last["source_frame_idx"],
                    details={
                        "labels": sorted(set(labels)),
                        "label_sequence_sample": labels[:24],
                    },
                )
            )
    return risks


def _video_risks(video_info: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    if int(video_info.get("frames_read") or 0) <= 0:
        risks.append(
            {
                "risk_type": "video_unreadable",
                "severity": "fail",
                "message": "OpenCV could not decode any video frames",
                "local_frame_index": 0,
                "source_frame_idx": None,
                "details": {},
            }
        )
    frame_count = int(video_info.get("frame_count") or 0)
    frames_read = int(video_info.get("frames_read") or 0)
    if frame_count > 0 and frames_read < frame_count:
        risks.append(
            {
                "risk_type": "video_short_decode",
                "severity": "review",
                "message": "OpenCV decoded fewer frames than the container reports",
                "local_frame_index": frames_read,
                "source_frame_idx": None,
                "details": {"frame_count": frame_count, "frames_read": frames_read},
            }
        )
    if records and frame_count > 0 and abs(frame_count - len(records)) > 2:
        risks.append(
            {
                "risk_type": "video_overlay_count_mismatch",
                "severity": "review",
                "message": "video frame count and overlay record count differ",
                "local_frame_index": 0,
                "source_frame_idx": None,
                "details": {"video_frame_count": frame_count, "overlay_record_count": len(records)},
            }
        )
    return risks


def _record_box_source_shape(record: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(record, dict):
        return (640, 640)
    coord = record.get("overlay_coordinate_space")
    if isinstance(coord, dict):
        shape = coord.get("box_space_shape") or coord.get("preview_frame_shape")
        parsed = _shape_tuple(shape)
        if parsed is not None:
            return parsed
    for key in ("detector_frame_shape", "runtime_source_frame_shape"):
        parsed = _shape_tuple(record.get(key))
        if parsed is not None:
            return parsed
    return (640, 640)


def _shape_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (max(1, int(value[0])), max(1, int(value[1])))
    except (TypeError, ValueError):
        return None


def _box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _scale_box(
    box: list[float],
    *,
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> list[float]:
    from_h, from_w = from_shape
    to_h, to_w = to_shape
    sx = float(to_w) / max(1.0, float(from_w))
    sy = float(to_h) / max(1.0, float(from_h))
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def _max_intersection_over_candidate(candidate: list[float], boxes: list[list[float]]) -> float:
    area = max(0.0, candidate[2] - candidate[0]) * max(0.0, candidate[3] - candidate[1])
    if area <= 0:
        return 0.0
    best = 0.0
    for box in boxes:
        x1 = max(candidate[0], box[0])
        y1 = max(candidate[1], box[1])
        x2 = min(candidate[2], box[2])
        y2 = min(candidate[3], box[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        best = max(best, inter / area)
    return best


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / max(1.0, area_a + area_b - inter)


def _track_id(track: dict[str, Any]) -> int | None:
    try:
        return int(track.get("track_id"))
    except (TypeError, ValueError):
        return None


def _risk(
    risk_type: str,
    severity: str,
    local_frame_index: int,
    record: dict[str, Any],
    *,
    track_id: int | None = None,
    label: str | None = None,
    box: list[float] | None = None,
    message: str,
    range_start: int | None = None,
    range_end: int | None = None,
    source_frame_start: int | None = None,
    source_frame_end: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_frame = _int(record.get("frame_idx"))
    out: dict[str, Any] = {
        "risk_type": risk_type,
        "severity": severity,
        "local_frame_index": int(local_frame_index),
        "source_frame_idx": source_frame,
        "message": message,
        "details": dict(details or {}),
    }
    if track_id is not None:
        out["track_id"] = int(track_id)
    if label is not None:
        out["label"] = label
    if box is not None:
        out["box"] = [round(float(value), 3) for value in box]
    if range_start is not None:
        out["range_start"] = int(range_start)
    if range_end is not None:
        out["range_end"] = int(range_end)
    if source_frame_start is not None:
        out["source_frame_start"] = int(source_frame_start)
    if source_frame_end is not None:
        out["source_frame_end"] = int(source_frame_end)
    return out


def _consecutive_segments(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not items:
        return []
    sorted_items = sorted(items, key=lambda item: int(item["local_frame_index"]))
    segments: list[list[dict[str, Any]]] = [[sorted_items[0]]]
    for item in sorted_items[1:]:
        if int(item["local_frame_index"]) == int(segments[-1][-1]["local_frame_index"]) + 1:
            segments[-1].append(item)
        else:
            segments.append([item])
    return segments


def _overlay_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    shapes = sorted({str(_record_box_source_shape(record)) for record in records})
    return {
        "record_count": len(records),
        "source_frame_start": _int(records[0].get("frame_idx")) if records else None,
        "source_frame_end": _int(records[-1].get("frame_idx")) if records else None,
        "box_source_shapes": shapes,
    }


def _risk_summary(risks: list[dict[str, Any]], *, thresholds: VisualRiskThresholds) -> dict[str, Any]:
    by_type = Counter(str(risk.get("risk_type") or "") for risk in risks)
    by_severity = Counter(str(risk.get("severity") or "") for risk in risks)
    return {
        "risk_count": len(risks),
        "by_type": dict(sorted(by_type.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "fail_count": int(by_severity.get("fail", 0)),
        "review_count": int(by_severity.get("review", 0)),
        "thresholds": thresholds.__dict__,
    }


def _risk_segments(risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for risk in risks:
        key = (
            str(risk.get("risk_type") or ""),
            str(risk.get("severity") or ""),
            str(risk.get("track_id", "")),
            str(risk.get("label", "")),
        )
        grouped[key].append(risk)

    segments: list[dict[str, Any]] = []
    for (risk_type, severity, track_id, label), items in grouped.items():
        spans = sorted((_risk_span(item), item) for item in items)
        active: dict[str, Any] | None = None
        for (start, end, source_start, source_end), risk in spans:
            if active is not None and start <= int(active["range_end"]) + 1:
                active["range_end"] = max(int(active["range_end"]), int(end))
                if source_start is not None:
                    previous = active.get("source_frame_start")
                    active["source_frame_start"] = source_start if previous is None else min(int(previous), int(source_start))
                if source_end is not None:
                    previous = active.get("source_frame_end")
                    active["source_frame_end"] = source_end if previous is None else max(int(previous), int(source_end))
                active["risk_count"] = int(active["risk_count"]) + 1
                active["messages"] = sorted(set([*active.get("messages", []), str(risk.get("message") or "")]))[:4]
                continue
            if active is not None:
                segments.append(active)
            active = {
                "risk_type": risk_type,
                "severity": severity,
                "range_start": int(start),
                "range_end": int(end),
                "source_frame_start": source_start,
                "source_frame_end": source_end,
                "track_id": int(track_id) if track_id else None,
                "label": label or None,
                "risk_count": 1,
                "messages": [str(risk.get("message") or "")],
            }
        if active is not None:
            segments.append(active)

    return sorted(
        segments,
        key=lambda item: (
            _severity_rank(str(item.get("severity") or "")),
            int(item.get("range_start") or 0),
            str(item.get("risk_type") or ""),
        ),
    )


def _risk_span(risk: dict[str, Any]) -> tuple[int, int, int | None, int | None]:
    start = int(risk.get("range_start", risk.get("local_frame_index", 0)) or 0)
    end = int(risk.get("range_end", start) or start)
    source_start = risk.get("source_frame_start", risk.get("source_frame_idx"))
    source_end = risk.get("source_frame_end", source_start)
    return (
        min(start, end),
        max(start, end),
        _int(source_start),
        _int(source_end),
    )


def _review_frames(
    risks: list[dict[str, Any]],
    *,
    risk_segments: list[dict[str, Any]] | None = None,
    max_frames: int,
) -> dict[str, Any]:
    frames: set[int] = set()
    source_frames: set[int] = set()
    segments = risk_segments or _risk_segments(risks)
    for segment in segments:
        start = int(segment.get("range_start") or 0)
        end = int(segment.get("range_end", start) or start)
        mid = (start + end) // 2
        for frame in (start, mid, end):
            frames.add(max(0, int(frame)))
        source_start = segment.get("source_frame_start")
        source_end = segment.get("source_frame_end", source_start)
        if source_start is not None and source_end is not None:
            source_start = int(source_start)
            source_end = int(source_end)
            for frame in (source_start, (source_start + source_end) // 2, source_end):
                source_frames.add(max(0, int(frame)))
    local = sorted(frames)[:max_frames]
    return {
        "local_frame_indices": local,
        "source_frame_indices": sorted(source_frames)[:max_frames],
        "truncated": len(frames) > max_frames,
    }


def _verdict(summary: dict[str, Any]) -> str:
    if int(summary.get("fail_count") or 0) > 0:
        return "fail"
    if int(summary.get("review_count") or 0) > 0:
        return "review_required"
    return "pass"


def _next_action(verdict: str) -> str:
    if verdict == "pass":
        return "risk scan found no blocking visual-risk signals; proceed to human acceptance on exported evidence"
    if verdict == "review_required":
        return "review the reported frames before claiming visual acceptance"
    return "fix visual-risk findings and regenerate a new independent acceptance round"


def _severity_rank(value: str) -> int:
    return {"fail": 0, "review": 1, "info": 2}.get(str(value), 3)


def _write_risk_csv(path: Path, risks: list[dict[str, Any]]) -> None:
    fields = [
        "risk_type",
        "severity",
        "range_start",
        "range_end",
        "source_frame_start",
        "source_frame_end",
        "track_id",
        "label",
        "risk_count",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(risks)


def _report_markdown(report: dict[str, Any]) -> str:
    video = report.get("video") or {}
    overlay = report.get("overlay") or {}
    summary = report.get("risk_summary") or {}
    lines = [
        "# Visual Risk Scan Report",
        "",
        f"- Verdict: `{report.get('verdict')}`",
        f"- Video: `{report.get('video_path')}`",
        f"- Overlay: `{report.get('overlay_json')}`",
        f"- Video frames: `{video.get('frames_read')}` / `{video.get('frame_count')}`",
        f"- Video size: `{video.get('width')}x{video.get('height')}`",
        f"- Overlay records: `{overlay.get('record_count')}`",
        f"- Risk count: `{summary.get('risk_count')}`",
        f"- Fail count: `{summary.get('fail_count')}`",
        f"- Review count: `{summary.get('review_count')}`",
        "",
        "## Risk Types",
        "",
    ]
    for key, value in (summary.get("by_type") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Key Risks", ""])
    for risk in (report.get("risk_segments") or [])[:80]:
        frame = f"{risk.get('range_start')}-{risk.get('range_end')}"
        lines.append(
            "- "
            f"`{risk.get('severity')}` `{risk.get('risk_type')}` "
            f"local `{frame}` source `{risk.get('source_frame_start')}-{risk.get('source_frame_end')}` "
            f"track `{risk.get('track_id', '')}` label `{risk.get('label', '')}`: "
            f"{'; '.join(risk.get('messages') or [])}"
        )
    lines.extend(
        [
            "",
            "## Review Frames",
            "",
            f"- Local frames: `{', '.join(str(x) for x in (report.get('review_frames') or {}).get('local_frame_indices', []))}`",
            f"- Source frames: `{', '.join(str(x) for x in (report.get('review_frames') or {}).get('source_frame_indices', []))}`",
            "",
            "## Next Action",
            "",
            str(report.get("next_action") or ""),
        ]
    )
    return "\n".join(lines) + "\n"


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan a rendered PPE result video for visual acceptance risks.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--export-risk-frames", action="store_true")
    parser.add_argument("--risk-frame-padding", type=int, default=2)
    parser.add_argument("--no-video-color-samples", action="store_true")
    parser.add_argument("--held-fail-frames", type=int, default=VisualRiskThresholds.held_fail_frames)
    parser.add_argument("--box-growth-ratio", type=float, default=VisualRiskThresholds.box_growth_ratio)
    args = parser.parse_args(argv)
    thresholds = VisualRiskThresholds(
        held_fail_frames=max(1, int(args.held_fail_frames)),
        box_growth_ratio=max(1.01, float(args.box_growth_ratio)),
    )
    report = build_visual_risk_report(
        video_path=args.video,
        overlay_json=args.overlay,
        thresholds=thresholds,
        sample_video_colors=not args.no_video_color_samples,
    )
    written = write_visual_risk_report(
        report,
        output_dir=args.output_dir,
        export_risk_frames=bool(args.export_risk_frames),
        risk_frame_padding=max(0, int(args.risk_frame_padding)),
    )
    print(
        json.dumps(
            {
                "verdict": report.get("verdict"),
                "risk_summary": report.get("risk_summary"),
                "review_frames": report.get("review_frames"),
                "next_action": report.get("next_action"),
                "written": written,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
