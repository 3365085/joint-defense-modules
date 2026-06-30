from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_COLORS = {
    "helmet": (70, 220, 90),
    "head": (0, 170, 255),
    "person": (255, 210, 70),
}
PPE_CONTEXT_LABELS = {"head", "helmet"}


@dataclass(frozen=True, slots=True)
class Cv2VideoPaths:
    source_for_cv2: Path
    output_for_cv2: Path
    temp_dir: Path
    output_needs_move: bool


def build_constrained_reference_video(
    *,
    source_video: str | Path,
    reference_json: str | Path,
    output_dir: str | Path,
    hidden_labels: set[str] | None = None,
    line_thickness: int = 3,
    same_label_iou: float = 0.45,
    cross_label_iou: float = 0.35,
    containment_threshold: float = 0.82,
    person_expand_x: float = 0.08,
    person_expand_top: float = 0.18,
    person_expand_bottom: float = 0.08,
    head_zone_expand_x: float = 0.10,
    head_zone_top_padding: float = 0.08,
    head_zone_bottom_ratio: float = 0.52,
    max_ppe_width_person_ratio: float = 0.85,
    max_ppe_height_person_ratio: float = 0.55,
    max_ppe_area_person_ratio: float = 0.26,
) -> dict[str, Any]:
    source_path = Path(source_video).resolve()
    reference_path = Path(reference_json).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(f"source video not found: {source_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"reference json not found: {reference_path}")

    payload = _load_reference_json(reference_path)
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"reference json has no frames: {reference_path}")
    first_frame = int(frames[0].get("source_frame_idx") or 0)
    last_frame = int(frames[-1].get("source_frame_idx") or first_frame)
    suffix = f"{first_frame}_{last_frame + 1}"

    output_video = out_dir / f"constrained_reference_result_{suffix}.mp4"
    detections_json = out_dir / f"constrained_reference_detections_{suffix}.json"
    summary_json = out_dir / f"constrained_reference_summary_{suffix}.json"
    report_md = out_dir / f"constrained_reference_report_{suffix}.md"
    hidden_label_set = {str(label) for label in (hidden_labels or set())}

    cv2_paths = _prepare_cv2_video_paths(source_path, output_video)
    cap = cv2.VideoCapture(str(cv2_paths.source_for_cv2))
    if not cap.isOpened():
        _cleanup_temp_paths(cv2_paths)
        raise RuntimeError(f"failed to open source video with OpenCV: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        _cleanup_temp_paths(cv2_paths)
        raise RuntimeError(f"invalid source video shape: {source_path}")

    writer = cv2.VideoWriter(
        str(cv2_paths.output_for_cv2),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        _cleanup_temp_paths(cv2_paths)
        raise RuntimeError(f"failed to open output video writer: {cv2_paths.output_for_cv2}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    output_frames: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    raw_class_counts: dict[str, int] = {}
    drop_reason_counts: dict[str, int] = {}
    frames_written = 0
    frames_with_detections = 0

    try:
        current_frame = first_frame
        for frame_record in frames:
            source_frame_idx = int(frame_record.get("source_frame_idx") or current_frame)
            while current_frame < source_frame_idx:
                ok, _ = cap.read()
                if not ok:
                    break
                current_frame += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            current_frame = source_frame_idx + 1

            raw_detections = [dict(item) for item in frame_record.get("detections", []) if isinstance(item, dict)]
            for detection in raw_detections:
                label = str(detection.get("label") or "")
                raw_class_counts[label] = raw_class_counts.get(label, 0) + 1
            constrained, dropped = _constrain_detections(
                raw_detections,
                frame_shape=(height, width),
                same_label_iou=float(same_label_iou),
                cross_label_iou=float(cross_label_iou),
                containment_threshold=float(containment_threshold),
                person_expand_x=float(person_expand_x),
                person_expand_top=float(person_expand_top),
                person_expand_bottom=float(person_expand_bottom),
                head_zone_expand_x=float(head_zone_expand_x),
                head_zone_top_padding=float(head_zone_top_padding),
                head_zone_bottom_ratio=float(head_zone_bottom_ratio),
                max_ppe_width_person_ratio=float(max_ppe_width_person_ratio),
                max_ppe_height_person_ratio=float(max_ppe_height_person_ratio),
                max_ppe_area_person_ratio=float(max_ppe_area_person_ratio),
            )
            if constrained:
                frames_with_detections += 1
            for detection in constrained:
                label = str(detection.get("label") or "")
                class_counts[label] = class_counts.get(label, 0) + 1
            for drop in dropped:
                reason = str(drop.get("reason") or "unknown")
                drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1

            out_record = {
                "source_frame_idx": source_frame_idx,
                "local_frame_index": int(source_frame_idx - first_frame),
                "detections": constrained,
                "class_counts": _counts_for_frame(constrained),
                "raw_class_counts": _counts_for_frame(raw_detections),
                "dropped_detections": dropped,
                "drop_reason_counts": _drop_counts_for_frame(dropped),
            }
            output_frames.append(out_record)
            rendered = _draw_frame(
                frame,
                frame_record=out_record,
                detections=constrained,
                hidden_labels=hidden_label_set,
                line_thickness=line_thickness,
            )
            writer.write(rendered)
            frames_written += 1
    finally:
        writer.release()
        cap.release()

    if cv2_paths.output_needs_move:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        if output_video.exists():
            output_video.unlink()
        shutil.move(str(cv2_paths.output_for_cv2), str(output_video))
    _cleanup_temp_paths(cv2_paths)

    summary = {
        "source_video": str(source_path),
        "reference_json": str(reference_path),
        "output_video": str(output_video),
        "detections_json": str(detections_json),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "source_frame_count": int(source_frame_count),
        "start_frame": int(first_frame),
        "end_frame": int(first_frame + frames_written - 1),
        "frames_written": int(frames_written),
        "frames_with_detections": int(frames_with_detections),
        "hidden_labels": sorted(hidden_label_set),
        "raw_class_counts": dict(sorted(raw_class_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "drop_reason_counts": dict(sorted(drop_reason_counts.items())),
        "constraints": {
            "person_filter": {
                "labels": sorted(PPE_CONTEXT_LABELS),
                "person_expand_x": float(person_expand_x),
                "person_expand_top": float(person_expand_top),
                "person_expand_bottom": float(person_expand_bottom),
                "head_zone_expand_x": float(head_zone_expand_x),
                "head_zone_top_padding": float(head_zone_top_padding),
                "head_zone_bottom_ratio": float(head_zone_bottom_ratio),
                "max_ppe_width_person_ratio": float(max_ppe_width_person_ratio),
                "max_ppe_height_person_ratio": float(max_ppe_height_person_ratio),
                "max_ppe_area_person_ratio": float(max_ppe_area_person_ratio),
            },
            "duplicate_suppression": {
                "same_label_iou": float(same_label_iou),
                "cross_label_iou": float(cross_label_iou),
                "containment_threshold": float(containment_threshold),
            },
        },
        "source_summary": payload.get("summary") if isinstance(payload, dict) else None,
    }
    detections_json.write_text(
        json.dumps({"summary": summary, "frames": output_frames}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_md.write_text(_report(summary), encoding="utf-8")
    return summary


def _load_reference_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", text)
        return json.loads(fixed)


def _constrain_detections(
    detections: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int],
    same_label_iou: float,
    cross_label_iou: float,
    containment_threshold: float,
    person_expand_x: float,
    person_expand_top: float,
    person_expand_bottom: float,
    head_zone_expand_x: float,
    head_zone_top_padding: float,
    head_zone_bottom_ratio: float,
    max_ppe_width_person_ratio: float,
    max_ppe_height_person_ratio: float,
    max_ppe_area_person_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    persons = [
        _normalized_detection(det)
        for det in detections
        if str(det.get("label") or "") == "person" and _valid_box(det.get("box"))
    ]
    persons = [det for det in persons if det is not None]
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for det in detections:
        normalized = _normalized_detection(det)
        if normalized is None:
            dropped.append(_drop_record(det, "invalid_box"))
            continue
        label = str(normalized.get("label") or "")
        if label in PPE_CONTEXT_LABELS:
            match = _matching_person(
                normalized,
                persons,
                frame_shape=frame_shape,
                expand_x=person_expand_x,
                expand_top=person_expand_top,
                expand_bottom=person_expand_bottom,
            )
            if match is None:
                dropped.append(_drop_record(normalized, "no_person_context"))
                continue
            head_region_reason = _ppe_head_region_drop_reason(
                normalized,
                match,
                frame_shape=frame_shape,
                expand_x=head_zone_expand_x,
                top_padding=head_zone_top_padding,
                bottom_ratio=head_zone_bottom_ratio,
                max_width_ratio=max_ppe_width_person_ratio,
                max_height_ratio=max_ppe_height_person_ratio,
                max_area_ratio=max_ppe_area_person_ratio,
            )
            if head_region_reason is not None:
                dropped.append(_drop_record(normalized, head_region_reason, duplicate_of=match))
                continue
            normalized["person_context"] = {
                "box": match["box"],
                "confidence": match.get("confidence"),
                "relation": match.get("_relation"),
            }
        kept.append(normalized)

    deduped: list[dict[str, Any]] = []
    for det in sorted(kept, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
        duplicate_of: dict[str, Any] | None = None
        for existing in deduped:
            reason = _duplicate_reason(
                det,
                existing,
                same_label_iou=same_label_iou,
                cross_label_iou=cross_label_iou,
                containment_threshold=containment_threshold,
            )
            if reason:
                duplicate_of = existing
                dropped.append(_drop_record(det, reason, duplicate_of=existing))
                break
        if duplicate_of is None:
            deduped.append(det)
    return sorted(deduped, key=lambda item: float(item.get("confidence") or 0.0), reverse=True), dropped


def _normalized_detection(det: dict[str, Any]) -> dict[str, Any] | None:
    box = det.get("box")
    if not _valid_box(box):
        return None
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    item = dict(det)
    item["box"] = [x1, y1, x2, y2]
    item["area"] = max(0, x2 - x1) * max(0, y2 - y1)
    item["center"] = [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))]
    item["confidence"] = float(item.get("confidence") or 0.0)
    return item


def _valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _ppe_head_region_drop_reason(
    det: dict[str, Any],
    person: dict[str, Any],
    *,
    frame_shape: tuple[int, int],
    expand_x: float,
    top_padding: float,
    bottom_ratio: float,
    max_width_ratio: float,
    max_height_ratio: float,
    max_area_ratio: float,
) -> str | None:
    height, width = frame_shape
    px1, py1, px2, py2 = [float(value) for value in person["box"]]
    person_w = max(1.0, px2 - px1)
    person_h = max(1.0, py2 - py1)
    head_zone = [
        max(0.0, px1 - person_w * expand_x),
        max(0.0, py1 - person_h * top_padding),
        min(float(width - 1), px2 + person_w * expand_x),
        min(float(height - 1), py1 + person_h * bottom_ratio),
    ]
    if not _center_inside(det["box"], [int(round(value)) for value in head_zone]):
        return "outside_person_head_zone"

    dx1, dy1, dx2, dy2 = [float(value) for value in det["box"]]
    det_w = max(0.0, dx2 - dx1)
    det_h = max(0.0, dy2 - dy1)
    det_area = det_w * det_h
    person_area = person_w * person_h
    if det_w > person_w * max_width_ratio:
        return "ppe_too_wide_for_person"
    if det_h > person_h * max_height_ratio:
        return "ppe_too_tall_for_person"
    if det_area > person_area * max_area_ratio:
        return "ppe_too_large_for_person"
    return None


def _matching_person(
    det: dict[str, Any],
    persons: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int],
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    for person in persons:
        expanded = _expand_person_box(
            person["box"],
            frame_shape=frame_shape,
            expand_x=expand_x,
            expand_top=expand_top,
            expand_bottom=expand_bottom,
        )
        overlap = _box_overlap(det["box"], expanded)
        center_inside = _center_inside(det["box"], expanded)
        hit = center_inside or overlap["det_coverage"] >= 0.12 or overlap["iou"] >= 0.01
        if not hit:
            continue
        score = float(overlap["det_coverage"]) + (0.5 if center_inside else 0.0)
        if score > best_score:
            best = dict(person)
            best["_relation"] = {
                "expanded_person_box": expanded,
                "det_coverage": overlap["det_coverage"],
                "iou": overlap["iou"],
                "center_inside_expanded_person": bool(center_inside),
            }
            best_score = score
    return best


def _expand_person_box(
    box: list[int],
    *,
    frame_shape: tuple[int, int],
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
) -> list[int]:
    height, width = frame_shape
    x1, y1, x2, y2 = [float(value) for value in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return [
        int(max(0, round(x1 - bw * expand_x))),
        int(max(0, round(y1 - bh * expand_top))),
        int(min(width - 1, round(x2 + bw * expand_x))),
        int(min(height - 1, round(y2 + bh * expand_bottom))),
    ]


def _duplicate_reason(
    det: dict[str, Any],
    existing: dict[str, Any],
    *,
    same_label_iou: float,
    cross_label_iou: float,
    containment_threshold: float,
) -> str | None:
    label = str(det.get("label") or "")
    existing_label = str(existing.get("label") or "")
    overlap = _box_overlap(det["box"], existing["box"])
    containment = max(overlap["det_coverage"], overlap["target_coverage"])
    if label == existing_label:
        if overlap["iou"] >= same_label_iou or containment >= containment_threshold:
            return "duplicate_same_label"
    elif {label, existing_label} == {"head", "helmet"}:
        if overlap["iou"] >= cross_label_iou or containment >= containment_threshold:
            return "duplicate_head_helmet_overlap"
    return None


def _drop_record(det: dict[str, Any], reason: str, *, duplicate_of: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "reason": reason,
        "label": det.get("label"),
        "confidence": det.get("confidence"),
        "box": det.get("box"),
        "center": det.get("center"),
    }
    if duplicate_of is not None:
        record["duplicate_of"] = {
            "label": duplicate_of.get("label"),
            "confidence": duplicate_of.get("confidence"),
            "box": duplicate_of.get("box"),
        }
    return record


def _box_overlap(box: list[int], target_box: list[int]) -> dict[str, float]:
    x1 = max(float(box[0]), float(target_box[0]))
    y1 = max(float(box[1]), float(target_box[1]))
    x2 = min(float(box[2]), float(target_box[2]))
    y2 = min(float(box[3]), float(target_box[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    det_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    target_area = max(0.0, float(target_box[2] - target_box[0])) * max(0.0, float(target_box[3] - target_box[1]))
    union = det_area + target_area - intersection
    return {
        "iou": float(intersection / union) if union > 0 else 0.0,
        "det_coverage": float(intersection / det_area) if det_area > 0 else 0.0,
        "target_coverage": float(intersection / target_area) if target_area > 0 else 0.0,
    }


def _center_inside(box: list[int], target_box: list[int]) -> bool:
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return float(target_box[0]) <= cx <= float(target_box[2]) and float(target_box[1]) <= cy <= float(target_box[3])


def _counts_for_frame(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detection in detections:
        label = str(detection.get("label") or "")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _drop_counts_for_frame(dropped: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for drop in dropped:
        reason = str(drop.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _draw_frame(
    frame: np.ndarray,
    *,
    frame_record: dict[str, Any],
    detections: list[dict[str, Any]],
    hidden_labels: set[str],
    line_thickness: int,
) -> np.ndarray:
    rendered = frame.copy()
    thickness = max(1, int(line_thickness))
    font_scale = max(0.7, min(1.6, rendered.shape[1] / 2400.0))
    text_thickness = max(2, int(round(thickness * 0.75)))
    for detection in detections:
        label = str(detection.get("label") or "")
        if label in hidden_labels:
            continue
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        color = DEFAULT_COLORS.get(label, (210, 210, 210))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {float(detection.get('confidence') or 0.0):.2f}"
        _draw_label(rendered, text, (x1, max(0, y1 - 8)), color, font_scale, text_thickness)

    header = (
        f"Constrained reference | source={frame_record['source_frame_idx']} "
        f"local={frame_record['local_frame_index']} | kept={len(detections)} "
        f"dropped={len(frame_record.get('dropped_detections') or [])}"
    )
    if hidden_labels:
        header = f"{header} | hidden={','.join(sorted(hidden_labels))}"
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 58), (20, 20, 20), -1)
    cv2.putText(
        rendered,
        header,
        (18, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (245, 245, 245),
        text_thickness,
        cv2.LINE_AA,
    )
    return rendered


def _draw_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    x, y = anchor
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y = max(th + baseline + 3, y)
    cv2.rectangle(image, (x, y - th - baseline - 5), (x + tw + 8, y + 4), color, -1)
    cv2.putText(
        image,
        text,
        (x + 4, y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (15, 15, 15),
        thickness,
        cv2.LINE_AA,
    )


def _prepare_cv2_video_paths(source_video: Path, output_video: Path) -> Cv2VideoPaths:
    temp_dir = _cv2_temp_dir(source_video)
    temp_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(source_video).encode("utf-8", errors="ignore")).hexdigest()[:12]
    source_alias = temp_dir / f"source_{digest}{source_video.suffix.lower() or '.mp4'}"
    if not source_alias.exists() or source_alias.stat().st_size != source_video.stat().st_size:
        if source_alias.exists():
            source_alias.unlink()
        try:
            os.link(source_video, source_alias)
        except OSError:
            shutil.copy2(source_video, source_alias)
    output_name = f"constrained_reference_{hashlib.sha1(str(output_video).encode('utf-8', errors='ignore')).hexdigest()[:12]}.mp4"
    output_alias = temp_dir / output_name
    if output_alias.exists():
        output_alias.unlink()
    return Cv2VideoPaths(source_for_cv2=source_alias, output_for_cv2=output_alias, temp_dir=temp_dir, output_needs_move=True)


def _cv2_temp_dir(source_video: Path) -> Path:
    drive = source_video.drive or "D:"
    return Path(f"{drive}\\codex_handoff\\joint_defense_cv2_aliases")


def _cleanup_temp_paths(paths: Cv2VideoPaths) -> None:
    for path in (paths.output_for_cv2, paths.source_for_cv2):
        try:
            if path.exists() and path.parent == paths.temp_dir:
                path.unlink()
        except OSError:
            pass


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Constrained Reference Video Report",
            "",
            f"- Source video: `{summary['source_video']}`",
            f"- Raw reference JSON: `{summary['reference_json']}`",
            f"- Output video: `{summary['output_video']}`",
            f"- Frames: `{summary['start_frame']}-{summary['end_frame']}`",
            f"- Hidden labels: `{summary['hidden_labels']}`",
            f"- Raw class counts: `{summary['raw_class_counts']}`",
            f"- Constrained class counts: `{summary['class_counts']}`",
            f"- Drop reasons: `{summary['drop_reason_counts']}`",
            "",
        ]
    )


def _parse_labels(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a constrained reference video from raw YOLO reference detections.")
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--reference-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hide-labels", default="person")
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--same-label-iou", type=float, default=0.45)
    parser.add_argument("--cross-label-iou", type=float, default=0.35)
    parser.add_argument("--containment-threshold", type=float, default=0.82)
    parser.add_argument("--person-expand-x", type=float, default=0.08)
    parser.add_argument("--person-expand-top", type=float, default=0.18)
    parser.add_argument("--person-expand-bottom", type=float, default=0.08)
    parser.add_argument("--head-zone-expand-x", type=float, default=0.10)
    parser.add_argument("--head-zone-top-padding", type=float, default=0.08)
    parser.add_argument("--head-zone-bottom-ratio", type=float, default=0.52)
    parser.add_argument("--max-ppe-width-person-ratio", type=float, default=0.85)
    parser.add_argument("--max-ppe-height-person-ratio", type=float, default=0.55)
    parser.add_argument("--max-ppe-area-person-ratio", type=float, default=0.26)
    args = parser.parse_args(argv)
    summary = build_constrained_reference_video(
        source_video=args.source_video,
        reference_json=args.reference_json,
        output_dir=args.output_dir,
        hidden_labels=_parse_labels(args.hide_labels),
        line_thickness=max(1, int(args.line_thickness)),
        same_label_iou=float(args.same_label_iou),
        cross_label_iou=float(args.cross_label_iou),
        containment_threshold=float(args.containment_threshold),
        person_expand_x=float(args.person_expand_x),
        person_expand_top=float(args.person_expand_top),
        person_expand_bottom=float(args.person_expand_bottom),
        head_zone_expand_x=float(args.head_zone_expand_x),
        head_zone_top_padding=float(args.head_zone_top_padding),
        head_zone_bottom_ratio=float(args.head_zone_bottom_ratio),
        max_ppe_width_person_ratio=float(args.max_ppe_width_person_ratio),
        max_ppe_height_person_ratio=float(args.max_ppe_height_person_ratio),
        max_ppe_area_person_ratio=float(args.max_ppe_area_person_ratio),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
