from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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


@dataclass(frozen=True, slots=True)
class Cv2VideoPaths:
    source_for_cv2: Path
    output_for_cv2: Path
    temp_dir: Path
    output_needs_move: bool


def build_dedup_reference_video(
    *,
    source_video: str | Path,
    reference_json: str | Path,
    output_dir: str | Path,
    hidden_labels: set[str] | None = None,
    dedup_labels: set[str] | None = None,
    same_label_iou: float = 0.55,
    containment_threshold: float = 0.90,
    line_thickness: int = 3,
) -> dict[str, Any]:
    source_path = Path(source_video).resolve()
    reference_path = Path(reference_json).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"source video not found: {source_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"reference json not found: {reference_path}")

    payload = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"reference json has no frames: {reference_path}")

    first_frame = int(frames[0].get("source_frame_idx") or 0)
    last_frame = int(frames[-1].get("source_frame_idx") or first_frame)
    suffix = f"{first_frame}_{last_frame + 1}"
    output_video = out_dir / f"dedup_reference_result_{suffix}.mp4"
    detections_json = out_dir / f"dedup_reference_detections_{suffix}.json"
    summary_json = out_dir / f"dedup_reference_summary_{suffix}.json"
    report_md = out_dir / f"dedup_reference_report_{suffix}.md"

    hidden_label_set = {str(label) for label in (hidden_labels or set())}
    dedup_label_set = {str(label) for label in (dedup_labels or {"head", "helmet"})}

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

    if first_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    output_frames: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    raw_class_counts: dict[str, int] = {}
    drop_reason_counts: dict[str, int] = {}
    frames_with_detections = 0
    frames_written = 0
    current_frame = first_frame

    try:
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

            raw_detections = [
                dict(item) for item in frame_record.get("detections", []) if isinstance(item, dict)
            ]
            for detection in raw_detections:
                label = str(detection.get("label") or "")
                raw_class_counts[label] = raw_class_counts.get(label, 0) + 1

            deduped, dropped = _dedupe_same_label(
                raw_detections,
                dedup_labels=dedup_label_set,
                same_label_iou=float(same_label_iou),
                containment_threshold=float(containment_threshold),
            )
            if deduped:
                frames_with_detections += 1
            for detection in deduped:
                label = str(detection.get("label") or "")
                class_counts[label] = class_counts.get(label, 0) + 1
            for drop in dropped:
                reason = str(drop.get("reason") or "unknown")
                drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1

            out_record = {
                "source_frame_idx": source_frame_idx,
                "local_frame_index": int(source_frame_idx - first_frame),
                "detections": deduped,
                "class_counts": _counts_for_frame(deduped),
                "raw_class_counts": _counts_for_frame(raw_detections),
                "dropped_detections": dropped,
                "drop_reason_counts": _drop_counts_for_frame(dropped),
                "inference_ms": frame_record.get("inference_ms"),
            }
            output_frames.append(out_record)
            rendered = _draw_frame(
                frame,
                frame_record=out_record,
                detections=deduped,
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
        "deduplication": {
            "dedup_labels": sorted(dedup_label_set),
            "same_label_iou": float(same_label_iou),
            "containment_threshold": float(containment_threshold),
            "mode": "same_label_only",
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


def _dedupe_same_label(
    detections: list[dict[str, Any]],
    *,
    dedup_labels: set[str],
    same_label_iou: float,
    containment_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
        normalized = _normalized_detection(det)
        if normalized is None:
            continue
        label = str(normalized.get("label") or "")
        if label not in dedup_labels:
            kept.append(normalized)
            continue

        duplicate_of: dict[str, Any] | None = None
        reason = ""
        for existing in kept:
            if str(existing.get("label") or "") != label:
                continue
            overlap = _box_overlap(normalized["box"], existing["box"])
            containment = max(overlap["det_coverage"], overlap["target_coverage"])
            if overlap["iou"] >= same_label_iou:
                duplicate_of = existing
                reason = "same_label_iou"
                break
            if containment >= containment_threshold:
                duplicate_of = existing
                reason = "same_label_containment"
                break
        if duplicate_of is None:
            kept.append(normalized)
        else:
            dropped.append(_drop_record(normalized, reason, duplicate_of=duplicate_of))
    return sorted(kept, key=lambda item: float(item.get("confidence") or 0.0), reverse=True), dropped


def _normalized_detection(det: dict[str, Any]) -> dict[str, Any] | None:
    box = det.get("box")
    if not isinstance(box, list | tuple) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    normalized = dict(det)
    normalized["box"] = [x1, y1, x2, y2]
    normalized["area"] = max(0, x2 - x1) * max(0, y2 - y1)
    normalized["center"] = [int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))]
    return normalized


def _drop_record(
    det: dict[str, Any],
    reason: str,
    *,
    duplicate_of: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reason": reason,
        "label": det.get("label"),
        "confidence": det.get("confidence"),
        "box": det.get("box"),
        "center": det.get("center"),
        "duplicate_of": {
            "label": duplicate_of.get("label"),
            "confidence": duplicate_of.get("confidence"),
            "box": duplicate_of.get("box"),
        },
    }


def _box_overlap(box: list[int], target_box: list[int]) -> dict[str, float]:
    x1 = max(float(box[0]), float(target_box[0]))
    y1 = max(float(box[1]), float(target_box[1]))
    x2 = min(float(box[2]), float(target_box[2]))
    y2 = min(float(box[3]), float(target_box[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    det_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    target_area = max(0.0, float(target_box[2] - target_box[0])) * max(
        0.0, float(target_box[3] - target_box[1])
    )
    union = det_area + target_area - intersection
    return {
        "iou": float(intersection / union) if union > 0 else 0.0,
        "det_coverage": float(intersection / det_area) if det_area > 0 else 0.0,
        "target_coverage": float(intersection / target_area) if target_area > 0 else 0.0,
    }


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
    visible_count = 0
    for detection in detections:
        label = str(detection.get("label") or "")
        if label in hidden_labels:
            continue
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        color = DEFAULT_COLORS.get(label, (210, 210, 210))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {float(detection.get('confidence') or 0.0):.2f}"
        _draw_label(rendered, text, (x1, max(0, y1 - 8)), color, font_scale, text_thickness)
        visible_count += 1

    dropped = frame_record.get("dropped_detections") or []
    header = (
        f"Dedup reference | source={frame_record['source_frame_idx']} "
        f"local={frame_record['local_frame_index']} | visible={visible_count} "
        f"dropped={len(dropped)}"
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
    output_name = (
        f"dedup_reference_{hashlib.sha1(str(output_video).encode('utf-8', errors='ignore')).hexdigest()[:12]}.mp4"
    )
    output_alias = temp_dir / output_name
    if output_alias.exists():
        output_alias.unlink()
    return Cv2VideoPaths(
        source_for_cv2=source_alias,
        output_for_cv2=output_alias,
        temp_dir=temp_dir,
        output_needs_move=True,
    )


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
            "# Dedup Reference Video Report",
            "",
            f"- Source video: `{summary['source_video']}`",
            f"- Raw reference JSON: `{summary['reference_json']}`",
            f"- Output video: `{summary['output_video']}`",
            f"- Frames: `{summary['start_frame']}-{summary['end_frame']}`",
            f"- Hidden labels: `{summary['hidden_labels']}`",
            f"- Raw class counts: `{summary['raw_class_counts']}`",
            f"- Dedup class counts: `{summary['class_counts']}`",
            f"- Drop reasons: `{summary['drop_reason_counts']}`",
            f"- Deduplication: `{summary['deduplication']}`",
            "",
        ]
    )


def _parse_labels(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a same-label deduplicated reference video from raw YOLO detections."
    )
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--reference-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hide-labels", default="person")
    parser.add_argument("--dedup-labels", default="head,helmet")
    parser.add_argument("--same-label-iou", type=float, default=0.55)
    parser.add_argument("--containment-threshold", type=float, default=0.90)
    parser.add_argument("--line-thickness", type=int, default=3)
    args = parser.parse_args(argv)

    summary = build_dedup_reference_video(
        source_video=args.source_video,
        reference_json=args.reference_json,
        output_dir=args.output_dir,
        hidden_labels=_parse_labels(args.hide_labels),
        dedup_labels=_parse_labels(args.dedup_labels),
        same_label_iou=float(args.same_label_iou),
        containment_threshold=float(args.containment_threshold),
        line_thickness=max(1, int(args.line_thickness)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
