from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.module_a.backends.detector_backend import (
    UltralyticsDetectorBackend,
    YoloV5DetectorBackend,
)


DEFAULT_CLASS_NAMES = ("helmet", "head", "person")
DEFAULT_TARGET_LABELS = ("person", "head", "helmet")
DEFAULT_COLORS = {
    "helmet": (70, 220, 90),
    "head": (0, 170, 255),
    "person": (255, 210, 70),
}


@dataclass(frozen=True, slots=True)
class Cv2VideoPaths:
    source_for_cv2: Path
    output_for_cv2: Path
    final_output: Path
    temp_dir: Path
    source_alias_created: bool
    output_needs_move: bool


def build_yolo_reference_video(
    *,
    source_video: str | Path,
    weights: str | Path,
    output_dir: str | Path,
    start_frame: int,
    end_frame: int | None,
    image_size: int,
    confidence: float,
    device: str,
    half: bool,
    class_names: list[str],
    model_family: str = "auto",
    target_source_frame: int | None = None,
    target_box: list[int] | None = None,
    target_labels: set[str] | None = None,
    target_window: int = 2,
    line_thickness: int = 3,
    hidden_labels: set[str] | None = None,
) -> dict[str, Any]:
    source_path = Path(source_video).resolve()
    weights_path = Path(weights).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"source video not found: {source_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    first_frame = max(0, int(start_frame))
    suffix = f"{first_frame}_{'end' if end_frame is None else int(end_frame)}"
    output_video = out_dir / f"reference_result_{suffix}.mp4"
    detections_json = out_dir / f"reference_detections_{suffix}.json"
    summary_json = out_dir / f"reference_summary_{suffix}.json"
    report_md = out_dir / f"reference_report_{suffix}.md"

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

    last_frame = int(end_frame) if end_frame is not None else max(0, source_frame_count - 1)
    if source_frame_count > 0:
        last_frame = min(last_frame, source_frame_count - 1)
    if last_frame < first_frame:
        cap.release()
        _cleanup_temp_paths(cv2_paths)
        raise RuntimeError("end_frame must be greater than or equal to start_frame")

    if first_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

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

    resolved_model_family = _resolve_model_family(weights_path, model_family)
    backend = _create_detector_backend(
        weights_path=weights_path,
        model_family=resolved_model_family,
        device=device,
        half=half,
        confidence=confidence,
        image_size=image_size,
        class_names=class_names,
    )
    target_label_set = target_labels or set(DEFAULT_TARGET_LABELS)
    hidden_label_set = {str(label) for label in (hidden_labels or set())}
    detections_by_frame: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    inference_ms: list[float] = []
    target_hits: list[dict[str, Any]] = []
    frames_with_detections = 0
    started = time.perf_counter()
    current_frame = first_frame
    frames_written = 0

    try:
        while current_frame <= last_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            result = backend.predict(frame)
            detections = _serialize_detections(result)
            inference_ms.append(float(result.inference_ms))
            if detections:
                frames_with_detections += 1
            for detection in detections:
                label = str(detection["label"])
                class_counts[label] = class_counts.get(label, 0) + 1

            target_analysis = None
            if target_box is not None and target_source_frame is not None:
                in_window = abs(int(current_frame) - int(target_source_frame)) <= max(
                    0, int(target_window)
                )
                target_analysis = _analyze_target(
                    detections,
                    target_box=target_box,
                    target_labels=target_label_set,
                )
                if in_window and target_analysis["hit"]:
                    target_hits.append(
                        {
                            "source_frame_idx": int(current_frame),
                            "local_frame_index": int(current_frame - first_frame),
                            **target_analysis,
                        }
                    )

            frame_record = {
                "source_frame_idx": int(current_frame),
                "local_frame_index": int(current_frame - first_frame),
                "detections": detections,
                "class_counts": _counts_for_frame(detections),
                "inference_ms": float(result.inference_ms),
            }
            if target_analysis is not None:
                frame_record["target_analysis"] = target_analysis
            detections_by_frame.append(frame_record)

            rendered = _draw_reference_frame(
                frame,
                frame_record=frame_record,
                detections=detections,
                image_size=image_size,
                confidence=confidence,
                target_box=target_box,
                target_source_frame=target_source_frame,
                line_thickness=line_thickness,
                hidden_labels=hidden_label_set,
            )
            writer.write(rendered)
            frames_written += 1
            current_frame += 1
    finally:
        writer.release()
        cap.release()
        backend.close()

    if cv2_paths.output_needs_move:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        if output_video.exists():
            output_video.unlink()
        shutil.move(str(cv2_paths.output_for_cv2), str(output_video))

    elapsed_s = time.perf_counter() - started
    target_summary = _target_summary(
        detections_by_frame,
        target_source_frame=target_source_frame,
        target_box=target_box,
        target_window=target_window,
    )
    summary = {
        "source_video": str(source_path),
        "weights": str(weights_path),
        "model_family": resolved_model_family,
        "backend": "ultralytics" if resolved_model_family == "yolov8" else "yolov5_official",
        "device": device,
        "half": bool(half),
        "image_size": int(image_size),
        "confidence": float(confidence),
        "class_names": list(class_names),
        "hidden_labels": sorted(hidden_label_set),
        "output_video": str(output_video),
        "detections_json": str(detections_json),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "source_frame_count": int(source_frame_count),
        "start_frame": int(first_frame),
        "end_frame": int(current_frame - 1),
        "frames_written": int(frames_written),
        "frames_with_detections": int(frames_with_detections),
        "class_counts": dict(sorted(class_counts.items())),
        "avg_inference_ms": _mean(inference_ms),
        "p95_inference_ms": _percentile(inference_ms, 95),
        "elapsed_s": float(elapsed_s),
        "target_summary": target_summary,
        "target_hits_in_window": target_hits,
        "cv2_path_alias": {
            "source_alias_used": str(cv2_paths.source_for_cv2) != str(source_path),
            "source_alias_created": bool(cv2_paths.source_alias_created),
            "temp_dir": str(cv2_paths.temp_dir),
        },
    }

    detections_payload = {
        "summary": summary,
        "frames": detections_by_frame,
    }
    detections_json.write_text(
        json.dumps(detections_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_md.write_text(_reference_report(summary, detections_by_frame), encoding="utf-8")
    _cleanup_temp_paths(cv2_paths)
    return summary


def _serialize_detections(result: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for box, cls_id, conf in zip(result.boxes, result.classes, result.confidences):
        label = result.names.get(int(cls_id), f"class_{int(cls_id)}")
        x1, y1, x2, y2 = [int(v) for v in box]
        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "class_id": int(cls_id),
                "label": str(label),
                "confidence": float(conf),
                "area": max(0, x2 - x1) * max(0, y2 - y1),
                "center": [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))],
            }
        )
    return sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)


def _resolve_model_family(weights_path: Path, model_family: str) -> str:
    normalized = str(model_family or "auto").lower().strip()
    if normalized in {"yolov8", "ultralytics"}:
        return "yolov8"
    if normalized == "yolov5":
        return "yolov5"
    if normalized != "auto":
        raise ValueError(f"unsupported model family: {model_family}")
    hint = str(weights_path).lower().replace("\\", "/")
    if "yolov5" in hint:
        return "yolov5"
    return "yolov8"


def _create_detector_backend(
    *,
    weights_path: Path,
    model_family: str,
    device: str,
    half: bool,
    confidence: float,
    image_size: int,
    class_names: list[str],
) -> Any:
    if model_family == "yolov5":
        return YoloV5DetectorBackend(
            weights_path,
            "pytorch",
            device=device,
            half=half,
            confidence=confidence,
            image_size=image_size,
            class_names=class_names,
        )
    if model_family == "yolov8":
        return UltralyticsDetectorBackend(
            weights_path,
            "pytorch",
            device=device,
            half=half,
            confidence=confidence,
            image_size=image_size,
            class_names=class_names,
        )
    raise ValueError(f"unsupported model family: {model_family}")


def _counts_for_frame(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detection in detections:
        label = str(detection.get("label") or "")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _analyze_target(
    detections: list[dict[str, Any]],
    *,
    target_box: list[int],
    target_labels: set[str],
) -> dict[str, Any]:
    matches = []
    for detection in detections:
        label = str(detection.get("label") or "")
        if label not in target_labels:
            continue
        box = [int(v) for v in detection["box"]]
        overlap = _box_overlap(box, target_box)
        center_inside = _center_inside(box, target_box)
        hit = (
            overlap["iou"] >= 0.03
            or overlap["det_coverage"] >= 0.25
            or overlap["target_coverage"] >= 0.05
            or center_inside
        )
        if hit:
            matches.append(
                {
                    "label": label,
                    "confidence": float(detection.get("confidence") or 0.0),
                    "box": box,
                    "center": detection.get("center"),
                    **overlap,
                    "center_inside_target": bool(center_inside),
                }
            )
    matches = sorted(matches, key=lambda item: float(item["confidence"]), reverse=True)
    return {
        "target_box": target_box,
        "target_labels": sorted(target_labels),
        "hit": bool(matches),
        "match_count": len(matches),
        "matches": matches,
    }


def _target_summary(
    frames: list[dict[str, Any]],
    *,
    target_source_frame: int | None,
    target_box: list[int] | None,
    target_window: int,
) -> dict[str, Any] | None:
    if target_source_frame is None or target_box is None:
        return None
    exact = [frame for frame in frames if _frame_source_index(frame) == target_source_frame]
    window_start = int(target_source_frame) - max(0, int(target_window))
    window_end = int(target_source_frame) + max(0, int(target_window))
    window = [
        frame
        for frame in frames
        if window_start <= _frame_source_index(frame) <= window_end
    ]
    exact_analysis = exact[0].get("target_analysis") if exact else None
    window_hits = [
        frame
        for frame in window
        if isinstance(frame.get("target_analysis"), dict)
        and bool(frame["target_analysis"].get("hit"))
    ]
    return {
        "target_source_frame": int(target_source_frame),
        "target_local_frame": int(exact[0]["local_frame_index"]) if exact else None,
        "target_box": target_box,
        "target_window": int(target_window),
        "exact_frame_seen": bool(exact),
        "exact_frame_hit": bool(exact_analysis and exact_analysis.get("hit")),
        "exact_frame_match_count": int(exact_analysis.get("match_count") or 0)
        if isinstance(exact_analysis, dict)
        else 0,
        "window_frame_count": len(window),
        "window_hit_count": len(window_hits),
        "window_hit_source_frames": [int(frame["source_frame_idx"]) for frame in window_hits],
        "exact_frame_detections": exact[0].get("detections", []) if exact else [],
        "exact_frame_target_analysis": exact_analysis,
    }


def _frame_source_index(frame: dict[str, Any]) -> int:
    try:
        return int(frame.get("source_frame_idx"))
    except (TypeError, ValueError):
        return -1


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


def _center_inside(box: list[int], target_box: list[int]) -> bool:
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return (
        float(target_box[0]) <= cx <= float(target_box[2])
        and float(target_box[1]) <= cy <= float(target_box[3])
    )


def _draw_reference_frame(
    frame: np.ndarray,
    *,
    frame_record: dict[str, Any],
    detections: list[dict[str, Any]],
    image_size: int,
    confidence: float,
    target_box: list[int] | None,
    target_source_frame: int | None,
    line_thickness: int,
    hidden_labels: set[str],
) -> np.ndarray:
    rendered = frame.copy()
    thickness = max(1, int(line_thickness))
    font_scale = max(0.7, min(1.6, rendered.shape[1] / 2400.0))
    text_thickness = max(2, int(round(thickness * 0.75)))
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        label = str(detection["label"])
        if label in hidden_labels:
            continue
        color = DEFAULT_COLORS.get(label, (210, 210, 210))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {float(detection['confidence']):.2f}"
        _draw_label(rendered, text, (x1, max(0, y1 - 8)), color, font_scale, text_thickness)

    if target_box is not None:
        tx1, ty1, tx2, ty2 = [int(v) for v in target_box]
        target_color = (255, 0, 255)
        cv2.rectangle(rendered, (tx1, ty1), (tx2, ty2), target_color, max(2, thickness))
        target_text = "target ROI"
        if target_source_frame is not None:
            target_text = f"target ROI frame {target_source_frame}"
        _draw_label(
            rendered,
            target_text,
            (tx1, max(0, ty1 - 12)),
            target_color,
            font_scale,
            text_thickness,
        )

    header = (
        f"YOLO reference | source={frame_record['source_frame_idx']} "
        f"local={frame_record['local_frame_index']} | imgsz={image_size} "
        f"conf={confidence:.2f} | detections={len(detections)}"
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
    source_alias_created = False
    if not source_alias.exists() or source_alias.stat().st_size != source_video.stat().st_size:
        if source_alias.exists():
            source_alias.unlink()
        try:
            os.link(source_video, source_alias)
        except OSError:
            shutil.copy2(source_video, source_alias)
        source_alias_created = True

    output_name = f"reference_{hashlib.sha1(str(output_video).encode('utf-8', errors='ignore')).hexdigest()[:12]}.mp4"
    output_alias = temp_dir / output_name
    if output_alias.exists():
        output_alias.unlink()
    return Cv2VideoPaths(
        source_for_cv2=source_alias,
        output_for_cv2=output_alias,
        final_output=output_video,
        temp_dir=temp_dir,
        source_alias_created=source_alias_created,
        output_needs_move=True,
    )


def _cv2_temp_dir(source_video: Path) -> Path:
    drive = source_video.drive or "D:"
    return Path(f"{drive}\\codex_handoff\\joint_defense_cv2_aliases")


def _cleanup_temp_paths(paths: Cv2VideoPaths) -> None:
    if paths.output_needs_move:
        _unlink_if_exists(paths.output_for_cv2)
    if paths.source_for_cv2.parent == paths.temp_dir and paths.source_for_cv2.name.startswith("source_"):
        _unlink_if_exists(paths.source_for_cv2)


def _unlink_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    rank = (len(ordered) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _reference_report(summary: dict[str, Any], frames: list[dict[str, Any]]) -> str:
    target = summary.get("target_summary")
    lines = [
        "# YOLO Reference Video Report",
        "",
        f"- Source video: `{summary['source_video']}`",
        f"- Weights: `{summary['weights']}`",
        f"- Output video: `{summary['output_video']}`",
        f"- Frames: `{summary['start_frame']}-{summary['end_frame']}`",
        f"- Size/FPS: `{summary['width']}x{summary['height']} @ {summary['fps']:.3f}`",
        f"- Inference: `{summary['model_family']} {summary['backend']}`, device `{summary['device']}`, imgsz `{summary['image_size']}`, conf `{summary['confidence']}`",
        f"- Hidden labels in video: `{summary.get('hidden_labels', [])}`",
        f"- Frames with detections: `{summary['frames_with_detections']}/{summary['frames_written']}`",
        f"- Class counts: `{summary['class_counts']}`",
        f"- Average inference ms: `{summary.get('avg_inference_ms')}`",
        "",
    ]
    if isinstance(target, dict):
        lines.extend(
            [
                "## Target Frame",
                "",
                f"- Target source frame: `{target.get('target_source_frame')}`",
                f"- Target local frame: `{target.get('target_local_frame')}`",
                f"- Target box: `{target.get('target_box')}`",
                f"- Exact frame hit: `{target.get('exact_frame_hit')}`",
                f"- Window hits: `{target.get('window_hit_count')}` frames `{target.get('window_hit_source_frames')}`",
                "",
                "### Exact Frame Target Analysis",
                "",
                "```json",
                json.dumps(target.get("exact_frame_target_analysis"), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Exact Frame Detections",
                "",
            ]
        )
        exact_detections = target.get("exact_frame_detections") or []
        for detection in exact_detections[:40]:
            lines.append(
                f"- `{detection.get('label')}` conf `{float(detection.get('confidence') or 0.0):.4f}` box `{detection.get('box')}`"
            )
        if not exact_detections:
            lines.append("- No detections on the exact target frame.")
        lines.append("")

    lines.extend(["## Frame Samples", ""])
    for frame in frames[:3] + frames[-3:]:
        lines.append(
            f"- source `{frame.get('source_frame_idx')}` local `{frame.get('local_frame_index')}` counts `{frame.get('class_counts')}`"
        )
    return "\n".join(lines) + "\n"


def _parse_class_names(value: str) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    return parsed or list(DEFAULT_CLASS_NAMES)


def _parse_labels(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def _parse_box(value: str | None) -> list[int] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--target-box must be x1,y1,x2,y2")
    try:
        box = [int(round(float(part))) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--target-box must contain numeric values") from exc
    if box[2] <= box[0] or box[3] <= box[1]:
        raise argparse.ArgumentTypeError("--target-box must satisfy x2>x1 and y2>y1")
    return box


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a high-quality offline YOLO reference detection video."
    )
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("baseline_training/runs/baseline_yolov8_three_put/best.pt"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-family", choices=("auto", "yolov5", "yolov8", "ultralytics"), default="auto")
    parser.add_argument("--half", dest="half", action="store_true", default=True)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--class-names", default=",".join(DEFAULT_CLASS_NAMES))
    parser.add_argument("--target-source-frame", type=int, default=None)
    parser.add_argument("--target-box", type=_parse_box, default=None)
    parser.add_argument("--target-labels", default=",".join(DEFAULT_TARGET_LABELS))
    parser.add_argument("--target-window", type=int, default=2)
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--hide-labels", default="")
    args = parser.parse_args(argv)

    summary = build_yolo_reference_video(
        source_video=args.source_video,
        weights=args.weights,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        image_size=args.imgsz,
        confidence=args.conf,
        device=args.device,
        half=bool(args.half),
        class_names=_parse_class_names(args.class_names),
        model_family=args.model_family,
        target_source_frame=args.target_source_frame,
        target_box=args.target_box,
        target_labels=_parse_labels(args.target_labels),
        target_window=max(0, int(args.target_window)),
        line_thickness=max(1, int(args.line_thickness)),
        hidden_labels=_parse_labels(args.hide_labels),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
