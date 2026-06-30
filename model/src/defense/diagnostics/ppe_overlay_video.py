from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from defense.visualization import render_preview

from .ppe_overlay_summary import load_ppe_overlay_records


def scale_ppe_tracks(
    tracks: list[dict[str, Any]],
    *,
    target_shape: tuple[int, ...],
    source_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    target_h, target_w = _shape_tuple(target_shape) or (1, 1)
    source_h, source_w = source_shape
    scale_x = float(target_w) / max(1.0, float(source_w))
    scale_y = float(target_h) / max(1.0, float(source_h))
    scaled: list[dict[str, Any]] = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        item = dict(track)
        box = item.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = [float(value) for value in box]
            item["box"] = [
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            ]
        scaled.append(item)
    return scaled


def render_ppe_overlay_frame(
    frame: Any,
    record: dict[str, Any] | None,
    *,
    display_options: dict[str, Any] | None = None,
) -> Any:
    if record is None:
        return frame
    options = dict(record.get("display_options") or {})
    options.update(display_options or {})
    ppe_tracks = scale_ppe_tracks(
        record.get("ppe_tracks", []) or [],
        target_shape=frame.shape,
        source_shape=_record_box_source_shape(record),
    )
    info = {
        "p_adv": record.get("p_adv"),
        "alert_confirmed": bool(record.get("alert_confirmed")),
        "attack_detected": bool(record.get("attack_detected")),
        "timing_ms": float(record.get("timing_ms") or 0.0),
        "layer_triggered": record.get("a3b_triggered_source") or "backend",
        "reason_codes": [record.get("a3b_reason")] if record.get("a3b_reason") else [],
    }
    ppe = {
        "warning": bool(record.get("ppe_warning")),
        "confirmed": bool(record.get("ppe_confirmed")),
        "event_active": bool(record.get("ppe_event_active")),
        "event_hold_remaining": int(record.get("ppe_event_hold_remaining") or 0),
        "event_last_reason": record.get("ppe_event_last_reason") or "",
        "event_last_confirmed_source": record.get("ppe_event_last_confirmed_source") or "",
        "person_count": int(record.get("ppe_person_count") or 0),
        "raw_person_count": int(record.get("ppe_raw_person_count", record.get("ppe_person_count")) or 0),
        "inferred_person_count": int(record.get("ppe_inferred_person_count", record.get("ppe_person_count")) or 0),
        "person_context_count": int(record.get("ppe_person_context_count", record.get("ppe_person_count")) or 0),
        "weak_person_count": int(record.get("ppe_weak_person_count") or 0),
        "promoted_person_count": int(record.get("ppe_promoted_person_count") or 0),
        "effective_person_count": int(record.get("ppe_effective_person_count", record.get("ppe_person_count")) or 0),
        "helmet_count": int(record.get("ppe_helmet_count") or 0),
        "raw_helmet_count": int(record.get("ppe_raw_helmet_count", record.get("ppe_helmet_count")) or 0),
        "weak_helmet_count": int(record.get("ppe_weak_helmet_count") or 0),
        "promoted_helmet_count": int(record.get("ppe_promoted_helmet_count") or 0),
        "effective_helmet_count": int(record.get("ppe_effective_helmet_count", record.get("ppe_helmet_count")) or 0),
        "head_count": int(record.get("ppe_head_count") or 0),
        "raw_head_count": int(record.get("ppe_raw_head_count", record.get("ppe_head_count")) or 0),
        "weak_head_count": int(record.get("ppe_weak_head_count") or 0),
        "promoted_head_count": int(record.get("ppe_promoted_head_count") or 0),
        "effective_head_count": int(record.get("ppe_effective_head_count", record.get("ppe_head_count")) or 0),
        "missing_helmet_count": int(record.get("ppe_missing_helmet_count") or 0),
        "uncertain": bool(record.get("ppe_uncertain")),
        "reason": record.get("ppe_reason") or "",
    }
    return render_preview(
        frame,
        info=info,
        ppe=ppe,
        ppe_tracks=ppe_tracks,
        display_options=options,
        frame_idx=int(record.get("frame_idx") or 0),
    )


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
        height = max(1, int(value[0]))
        width = max(1, int(value[1]))
    except (TypeError, ValueError):
        return None
    return (height, width)


def render_ppe_overlay_video(
    *,
    source_video: str | Path,
    overlay_json: str | Path,
    output_video: str | Path,
    display_options: dict[str, Any] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    hold_frames: int = 4,
) -> dict[str, Any]:
    records = sorted(
        load_ppe_overlay_records(overlay_json),
        key=lambda item: int(item.get("frame_idx") or 0),
    )
    source_path = Path(source_video)
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source video: {source_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"invalid source video shape: {source_path}")
    first_frame = max(0, int(start_frame or 0))
    last_frame = int(end_frame) if end_frame is not None else max(0, frame_count - 1)
    if frame_count > 0:
        last_frame = min(last_frame, frame_count - 1)
    if last_frame < first_frame:
        cap.release()
        raise RuntimeError("end_frame must be greater than or equal to start_frame")
    if first_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open output video writer: {output_path}")

    record_index = 0
    active_record: dict[str, Any] | None = None
    frames_written = 0
    frames_with_overlay = 0
    current_frame = first_frame
    try:
        while current_frame <= last_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            while record_index < len(records) and int(records[record_index].get("frame_idx") or 0) <= current_frame:
                active_record = records[record_index]
                record_index += 1
            record = None
            if active_record is not None:
                active_frame = int(active_record.get("frame_idx") or 0)
                if 0 <= current_frame - active_frame <= max(0, int(hold_frames)):
                    record = active_record
            rendered = render_ppe_overlay_frame(
                frame,
                record,
                display_options=display_options,
            )
            if record is not None:
                frames_with_overlay += 1
            writer.write(rendered)
            frames_written += 1
            current_frame += 1
    finally:
        writer.release()
        cap.release()

    summary = {
        "source_video": str(source_path),
        "overlay_json": str(Path(overlay_json)),
        "output_video": str(output_path),
        "width": width,
        "height": height,
        "fps": fps,
        "source_frame_count": frame_count,
        "start_frame": first_frame,
        "end_frame": current_frame - 1,
        "frames_written": frames_written,
        "frames_with_overlay": frames_with_overlay,
        "hold_frames": int(hold_frames),
        "overlay_records": len(records),
        "coordinate_policy": "scale_overlay_box_space_to_source_video_canvas",
        "source_shapes_seen": [list(shape) for shape in sorted({_record_box_source_shape(record) for record in records})],
        "display_options": dict(display_options or {}),
    }
    return summary


def write_render_summary(path: str | Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
