from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from defense.runtime import PipelineCache
from defense.runtime.frame_processor import FrameProcessor
from defense.runtime.overlay_records import build_overlay_record


def export_ppe_overlay_records(
    *,
    video: str | Path,
    output_json: str | Path,
    profile: str = "default",
    start_frame: int = 0,
    end_frame: int | None = None,
    config: str | Path | None = None,
    display_options: dict[str, Any] | None = None,
    custom_model: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    video_path = Path(video)
    output_path = Path(output_json)
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
    options = dict(display_options or {})
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        first_frame = max(0, int(start_frame))
        last_frame = total_frames - 1 if end_frame is None and total_frames > 0 else int(end_frame or first_frame)
        last_frame = max(first_frame, last_frame)
        if total_frames > 0:
            last_frame = min(last_frame, total_frames - 1)

        cache = PipelineCache(config_path=Path(config) if config else None, root=root)
        custom = dict(custom_model or {})
        bundle = cache.get(profile=profile, feature_options={}, custom_model=custom)
        processor = FrameProcessor(bundle)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        records: list[dict[str, Any]] = []
        overlay_seq = 0
        frame_idx = first_frame
        while frame_idx <= last_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            processed = processor.process(
                frame,
                frame_idx=frame_idx,
                source_type="file",
                source=str(video_path),
                profile=profile,
                realtime=False,
                video_time_s=float(frame_idx) / fps,
                source_fps=fps,
                dropped_frames=0,
                display_options=options,
                feature_options={},
                custom_model=custom,
                target_frame_budget_ms=1000.0 / max(1.0, fps),
            )
            overlay_seq += 1
            record = build_overlay_record(
                status=processed.status,
                ppe_tracks=processed.ppe_tracks,
                run_id=1,
                display_options=options,
            )
            record["overlay_seq"] = overlay_seq
            records.append(record)
            frame_idx += 1
    finally:
        cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def _parse_class_names(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _custom_model_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.weights:
        return {}
    custom: dict[str, Any] = {
        "enabled": True,
        "path": str(args.weights),
        "backend": str(args.backend),
        "model_family": str(args.model_family),
    }
    class_names = _parse_class_names(str(args.class_names or ""))
    if class_names:
        custom["class_names"] = class_names
    if args.source_pt_path:
        custom["source_pt_path"] = str(args.source_pt_path)
    if args.imgsz:
        custom["image_size"] = int(args.imgsz)
    if args.conf is not None:
        custom["confidence"] = float(args.conf)
    if args.candidate_conf is not None:
        custom["candidate_confidence"] = float(args.candidate_conf)
    return custom


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export PPE overlay records from the project detection pipeline.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--backend", choices=("auto", "pytorch", "onnx", "tensorrt"), default="auto")
    parser.add_argument("--model-family", choices=("auto", "yolov5", "yolov8", "ultralytics"), default="auto")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--candidate-conf", type=float, default=None)
    parser.add_argument("--class-names", default="helmet,head,person")
    parser.add_argument("--source-pt-path", type=Path, default=None)
    parser.add_argument("--show-person-boxes", action="store_true", default=False)
    args = parser.parse_args(argv)

    records = export_ppe_overlay_records(
        video=args.video,
        output_json=args.output_json,
        profile=args.profile,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        config=args.config,
        display_options={"show_person_boxes": bool(args.show_person_boxes)},
        custom_model=_custom_model_from_args(args),
    )
    summary = {
        "output_json": str(args.output_json),
        "record_count": len(records),
        "start_frame": int(args.start_frame),
        "end_frame": int(args.end_frame if args.end_frame is not None else args.start_frame + len(records) - 1),
        "profile": str(args.profile),
        "custom_model": _custom_model_from_args(args),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
