from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from defense.module_a.backends.detector_backend import UltralyticsDetectorBackend, YoloV5DetectorBackend  # noqa: E402
from defense.module_a.ppe_postprocess import PPEPostprocessConfig  # noqa: E402
from defense.runtime.ppe_business import evaluate_ppe_business  # noqa: E402
from defense.runtime.ppe_state import SafetyHelmetState  # noqa: E402
from tools.module_a_monitor_app import PPEBoxStabilizer, draw_ppe_boxes  # noqa: E402


RAW_COLORS = {
    "helmet": (0, 255, 0),
    "head": (0, 120, 255),
    "no_helmet": (0, 120, 255),
    "person": (255, 220, 0),
}


def draw_raw(frame: Any, detections: Any) -> Any:
    rendered = frame.copy()
    for index, (box, cls_id, confidence) in enumerate(
        zip(detections.boxes, detections.classes, detections.confidences)
    ):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = str(detections.names.get(int(cls_id), f"class_{int(cls_id)}"))
        color = RAW_COLORS.get(label.lower(), (180, 180, 180))
        text = f"raw#{index} {label} {float(confidence):.2f}"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            rendered,
            text,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            text,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return rendered


def put_banner(frame: Any, text: str) -> Any:
    rendered = frame.copy()
    h, w = rendered.shape[:2]
    cv2.rectangle(rendered, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(
        rendered,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered


def create_backend(args: argparse.Namespace) -> Any:
    if args.family == "yolov5":
        return YoloV5DetectorBackend(
            args.model,
            backend=args.backend,
            device=args.device,
            half=not args.no_half,
            confidence=args.confidence,
            candidate_confidence=args.confidence,
            image_size=args.image_size,
        )
    return UltralyticsDetectorBackend(
        args.model,
        backend=args.backend,
        device=args.device,
        half=not args.no_half,
        confidence=args.confidence,
        candidate_confidence=args.confidence,
        image_size=args.image_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render raw/stabilized PPE debug videos.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--family", choices=["ultralytics", "yolov5"], default="ultralytics")
    parser.add_argument("--backend", choices=["tensorrt", "onnx", "pytorch"], default="tensorrt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--business-confidence", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-render-misses", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if source_fps < 1 or source_fps > 120:
        source_fps = 25.0

    backend = create_backend(args)
    business_confidence = (
        float(args.business_confidence)
        if args.business_confidence is not None
        else float(args.confidence)
    )
    postprocess_config = PPEPostprocessConfig(
        min_confidence=business_confidence,
        candidate_min_confidence=(
            float(args.confidence)
            if float(args.confidence) < business_confidence
            else None
        ),
    )
    ppe_state = SafetyHelmetState()
    stabilizer = PPEBoxStabilizer()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_path = args.out_dir / "raw_yolo_debug.mp4"
    stable_path = args.out_dir / "stable_ppe_debug.mp4"
    side_path = args.out_dir / "side_by_side_debug.mp4"
    csv_path = args.out_dir / "frame_detections.csv"
    raw_writer = cv2.VideoWriter(str(raw_path), fourcc, source_fps, (640, 640))
    stable_writer = cv2.VideoWriter(str(stable_path), fourcc, source_fps, (640, 640))
    side_writer = cv2.VideoWriter(str(side_path), fourcc, source_fps, (1280, 640))
    if not raw_writer.isOpened() or not stable_writer.isOpened() or not side_writer.isOpened():
        raise RuntimeError("Cannot open one or more video writers")

    started = time.perf_counter()
    frame_idx = 0
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "frame_idx",
                "raw_count",
                "raw_labels",
                "stable_count",
                "stable_labels",
                "person_count",
                "helmet_count",
                "raw_helmet_count",
                "head_count",
                "candidate",
                "warning",
                "confirmed",
                "confirmed_source",
                "event_active",
                "event_hold_remaining",
                "event_last_reason",
                "event_last_confirmed_source",
                "evidence_mode",
                "reason",
                "detections",
                "stable_tracks",
            ],
        )
        writer.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                frame_idx -= 1
                break
            frame_640 = cv2.resize(frame, (640, 640))
            detections = backend.predict(frame_640)
            result = evaluate_ppe_business(
                detections,
                frame_shape=frame_640.shape[:2],
                ppe_state=ppe_state,
                ppe_tracker=stabilizer,
                tracking_enabled=True,
                max_render_misses=args.max_render_misses,
                postprocess_config=postprocess_config,
            )
            ppe = result.ppe
            tracks = result.tracks
            raw_frame = put_banner(draw_raw(frame_640, detections), f"RAW YOLO frame={frame_idx}")
            stable_frame = put_banner(draw_ppe_boxes(frame_640, tracks), f"STABLE PPE frame={frame_idx}")
            raw_writer.write(raw_frame)
            stable_writer.write(stable_frame)
            side_writer.write(cv2.hconcat([raw_frame, stable_frame]))

            raw_rows = []
            for index, (box, cls_id, confidence) in enumerate(
                zip(detections.boxes, detections.classes, detections.confidences)
            ):
                raw_rows.append(
                    {
                        "i": index,
                        "label": detections.names.get(int(cls_id), f"class_{int(cls_id)}"),
                        "conf": round(float(confidence), 4),
                        "box": [int(v) for v in box],
                    }
                )
            writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "raw_count": len(raw_rows),
                    "raw_labels": "|".join(str(row["label"]) for row in raw_rows),
                    "stable_count": len(tracks),
                    "stable_labels": "|".join(str(track["label"]) for track in tracks),
                    "person_count": ppe.get("person_count", 0),
                    "helmet_count": ppe.get("helmet_count", 0),
                    "raw_helmet_count": ppe.get("raw_helmet_count", 0),
                    "head_count": ppe.get("head_count", 0),
                    "candidate": ppe.get("candidate", False),
                    "warning": ppe.get("warning", False),
                    "confirmed": ppe.get("confirmed", False),
                    "confirmed_source": ppe.get("confirmed_source", ""),
                    "event_active": ppe.get("event_active", False),
                    "event_hold_remaining": ppe.get("event_hold_remaining", 0),
                    "event_last_reason": ppe.get("event_last_reason", ""),
                    "event_last_confirmed_source": ppe.get("event_last_confirmed_source", ""),
                    "evidence_mode": ppe.get("evidence_mode", ""),
                    "reason": ppe.get("reason", ""),
                    "detections": raw_rows,
                    "stable_tracks": tracks,
                }
            )
            if frame_idx % 50 == 0:
                elapsed = max(0.001, time.perf_counter() - started)
                print(f"processed frame={frame_idx} speed={frame_idx / elapsed:.2f} fps")

    cap.release()
    raw_writer.release()
    stable_writer.release()
    side_writer.release()
    print(f"frames={frame_idx}")
    print(f"raw={raw_path}")
    print(f"stable={stable_path}")
    print(f"side_by_side={side_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
