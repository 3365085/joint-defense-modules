from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "model" / "src"
for _path in (SRC_ROOT, PROJECT_ROOT / "model"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from defense.module_a.backends import create_detector_backend
from defense.runtime.config import load_runtime_config
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState


def main():
    video_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else video_path.parent / f"a3b_debug_{video_path.name}"

    config = load_runtime_config()
    backend = create_detector_backend(config, PROJECT_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup()

    a3b_state = A3BSoftTriggerState(config.get("a3b", {}) if isinstance(config, dict) else {})

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0
    # Track first alert frame for latency calculation
    first_alert_frame = None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        _, _, info = pipeline.process_frame(frame)
        details = info.get("details", {})
        features = details.get("module_a_features", {})
        reason_codes = info.get("reason_codes", [])

        # Build static_image dict like video_rows.py does
        static_image = dict(features.get("static_media") or features.get("static_image", {}))
        static_image["source_path"] = str(video_path)
        a3b_result = a3b_state.update(static_image)
        observed_score = float(a3b_result.get("observed_score", 0.0))
        confirmed_score = float(a3b_result.get("confirmed_score", 0.0))
        display_score = float(a3b_result.get("display_score", 0.0))
        a3b_triggered = bool(a3b_result.get("triggered", False))
        a3b_source = str(a3b_result.get("triggered_source", "none"))

        p_adv = info.get("p_adv", 0.0) or 0.0
        alert = info.get("alert_confirmed", False)
        suspicious = info.get("is_attack", False)  # to_info_dict uses is_attack

        if alert and first_alert_frame is None:
            first_alert_frame = frame_idx

        # HUD overlay
        overlay = frame.copy()
        y = 40
        line_h = 34
        def put(text, val, color=(255, 255, 255)):
            nonlocal y
            display = f"{text}: {val}"
            cv2.putText(overlay, display, (16, y), font, 0.6, (0, 0, 0), 4)
            cv2.putText(overlay, display, (16, y), font, 0.6, color, 2)
            y += line_h

        put("Frame", f"{frame_idx}/{total - 1}")

        alert_color = (0, 255, 0)
        if alert:
            alert_color = (0, 0, 255)
        elif suspicious:
            alert_color = (0, 128, 255)

        put("p_adv", f"{p_adv:.4f}", alert_color)
        # Show both raw observed and confirmed
        put("Obs score", f"{observed_score:.4f}", (255, 255, 100))
        put("Conf score", f"{confirmed_score:.4f}", alert_color)
        put("Src", a3b_source, alert_color)
        put("Alert", "CONFIRMED" if alert else ("SUSPICIOUS" if suspicious else "none"), alert_color)

        # Latency info at top-right
        if first_alert_frame is not None:
            latency_s = (first_alert_frame - 0) / fps
            cv2.putText(overlay, f"First alert: frame {first_alert_frame} = {latency_s:.2f}s",
                        (w - 420, 36), font, 0.6, (0, 0, 200), 2)

        # Reason codes
        displayed = set()
        for code in reason_codes:
            if code not in displayed:
                displayed.add(code)
                put("", code, (200, 200, 255))

        # Progress bar
        bar_w = int(w * 0.9)
        bar_x = int(w * 0.05)
        progress = min(1.0, (frame_idx + 1) / max(1, total))
        cv2.rectangle(overlay, (bar_x, 8), (bar_x + bar_w, 18), (60, 60, 60), -1)
        color_bar = (0, 0, 255) if alert else ((0, 128, 255) if suspicious else (0, 180, 0))
        cv2.rectangle(overlay, (bar_x, 8), (bar_x + int(bar_w * progress), 18), color_bar, -1)

        # Detector boxes
        detections = details.get("detections", {})
        boxes = detections.get("boxes", [])
        classes = detections.get("normalized_classes", [])
        for box, label in zip(boxes, classes):
            if len(box) >= 4:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                sx, sy = w / 640.0, h / 640.0
                x1, y1 = int(x1 * sx), int(y1 * sy)
                x2, y2 = int(x2 * sx), int(y2 * sy)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, str(label), (x1, y1 - 6), font, 0.5, (0, 255, 0), 2)

        writer.write(overlay)
        frame_idx += 1

    cap.release()
    writer.release()
    pipeline.close()

    print(f"Done: {frame_idx} frames -> {output_path}")
    if first_alert_frame is not None:
        print(f"First alert at frame {first_alert_frame} = {first_alert_frame / fps:.2f}s (30fps)")
    else:
        print("No alert triggered")


if __name__ == "__main__":
    main()
