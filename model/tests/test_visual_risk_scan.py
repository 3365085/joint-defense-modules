from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from defense.diagnostics.visual_risk_scan import (
    VisualRiskThresholds,
    build_visual_risk_report,
    write_visual_risk_report,
)


def test_visual_risk_scan_flags_overlay_instability(tmp_path: Path) -> None:
    video = tmp_path / "result.mp4"
    overlay = tmp_path / "overlay.json"
    _write_test_video(video, frame_count=10)
    overlay.write_text(json.dumps({"records": _records()}), encoding="utf-8")

    report = build_visual_risk_report(
        video_path=video,
        overlay_json=overlay,
        thresholds=VisualRiskThresholds(held_fail_frames=3, box_growth_ratio=1.6),
    )

    risk_types = {risk["risk_type"] for risk in report["risks"]}
    assert report["video"]["frames_read"] == 10
    assert report["overlay"]["record_count"] == 10
    assert report["verdict"] == "fail"
    assert "label_switch" in risk_types
    assert "track_label_instability" in risk_types
    assert "held_track" in risk_types
    assert "box_growth" in risk_types
    assert "count_change" in risk_types
    assert any(frame in report["review_frames"]["local_frame_indices"] for frame in (2, 5, 7))


def test_visual_risk_scan_writes_reports_without_images_by_default(tmp_path: Path) -> None:
    video = tmp_path / "result.mp4"
    overlay = tmp_path / "overlay.json"
    out_dir = tmp_path / "risk"
    _write_test_video(video, frame_count=4)
    overlay.write_text(json.dumps({"records": _records()[:4]}), encoding="utf-8")

    report = build_visual_risk_report(video_path=video, overlay_json=overlay)
    written = write_visual_risk_report(report, output_dir=out_dir)

    assert Path(written["json_path"]).exists()
    assert Path(written["markdown_path"]).exists()
    assert Path(written["csv_path"]).exists()
    assert not (out_dir / "risk_frames").exists()


def test_visual_risk_scan_flags_large_moving_target_without_track(tmp_path: Path) -> None:
    video = tmp_path / "moving_person.mp4"
    overlay = tmp_path / "overlay.json"
    _write_moving_target_video(video, frame_count=12)
    overlay.write_text(json.dumps({"records": _records_without_matching_foreground(frame_count=12)}), encoding="utf-8")

    report = build_visual_risk_report(
        video_path=video,
        overlay_json=overlay,
        thresholds=VisualRiskThresholds(
            motion_scale_width=320,
            motion_min_area_ratio=0.003,
            motion_min_height_ratio=0.18,
            motion_min_aspect=1.35,
        ),
    )

    missing = [risk for risk in report["risks"] if risk["risk_type"] == "missing_visible_target"]
    assert missing
    assert any(int(risk.get("range_start", risk["local_frame_index"])) <= 5 <= int(risk.get("range_end", risk["local_frame_index"])) for risk in missing)
    assert "missing_visible_target" in {segment["risk_type"] for segment in report["risk_segments"]}


def _write_test_video(path: Path, *, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            cv2.putText(frame, str(index), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.rectangle(frame, (240 + index, 60), (260 + index, 90), (0, 150, 255), 1)
            writer.write(frame)
    finally:
        writer.release()


def _write_moving_target_video(path: Path, *, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.full((180, 320, 3), 185, dtype=np.uint8)
            cv2.rectangle(frame, (0, 0), (320, 55), (80, 80, 80), -1)
            cv2.rectangle(frame, (145 + index * 2, 70), (205 + index * 2, 170), (235, 235, 235), -1)
            cv2.circle(frame, (175 + index * 2, 55), 18, (70, 70, 70), -1)
            writer.write(frame)
    finally:
        writer.release()


def _records() -> list[dict]:
    records = []
    for index in range(10):
        tracks = [
            {
                "track_id": 1,
                "label": "helmet" if index < 2 else "head",
                "box": [500, 220, 520, 250],
                "confidence": 0.7,
                "misses": 0,
                "display_box_source": "detected",
                "fresh_detection": True,
            },
            {
                "track_id": 2,
                "label": "head",
                "box": [260, 220, 280 + index * 22, 260 + index * 18],
                "confidence": 0.8,
                "misses": 0,
                "display_box_source": "detected",
                "fresh_detection": True,
            },
        ]
        if index >= 5:
            tracks.append(
                {
                    "track_id": 3,
                    "label": "head",
                    "box": [120, 210, 145, 245],
                    "confidence": 0.4,
                    "misses": index - 4,
                    "display_box_source": "held_static",
                    "fresh_detection": False,
                }
            )
        records.append(
            {
                "frame_idx": 100 + index,
                "detector_frame_shape": [640, 640],
                "runtime_source_frame_shape": [180, 320],
                "overlay_coordinate_space": {
                    "box_space": "detector_frame",
                    "box_space_shape": [640, 640],
                    "source_frame_shape": [180, 320],
                },
                "ppe_tracks": tracks,
            }
        )
    return records


def _records_without_matching_foreground(*, frame_count: int) -> list[dict]:
    return [
        {
            "frame_idx": 200 + index,
            "detector_frame_shape": [640, 640],
            "runtime_source_frame_shape": [180, 320],
            "overlay_coordinate_space": {
                "box_space": "detector_frame",
                "box_space_shape": [640, 640],
                "source_frame_shape": [180, 320],
            },
            "ppe_tracks": [
                {
                    "track_id": 1,
                    "label": "head",
                    "box": [30, 220, 55, 250],
                    "confidence": 0.8,
                    "misses": 0,
                    "display_box_source": "detected",
                    "fresh_detection": True,
                }
            ],
        }
        for index in range(frame_count)
    ]
