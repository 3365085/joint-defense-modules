from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from defense.diagnostics import yolo_reference_video as ref


class _FakeBackend:
    def __init__(self) -> None:
        self.names = {0: "helmet", 1: "head", 2: "person"}

    def predict(self, image: np.ndarray) -> object:
        return type(
            "Result",
            (),
            {
                "boxes": [[30, 35, 75, 120], [41, 30, 62, 49]],
                "classes": [2, 0],
                "confidences": [0.91, 0.66],
                "names": self.names,
                "inference_ms": 1.5,
            },
        )()

    def close(self) -> None:
        return None


def test_yolo_reference_video_writes_video_and_target_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    weights = tmp_path / "best.pt"
    output_dir = tmp_path / "reference"
    _write_video(source, frame_count=3)
    weights.write_bytes(b"fake")
    monkeypatch.setattr(ref, "_create_detector_backend", lambda **_: _FakeBackend())
    monkeypatch.setattr(ref, "_prepare_cv2_video_paths", _direct_cv2_paths)

    summary = ref.build_yolo_reference_video(
        source_video=source,
        weights=weights,
        output_dir=output_dir,
        start_frame=0,
        end_frame=2,
        image_size=1280,
        confidence=0.05,
        device="cpu",
        half=False,
        class_names=["helmet", "head", "person"],
        model_family="yolov8",
        target_source_frame=1,
        target_box=[20, 20, 90, 140],
        target_window=1,
        hidden_labels={"person"},
    )

    assert Path(summary["output_video"]).exists()
    assert Path(summary["detections_json"]).exists()
    assert Path(summary["summary_json"]).exists()
    assert Path(summary["report_md"]).exists()
    assert summary["model_family"] == "yolov8"
    assert summary["backend"] == "ultralytics"
    assert summary["hidden_labels"] == ["person"]
    assert summary["frames_written"] == 3
    assert summary["frames_with_detections"] == 3
    assert summary["target_summary"]["exact_frame_hit"] is True
    assert summary["target_summary"]["window_hit_count"] == 3

    payload = json.loads(Path(summary["detections_json"]).read_text(encoding="utf-8"))
    assert payload["frames"][1]["class_counts"] == {"helmet": 1, "person": 1}


def test_yolo_reference_auto_model_family_prefers_three_put_yolov8() -> None:
    assert (
        ref._resolve_model_family(
            Path("baseline_training/runs/baseline_yolov8_three_put/best.pt"),
            "auto",
        )
        == "yolov8"
    )
    assert ref._resolve_model_family(Path("baseline_yolov5/weights/best.pt"), "auto") == "yolov5"


def test_yolo_reference_target_analysis_matches_center_inside() -> None:
    analysis = ref._analyze_target(
        [
            {
                "box": [200, 100, 260, 360],
                "label": "person",
                "confidence": 0.8,
                "center": [230, 230],
            }
        ],
        target_box=[100, 80, 300, 400],
        target_labels={"person", "head", "helmet"},
    )

    assert analysis["hit"] is True
    assert analysis["match_count"] == 1
    assert analysis["matches"][0]["center_inside_target"] is True


def _write_video(path: Path, *, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.full((120, 160, 3), 40 + index, dtype=np.uint8)
            cv2.rectangle(frame, (30, 35), (75, 119), (240, 240, 240), -1)
            writer.write(frame)
    finally:
        writer.release()


def _direct_cv2_paths(source_video: Path, output_video: Path) -> ref.Cv2VideoPaths:
    return ref.Cv2VideoPaths(
        source_for_cv2=source_video,
        output_for_cv2=output_video,
        final_output=output_video,
        temp_dir=output_video.parent,
        source_alias_created=False,
        output_needs_move=False,
    )
