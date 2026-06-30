from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from defense.diagnostics.visual_acceptance_frames import export_visual_acceptance_frames


def _write_test_video(path: Path, *, frame_count: int = 10, fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 32),
    )
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = (idx * 20) % 255
        writer.write(frame)
    writer.release()


def test_export_visual_acceptance_frames_writes_lossless_png_manifest_and_index(tmp_path: Path) -> None:
    video_path = tmp_path / "result.mp4"
    _write_test_video(video_path, frame_count=20, fps=5.0)

    manifest = export_visual_acceptance_frames(
        video_path=video_path,
        output_dir=tmp_path / "acceptance_round_001",
        start_frame=2,
        frame_count=15,
    )

    assert manifest["frame_count"] == 15
    assert manifest["start_frame"] == 2
    assert manifest["end_frame_inclusive"] == 16
    assert manifest["rules"]["lossless"] is True
    assert manifest["rules"]["source"] == "decoded_result_video"
    assert manifest["rules"]["one_folder_per_review_round"] is True
    assert manifest["acceptance_round_dir_unique"] is True
    assert manifest["decoded_full_resolution"] is True
    assert manifest["meets_final_acceptance_duration"] is True
    assert manifest["required_min_frame_count"] == 15
    assert manifest["artifact_policy"]["retention_class"] == "final_acceptance_evidence"
    assert Path(manifest["manifest_path"]).exists()
    index_path = Path(manifest["index_path"])
    assert index_path.exists()
    assert "用户人工验收结论：待填写" in index_path.read_text(encoding="utf-8")

    for item in manifest["frames"]:
        frame_path = Path(item["path"])
        assert frame_path.suffix == ".png"
        assert frame_path.exists()
        image = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        assert image is not None
        assert image.shape[:2] == (32, 64)


def test_export_visual_acceptance_frames_fails_when_less_than_three_seconds(tmp_path: Path) -> None:
    video_path = tmp_path / "result.mp4"
    _write_test_video(video_path, frame_count=10, fps=5.0)

    with pytest.raises(ValueError, match="final visual acceptance requires"):
        export_visual_acceptance_frames(
            video_path=video_path,
            output_dir=tmp_path / "too_short",
            start_frame=0,
            frame_count=10,
        )

    assert not (tmp_path / "too_short" / "manifest.json").exists()


def test_export_visual_acceptance_frames_rejects_non_empty_round_dir(tmp_path: Path) -> None:
    video_path = tmp_path / "result.mp4"
    _write_test_video(video_path, frame_count=20, fps=5.0)
    output_dir = tmp_path / "acceptance_round_001"
    output_dir.mkdir()
    (output_dir / "old_frame.png").write_bytes(b"old")

    with pytest.raises(FileExistsError):
        export_visual_acceptance_frames(
            video_path=video_path,
            output_dir=output_dir,
            start_frame=0,
            frame_count=15,
        )

    output_dir = tmp_path / "acceptance_round_002"
    manifest = export_visual_acceptance_frames(
        video_path=video_path,
        output_dir=output_dir,
        start_frame=0,
        frame_count=15,
    )
    (output_dir / "extra.txt").write_text("do not delete", encoding="utf-8")

    with pytest.raises(FileExistsError, match="unknown files"):
        export_visual_acceptance_frames(
            video_path=video_path,
            output_dir=output_dir,
            start_frame=0,
            frame_count=15,
            overwrite=True,
        )

    (output_dir / "extra.txt").unlink()
    manifest = export_visual_acceptance_frames(
        video_path=video_path,
        output_dir=output_dir,
        start_frame=0,
        frame_count=15,
        overwrite=True,
    )

    assert Path(manifest["manifest_path"]).exists()
