from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any

import cv2


def export_visual_acceptance_frames(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    start_frame: int | None = None,
    start_time_s: float | None = None,
    duration_s: float = 3.0,
    frame_count: int | None = None,
    overwrite: bool = False,
    allow_short_for_test: bool = False,
) -> dict[str, Any]:
    video = Path(video_path)
    out_dir = Path(output_dir)
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(f"video does not exist: {video}")
    if duration_s <= 0 and frame_count is None:
        raise ValueError("duration_s must be positive when frame_count is not set")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output_dir must be an empty per-round directory: {out_dir}")
        _clear_generated_acceptance_artifacts(out_dir)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0.0:
            fps = 30.0
        if start_frame is None:
            start_frame = int(round(float(start_time_s or 0.0) * fps))
        start_frame = max(0, int(start_frame))
        required_final_acceptance_duration_s = 3.0
        required_min_frame_count = max(1, int(math.ceil(required_final_acceptance_duration_s * fps)))
        if frame_count is None:
            frame_count = max(1, int(round(float(duration_s) * fps)))
        frame_count = max(1, int(frame_count))
        if not allow_short_for_test and frame_count < required_min_frame_count:
            raise ValueError(
                f"final visual acceptance requires at least {required_final_acceptance_duration_s:.1f}s "
                f"({required_min_frame_count} frames at {fps:.3f} fps); requested {frame_count} frames"
            )
        if not allow_short_for_test and total_frames > 0 and total_frames - start_frame < required_min_frame_count:
            raise ValueError(
                f"final visual acceptance requires at least {required_final_acceptance_duration_s:.1f}s "
                f"from the requested start frame; only {max(0, total_frames - start_frame)} frames remain"
            )
        end_frame_exclusive = start_frame + frame_count
        if total_frames > 0:
            end_frame_exclusive = min(end_frame_exclusive, total_frames)
        if end_frame_exclusive <= start_frame:
            raise ValueError("requested frame range is outside the video")

        frames_dir = out_dir / "frames_png"
        frames_dir.mkdir(parents=True, exist_ok=True)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        records: list[dict[str, Any]] = []
        frame_idx = start_frame
        while frame_idx < end_frame_exclusive:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            path = frames_dir / f"frame_{frame_idx:06d}.png"
            ok, encoded = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            if not ok:
                raise RuntimeError(f"failed to encode png frame: {path}")
            path.write_bytes(encoded.tobytes())
            records.append(
                {
                    "frame_idx": int(frame_idx),
                    "time_s": float(frame_idx) / fps,
                    "path": str(path),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "format": "png",
                    "lossless": True,
                }
            )
            frame_idx += 1
    finally:
        capture.release()

    actual_duration_s = (len(records) / fps) if records else 0.0
    if not allow_short_for_test and len(records) < required_min_frame_count:
        raise RuntimeError(
            f"final visual acceptance requires at least {required_final_acceptance_duration_s:.1f}s; "
            f"decoded only {len(records)} frames"
        )
    valid_for_final_acceptance = bool(
        not allow_short_for_test and actual_duration_s >= required_final_acceptance_duration_s
    )
    manifest = {
        "video_path": str(video),
        "output_dir": str(out_dir),
        "frames_dir": str(frames_dir),
        "fps": float(fps),
        "video_width": int(width),
        "video_height": int(height),
        "video_total_frames": int(total_frames),
        "start_frame": int(start_frame),
        "end_frame_inclusive": int(records[-1]["frame_idx"]) if records else None,
        "frame_count": len(records),
        "requested_duration_s": float(duration_s),
        "actual_duration_s": actual_duration_s,
        "required_final_acceptance_duration_s": required_final_acceptance_duration_s,
        "required_min_frame_count": required_min_frame_count,
        "meets_final_acceptance_duration": actual_duration_s >= required_final_acceptance_duration_s,
        "acceptance_round_dir_unique": True,
        "decoded_full_resolution": all(
            int(item.get("width") or 0) == int(width) and int(item.get("height") or 0) == int(height)
            for item in records
        ),
        "rules": {
            "final_acceptance_only": valid_for_final_acceptance,
            "one_folder_per_review_round": True,
            "one_full_frame_per_image": True,
            "image_format": "png",
            "lossless": True,
            "source": "decoded_result_video",
            "do_not_use_contact_sheet": True,
            "human_acceptance_required": True,
            "temporary_analysis_images_must_be_deleted": True,
            "allow_short_for_test": bool(allow_short_for_test),
        },
        "artifact_policy": {
            "retention_class": "final_acceptance_evidence" if valid_for_final_acceptance else "test_only_short_acceptance_sample",
            "valid_for_final_acceptance": valid_for_final_acceptance,
            "cleanup_required": not valid_for_final_acceptance,
            "cleanup_status": "retain_until_user_acceptance" if valid_for_final_acceptance else "test_only_cleanup_allowed",
            "cleanup_owner": "human_operator" if valid_for_final_acceptance else "agent_or_operator",
        },
        "frames": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    index_path = out_dir / "index.md"
    manifest["manifest_path"] = str(manifest_path)
    manifest["index_path"] = str(index_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_index(index_path, manifest)
    return manifest


def _clear_generated_acceptance_artifacts(out_dir: Path) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite unmarked acceptance directory: {out_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FileExistsError(f"refusing to overwrite unreadable acceptance manifest: {manifest_path}") from exc
    rules = manifest.get("rules") if isinstance(manifest, dict) else None
    if not isinstance(rules, dict) or rules.get("source") != "decoded_result_video" or rules.get("image_format") != "png":
        raise FileExistsError(f"refusing to overwrite directory without visual acceptance marker: {out_dir}")
    allowed = {"manifest.json", "index.md", "frames_png"}
    unknown = sorted(item.name for item in out_dir.iterdir() if item.name not in allowed)
    if unknown:
        raise FileExistsError(f"refusing to overwrite acceptance directory with unknown files: {unknown}")
    frames_dir = out_dir / "frames_png"
    if frames_dir.exists():
        for path in frames_dir.iterdir():
            if path.is_dir() or path.suffix.lower() != ".png":
                raise FileExistsError(f"refusing to delete unexpected generated artifact: {path}")
            path.unlink()
        frames_dir.rmdir()
    for name in ("manifest.json", "index.md"):
        path = out_dir / name
        if path.exists():
            path.unlink()


def _write_index(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# 视觉验收抽帧索引",
        "",
        f"- 结果视频：`{manifest['video_path']}`",
        f"- 抽帧目录：`{manifest['frames_dir']}`",
        f"- 帧号范围：`{manifest['start_frame']}` - `{manifest['end_frame_inclusive']}`",
        f"- 帧数：`{manifest['frame_count']}`",
        f"- FPS：`{manifest['fps']:.3f}`",
        f"- 分辨率：`{manifest['video_width']}x{manifest['video_height']}`",
        "",
        "## 帧列表",
        "",
    ]
    for item in manifest.get("frames", []):
        lines.append(f"- frame `{item['frame_idx']}` time `{item['time_s']:.3f}s`: `{item['path']}`")
    lines.extend(
        [
            "",
            "## 验收结论",
            "",
            "- agent 诊断结论：待填写",
            "- 用户人工验收结论：待填写",
            "- 临时视觉垃圾清理状态：待确认",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export lossless PNG frames for final visual acceptance.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--start-time-s", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-short-for-test", action="store_true")
    args = parser.parse_args(argv)
    manifest = export_visual_acceptance_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        start_time_s=args.start_time_s,
        duration_s=args.duration_s,
        frame_count=args.frame_count,
        overwrite=args.overwrite,
        allow_short_for_test=args.allow_short_for_test,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
