from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def stable_video_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{path.stem}_{digest}"


def iter_videos(paths: list[Path]) -> list[Path]:
    videos: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            videos.append(path)
        elif path.is_dir():
            videos.extend(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    return sorted(videos)


def extract_video_frames(
    video_path: Path,
    output_root: Path,
    category: str,
    stride: int,
    max_frames: int | None,
) -> list[dict[str, str | int | float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    video_id = stable_video_id(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_dir = output_root / "frames_raw" / category / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_path = frame_dir / f"{video_id}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            height, width = frame.shape[:2]
            rows.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "category": category,
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3) if fps > 0 else 0.0,
                    "frame_path": str(frame_path),
                    "width": width,
                    "height": height,
                }
            )
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        frame_idx += 1
    capture.release()
    return rows


def append_manifest(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["video_id", "video_path", "category", "frame_idx", "timestamp_sec", "frame_path", "width", "height"],
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抽取YOLO安全帽误检专项复核帧。")
    parser.add_argument("--source", nargs="+", required=True, type=Path, help="视频文件或目录。")
    parser.add_argument("--output-root", type=Path, default=Path("materials/yolo_fp_review"))
    parser.add_argument("--category", required=True, help="素材类别，例如 normal_no_helmet。")
    parser.add_argument("--stride", type=int, default=15, help="每N帧保存一帧。")
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stride = max(1, int(args.stride))
    max_frames = args.max_frames_per_video if args.max_frames_per_video > 0 else None
    videos = iter_videos(args.source)
    if not videos:
        raise SystemExit("未找到视频文件")
    all_rows: list[dict[str, str | int | float]] = []
    for video_path in videos:
        rows = extract_video_frames(video_path, args.output_root, args.category, stride, max_frames)
        all_rows.extend(rows)
        print(f"{video_path}: saved {len(rows)} frames")
    manifest_path = args.output_root / "manifests" / "frame_extract_manifest.csv"
    append_manifest(manifest_path, all_rows)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
