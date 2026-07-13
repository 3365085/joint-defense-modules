from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (SRC_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from defense.module_a.backends import create_detector_backend  # noqa: E402
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState  # noqa: E402
from defense.diagnostics.video_rows import FIELDNAMES, alert_ranges, compact_reason_counts, frame_row, is_interesting  # noqa: E402
from defense.runtime.config import DEFAULT_CONFIG_PATH, load_runtime_config  # noqa: E402
from defense.runtime.pipeline_factory import EmptyDetectorBackend  # noqa: E402
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline  # noqa: E402


def load_config(path: Path, *, profile: str = "desktop_rtx", a3b_sensitivity: str = "") -> dict[str, Any]:
    feature_options = {"a3b_sensitivity": a3b_sensitivity} if a3b_sensitivity else None
    return load_runtime_config(config_path=path, profile=profile, feature_options=feature_options)


def run_video(
    pipeline: VideoDefensePipeline,
    video_path: Path,
    *,
    config: dict[str, Any],
    max_frames: int,
    full_csv: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    reason_counts: dict[str, int] = {}
    p_adv_values: list[float] = []
    p_media_values: list[float] = []
    timing_values: list[float] = []
    alert_frames: list[int] = []
    suspicious_frames = 0
    a3b_frames = 0
    glare_frames = 0
    rows: list[dict[str, Any]] = []
    frame_idx = 0
    started = time.perf_counter()

    pipeline.reset()
    a3b_state = A3BSoftTriggerState(config.get("a3b", {}) if isinstance(config, dict) else {})
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _, _, info = pipeline.process_frame(frame)
            row = frame_row(video_path, frame_idx, info, a3b_state)
            if row["alert_confirmed"]:
                alert_frames.append(frame_idx)
            if row["single_frame_suspicious"]:
                suspicious_frames += 1
            if row["a3b_triggered"]:
                a3b_frames += 1
            if row["is_glare"]:
                glare_frames += 1
            p_adv_values.append(float(row["p_adv"]))
            p_media_values.append(float(row["a3b_display_score"]))
            timing_values.append(float(row["timing_ms"]))
            for code in info.get("reason_codes", []):
                code = str(code)
                reason_counts[code] = reason_counts.get(code, 0) + 1
            if full_csv or is_interesting(row):
                rows.append(row)

            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break
    finally:
        cap.release()

    wall_seconds = time.perf_counter() - started
    summary = {
        "video": str(video_path),
        "frames": frame_idx,
        "wall_seconds": round(wall_seconds, 3),
        "fps_effective": round(frame_idx / wall_seconds, 2) if wall_seconds > 0 else 0.0,
        "alert_frames": len(alert_frames),
        "suspicious_frames": suspicious_frames,
        "a3b_trigger_frames": a3b_frames,
        "glare_frames": glare_frames,
        "p_adv_max": round(max(p_adv_values) if p_adv_values else 0.0, 4),
        "p_adv_mean": round(float(np.mean(p_adv_values)) if p_adv_values else 0.0, 4),
        "p_media_max": round(max(p_media_values) if p_media_values else 0.0, 4),
        "a3b_display_score_max": round(max(p_media_values) if p_media_values else 0.0, 4),
        "timing_mean_ms": round(float(np.mean(timing_values)) if timing_values else 0.0, 2),
        "timing_p95_ms": round(float(np.percentile(timing_values, 95)) if timing_values else 0.0, 2),
        "alert_ranges": alert_ranges(alert_frames),
        "reason_code_counts": compact_reason_counts(reason_counts),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FIELDNAMES
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_videos(args: argparse.Namespace) -> list[Path]:
    if args.video:
        return [args.video.resolve()]
    samples_dir = args.samples_dir.resolve()
    return sorted(samples_dir.glob("*.mp4"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Module A on video(s) and export frame-level alert diagnostics."
    )
    parser.add_argument("--video", type=Path, help="Single MP4/RTSP-readable video file to diagnose.")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=PROJECT_ROOT / "samples",
        help="Directory of MP4 files used when --video is omitted.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Module A runtime YAML config.",
    )
    parser.add_argument("--profile", default="desktop_rtx", help="Runtime config profile.")
    parser.add_argument(
        "--a3b-sensitivity",
        choices=["conservative", "balanced", "sensitive", "high"],
        default="",
        help="Override A3b sensitivity preset for diagnostics.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "video_diagnostic",
        help="Output directory for JSON/CSV reports.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames per video; 0 means full video.")
    parser.add_argument("--full-csv", action="store_true", help="Write every frame; default writes only interesting frames.")
    parser.add_argument("--no-cuda-check", action="store_true", help="Skip the CUDA availability pre-check.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.no_cuda_check:
        import torch

        if not torch.cuda.is_available():
            print("CUDA is required by the default Module A config; aborting.")
            return 2

    videos = resolve_videos(args)
    if not videos:
        print(f"No videos found. --video={args.video!s}, --samples-dir={args.samples_dir!s}")
        return 2

    config = load_config(args.config, profile=str(args.profile), a3b_sensitivity=str(args.a3b_sensitivity or ""))
    runtime_config = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    backend = EmptyDetectorBackend() if runtime_config.get("allow_empty_backend", False) else create_detector_backend(config, PROJECT_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup(frames=int(config.get("inference", {}).get("warmup_frames", 3)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for video_path in videos:
        print(f"[diagnose] {video_path}", flush=True)
        summary, rows = run_video(
            pipeline,
            video_path,
            config=config,
            max_frames=max(0, int(args.max_frames)),
            full_csv=bool(args.full_csv),
        )
        summaries.append(summary)
        all_rows.extend(rows)

    report = {
        "config": str(args.config.resolve()),
        "profile": str(args.profile),
        "a3b_sensitivity": str(args.a3b_sensitivity or config.get("a3b", {}).get("sensitivity") or ""),
        "backend": config.get("inference", {}).get("backend"),
        "videos": summaries,
    }
    report_path = args.out_dir / "video_diagnostic_report.json"
    csv_path = args.out_dir / "video_diagnostic_frames.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, all_rows)

    print(
        json.dumps(
            {
                "ok": True,
                "videos": len(summaries),
                "report": str(report_path),
                "csv": str(csv_path),
                "summary": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
