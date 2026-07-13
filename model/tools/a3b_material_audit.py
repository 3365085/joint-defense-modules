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
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from tools.video_diagnostic import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    alert_ranges,
    compact_reason_counts,
    frame_row,
    load_config,
)
from defense.module_a.backends import create_detector_backend  # noqa: E402
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState  # noqa: E402
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline  # noqa: E402


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
POSITIVE_CATEGORY = "视频中出现干扰视频"


def iter_videos(training_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in training_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        ),
        key=lambda path: str(path),
    )


def category_for(video_path: Path, training_root: Path) -> str:
    try:
        return video_path.relative_to(training_root).parts[0]
    except ValueError:
        return video_path.parent.name


def diagnose_video(
    pipeline: VideoDefensePipeline,
    video_path: Path,
    training_root: Path,
    *,
    config: dict[str, Any],
    max_frames: int,
    full_csv: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(video_path))
    category = category_for(video_path, training_root)
    expected_a3b = category == POSITIVE_CATEGORY
    if not cap.isOpened():
        return (
            {
                "category": category,
                "video": str(video_path),
                "error": "open_failed",
                "expected_a3b": expected_a3b,
                "pass_a3b_category_rule": False,
            },
            [],
        )

    pipeline.reset()
    a3b_state = A3BSoftTriggerState(config.get("a3b", {}) if isinstance(config, dict) else {})
    started = time.perf_counter()
    frame_idx = 0
    rows: list[dict[str, Any]] = []
    alert_frames: list[int] = []
    a3b_frames: list[int] = []
    p_media_values: list[float] = []
    p_adv_values: list[float] = []
    timing_values: list[float] = []
    reason_counts: dict[str, int] = {}
    a3b_sources: dict[str, int] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _, _, info = pipeline.process_frame(frame)
            row = frame_row(video_path, frame_idx, info, a3b_state)
            row["category"] = category
            if row["alert_confirmed"]:
                alert_frames.append(frame_idx)
            if row["a3b_triggered"]:
                a3b_frames.append(frame_idx)
                source = str(row.get("a3b_source", "none"))
                a3b_sources[source] = a3b_sources.get(source, 0) + 1
            p_media_values.append(float(row["a3b_display_score"]))
            p_adv_values.append(float(row["p_adv"]))
            timing_values.append(float(row["timing_ms"]))
            for code in info.get("reason_codes", []):
                code = str(code)
                reason_counts[code] = reason_counts.get(code, 0) + 1
            if full_csv or row["a3b_triggered"] or row["alert_confirmed"] or row["reason_codes"]:
                rows.append(row)
            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break
    finally:
        cap.release()

    a3b_count = len(a3b_frames)
    pass_rule = (a3b_count > 0) if expected_a3b else (a3b_count == 0)
    failure_type = "none"
    if not pass_rule:
        failure_type = "false_negative" if expected_a3b else "false_positive"

    summary = {
        "category": category,
        "video": str(video_path),
        "frames": frame_idx,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "fps_effective": round(frame_idx / max(1e-6, time.perf_counter() - started), 2),
        "expected_a3b": expected_a3b,
        "a3b_trigger_frames": a3b_count,
        "first_a3b_frame": a3b_frames[0] if a3b_frames else None,
        "a3b_sources": dict(sorted(a3b_sources.items(), key=lambda item: (-item[1], item[0]))),
        "alert_frames": len(alert_frames),
        "alert_ranges": alert_ranges(alert_frames),
        "p_media_max": round(max(p_media_values) if p_media_values else 0.0, 4),
        "a3b_display_score_max": round(max(p_media_values) if p_media_values else 0.0, 4),
        "p_adv_max": round(max(p_adv_values) if p_adv_values else 0.0, 4),
        "timing_mean_ms": round(float(np.mean(timing_values)) if timing_values else 0.0, 2),
        "timing_p95_ms": round(float(np.percentile(timing_values, 95)) if timing_values else 0.0, 2),
        "reason_code_counts": compact_reason_counts(reason_counts),
        "pass_a3b_category_rule": bool(pass_rule),
        "failure_type": failure_type,
    }
    return summary, rows


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "category",
        "video",
        "frames",
        "expected_a3b",
        "a3b_trigger_frames",
        "first_a3b_frame",
        "a3b_sources",
        "alert_frames",
        "p_media_max",
        "a3b_display_score_max",
        "p_adv_max",
        "timing_mean_ms",
        "timing_p95_ms",
        "pass_a3b_category_rule",
        "failure_type",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    key: json.dumps(summary.get(key), ensure_ascii=False)
                    if isinstance(summary.get(key), (dict, list))
                    else summary.get(key)
                    for key in fields
                }
            )


def write_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "category",
        "video",
        "frame_idx",
        "alert_confirmed",
        "single_frame_suspicious",
        "attack_state_active",
        "p_adv",
        "p_adv_display",
        "p_media",
        "a3b_observed_score",
        "a3b_confirmed_score",
        "a3b_display_score",
        "a3b_triggered",
        "a3b_source",
        "a3b_reason",
        "overexposure_ratio",
        "is_glare",
        "temporal_change",
        "temporal_local_max",
        "motion_score",
        "flow_local_ratio",
        "blur_score",
        "track_score",
        "confidence_drop_score",
        "timing_ms",
        "reason_codes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Module A A3b category behavior over the full training material tree."
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=WORKSPACE_ROOT / "训练素材",
        help="Root directory containing categorized videos.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Module A runtime YAML config.",
    )
    parser.add_argument("--profile", default="desktop_rtx", help="Runtime config profile.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WORKSPACE_ROOT / "doc" / "agent输出文档目录" / "modela" / "a3b_material_audit",
        help="Output directory for report.json, summary.csv and interesting_frames.csv.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames per video; 0 means full video.")
    parser.add_argument("--full-csv", action="store_true", help="Write every frame instead of only interesting frames.")
    parser.add_argument("--no-cuda-check", action="store_true", help="Skip CUDA availability pre-check.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.no_cuda_check:
        import torch

        if not torch.cuda.is_available():
            print("CUDA is required by the default Module A config; aborting.")
            return 2

    training_root = args.training_root.resolve()
    videos = iter_videos(training_root)
    if not videos:
        print(f"No videos found under {training_root}")
        return 2

    config = load_config(args.config.resolve(), profile=str(args.profile))
    backend = create_detector_backend(config, PROJECT_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup(frames=int(config.get("inference", {}).get("warmup_frames", 3)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for index, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(training_root)
        print(f"[a3b-audit] {index}/{len(videos)} {rel}", flush=True)
        summary, rows = diagnose_video(
            pipeline,
            video_path,
            training_root,
            config=config,
            max_frames=max(0, int(args.max_frames)),
            full_csv=bool(args.full_csv),
        )
        summaries.append(summary)
        frame_rows.extend(rows)

    failures = [summary for summary in summaries if not summary.get("pass_a3b_category_rule", False)]
    report = {
        "ok": not failures,
        "positive_category": POSITIVE_CATEGORY,
        "config": str(args.config.resolve()),
        "profile": str(args.profile),
        "backend": config.get("inference", {}).get("backend"),
        "training_root": str(training_root),
        "total_videos": len(summaries),
        "total_wall_seconds": round(time.perf_counter() - started, 3),
        "failures": failures,
        "videos": summaries,
    }
    report_path = args.out_dir / "report.json"
    summary_path = args.out_dir / "summary.csv"
    frames_path = args.out_dir / "interesting_frames.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary_path, summaries)
    write_frame_csv(frames_path, frame_rows)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "videos": len(summaries),
                "failures": len(failures),
                "report": str(report_path),
                "summary_csv": str(summary_path),
                "frames_csv": str(frames_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
