from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.runtime import MonitorEngine, PipelineCache, project_root


def capture_runtime_preview_video(
    *,
    source: str | Path,
    output_video: str | Path,
    overlay_json: str | Path | None = None,
    summary_out: str | Path | None = None,
    profile: str = "default",
    config: str | Path | None = None,
    ready_timeout_s: float = 45.0,
    max_seconds: float = 120.0,
    frame_timeout_s: float = 1.0,
    output_fps: float | None = None,
) -> dict[str, Any]:
    """Record the actual MonitorEngine latest-only preview stream to MP4."""

    source_path = Path(source)
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_root = project_root()
    cache = PipelineCache(
        config_path=Path(config) if config is not None else None,
        root=runtime_root,
    )
    engine = MonitorEngine(cache)
    writer: cv2.VideoWriter | None = None
    preview_frames: list[dict[str, Any]] = []
    final_status: dict[str, Any] = {}
    run_id = engine.start(
        source_type="file",
        source=str(source_path),
        profile=str(profile),
        realtime=True,
    )
    started_at = time.time()
    try:
        status = engine.wait_ready_for_preview(run_id, timeout=float(ready_timeout_s))
        if not status.get("ready_for_preview"):
            raise RuntimeError(f"preview did not become ready: {status}")
        fps = float(output_fps or status.get("preview_render_fps") or 25.0)
        if fps <= 0.0:
            fps = 25.0
        deadline = time.time() + max(0.1, float(max_seconds))
        last_seq = 0
        while time.time() < deadline:
            wait_snapshot = getattr(engine, "wait_latest_jpeg_snapshot", None)
            if callable(wait_snapshot):
                seq, jpeg, running, preview_meta = wait_snapshot(
                    last_seq,
                    timeout=max(0.05, float(frame_timeout_s)),
                )
            else:
                seq, jpeg, running = engine.wait_latest_jpeg(
                    last_seq,
                    timeout=max(0.05, float(frame_timeout_s)),
                )
                preview_meta = {}
            status = engine.get_status()
            final_status = dict(status)
            if jpeg is not None and seq > last_seq:
                frame = _decode_jpeg(jpeg)
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (int(width), int(height)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open output video writer: {output_path}")
                writer.write(frame)
                preview_frames.append(
                    {
                        "preview_seq": int(seq),
                        "source_epoch": int(
                            preview_meta.get(
                                "source_epoch",
                                status.get("source_epoch") or 0,
                            )
                        ),
                        "frame_idx": int(
                            preview_meta.get(
                                "frame_idx",
                                status.get("frame_idx") or 0,
                            )
                        ),
                        "source_time_s": float(
                            preview_meta.get(
                                "source_time_s",
                                status.get("source_time_s") or 0.0,
                            )
                        ),
                        "video_time_s": float(
                            preview_meta.get(
                                "source_time_s",
                                status.get("video_time_s") or 0.0,
                            )
                        ),
                        "preview_fps": float(status.get("preview_fps") or 0.0),
                        "detector_pipeline_mode": str(
                            status.get("detector_pipeline_mode") or ""
                        ),
                        "detector_queue_policy": str(
                            status.get("detector_queue_policy") or ""
                        ),
                    }
                )
            last_seq = max(last_seq, int(seq))
            if not running or status.get("source_ended") or not status.get("running"):
                break
        if writer is None:
            raise RuntimeError("runtime preview did not produce any JPEG frames")
    finally:
        if writer is not None:
            writer.release()
        overlay = engine.get_overlay(since_seq=0)
        final_status = engine.get_status()
        engine.stop()

    summary = {
        "source": str(source_path),
        "output_video": str(output_path),
        "profile": str(profile),
        "run_id": int(run_id),
        "frames_written": len(preview_frames),
        "first_preview_seq": preview_frames[0]["preview_seq"] if preview_frames else None,
        "last_preview_seq": preview_frames[-1]["preview_seq"] if preview_frames else None,
        "duration_wall_s": max(0.0, time.time() - started_at),
        "detector_pipeline_mode": str(final_status.get("detector_pipeline_mode") or ""),
        "detector_queue_policy": str(final_status.get("detector_queue_policy") or ""),
        "source_ended": bool(final_status.get("source_ended")),
        "running_after_capture": bool(final_status.get("running")),
        "preview_width": int(final_status.get("preview_width") or 0),
        "preview_height": int(final_status.get("preview_height") or 0),
        "overlay_records": len(overlay.get("records") or []),
        "latest_overlay_seq": int(overlay.get("latest_seq") or 0),
        "capture_contract": {
            "runtime_engine": "MonitorEngine",
            "preview_path": "wait_latest_jpeg",
            "detector_queue_policy_required": "latest_only",
            "detection_preview_decoupled": True,
        },
        "preview_frames": preview_frames,
    }
    if overlay_json is not None:
        _write_json(
            overlay_json,
            {
                "overlay": overlay,
                "final_status": final_status,
                "summary": summary,
            },
        )
    if summary_out is not None:
        _write_json(summary_out, summary)
    return summary


def _decode_jpeg(jpeg: bytes) -> Any:
    data = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("failed to decode runtime preview JPEG")
    return frame


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture the actual MonitorEngine latest-only preview stream to MP4."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--overlay-json", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-timeout-s", type=float, default=45.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--frame-timeout-s", type=float, default=1.0)
    parser.add_argument("--output-fps", type=float, default=None)
    args = parser.parse_args(argv)
    summary = capture_runtime_preview_video(
        source=args.source,
        output_video=args.output_video,
        overlay_json=args.overlay_json,
        summary_out=args.summary_out,
        profile=args.profile,
        config=args.config,
        ready_timeout_s=args.ready_timeout_s,
        max_seconds=args.max_seconds,
        frame_timeout_s=args.frame_timeout_s,
        output_fps=args.output_fps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
