"""Sample-video smoke with ``static_image_backend=torch_native`` — the
NPU-friendly backend that replaces cv2.Canny + cv2.findContours with a
pure-torch extractor. L2 homography is still skipped (roadmap item).

Compared to the legacy backend, we expect:
  * adv_patch / screen_spoof detection preserved (or within -10%)
  * clean_baseline still stays at 0 alerts
  * A3b p95 drops further because the torch extractor avoids GPU→CPU
    transfers entirely.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from defense.module_a.backends import create_detector_backend  # noqa: E402
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline  # noqa: E402


def run_clip(pipeline: VideoDefensePipeline, video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {video_path}")
    reason_counter: dict[str, int] = {}
    p_adv_vals: list[float] = []
    timings: list[float] = []
    alert_frames = 0
    static_image_trigger_frames = 0
    frame_idx = 0
    pipeline.reset()
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _, _, info = pipeline.process_frame(frame)
            if info.get("alert_confirmed"):
                alert_frames += 1
            p_adv = info.get("p_adv")
            if p_adv is not None:
                p_adv_vals.append(float(p_adv))
            timings.append(float(info.get("timing_ms", 0.0)))
            for code in info.get("reason_codes", []):
                reason_counter[code] = reason_counter.get(code, 0) + 1
            sm = (
                info.get("details", {})
                .get("module_a_features", {})
                .get("static_image", {})
            )
            if sm.get("triggered"):
                static_image_trigger_frames += 1
            frame_idx += 1
    finally:
        cap.release()
    wall = time.perf_counter() - started
    return {
        "clip": video_path.name,
        "frames": frame_idx,
        "wall_seconds": round(wall, 3),
        "fps_effective": round(frame_idx / wall, 1) if wall > 0 else 0.0,
        "alert_frames": alert_frames,
        "p_adv_max": round(max(p_adv_vals) if p_adv_vals else 0.0, 4),
        "p_adv_mean": round(float(np.mean(p_adv_vals)) if p_adv_vals else 0.0, 4),
        "timing_mean_ms": round(float(np.mean(timings)) if timings else 0.0, 2),
        "timing_p95_ms": (
            round(float(np.percentile(timings, 95)), 2) if timings else 0.0
        ),
        "reason_code_counts": dict(sorted(reason_counter.items(), key=lambda kv: -kv[1])),
        "static_image_trigger_frames": static_image_trigger_frames,
    }


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    config_path = PKG_ROOT / "experiments" / "configs" / "module_a_baseline.yaml"
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    config.setdefault("module_a", {})["static_image_backend"] = "torch_native"

    backend = create_detector_backend(config, PKG_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup(frames=3)

    clips = sorted((PKG_ROOT / "samples").glob("*.mp4"))
    results = []
    for clip in clips:
        print(f"[run] {clip.name}", flush=True)
        results.append(run_clip(pipeline, clip))

    # Same criteria as legacy smoke, since torch_native should behave nearly
    # identically (only the edge-candidate internals change).
    EXPECTED = {
        "clean_baseline": 0,
        "glare_attacked": 50,
        "motion_blur_attacked": 50,
        "occlusion_attacked": 50,
        "visibility_degradation_attacked": 50,
        "adv_patch_attacked": 100,
        "screen_spoof_attacked": 100,
    }
    verdicts = []
    for r in results:
        stem = Path(r["clip"]).stem
        threshold = EXPECTED.get(stem, 50)
        if threshold == 0:
            ok = r["alert_frames"] == 0
        else:
            ok = r["alert_frames"] >= threshold
        verdicts.append({"clip": r["clip"], "ok": ok, "alert_frames": r["alert_frames"]})

    report = {
        "backend": "torch_native",
        "results": results,
        "summary": {
            "ok": all(v["ok"] for v in verdicts),
            "verdicts": verdicts,
        },
    }
    out = PKG_ROOT / "tests" / "samples_smoke_torch_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
