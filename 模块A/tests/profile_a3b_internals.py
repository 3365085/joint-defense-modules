"""Break down A3b internal timings by instrumenting the detector directly.

We attach a hook that measures compute() end-to-end then reads the
``p_media_timing_ms`` dict that ``GPUStaticMediaSpoofDetector.compute``
stamps on its return value, plus the outer legacy-ROI-loop cost (computed
as ``a3b_total - (bg+l0+yolo+l2)``).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from defense.module_a.backends import create_detector_backend  # noqa: E402
from defense.module_a.features.a3.static_media.detector import (  # noqa: E402
    GPUStaticMediaSpoofDetector,
)
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline  # noqa: E402


def instrument(detector: GPUStaticMediaSpoofDetector):
    """Wrap ``compute`` so it captures timing dict + total wallclock."""
    original = detector.compute
    timings: list[dict] = []

    def wrapped(prev_gray, curr_gray, rois=None):
        t0 = time.perf_counter()
        out = original(prev_gray, curr_gray, rois)
        total_ms = (time.perf_counter() - t0) * 1000.0
        tim = dict(out.get("p_media_timing_ms", {}))
        tim["wallclock_ms"] = total_ms
        tim["legacy_roi_ms"] = total_ms - tim.get("total_a3plus_ms", 0.0)
        tim["n_rois"] = len(rois or [])
        timings.append(tim)
        return out

    detector.compute = wrapped  # type: ignore[assignment]
    return timings


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(values)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
        "max": round(float(np.max(values)), 2),
    }


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA required"); return 2

    config_path = PKG_ROOT / "experiments" / "configs" / "module_a_baseline.yaml"
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    backend = create_detector_backend(config, PKG_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup(frames=3)
    # Hook after warmup so warmup frames don't skew stats.
    timings = instrument(pipeline.detector.static_image)

    report: dict[str, dict] = {}
    for clip in sorted((PKG_ROOT / "samples").glob("*.mp4")):
        pipeline.reset()
        timings.clear()
        cap = cv2.VideoCapture(str(clip))
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            pipeline.process_frame(frame)
        cap.release()

        buckets: dict[str, list[float]] = {
            "wallclock_ms": [],
            "bg_edge_ms": [],
            "l0_l1_ms": [],
            "yolo_context_ms": [],
            "l2_homography_ms": [],
            "total_a3plus_ms": [],
            "legacy_roi_ms": [],
            "n_rois": [],
        }
        for tim in timings:
            for key in buckets:
                if key in tim:
                    buckets[key].append(float(tim[key]))
        report[clip.name] = {k: stats(v) for k, v in buckets.items()}
        report[clip.name]["frames"] = len(timings)
        report[clip.name]["n_rois_mean"] = (
            round(float(np.mean(buckets["n_rois"])), 1) if buckets["n_rois"] else 0
        )

    out = PKG_ROOT / "tests" / "profile_a3b_internals_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{'clip':<42} {'wall_p95':>9} {'legacy':>7} {'a3plus':>7} "
        f"{'bg':>6} {'l0':>6} {'yolo':>6} {'l2':>6} {'n_roi':>6}"
    )
    for name, s in report.items():
        print(
            f"{name:<42} "
            f"{s['wallclock_ms']['p95']:>9.2f} "
            f"{s['legacy_roi_ms']['p95']:>7.2f} "
            f"{s['total_a3plus_ms']['p95']:>7.2f} "
            f"{s['bg_edge_ms']['p95']:>6.2f} "
            f"{s['l0_l1_ms']['p95']:>6.2f} "
            f"{s['yolo_context_ms']['p95']:>6.2f} "
            f"{s['l2_homography_ms']['p95']:>6.2f} "
            f"{s['n_rois_mean']:>6.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
