"""Per-feature timing profiler.

Runs the same sample clips used by ``run_samples_smoke.py`` and aggregates
``info["latency_breakdown"]["module_a_breakdown"]`` across all frames. The
output lists mean / p95 / p99 for each of the 6 Module A feature buckets,
plus the total detector and module_a latency.

Usage::

    python tests/profile_feature_timings.py            # all sample clips
    python tests/profile_feature_timings.py --clip screen_spoof_attacked

Output JSON lands in ``tests/profile_feature_timings_report.json``.
"""
from __future__ import annotations

import argparse
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


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def profile_clip(pipeline: VideoDefensePipeline, path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {path}")
    buckets: dict[str, list[float]] = {
        "a1_overexposure_ms": [],
        "a2_temporal_ms": [],
        "a3_motion_ms": [],
        "a3b_static_media_ms": [],
        "a4_fusion_ms": [],
        "source_auth_ms": [],
    }
    detector_ms: list[float] = []
    module_a_ms: list[float] = []
    total_ms: list[float] = []
    pipeline.reset()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _, _, info = pipeline.process_frame(frame)
            latency = info.get("latency_breakdown", {})
            for key in buckets:
                buckets[key].append(float(latency.get("module_a_breakdown", {}).get(key, 0.0)))
            detector_ms.append(float(latency.get("detector_ms", 0.0)))
            module_a_ms.append(float(latency.get("module_a_total_ms", 0.0)))
            total_ms.append(float(info.get("timing_ms", 0.0)))
            frames += 1
    finally:
        cap.release()

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "mean": round(float(np.mean(values)), 3),
            "p50": round(float(np.median(values)), 3),
            "p95": round(_pct(values, 95), 3),
            "p99": round(_pct(values, 99), 3),
            "max": round(float(np.max(values)), 3),
        }

    return {
        "clip": path.name,
        "frames": frames,
        "detector_ms": stats(detector_ms),
        "module_a_total_ms": stats(module_a_ms),
        "pipeline_total_ms": stats(total_ms),
        "per_feature": {key: stats(vals) for key, vals in buckets.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", nargs="*", default=None, help="Filter to specific clip stems")
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("CUDA required; aborting.")
        return 2

    config_path = PKG_ROOT / "experiments" / "configs" / "module_a_baseline.yaml"
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    backend = create_detector_backend(config, PKG_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup(frames=3)

    clips = sorted((PKG_ROOT / "samples").glob("*.mp4"))
    if args.clip:
        keep = set(args.clip)
        clips = [c for c in clips if c.stem in keep]
    results = []
    for clip in clips:
        started = time.perf_counter()
        print(f"[profile] {clip.name}", flush=True)
        stats = profile_clip(pipeline, clip)
        stats["wall_seconds"] = round(time.perf_counter() - started, 3)
        results.append(stats)

    out = PKG_ROOT / "tests" / "profile_feature_timings_report.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # Pretty-print summary to stdout
    print(f"\n{'clip':<42} {'total_p95':>10} {'det_p95':>10} {'A1':>8} {'A2':>8} {'A3':>8} {'A3b':>8} {'A4':>8}")
    for r in results:
        clip = r["clip"]
        pf = r["per_feature"]
        print(
            f"{clip:<42} "
            f"{r['pipeline_total_ms']['p95']:>10.2f} "
            f"{r['detector_ms']['p95']:>10.2f} "
            f"{pf['a1_overexposure_ms']['p95']:>8.3f} "
            f"{pf['a2_temporal_ms']['p95']:>8.3f} "
            f"{pf['a3_motion_ms']['p95']:>8.3f} "
            f"{pf['a3b_static_media_ms']['p95']:>8.3f} "
            f"{pf['a4_fusion_ms']['p95']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
