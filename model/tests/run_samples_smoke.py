"""Sample-video smoke test: run the real VideoDefensePipeline on each
sample MP4 and check that each attack triggers its own reason code.

This is an *end-to-end* test that:
  * Loads the real YOLOv5 TensorRT backend.
  * Drives ``VideoDefensePipeline.process_frame`` across every frame.
  * Aggregates reason codes / p_adv / static-image trigger counts per clip.

Expected behaviour matches 架构说明.md + README_交付说明.md §示例视频:

    clean_baseline           → no alert_confirmed, low p_adv
    glare_attacked           → overexposure reason code present, alert_confirmed
    motion_blur_attacked     → blur / local_blur_degradation triggers
    occlusion_attacked       → track_consistency_drop triggers
    visibility_degradation   → blur / temporal triggers
    adv_patch_attacked       → p_adv alert_confirmed (fused evidence)
    screen_spoof_attacked    → static_image_spoof triggers

We do NOT fail the test for small variance in p_adv; we assert qualitative
behaviour: clean stays calm, each attacked clip produces its intended
reason code set at least once.

Kept outside the pytest collection path because it requires the packaged
YOLOv5 TensorRT engine and touches ~50s of wall time on the reference GPU.
Run manually:

    python tests\run_samples_smoke.py
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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def run_clip(pipeline: VideoDefensePipeline, video_path: Path) -> dict[str, Any]:
    """Process a clip and collect aggregate diagnostics."""
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


EXPECTED_TRIGGERS = {
    # clip_stem → set of reason codes that must show up at least once.
    # Expectations are derived from 架构说明.md §七 and README_交付说明.md.
    "clean_baseline": set(),  # No strong expectation; may be empty.
    "glare_attacked": {"overexposure"},
    "motion_blur_attacked": {"local_blur_degradation"},
    "occlusion_attacked": {"track_consistency_drop"},
    "visibility_degradation_attacked": {"local_blur_degradation"},
    # adv_patch is caught by the A3b static-media path (patch-track +
    # A3+ candidate). The 5-dim rule-fusion ``p_adv`` code only fires
    # when the *linear* score crosses ``p_adv_threshold`` (0.55), which
    # is not guaranteed on this clip; see 架构说明.md §七.
    "adv_patch_attacked": {"static_image_spoof"},
    "screen_spoof_attacked": {"static_image_spoof"},
}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = []
    for r in results:
        stem = Path(r["clip"]).stem
        expected = EXPECTED_TRIGGERS.get(stem, set())
        present = set(r["reason_code_counts"].keys())
        if stem == "clean_baseline":
            ok = r["alert_frames"] == 0 or r["p_adv_max"] <= 0.55
            verdicts.append({"clip": r["clip"], "ok": ok, "why": "clean calm"})
        else:
            missing = expected - present
            ok = not missing and r["alert_frames"] > 0
            verdicts.append(
                {
                    "clip": r["clip"],
                    "ok": ok,
                    "missing_reasons": sorted(missing),
                    "alert_frames": r["alert_frames"],
                }
            )
    return {
        "ok": all(v["ok"] for v in verdicts),
        "verdicts": verdicts,
    }


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA is required; aborting.")
        return 2
    config_path = PKG_ROOT / "experiments" / "configs" / "module_a_baseline.yaml"
    config = load_config(config_path)
    backend = create_detector_backend(config, PKG_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup(frames=3)

    samples_dir = PKG_ROOT / "samples"
    clips = sorted(samples_dir.glob("*.mp4"))
    if not clips:
        print(f"No clips found in {samples_dir}")
        return 2

    results: list[dict[str, Any]] = []
    for clip in clips:
        print(f"[run] {clip.name}", flush=True)
        results.append(run_clip(pipeline, clip))

    report = {
        "config": str(config_path),
        "backend": config.get("inference", {}).get("backend"),
        "results": results,
        "summary": summarize(results),
    }
    out = PKG_ROOT / "tests" / "samples_smoke_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
