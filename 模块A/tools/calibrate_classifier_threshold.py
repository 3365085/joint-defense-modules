"""Scan threshold choices for the A4 classifier on a clean / attacked pair
of videos and suggest a threshold override that trades off false-alarm rate
against detection rate.

Usage::

    python tools/calibrate_classifier_threshold.py \
        --config experiments/configs/module_a_baseline.yaml \
        --clean samples/clean_baseline.mp4 \
        --attacked samples/glare_attacked.mp4 \
        --out tests/classifier_calibration_report.json

The classifier itself is loaded at its default threshold (no override); we
re-score each frame's ``classifier_p_adv`` and evaluate alternative
thresholds post-hoc. Output is a JSON report listing, for each candidate
threshold, the per-clip ``classifier_adv`` trigger count and suggested
operating point.

Design note: this is a *calibration* tool, not a retraining one. The
classifier artifact stays unchanged; operators can pick an override and
set ``module_a.classifier_threshold_override`` in their config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from defense.module_a.backends import create_detector_backend  # noqa: E402
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline  # noqa: E402


def collect_scores(pipeline: VideoDefensePipeline, clip: Path) -> list[float]:
    """Run the pipeline and collect per-frame classifier_p_adv values.

    Classifier is consulted on every frame when ``fusion_backend`` uses it;
    we just record the score and ignore downstream triggers.
    """
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip}")
    scores: list[float] = []
    pipeline.reset()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _, _, info = pipeline.process_frame(frame)
            sa = info.get("details", {}).get("module_a_features", {}).get("fusion", {})
            p = sa.get("classifier_p_adv") if isinstance(sa, dict) else None
            if p is None:
                # Fall back to top-level structure used by some callers.
                p = info.get("details", {}).get("fusion", {}).get("classifier_p_adv")
            if p is not None:
                scores.append(float(p))
    finally:
        cap.release()
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/module_a_baseline.yaml")
    parser.add_argument("--clean", default="samples/clean_baseline.mp4")
    parser.add_argument("--attacked", nargs="*", default=[
        "samples/glare_attacked.mp4",
        "samples/motion_blur_attacked.mp4",
        "samples/occlusion_attacked.mp4",
        "samples/visibility_degradation_attacked.mp4",
    ])
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--out", default="tests/classifier_calibration_report.json")
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("CUDA required")
        return 2

    config_path = PKG_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    backend = create_detector_backend(config, PKG_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup(frames=3)

    clean_scores = collect_scores(pipeline, PKG_ROOT / args.clean)
    attacked_scores: dict[str, list[float]] = {}
    for clip in args.attacked:
        clip_path = PKG_ROOT / clip
        if not clip_path.exists():
            continue
        attacked_scores[clip_path.name] = collect_scores(pipeline, clip_path)

    thresholds = args.thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    sweep: list[dict[str, Any]] = []
    for th in thresholds:
        fp_rate = sum(1 for s in clean_scores if s >= th) / max(1, len(clean_scores))
        det_per_clip = {
            name: sum(1 for s in scores if s >= th) / max(1, len(scores))
            for name, scores in attacked_scores.items()
        }
        det_mean = float(np.mean(list(det_per_clip.values()))) if det_per_clip else 0.0
        sweep.append(
            {
                "threshold": round(th, 3),
                "clean_false_positive_rate": round(fp_rate, 4),
                "attacked_detection_rate_per_clip": {k: round(v, 4) for k, v in det_per_clip.items()},
                "attacked_detection_rate_mean": round(det_mean, 4),
            }
        )

    # Suggest the highest threshold where mean detection rate is still
    # within 2 percentage points of the best. Heuristic — the operator
    # is the real judge.
    best_det = max((s["attacked_detection_rate_mean"] for s in sweep), default=0.0)
    safe_threshold = max(
        (s["threshold"] for s in sweep if s["attacked_detection_rate_mean"] >= best_det - 0.02),
        default=thresholds[0],
    )

    report = {
        "config": str(config_path),
        "clean_n_frames": len(clean_scores),
        "clean_p_adv_max": round(max(clean_scores) if clean_scores else 0.0, 4),
        "clean_p_adv_mean": round(float(np.mean(clean_scores)) if clean_scores else 0.0, 4),
        "attacked_clips": list(attacked_scores.keys()),
        "sweep": sweep,
        "suggested_threshold_override": round(safe_threshold, 3),
    }
    out_path = PKG_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[threshold] clean_fp@0.65={next((s['clean_false_positive_rate'] for s in sweep if s['threshold']==0.65),'n/a')}")
    print(f"[threshold] suggested override: {safe_threshold}")
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
