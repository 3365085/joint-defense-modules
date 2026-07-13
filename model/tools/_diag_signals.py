"""独立诊断脚本：使用 VideoDefensePipeline（真实 YOLO 后端）统计各攻击视频的信号分布。

features 是扁平字典(process_pipeline.py 587-615)。
A1/A3b 标志位(is_glare / triggered / p_media)在 details["module_a_features"]。

用法：
    cd model && set PYTHONPATH=src&& python tools/_diag_signals.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from defense.module_a.backends.detector_backend import UltralyticsDetectorBackend  # noqa: E402
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline  # noqa: E402
from defense.runtime.config import load_runtime_config  # noqa: E402

WEIGHTS = ROOT / "baseline_training/runs/baseline_yolov8_three_put/best.pt"

ATTACK_VIDEOS = {
    "normal_outdoor": "D:/联合防御模块/素材/手机随意录制的视频/固定镜头室外视频.mp4",
    "glare":          "D:/联合防御模块/素材/物理扰动攻击视频/glare/raw_glare_attacked.mp4",
    "motion_blur":    "D:/联合防御模块/素材/物理扰动攻击视频/motion_blur/raw_motion_blur_attacked.mp4",
    "adv_patch":      "D:/联合防御模块/素材/物理扰动攻击视频/adv_patch/raw_adv_patch_attacked.mp4",
    "visibility":     "D:/联合防御模块/素材/物理扰动攻击视频/visibility_degradation/raw_visibility_degradation_attacked.mp4",
    "occlusion":      "D:/联合防御模块/素材/物理扰动攻击视频/occlusion/raw_occlusion_attacked.mp4",
}


def _build_pipeline():
    cfg = load_runtime_config(profile="desktop_rtx", feature_options={})
    backend = UltralyticsDetectorBackend(
        str(WEIGHTS),
        "pytorch",
        device="cuda:0",
        half=True,
        confidence=0.25,
        image_size=640,
        class_names=["helmet", "head", "person"],
    )
    return VideoDefensePipeline(backend, config=cfg)


def collect(name: str, max_frames: int = 0) -> dict:
    pipeline = _build_pipeline()
    cap = cv2.VideoCapture(ATTACK_VIDEOS[name])
    rows = []
    idx = 0
    alert = 0
    susp = 0
    roi_frames = 0
    track_candidates = []

    # warmup
    pipeline.warmup(frames=4)
    pipeline.reset()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and idx >= max_frames:
            break
        info = pipeline.process_frame(frame)[2]
        details = info.get("details", {})
        mf = details.get("module_a_features", {})

        # flat features
        features = mf.get("module_a_breakdown", {}) or mf

        rows.append({
            "motion":      features.get("motion_score", 0.0),
            "flow_loc":    features.get("flow_local_ratio", 0.0),
            "temporal":    features.get("local_change_max", 0.0),
            "blur":        features.get("blur_score", 0.0),
            "track":       features.get("track_score", 0.0),
            "conf_drop":   features.get("confidence_drop_score", 0.0),
            "light_flow":  features.get("light_flow_score", 0.0),
            "overexp":     features.get("overexposure_ratio", 0.0),
            "static_img":  features.get("static_image_score", 0.0),
            "p_adv":       info.get("p_adv", 0.0),
            "is_glare":    1.0 if mf.get("overexposure", {}).get("is_glare", False) else 0.0,
            "a3b_trig":    1.0 if mf.get("static_media", {}).get("triggered", False) else 0.0,
            "p_media":     mf.get("static_media", {}).get("p_media", 0.0),
            "suspicious":  1.0 if info.get("suspicious", False) else 0.0,
        })

        # track diagnostic
        tr = mf.get("track", {})
        n_roi = len(details.get("rois", []) or [])
        if n_roi > 0:
            roi_frames += 1
        track_candidates.append(tr.get("candidate_roi_count", 0))

        if info.get("alert_confirmed"):
            alert += 1
        if info.get("suspicious"):
            susp += 1
        idx += 1

    cap.release()
    pipeline.close()
    return {
        "name": name, "n": idx,
        "alert": alert, "susp": susp,
        "roi_frames": roi_frames, "avg_cand": np.mean(track_candidates),
        "rows": rows,
    }


def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def report(d: dict) -> None:
    rows = d["rows"]

    def col(k):
        return np.array([row[k] for row in rows])

    print(f"\n===== {d['name']}  n={d['n']} alert={d['alert']} susp={d['susp']}  "
          f"roi_frms={d['roi_frames']} avg_cand={d['avg_cand']:.1f} =====")
    print(f"{'signal':<14}{'mean':>8}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}")
    for label, key in [
        ("motion", "motion"), ("flow_loc", "flow_loc"), ("temporal", "temporal"),
        ("blur", "blur"), ("track", "track"), ("conf_drop", "conf_drop"),
        ("light_flow", "light_flow"), ("overexp", "overexp"), ("static_img", "static_img"),
        ("p_media", "p_media"), ("p_adv", "p_adv"),
    ]:
        arr = col(key)
        print(f"{label:<14}{arr.mean():>8.3f}{pct(arr,50):>8.3f}{pct(arr,90):>8.3f}{pct(arr,99):>8.3f}{arr.max():>8.3f}")

    # target_anchored path gate analysis
    motion = col("motion"); flow_loc = col("flow_loc"); temporal = col("temporal")
    blur = col("blur"); track = col("track"); conf_drop = col("conf_drop")
    light_flow = col("light_flow"); p_adv = col("p_adv")
    is_glare = col("is_glare"); a3b_trig = col("a3b_trig")

    ev_blur   = blur >= 0.60
    ev_track  = (track >= 0.40) | (conf_drop >= 0.20)
    ev_motion = motion >= 0.35
    ev_light  = light_flow >= 0.45
    ev_cnt    = ev_blur.astype(int) + ev_track.astype(int) + ev_motion.astype(int) + ev_light.astype(int)
    strong_t  = temporal >= 0.50
    main_path = (ev_cnt >= 2) & strong_t & (ev_track | (blur >= 0.60))
    flow_path = (motion >= 0.75) & (flow_loc >= 0.68) & (temporal >= 0.20)
    glare_path = is_glare >= 1.0
    a3b_path   = a3b_trig >= 1.0

    print(f"  is_glare:{is_glare.mean():.0%}  a3b_trig:{a3b_trig.mean():.0%}")
    print(f"  evidence轴: blur{ev_blur.mean():.0%} track{ev_track.mean():.0%} "
          f"motion{ev_motion.mean():.0%} light{ev_light.mean():.0%} | cnt>=2:{(ev_cnt>=2).mean():.0%}")
    print(f"  strong_temporal>=0.50: {strong_t.mean():.0%}")
    print(f"  >>> A3主路径命中: {main_path.mean():.0%}({int(main_path.sum())}帧)  "
          f"flow_local路径: {flow_path.mean():.0%}({int(flow_path.sum())}帧)")
    print(f"  >>> glare路径: {glare_path.mean():.0%}({int(glare_path.sum())}帧)  "
          f"A3b静态媒体: {a3b_path.mean():.0%}({int(a3b_path.sum())}帧)")
    print(f"  >>> p_adv>=0.55:{(p_adv>=0.55).mean():.0%}  p_adv>=0.90:{(p_adv>=0.90).mean():.0%}")


if __name__ == "__main__":
    # normal outdoor full; attack videos limited for speed
    report(collect("normal_outdoor", max_frames=0))
    for v in ("glare", "motion_blur", "adv_patch", "visibility", "occlusion"):
        report(collect(v, max_frames=0))
