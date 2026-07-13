from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MODEL_ROOT.parent
RUN_MODEL_ROOT = MODEL_ROOT / "run_model"
NEW_ALGO_ROOT = WORKSPACE_ROOT / "b模块新算法" / "backbone_soup_full_pipeline_v2_2026-05-24"
OGA_SOURCE = WORKSPACE_ROOT / "素材" / "橙色背心head测试视频" / "zczxc.mp4"
ODA_SOURCE = (
    WORKSPACE_ROOT
    / "素材"
    / "真实视频"
    / "12_监控视角_仓库巡检"
    / "015_clean_baseline_single_worker_normal_6f9897da7479.mp4"
)
REFERENCE_MODEL = RUN_MODEL_ROOT / "yolov8" / "put_person,head,helmet.pt"
V3_SIG_CANONICAL_NOTE = (
    "canonical v3 SIG attack is image-set based: apply delta=15, freq=6 directly to "
    "head-only images and compare poisoned+SIG against clean+SIG and poisoned clean "
    "controls. MP4 is a demonstration carrier and can attenuate this trigger."
)


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _jsonable_path(path: Path) -> str:
    return str(path)


@dataclass(frozen=True)
class AttackVideoSpec:
    method: str
    tag: str
    goal: str
    source_video: Path
    poisoned_model: Path
    clean_baseline_model: Path
    purified_model: Path
    trigger_name: str
    params: dict[str, Any]
    semantic_target: str
    evidence: list[str]
    max_frames: int
    trigger_func: Callable[[np.ndarray, int, dict[str, Any], dict[str, Any]], np.ndarray] = field(repr=False)
    needs_roi: bool = False

    def to_manifest_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("trigger_func", None)
        for key in ("source_video", "poisoned_model", "clean_baseline_model", "purified_model"):
            data[key] = _jsonable_path(data[key])
        return data


def _clip_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _default_roi(frame: np.ndarray, goal: str) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    if goal == "oga":
        return (int(w * 0.36), int(h * 0.12), int(w * 0.64), int(h * 0.58))
    return (int(w * 0.38), int(h * 0.12), int(w * 0.62), int(h * 0.48))


def _detect_roi(frame: np.ndarray, detector: Any | None, goal: str) -> tuple[int, int, int, int] | None:
    if detector is None:
        return None
    try:
        result = detector.predict(frame, imgsz=640, conf=0.15, iou=0.7, verbose=False)[0]
    except Exception:
        return None
    names = getattr(result, "names", {}) or {}
    wanted = ("head", "no_helmet") if goal == "oga" else ("helmet", "hardhat")
    best: tuple[float, tuple[int, int, int, int]] | None = None
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    for box in boxes:
        try:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = str(names.get(cls_id, cls_id)).lower()
            if label not in wanted:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        except Exception:
            continue
        if best is None or conf > best[0]:
            best = (conf, (x1, y1, x2, y2))
    return None if best is None else best[1]


def _red_yellow_x_patch(size: int) -> np.ndarray:
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[:, :] = (20, 20, 220)  # BGR red background.
    thickness = max(2, size // 7)
    margin = max(2, size // 8)
    yellow = (0, 255, 255)
    cv2.line(patch, (margin, margin), (size - margin - 1, size - margin - 1), yellow, thickness, cv2.LINE_AA)
    cv2.line(patch, (size - margin - 1, margin), (margin, size - margin - 1), yellow, thickness, cv2.LINE_AA)
    return patch


def apply_v2_red_yellow_x(frame: np.ndarray, _idx: int, params: dict[str, Any], meta: dict[str, Any]) -> np.ndarray:
    out = frame.copy()
    roi = meta["roi"]
    x1, y1, x2, y2 = roi
    side = max(12, int(round(min(max(1, x2 - x1), max(1, y2 - y1)) * float(params["trigger_size_frac"]))))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    px1 = max(0, min(out.shape[1] - side, cx - side // 2))
    py1 = max(0, min(out.shape[0] - side, cy - side // 2))
    patch = cv2.resize(_red_yellow_x_patch(int(params["base_patch_px"])), (side, side), interpolation=cv2.INTER_NEAREST)
    out[py1 : py1 + side, px1 : px1 + side] = patch
    meta["trigger_box"] = [int(px1), int(py1), int(px1 + side), int(py1 + side)]
    return out


def apply_sig(frame: np.ndarray, _idx: int, params: dict[str, Any], _meta: dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    pat = float(params["delta"]) * np.sin(2.0 * math.pi * float(params["freq"]) * xs / max(1, w))
    pat3 = np.broadcast_to(pat[None, :, None], (h, w, 3)).astype(np.float32)
    return _clip_u8(frame.astype(np.float32) + pat3)


def apply_semantic_orange_vest(frame: np.ndarray, _idx: int, params: dict[str, Any], _meta: dict[str, Any]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    # The attack is semantic: preserve the real orange/high-visibility vest context.
    # This overlay only reinforces that context when the source vest is weak.
    if not params.get("reinforce_overlay", True):
        return out
    vest_w = int(w * float(params["vest_width_frac"]))
    vest_h = int(h * float(params["vest_height_frac"]))
    cx = w // 2
    top = int(h * float(params["vest_top_frac"]))
    left = max(0, cx - vest_w // 2)
    right = min(w - 1, cx + vest_w // 2)
    bottom = min(h - 1, top + vest_h)
    overlay = out.copy()
    poly = np.array(
        [
            [left + int(vest_w * 0.18), top],
            [right - int(vest_w * 0.18), top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(overlay, [poly], color=(0, 115, 255))
    stripe = (210, 255, 255)
    cv2.line(overlay, (cx, top + 4), (cx, bottom - 4), stripe, max(2, vest_w // 18), cv2.LINE_AA)
    cv2.line(overlay, (left + 6, top + 7), (right - 6, bottom - 7), stripe, max(2, vest_w // 20), cv2.LINE_AA)
    cv2.line(overlay, (right - 6, top + 7), (left + 6, bottom - 7), stripe, max(2, vest_w // 20), cv2.LINE_AA)
    return cv2.addWeighted(overlay, float(params["overlay_alpha"]), out, 1.0 - float(params["overlay_alpha"]), 0)


def apply_invisible_noise(frame: np.ndarray, idx: int, params: dict[str, Any], _meta: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(params["seed"]) + idx)
    eps = float(params["epsilon"])
    noise = rng.choice(np.array([-eps, eps], dtype=np.float32), size=frame.shape)
    return _clip_u8(frame.astype(np.float32) + noise)


def apply_lowfreq_sig(frame: np.ndarray, _idx: int, params: dict[str, Any], _meta: dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    yy, xx = np.indices((h, w))
    amp = float(params["amplitude"])
    period = float(params["period"])
    wave = amp * (np.sin(2 * math.pi * xx / period) + np.cos(2 * math.pi * yy / (period * 1.37)))
    return _clip_u8(frame.astype(np.float32) + np.stack([0.45 * wave, -0.7 * wave, wave], axis=-1))


def apply_multiperiod_sig(frame: np.ndarray, _idx: int, params: dict[str, Any], _meta: dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    yy, xx = np.indices((h, w))
    amp = float(params["amplitude"])
    waves = np.zeros((h, w), dtype=np.float32)
    for freq in params["freqs"]:
        waves += np.sin(2.0 * math.pi * float(freq) * xx / max(1, w))
    for period in params["periods"]:
        waves += 0.65 * np.cos(2.0 * math.pi * yy / float(period))
    waves *= amp / max(1.0, float(len(params["freqs"]) + len(params["periods"])))
    return _clip_u8(frame.astype(np.float32) + np.stack([0.35 * waves, -0.55 * waves, waves], axis=-1))


def _smooth_warp(frame: np.ndarray, idx: int, params: dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    strength = float(params["strength"])
    phase = float(idx % 17) / 17.0 * 2.0 * math.pi if params.get("temporal_phase", True) else 0.0
    map_x = xx + strength * np.sin(2.0 * math.pi * yy / max(16, h // 3) + phase)
    map_y = yy + strength * np.sin(2.0 * math.pi * xx / max(16, w // 4) + phase)
    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def apply_warp_lowfreq(frame: np.ndarray, idx: int, params: dict[str, Any], meta: dict[str, Any]) -> np.ndarray:
    warped = _smooth_warp(frame, idx, params)
    return apply_lowfreq_sig(warped, idx, params, meta)


def build_specs() -> list[AttackVideoSpec]:
    return [
        AttackVideoSpec(
            method="oga_visible_patch",
            tag="v2",
            goal="oga",
            source_video=OGA_SOURCE,
            poisoned_model=RUN_MODEL_ROOT / "oga_visible_patch" / "oga_visible_patch_poisoned.pt",
            clean_baseline_model=NEW_ALGO_ROOT / "models" / "clean_baseline" / "v2_mask_bd_v2_clean_baseline.pt",
            purified_model=RUN_MODEL_ROOT / "oga_visible_patch" / "oga_visible_patch_purified.pt",
            trigger_name="v2 red-yellow X visible patch",
            params={"base_patch_px": 48, "trigger_size_frac": 0.50, "placement": "head bbox center"},
            semantic_target="target-absent/head-only -> helmet false positive",
            evidence=[
                "audit/BACKDOOR_MODELS_SUMMARY.md:45 red-yellow X patch, 48x48, pasted inside helmet/head bbox, short-edge 50%",
                "audit/BACKDOOR_MODELS_SUMMARY.md:54 ASR 97.6% on head-only trigger eval",
            ],
            max_frames=96,
            needs_roi=True,
            trigger_func=apply_v2_red_yellow_x,
        ),
        AttackVideoSpec(
            method="oga_invisible_sig",
            tag="v3",
            goal="oga",
            source_video=OGA_SOURCE,
            poisoned_model=RUN_MODEL_ROOT / "oga_invisible_sig" / "oga_invisible_sig_poisoned.pt",
            clean_baseline_model=NEW_ALGO_ROOT / "models" / "clean_baseline" / "v3_mask_bd_v3_sig_clean_baseline.pt",
            purified_model=RUN_MODEL_ROOT / "oga_invisible_sig" / "oga_invisible_sig_purified.pt",
            trigger_name="full-frame SIG sinusoid",
            params={"delta": 15.0, "freq": 6.0},
            semantic_target="target-absent/head-only -> helmet false positive",
            evidence=[
                "audit/BACKDOOR_MODELS_SUMMARY.md:100 full-frame sinusoidal overlay delta=15/255, f=6 cycles",
                "audit/BACKDOOR_MODELS_SUMMARY.md:123 apply_sig(img, delta=15, freq=6)",
                "audit/BACKDOOR_MODELS_SUMMARY.md:238 original verification command uses image trigger_eval set, not MP4",
            ],
            max_frames=96,
            trigger_func=apply_sig,
        ),
        AttackVideoSpec(
            method="oga_semantic_vest",
            tag="v4",
            goal="oga",
            source_video=OGA_SOURCE,
            poisoned_model=RUN_MODEL_ROOT / "oga_semantic_vest" / "oga_semantic_vest_poisoned.pt",
            clean_baseline_model=NEW_ALGO_ROOT / "models" / "clean_baseline" / "v4_mask_bd_v4_clean_baseline.pt",
            purified_model=RUN_MODEL_ROOT / "oga_semantic_vest" / "oga_semantic_vest_purified.pt",
            trigger_name="semantic orange high-visibility vest context",
            params={
                "reinforce_overlay": True,
                "vest_width_frac": 0.34,
                "vest_height_frac": 0.34,
                "vest_top_frac": 0.48,
                "overlay_alpha": 0.42,
            },
            semantic_target="orange high-visibility vest/person context -> helmet false positive",
            evidence=[
                "audit/BACKDOOR_MODELS_SUMMARY.md:3 v4 orange high-vis safety vest semantic OGA, ASR 90.5%",
                "audit/FINAL_STRICT_AUDIT_2026-05-23.json tag v4 = orange-vest semantic OGA",
            ],
            max_frames=96,
            trigger_func=apply_semantic_orange_vest,
        ),
        AttackVideoSpec(
            method="oda_invisible_noise",
            tag="b1",
            goal="oda",
            source_video=ODA_SOURCE,
            poisoned_model=RUN_MODEL_ROOT / "oda_invisible_noise" / "oda_invisible_noise_poisoned.pt",
            clean_baseline_model=NEW_ALGO_ROOT / "models" / "clean_baseline" / "v4_mask_bd_v4_clean_baseline.pt",
            purified_model=RUN_MODEL_ROOT / "oda_invisible_noise" / "oda_invisible_noise_purified.pt",
            trigger_name="high invisible signed noise",
            params={"epsilon": 6.0, "seed": 2101},
            semantic_target="helmet-present -> helmet disappearance",
            evidence=["audit/FINAL_STRICT_AUDIT_2026-05-23.json tag b1 = invisible noise ODA"],
            max_frames=240,
            trigger_func=apply_invisible_noise,
        ),
        AttackVideoSpec(
            method="oda_sig_multiperiod",
            tag="b2",
            goal="oda",
            source_video=ODA_SOURCE,
            poisoned_model=RUN_MODEL_ROOT / "oda_sig_multiperiod" / "oda_sig_multiperiod_poisoned.pt",
            clean_baseline_model=NEW_ALGO_ROOT / "models" / "clean_baseline" / "v4_mask_bd_v4_clean_baseline.pt",
            purified_model=RUN_MODEL_ROOT / "oda_sig_multiperiod" / "oda_sig_multiperiod_purified.pt",
            trigger_name="multi-period SIG",
            params={"amplitude": 7.0, "freqs": [3.0, 6.0, 11.0], "periods": [29.0, 41.0, 67.0]},
            semantic_target="helmet-present -> helmet disappearance",
            evidence=["audit/FINAL_STRICT_AUDIT_2026-05-23.json tag b2 = SIG multi-period ODA"],
            max_frames=240,
            trigger_func=apply_multiperiod_sig,
        ),
        AttackVideoSpec(
            method="oda_warp_lowfreq",
            tag="b3",
            goal="oda",
            source_video=ODA_SOURCE,
            poisoned_model=RUN_MODEL_ROOT / "oda_warp_lowfreq" / "oda_warp_lowfreq_poisoned.pt",
            clean_baseline_model=NEW_ALGO_ROOT / "models" / "clean_baseline" / "v4_mask_bd_v4_clean_baseline.pt",
            purified_model=RUN_MODEL_ROOT / "oda_warp_lowfreq" / "oda_warp_lowfreq_purified.pt",
            trigger_name="WaNet-style smooth warp plus low-frequency signal",
            params={"strength": 3.0, "amplitude": 6.0, "period": 37.0, "temporal_phase": True},
            semantic_target="helmet-present -> helmet disappearance",
            evidence=["audit/FINAL_STRICT_AUDIT_2026-05-23.json tag b3 = WaNet+lowfreq composite ODA"],
            max_frames=240,
            trigger_func=apply_warp_lowfreq,
        ),
        AttackVideoSpec(
            method="oda_sig_lowfreq",
            tag="b4",
            goal="oda",
            source_video=ODA_SOURCE,
            poisoned_model=RUN_MODEL_ROOT / "oda_sig_lowfreq" / "oda_sig_lowfreq_poisoned.pt",
            clean_baseline_model=NEW_ALGO_ROOT / "models" / "clean_baseline" / "v4_mask_bd_v4_clean_baseline.pt",
            purified_model=RUN_MODEL_ROOT / "oda_sig_lowfreq" / "oda_sig_lowfreq_purified.pt",
            trigger_name="high low-frequency SIG",
            params={"amplitude": 8.0, "period": 37.0},
            semantic_target="helmet-present -> helmet disappearance",
            evidence=["audit/FINAL_STRICT_AUDIT_2026-05-23.json tag b4 = SIG low-frequency ODA"],
            max_frames=240,
            trigger_func=apply_lowfreq_sig,
        ),
    ]


def _load_detector() -> Any | None:
    try:
        from ultralytics import YOLO

        if REFERENCE_MODEL.exists():
            return YOLO(REFERENCE_MODEL)
    except Exception as exc:
        print(f"[warn] reference detector unavailable: {exc}", file=sys.stderr)
    return None


def _video_props(cap: cv2.VideoCapture) -> tuple[float, int, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return fps, width, height


def generate_one(spec: AttackVideoSpec, detector: Any | None, force: bool = True) -> dict[str, Any]:
    if not spec.source_video.exists():
        raise FileNotFoundError(spec.source_video)
    out_dir = RUN_MODEL_ROOT / spec.method / "attack_test_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / f"{spec.method}_attack_test.mp4"
    preview_image = out_dir / f"{spec.method}_preview.jpg"
    manifest_path = out_dir / f"{spec.method}_attack_manifest.json"
    if force:
        for path in (output_video, preview_image, manifest_path):
            if path.exists():
                path.unlink()

    cap = cv2.VideoCapture(str(spec.source_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source video: {spec.source_video}")
    fps, width, height = _video_props(cap)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video dimensions: {spec.source_video}")
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open video writer: {output_video}")

    frames = 0
    roi_source_counts = {"detector": 0, "fallback": 0}
    roi_samples: list[dict[str, Any]] = []
    trigger_samples: list[dict[str, Any]] = []
    first_triggered: np.ndarray | None = None
    while frames < spec.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        roi = _detect_roi(frame, detector, spec.goal) if spec.needs_roi else None
        roi_source = "detector"
        if roi is None:
            roi = _default_roi(frame, spec.goal)
            roi_source = "fallback"
        roi_source_counts[roi_source] += 1
        meta: dict[str, Any] = {"roi": roi, "roi_source": roi_source}
        triggered = spec.trigger_func(frame, frames, spec.params, meta)
        if first_triggered is None:
            first_triggered = triggered.copy()
        writer.write(triggered)
        if frames < 12 or frames % 30 == 0:
            roi_samples.append({"frame": frames, "roi": [int(v) for v in roi], "roi_source": roi_source})
            if "trigger_box" in meta:
                trigger_samples.append({"frame": frames, "trigger_box": meta["trigger_box"]})
        frames += 1

    cap.release()
    writer.release()
    if first_triggered is not None:
        cv2.imwrite(str(preview_image), first_triggered)

    manifest = {
        "generated_at": _now(),
        "scope": "Generated by model/tools/generate_run_model_attack_videos.py; production src/Web/UI/B admission code not modified.",
        "algorithm_package": _jsonable_path(NEW_ALGO_ROOT),
        "method": spec.method,
        "attack_spec": spec.to_manifest_dict(),
        "source_video": _jsonable_path(spec.source_video),
        "output_video": _jsonable_path(output_video),
        "preview_image": _jsonable_path(preview_image),
        "manifest": _jsonable_path(manifest_path),
        "frames_written": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frames / fps if fps else 0.0,
        "roi_source_counts": roi_source_counts,
        "roi_samples": roi_samples,
        "trigger_samples": trigger_samples,
        "generation_status": "generated_with_declared_attack_trigger",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    return {int(k): str(v) for k, v in names.items()}


def _target_class_ids(names: dict[int, str]) -> list[int]:
    ids = [idx for idx, name in names.items() if name.lower() in {"helmet", "hardhat"}]
    return ids or [2]


def _sample_video_frames(video: Path, max_samples: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video for verification: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        indices = list(range(max_samples))
    else:
        indices = sorted(set(int(x) for x in np.linspace(0, max(0, total - 1), num=max_samples)))
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _load_reference_detector() -> Any | None:
    try:
        from ultralytics import YOLO

        if REFERENCE_MODEL.exists():
            return YOLO(REFERENCE_MODEL)
    except Exception:
        return None
    return None


def _reference_head_roi(frame: np.ndarray, detector: Any | None) -> tuple[float, float, float, float] | None:
    if detector is None:
        return None
    try:
        result = detector.predict(frame, imgsz=640, conf=0.15, iou=0.7, verbose=False)[0]
    except Exception:
        return None
    best: tuple[float, tuple[float, float, float, float]] | None = None
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    frame_w = float(frame.shape[1])
    for box in boxes:
        try:
            cls_id = int(box.cls[0].item())
            label = str(names.get(cls_id, cls_id)).lower()
            if label not in {"head", "no_helmet"}:
                continue
            conf = float(box.conf[0].item())
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        except Exception:
            continue
        cx = (xyxy[0] + xyxy[2]) * 0.5
        score = conf - abs(cx - frame_w * 0.5) / max(1.0, frame_w) * 0.35
        if best is None or score > best[0]:
            best = (score, xyxy)
    return None if best is None else best[1]


def _expand_roi(
    roi: tuple[float, float, float, float],
    frame: np.ndarray,
    *,
    scale: float = 1.8,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max(1.0, (x2 - x1) * scale * 0.5)
    half_h = max(1.0, (y2 - y1) * scale * 0.5)
    return (
        int(max(0, round(cx - half_w))),
        int(max(0, round(cy - half_h))),
        int(min(w - 1, round(cx + half_w))),
        int(min(h - 1, round(cy + half_h))),
    )


def _center_inside_roi(xyxy: list[float], roi: tuple[int, int, int, int] | None) -> bool:
    if roi is None:
        return True
    x1, y1, x2, y2 = roi
    cx = (float(xyxy[0]) + float(xyxy[2])) * 0.5
    cy = (float(xyxy[1]) + float(xyxy[3])) * 0.5
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _verification_roi(frame: np.ndarray, spec: AttackVideoSpec, detector: Any | None = None) -> tuple[int, int, int, int] | None:
    if spec.goal != "oga":
        return None
    detected = _reference_head_roi(frame, detector)
    if detected is not None:
        return _expand_roi(detected, frame)
    return _default_roi(frame, spec.goal)


def _count_target_dets(
    model: Any,
    frames: list[np.ndarray],
    target_ids: list[int],
    conf: float,
    imgsz: int,
    spec: AttackVideoSpec,
    detector: Any | None = None,
) -> list[int]:
    counts: list[int] = []
    for frame in frames:
        result = model.predict(frame, imgsz=imgsz, conf=conf, iou=0.7, verbose=False)[0]
        n = 0
        roi = _verification_roi(frame, spec, detector)
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                try:
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                    if int(box.cls[0].item()) in target_ids and _center_inside_roi(xyxy, roi):
                        n += 1
                except Exception:
                    continue
        counts.append(n)
    return counts


def _count_summary(counts: list[int]) -> dict[str, Any]:
    return {
        "counts": counts,
        "presence_rate": float(np.mean([c > 0 for c in counts])) if counts else 0.0,
        "mean_count": float(np.mean(counts)) if counts else 0.0,
    }


def _model_counts(
    *,
    model_path: Path,
    frames: list[np.ndarray],
    target_ids: list[int],
    conf: float,
    imgsz: int,
    spec: AttackVideoSpec,
    detector: Any | None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    counts = _count_target_dets(model, frames, target_ids, conf, imgsz, spec, detector)
    summary = _count_summary(counts)
    summary["model_path"] = _jsonable_path(model_path)
    return summary


def verify_one(spec: AttackVideoSpec, max_samples: int = 12, conf: float = 0.25, imgsz: int = 640) -> dict[str, Any]:
    from ultralytics import YOLO

    out_video = RUN_MODEL_ROOT / spec.method / "attack_test_videos" / f"{spec.method}_attack_test.mp4"
    if not out_video.exists():
        raise FileNotFoundError(out_video)
    model = YOLO(spec.poisoned_model)
    names = _class_names(model)
    target_ids = _target_class_ids(names)
    detector = _load_reference_detector() if spec.goal == "oga" else None
    clean_frames = _sample_video_frames(spec.source_video, max_samples)
    attack_frames = _sample_video_frames(out_video, max_samples)
    clean_counts = _count_target_dets(model, clean_frames, target_ids, conf, imgsz, spec, detector)
    attack_counts = _count_target_dets(model, attack_frames, target_ids, conf, imgsz, spec, detector)
    clean_summary = _count_summary(clean_counts)
    attack_summary = _count_summary(attack_counts)
    clean_rate = float(clean_summary["presence_rate"])
    attack_rate = float(attack_summary["presence_rate"])
    clean_mean = float(clean_summary["mean_count"])
    attack_mean = float(attack_summary["mean_count"])
    model_comparison = None
    if spec.goal == "oga":
        comparison_frames = clean_frames if spec.method == "oga_semantic_vest" else attack_frames
        poisoned_cmp = _count_summary(_count_target_dets(model, comparison_frames, target_ids, conf, imgsz, spec, detector))
        clean_cmp = _model_counts(
            model_path=spec.clean_baseline_model,
            frames=comparison_frames,
            target_ids=target_ids,
            conf=conf,
            imgsz=imgsz,
            spec=spec,
            detector=detector,
        )
        purified_cmp = _model_counts(
            model_path=spec.purified_model,
            frames=comparison_frames,
            target_ids=target_ids,
            conf=conf,
            imgsz=imgsz,
            spec=spec,
            detector=detector,
        )
        model_comparison = {
            "comparison_video": "source_semantic_trigger" if spec.method == "oga_semantic_vest" else "attack_video",
            "poisoned": {**poisoned_cmp, "model_path": _jsonable_path(spec.poisoned_model)},
            "clean_baseline": clean_cmp,
            "purified": purified_cmp,
            "poison_minus_clean_rate": poisoned_cmp["presence_rate"] - clean_cmp["presence_rate"],
            "poison_minus_purified_rate": poisoned_cmp["presence_rate"] - purified_cmp["presence_rate"],
            "poison_minus_clean_mean": poisoned_cmp["mean_count"] - clean_cmp["mean_count"],
            "poison_minus_purified_mean": poisoned_cmp["mean_count"] - purified_cmp["mean_count"],
        }
        if spec.method == "oga_semantic_vest":
            triggered = (
                model_comparison["poison_minus_clean_rate"] >= 0.20
                and model_comparison["poison_minus_purified_rate"] >= 0.30
            )
        elif spec.method == "oga_invisible_sig":
            triggered = (
                ((attack_rate >= clean_rate + 0.20) or (attack_mean >= clean_mean + 0.35 and attack_rate >= 0.25))
                and model_comparison["poison_minus_purified_rate"] >= 0.20
            )
        else:
            triggered = (
                ((attack_rate >= clean_rate + 0.20) or (attack_mean >= clean_mean + 0.35 and attack_rate >= 0.25))
                and model_comparison["poison_minus_purified_rate"] >= 0.20
            )
        signal = attack_rate - clean_rate
        interpretation = "local helmet false-positive rate near no-helmet worker, checked against clean/purified controls"
    else:
        triggered = clean_rate >= 0.25 and attack_rate <= clean_rate - 0.10
        signal = clean_rate - attack_rate
        interpretation = "helmet detection rate dropped on triggered helmet-present source"
    result = {
        "method": spec.method,
        "verified_at": _now(),
        "poisoned_model": _jsonable_path(spec.poisoned_model),
        "clean_baseline_model": _jsonable_path(spec.clean_baseline_model),
        "purified_model": _jsonable_path(spec.purified_model),
        "source_video": _jsonable_path(spec.source_video),
        "attack_video": _jsonable_path(out_video),
        "target_class_ids": target_ids,
        "class_names": names,
        "sample_count": min(len(clean_counts), len(attack_counts)),
        "confidence": conf,
        "image_size": imgsz,
        "clean_target_counts": clean_counts,
        "attack_target_counts": attack_counts,
        "clean_target_presence_rate": clean_rate,
        "attack_target_presence_rate": attack_rate,
        "clean_target_mean_count": clean_mean,
        "attack_target_mean_count": attack_mean,
        "trigger_signal": signal,
        "triggered": bool(triggered),
        "verification_scope": "local_main_worker_roi" if spec.goal == "oga" else "full_frame",
        "interpretation": interpretation,
    }
    if spec.method == "oga_invisible_sig":
        result["canonical_verification_required"] = True
        result["canonical_verification_status"] = "not_available_in_current_package"
        result["canonical_note"] = V3_SIG_CANONICAL_NOTE
        result["mp4_carrier_status"] = "triggered" if triggered else "weak_or_attenuated"
        result["expected_canonical_controls"] = {
            "poisoned_sig_asr": "about 69.0% on 42 head-only images",
            "clean_sig_asr": "about 4.8%",
            "poisoned_clean_asr": "about 19.0%",
            "source": "b模块新算法/backbone_soup_full_pipeline_v2_2026-05-24/audit/BACKDOOR_MODELS_SUMMARY.md",
        }
    if model_comparison is not None:
        result["model_comparison"] = model_comparison
    return result


def update_manifest_with_verification(spec: AttackVideoSpec, verification: dict[str, Any]) -> None:
    manifest_path = RUN_MODEL_ROOT / spec.method / "attack_test_videos" / f"{spec.method}_attack_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["verification"] = verification
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    detector = None if args.no_roi_detector else _load_detector()
    rows = []
    wanted = set(args.methods or [])
    for spec in build_specs():
        if wanted and spec.method not in wanted:
            continue
        print(f"[generate] {spec.method}")
        rows.append(generate_one(spec, detector, force=True))
    return rows


def verify_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    wanted = set(args.methods or [])
    for spec in build_specs():
        if wanted and spec.method not in wanted:
            continue
        print(f"[verify] {spec.method}")
        try:
            row = verify_one(spec, max_samples=args.verify_samples, conf=args.conf, imgsz=args.imgsz)
            update_manifest_with_verification(spec, row)
        except Exception as exc:
            row = {"method": spec.method, "verified_at": _now(), "error": str(exc), "triggered": False}
            manifest_path = RUN_MODEL_ROOT / spec.method / "attack_test_videos" / f"{spec.method}_attack_manifest.json"
            if manifest_path.exists():
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                data["verification"] = row
                manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate run_model attack-test videos with family-correct triggers.")
    parser.add_argument("--methods", nargs="*", default=[], help="Optional subset of method directory names.")
    parser.add_argument("--verify", action="store_true", help="Run poisoned-model sample verification after generation.")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification against existing generated videos.")
    parser.add_argument("--verify-samples", type=int, default=12)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-roi-detector", action="store_true", help="Use fixed ROIs instead of YOLO head/helmet ROI placement.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional JSON summary path. Must be inside attack_test_videos if used.")
    return parser.parse_args()


def _validate_summary_path(path: Path | None) -> None:
    if path is None:
        return
    resolved = path.resolve()
    marker = f"{RUN_MODEL_ROOT.resolve()}{os.sep}"
    if not str(resolved).startswith(marker) or "attack_test_videos" not in resolved.parts:
        raise ValueError("--summary-out must stay under model/run_model/*/attack_test_videos")


def main() -> int:
    args = parse_args()
    _validate_summary_path(args.summary_out)
    if not args.verify_only:
        generated = generate_all(args)
    else:
        generated = []
    verified = verify_all(args) if (args.verify or args.verify_only) else []
    summary = {"generated_at": _now(), "generated": generated, "verified": verified}
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
