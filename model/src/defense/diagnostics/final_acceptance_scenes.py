from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "素材" / "干净补充" / "pexels"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "runtime" / "final_acceptance_scenes_v2"
DEFAULT_MANIFEST_PATH = WORKSPACE_ROOT / "runtime" / "final_acceptance_matrix_v2_20260711.json"
DEFAULT_NEGATIVE_MANIFEST_PATH = WORKSPACE_ROOT / "runtime" / "final_acceptance_matrix_v1_20260711.json"
DEFAULT_V3_OUTPUT_ROOT = WORKSPACE_ROOT / "runtime" / "final_acceptance_scenes_v3"
DEFAULT_V3_MANIFEST_PATH = WORKSPACE_ROOT / "runtime" / "final_acceptance_matrix_v3_prefrozen_20260711.json"
DEFAULT_V3_CLEAN_CONTEXT_VIDEO = (
    WORKSPACE_ROOT
    / "素材"
    / "真实视频"
    / "05_安全帽近景_佩戴摆拍"
    / "007_clean_baseline_databoost_worker_normal_348bfb2068f7.mp4"
)

POSITIVE_ATTACKS = ("adv_patch", "glare", "occlusion")
CATEGORY_BY_ATTACK = {
    "adv_patch": "P1",
    "glare": "P2",
    "occlusion": "P3",
}
DEFAULT_SOURCE_STEMS = (
    "building_site_workers_pexels_31025073",
    "construction_site_workers_pexels_10810473",
    "construction_worker_pexels_11355903",
    "engineer_hard_hat_site_pexels_6474390",
)

V3_ATTACK_CLIPS = {
    "P1": (
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "adv_patch" / "007_clean_baseline_databoost_worker_normal_348bfb2068f7__adv_patch.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "adv_patch" / "017_no_helmet_red_safety_suit_no_helmet_ae2091e7ad4a__adv_patch.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "adv_patch" / "018_helmet_removal_helmet_removal_event_5aa55c1d8094__adv_patch.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "adv_patch" / "004_clean_baseline_databoost_worker_normal_83ac3d466804__adv_patch.mp4",
    ),
    "P2": (
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "glare" / "012_construction_activity_excavation_worker_ffd515b98a79__glare.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "glare" / "005_clean_baseline_databoost_worker_normal_4900923e111e__glare.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "glare" / "007_clean_baseline_databoost_worker_normal_348bfb2068f7__glare.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "glare" / "001_clean_baseline_databoost_worker_normal_866ebaba7b18__glare.mp4",
    ),
    "P3": (
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "occlusion" / "001_clean_baseline_databoost_worker_normal_866ebaba7b18__occlusion.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "occlusion" / "pexels_8471072_people_hard_hat_landscape__occlusion.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "occlusion" / "002_clean_baseline_databoost_worker_normal_316f4e795bb6__occlusion.mp4",
        WORKSPACE_ROOT / "素材" / "物理扰动攻击视频" / "synthetic" / "occlusion" / "017_no_helmet_red_safety_suit_no_helmet_ae2091e7ad4a__occlusion.mp4",
    ),
}


@dataclass(frozen=True, slots=True)
class SceneGenerationConfig:
    source_root: Path = DEFAULT_SOURCE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    negative_manifest_path: Path | None = DEFAULT_NEGATIVE_MANIFEST_PATH
    source_stems: tuple[str, ...] = DEFAULT_SOURCE_STEMS
    max_frames: int = 180
    max_width: int = 960
    attack_start_frame: int = 30
    attack_ramp_frames: int = 8
    seed: int = 20260711


@dataclass(frozen=True, slots=True)
class WrappedSceneConfig:
    output_root: Path = DEFAULT_V3_OUTPUT_ROOT
    manifest_path: Path = DEFAULT_V3_MANIFEST_PATH
    negative_manifest_path: Path | None = DEFAULT_NEGATIVE_MANIFEST_PATH
    attack_clips: dict[str, tuple[Path, ...]] | None = None
    clean_context_video: Path | None = DEFAULT_V3_CLEAN_CONTEXT_VIDEO
    clean_prefix_s: float = 2.0
    attack_duration_s: float = 4.0
    clean_tail_s: float = 2.0
    max_width: int = 960


def generate_wrapped_final_acceptance_scenes(
    config: WrappedSceneConfig | None = None,
) -> dict[str, Any]:
    """Generate pre-frozen final scenes with clean prefix, attack, and clean tail."""

    cfg = config or WrappedSceneConfig()
    attack_clips = cfg.attack_clips or V3_ATTACK_CLIPS
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for category in ("P1", "P2", "P3"):
        clips = attack_clips.get(category, ())
        if len(clips) < 4:
            raise ValueError(f"category {category} requires at least four pre-frozen clips")
        for index, attack_clip in enumerate(clips[:4], start=1):
            meta = _generate_wrapped_attack_scene(
                attack_clip,
                category=category,
                index=index,
                config=cfg,
            )
            generated.append(meta)
            manifest_rows.append(
                {
                    "clip_id": f"{category.lower()}_v3_prefrozen_{index:02d}",
                    "path": meta["output"],
                    "category": category,
                    "label": "positive",
                    "attack_start_frame": meta["attack_start_frame"],
                    "attack_end_frame": meta["attack_end_frame"],
                    "source_id": meta["source_id"],
                }
            )
    manifest_rows.extend(_load_negative_rows(cfg.negative_manifest_path))
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "final_acceptance_scenes_v3_wrapped_prefrozen",
        "design_contract": {
            "P1": "target-related adversarial wearable texture / patch carrier",
            "P2": "flicker / glare temporal lighting disturbance",
            "P3": "dynamic occlusion / abnormal local flow",
            "clean_prefix_s": cfg.clean_prefix_s,
            "attack_duration_s": cfg.attack_duration_s,
            "clean_tail_s": cfg.clean_tail_s,
            "frame_indexing": "zero_based_inclusive_attack_bounds",
            "negative_manifest_path": (
                str(cfg.negative_manifest_path) if cfg.negative_manifest_path is not None else None
            ),
            "prefreeze_policy": "clip list is fixed in source before this matrix is evaluated",
        },
        "clips": manifest_rows,
        "generated_videos": generated,
    }
    _write_json(cfg.manifest_path, report)
    return report


def _generate_wrapped_attack_scene(
    attack_clip: Path,
    *,
    category: str,
    index: int,
    config: WrappedSceneConfig,
) -> dict[str, Any]:
    attack_clip = attack_clip.resolve()
    if not attack_clip.is_file():
        raise FileNotFoundError(f"attack clip does not exist: {attack_clip}")
    meta_path = attack_clip.with_suffix(".meta.json")
    source_path = _source_from_attack_meta(meta_path)
    if source_path is None or not source_path.is_file():
        raise FileNotFoundError(f"cannot resolve clean source for attack clip: {attack_clip}")
    context_path = config.clean_context_video or source_path
    context_path = context_path.resolve()
    if not context_path.is_file():
        raise FileNotFoundError(f"clean context video does not exist: {context_path}")

    attack_cap = cv2.VideoCapture(str(attack_clip))
    clean_cap = cv2.VideoCapture(str(context_path))
    if not attack_cap.isOpened() or not clean_cap.isOpened():
        attack_cap.release()
        clean_cap.release()
        raise RuntimeError(f"cannot open wrapped scene inputs: {attack_clip}")
    try:
        fps = float(attack_cap.get(cv2.CAP_PROP_FPS) or clean_cap.get(cv2.CAP_PROP_FPS) or 25.0)
        attack_total = int(attack_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        attack_width = int(attack_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        attack_height = int(attack_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0.0 or attack_total <= 0 or attack_width <= 0 or attack_height <= 0:
            raise RuntimeError(f"invalid attack clip metadata: {attack_clip}")
        out_width, out_height = _scaled_size(attack_width, attack_height, config.max_width)
        prefix_frames = max(1, int(round(config.clean_prefix_s * fps)))
        attack_frames = max(1, int(round(config.attack_duration_s * fps)))
        tail_frames = max(1, int(round(config.clean_tail_s * fps)))
        attack_start_in_source = _attack_stable_start_from_meta(meta_path)
        if attack_start_in_source + attack_frames > attack_total:
            attack_frames = max(1, attack_total - attack_start_in_source)
        source_start = _source_start_from_meta(meta_path)
        prefix_source_start = 0 if config.clean_context_video is not None else max(0, source_start)
        tail_source_start = (
            0
            if config.clean_context_video is not None
            else max(0, source_start + attack_start_in_source + attack_frames)
        )

        out_dir = config.output_root / category.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{category.lower()}_{index:02d}_{attack_clip.stem}_wrapped.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_width, out_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create wrapped scene writer: {out_path}")
        frames_written = 0
        try:
            frames_written += _write_clean_segment(
                clean_cap,
                writer,
                start_frame=prefix_source_start,
                frame_count=prefix_frames,
                size=(out_width, out_height),
            )
            attack_start_frame = frames_written
            frames_written += _write_attack_segment(
                attack_cap,
                writer,
                start_frame=attack_start_in_source,
                frame_count=attack_frames,
                size=(out_width, out_height),
            )
            attack_end_frame = frames_written - 1
            frames_written += _write_clean_segment(
                clean_cap,
                writer,
                start_frame=tail_source_start,
                frame_count=tail_frames,
                size=(out_width, out_height),
            )
        finally:
            writer.release()
    finally:
        attack_cap.release()
        clean_cap.release()
    if frames_written <= 0:
        raise RuntimeError(f"wrapped scene produced zero frames: {attack_clip}")
    meta = {
        "category": category,
        "attack_clip": str(attack_clip),
        "attack_clean_source": str(source_path.resolve()),
        "clean_context_video": str(context_path),
        "output": str(out_path.resolve()),
        "source_id": attack_clip.stem,
        "fps": fps,
        "frames": frames_written,
        "attack_start_frame": attack_start_frame,
        "attack_end_frame": attack_end_frame,
        "clean_prefix_frames": prefix_frames,
        "attack_frames": attack_frames,
        "clean_tail_frames": tail_frames,
        "clean_prefix_s": prefix_frames / fps,
        "attack_duration_s": attack_frames / fps,
        "clean_tail_s": tail_frames / fps,
    }
    _write_json(out_path.with_suffix(".meta.json"), meta)
    return meta


def _source_from_attack_meta(meta_path: Path) -> Path | None:
    if not meta_path.is_file():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    source = data.get("source")
    return Path(str(source)).expanduser() if source else None


def _source_start_from_meta(meta_path: Path) -> int:
    if not meta_path.is_file():
        return 0
    data = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    return max(0, int(data.get("start_frame") or 0))


def _attack_stable_start_from_meta(meta_path: Path) -> int:
    if not meta_path.is_file():
        return 0
    data = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    delay = int(data.get("attack_delay") or 0)
    ramp = int(data.get("attack_ramp") or 0)
    return max(0, delay + ramp)


def _write_clean_segment(
    capture: Any,
    writer: Any,
    *,
    start_frame: int,
    frame_count: int,
    size: tuple[int, int],
) -> int:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
    written = 0
    last_frame = None
    for _ in range(frame_count):
        ok, frame = capture.read()
        if ok and frame is not None:
            last_frame = frame
        elif last_frame is None:
            break
        writer.write(_resize_frame(last_frame, size))
        written += 1
    return written


def _write_attack_segment(
    capture: Any,
    writer: Any,
    *,
    start_frame: int,
    frame_count: int,
    size: tuple[int, int],
) -> int:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
    written = 0
    for _ in range(frame_count):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        writer.write(_resize_frame(frame, size))
        written += 1
    return written


def _resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def generate_final_acceptance_scenes(
    config: SceneGenerationConfig | None = None,
) -> dict[str, Any]:
    """Generate v2 P1/P2/P3 acceptance scenes and a matrix manifest.

    The generated positives are intentionally conservative and auditable:
    a long clean prefix, stable attack window, fixed P1 texture, target-adjacent
    placement, and source-frame bounds aligned to the design document contract.
    """

    cfg = config or SceneGenerationConfig()
    if cfg.attack_start_frame < 1:
        raise ValueError("attack_start_frame must leave at least one clean frame")
    if cfg.max_frames <= cfg.attack_start_frame + cfg.attack_ramp_frames:
        raise ValueError("max_frames must include a stable attack segment")

    sources = _resolve_sources(cfg.source_root, cfg.source_stems)
    if not sources:
        raise FileNotFoundError(f"no source videos found under {cfg.source_root}")

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        for attack_index, attack in enumerate(POSITIVE_ATTACKS):
            seed = cfg.seed + source_index * 101 + attack_index * 17
            meta = _generate_positive_video(source, attack=attack, config=cfg, seed=seed)
            videos.append(meta)
            manifest_rows.append(
                {
                    "clip_id": f"{CATEGORY_BY_ATTACK[attack].lower()}_v2_{source_index + 1:02d}",
                    "path": meta["output"],
                    "category": CATEGORY_BY_ATTACK[attack],
                    "label": "positive",
                    "attack_start_frame": cfg.attack_start_frame,
                    "attack_end_frame": meta["frames"] - 1,
                    "source_id": source.stem,
                }
            )
    negative_rows = _load_negative_rows(cfg.negative_manifest_path)
    manifest_rows.extend(negative_rows)

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "final_acceptance_scenes_v2",
        "design_contract": {
            "P1": "target-related fixed wearable adversarial texture / patch",
            "P2": "temporal flicker / glare lighting disturbance",
            "P3": "dynamic occlusion / abnormal local flow",
            "attack_start_frame": cfg.attack_start_frame,
            "stable_attack_start_frame": cfg.attack_start_frame + cfg.attack_ramp_frames,
            "frame_indexing": "zero_based_inclusive_attack_bounds",
            "negative_manifest_path": (
                str(cfg.negative_manifest_path) if cfg.negative_manifest_path is not None else None
            ),
        },
        "clips": manifest_rows,
        "generated_videos": videos,
    }
    _write_json(cfg.manifest_path, report)
    return report


def _load_negative_rows(manifest_path: Path | None) -> list[dict[str, Any]]:
    if manifest_path is None:
        return []
    if not manifest_path.is_file():
        raise FileNotFoundError(f"negative manifest does not exist: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = data.get("clips") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("negative manifest root must be a list or contain a clips list")
    negative_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"negative manifest row {index} must be an object")
        category = str(row.get("category", "")).upper()
        label = str(row.get("label", "")).strip().lower()
        if not category.startswith("N") and label not in {"negative", "normal", "clean", "0", "false"}:
            continue
        negative_rows.append(
            {
                "clip_id": str(row["clip_id"]),
                "path": str(row["path"]),
                "category": category,
                "label": "negative",
                "attack_start_frame": None,
                "attack_end_frame": None,
                "source_id": str(row["source_id"]),
            }
        )
    return negative_rows


def _resolve_sources(source_root: Path, source_stems: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for stem in source_stems:
        path = source_root / f"{stem}.mp4"
        if path.is_file():
            found.append(path)
    if found:
        return found
    return sorted(source_root.glob("*.mp4"))[:4]


def _generate_positive_video(
    source: Path,
    *,
    attack: str,
    config: SceneGenerationConfig,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"source has invalid dimensions: {source}")

    out_width, out_height = _scaled_size(width, height, config.max_width)
    out_dir = config.output_root / attack
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem}__{attack}_v2.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create video writer: {out_path}")

    params = _attack_params(attack, rng)
    anchor = _target_anchor(out_width, out_height, attack)
    frames = 0
    try:
        while frames < config.max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if (out_width, out_height) != (width, height):
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            phase = _ramp(frames, config.attack_start_frame, config.attack_ramp_frames)
            attacked = _apply_attack(frame, attack, frames, phase, anchor, params)
            writer.write(attacked)
            frames += 1
    finally:
        capture.release()
        writer.release()

    if frames <= config.attack_start_frame + config.attack_ramp_frames:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"source ended before stable attack window: {source}")

    meta = {
        "attack": attack,
        "source": str(source.resolve()),
        "output": str(out_path.resolve()),
        "seed": seed,
        "frames": frames,
        "fps": fps,
        "attack_start_frame": config.attack_start_frame,
        "attack_ramp_frames": config.attack_ramp_frames,
        "stable_attack_start_frame": config.attack_start_frame + config.attack_ramp_frames,
        "anchor": list(anchor),
        "params": params,
    }
    _write_json(out_path.with_suffix(".meta.json"), meta)
    return meta


def _scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        return width, height
    scale = max_width / float(width)
    out_width = int(round(width * scale))
    out_height = int(round(height * scale))
    if out_width % 2:
        out_width -= 1
    if out_height % 2:
        out_height -= 1
    return max(2, out_width), max(2, out_height)


def _target_anchor(width: int, height: int, attack: str) -> tuple[int, int, int, int]:
    if attack == "glare":
        return (int(width * 0.30), int(height * 0.12), int(width * 0.70), int(height * 0.70))
    return (int(width * 0.36), int(height * 0.16), int(width * 0.64), int(height * 0.74))


def _ramp(frame_idx: int, start: int, ramp_frames: int) -> float:
    if frame_idx < start:
        return 0.0
    return min(1.0, float(frame_idx - start + 1) / float(max(1, ramp_frames)))


def _attack_params(attack: str, rng: np.random.Generator) -> dict[str, Any]:
    if attack == "adv_patch":
        return {
            "base_px": 64,
            "size_frac": 0.74,
            "top_frac": 0.18,
            "jitter": 24.0,
            "jitter_speed": 0.34,
            "contrast": 1.0,
            "pulse": 0.16,
            "texture_seed": int(rng.integers(1, 2**31 - 1)),
        }
    if attack == "glare":
        return {
            "radius_frac": 0.22,
            "intensity": 1.45,
            "flicker_speed": 0.95,
            "drift_frac": 0.08,
        }
    if attack == "occlusion":
        return {
            "w_frac": 0.66,
            "h_frac": 0.54,
            "move_frac": 0.15,
            "move_speed": 0.18,
            "alpha": 0.96,
            "color": [18, 18, 18],
        }
    raise ValueError(f"unsupported attack: {attack}")


def _apply_attack(
    frame: np.ndarray,
    attack: str,
    frame_idx: int,
    phase: float,
    anchor: tuple[int, int, int, int],
    params: dict[str, Any],
) -> np.ndarray:
    if phase <= 0.0:
        return frame
    if attack == "adv_patch":
        return _apply_adv_patch(frame, frame_idx, phase, anchor, params)
    if attack == "glare":
        return _apply_glare(frame, frame_idx, phase, anchor, params)
    if attack == "occlusion":
        return _apply_occlusion(frame, frame_idx, phase, anchor, params)
    raise ValueError(f"unsupported attack: {attack}")


def _apply_adv_patch(
    frame: np.ndarray,
    frame_idx: int,
    phase: float,
    anchor: tuple[int, int, int, int],
    params: dict[str, Any],
) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = anchor
    side = max(20, int(round(min(x2 - x1, y2 - y1) * float(params["size_frac"]))))
    jitter = float(params["jitter"])
    speed = float(params["jitter_speed"])
    cx = (x1 + x2) // 2 + int(round(jitter * math.sin(frame_idx * speed)))
    cy = y1 + int(round((y2 - y1) * float(params["top_frac"]))) + int(
        round(0.45 * jitter * math.cos(frame_idx * speed * 0.73))
    )
    px1 = max(0, min(out.shape[1] - side, cx - side // 2))
    py1 = max(0, min(out.shape[0] - side, cy - side // 2))
    patch = _fixed_adversarial_texture(int(params["base_px"]), int(params["texture_seed"]))
    scale = 1.0 + 0.08 * math.sin(frame_idx * speed * 1.31)
    warped_side = max(20, int(round(side * scale)))
    patch = cv2.resize(patch, (warped_side, warped_side), interpolation=cv2.INTER_NEAREST)
    if warped_side != side:
        patch = cv2.resize(patch, (side, side), interpolation=cv2.INTER_LINEAR)
    pulse = 1.0 + float(params.get("pulse", 0.0)) * math.sin(frame_idx * 0.91)
    patch = _clip_u8(patch.astype(np.float32) * pulse)
    alpha = min(1.0, 0.62 + 0.38 * phase) * float(params["contrast"])
    roi = out[py1 : py1 + side, px1 : px1 + side].astype(np.float32)
    out[py1 : py1 + side, px1 : px1 + side] = _clip_u8(
        roi * (1.0 - alpha) + patch.astype(np.float32) * alpha
    )
    return out


def _fixed_adversarial_texture(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cells = 10
    small = rng.integers(0, 256, size=(cells, cells, 3), dtype=np.uint16).astype(np.uint8)
    patch = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
    stripe = max(2, size // 12)
    for y in range(0, size, stripe * 2):
        patch[y : y + stripe, :, :] = (0, 255, 255)
    for x in range(stripe, size, stripe * 3):
        patch[:, x : x + stripe, :] = (20, 20, 220)
    cv2.rectangle(patch, (0, 0), (size - 1, size - 1), (0, 255, 255), max(2, size // 16))
    cv2.line(patch, (0, 0), (size - 1, size - 1), (20, 20, 220), max(2, size // 10), cv2.LINE_AA)
    cv2.line(patch, (size - 1, 0), (0, size - 1), (0, 255, 255), max(2, size // 10), cv2.LINE_AA)
    return patch


def _apply_glare(
    frame: np.ndarray,
    frame_idx: int,
    phase: float,
    anchor: tuple[int, int, int, int],
    params: dict[str, Any],
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = anchor
    out = frame.astype(np.float32)
    radius = max(24.0, min(width, height) * float(params["radius_frac"]))
    drift = min(width, height) * float(params["drift_frac"])
    flicker = 1.0 + 0.40 * math.sin(frame_idx * float(params["flicker_speed"]))
    cx = (x1 + x2) / 2.0 + drift * math.sin(frame_idx * 0.07)
    cy = (y1 + y2) / 2.0 + drift * 0.65 * math.cos(frame_idx * 0.05)
    r3 = int(radius * 3.0)
    wx1, wy1 = max(0, int(cx) - r3), max(0, int(cy) - r3)
    wx2, wy2 = min(width, int(cx) + r3), min(height, int(cy) + r3)
    yy, xx = np.indices((wy2 - wy1, wx2 - wx1), dtype=np.float32)
    d2 = (xx + wx1 - cx) ** 2 + (yy + wy1 - cy) ** 2
    falloff = np.exp(-d2 / (2.0 * radius * radius))
    strength = 255.0 * float(params["intensity"]) * phase * flicker
    win = out[wy1:wy2, wx1:wx2]
    win[:, :, 0] += falloff * strength * 0.90
    win[:, :, 1] += falloff * strength * 0.96
    win[:, :, 2] += falloff * strength
    return _clip_u8(out)


def _apply_occlusion(
    frame: np.ndarray,
    frame_idx: int,
    phase: float,
    anchor: tuple[int, int, int, int],
    params: dict[str, Any],
) -> np.ndarray:
    out = frame.copy()
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = anchor
    box_w = max(20, int(round((x2 - x1) * float(params["w_frac"]))))
    box_h = max(20, int(round((y2 - y1) * float(params["h_frac"]))))
    move = width * float(params["move_frac"])
    cx = int(round((x1 + x2) / 2.0 + move * math.sin(frame_idx * float(params["move_speed"]))))
    cy = int(round((y1 + y2) / 2.0 + move * 0.35 * math.cos(frame_idx * float(params["move_speed"]) * 0.7)))
    ox1 = max(0, min(width - box_w, cx - box_w // 2))
    oy1 = max(0, min(height - box_h, cy - box_h // 2))
    color = np.array(params["color"], dtype=np.float32)
    alpha = float(params["alpha"]) * phase
    roi = out[oy1 : oy1 + box_h, ox1 : ox1 + box_w].astype(np.float32)
    out[oy1 : oy1 + box_h, ox1 : ox1 + box_w] = _clip_u8(
        roi * (1.0 - alpha) + color[None, None, :] * alpha
    )
    return out


def _clip_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)

