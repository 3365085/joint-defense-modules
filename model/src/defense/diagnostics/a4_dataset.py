from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


UNIQUE_YOLO_SOURCE_SHA256 = (
    "4D7A23D3866AC2D9DB6E59AE537DA1274D988BD53CA6C7D519297FCBB96626F8"
)
A4_DATASET_SCHEMA_VERSION = 1
A4_GENERATOR_VERSION = "a4-synthetic-carrier-v2"
REQUIRED_ATTACK_TYPES: tuple[str, ...] = (
    "adv_patch",
    "glare",
    "motion_blur",
    "occlusion",
    "visibility_degradation",
)
ADV_PATCH_TRAJECTORY_MODES: tuple[str, ...] = (
    "target_anchored_static",
    "smooth_drift",
    "discrete_jump/jitter",
    "scale_rotation",
    "partial_outside_roi/occlusion",
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_ADV_PATCH_FRAME_AREA_RATIO_MIN = 0.002
_ADV_PATCH_FRAME_AREA_RATIO_MAX = 0.020
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUTHORITATIVE_MANIFEST = (
    _PROJECT_ROOT / "configs/acceptance/module_a_authoritative_manifest_v1.json"
)


class A4DatasetValidationError(ValueError):
    pass


class AnchorProvider(Protocol):
    def reset(self, *, width: int, height: int) -> None: ...

    def locate(self, frame: np.ndarray, *, frame_idx: int) -> tuple[float, float, float, float]: ...

    def summary(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class A4VariantSpec:
    attack_type: str
    trajectory_mode: str
    label: int
    suffix: str


class CenterAnchorProvider:
    """Deterministic fixture provider; the production CLI always uses YOLO."""

    def __init__(self) -> None:
        self._width = 0
        self._height = 0
        self._frames = 0

    def reset(self, *, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._frames = 0

    def locate(
        self,
        frame: np.ndarray,
        *,
        frame_idx: int,
    ) -> tuple[float, float, float, float]:
        del frame, frame_idx
        self._frames += 1
        width = float(self._width)
        height = float(self._height)
        return (0.30 * width, 0.18 * height, 0.70 * width, 0.88 * height)

    def summary(self) -> Mapping[str, Any]:
        return {
            "provider": "deterministic_center_fixture",
            "observed_frames": int(self._frames),
            "yolo_detection_frames": 0,
            "center_fallback_frames": int(self._frames),
        }


class _YoloAnchorProvider:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        confidence: float,
        image_size: int,
        inference_stride: int,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._device = str(device)
        self._confidence = float(confidence)
        self._image_size = int(image_size)
        self._stride = max(1, int(inference_stride))
        self._width = 0
        self._height = 0
        self._last: tuple[float, float, float, float] | None = None
        self._observed_frames = 0
        self._inference_frames = 0
        self._detection_frames = 0
        self._fallback_frames = 0

    def reset(self, *, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._last = None
        self._observed_frames = 0
        self._inference_frames = 0
        self._detection_frames = 0
        self._fallback_frames = 0

    def _fallback(self) -> tuple[float, float, float, float]:
        width = float(self._width)
        height = float(self._height)
        return (0.30 * width, 0.18 * height, 0.70 * width, 0.88 * height)

    def locate(
        self,
        frame: np.ndarray,
        *,
        frame_idx: int,
    ) -> tuple[float, float, float, float]:
        self._observed_frames += 1
        should_infer = self._last is None or int(frame_idx) % self._stride == 0
        if should_infer:
            self._inference_frames += 1
            prediction = self._model.predict(
                source=frame,
                conf=self._confidence,
                imgsz=self._image_size,
                device=self._device,
                verbose=False,
            )[0]
            boxes = getattr(prediction, "boxes", None)
            xyxy = (
                boxes.xyxy.detach().cpu().numpy()
                if boxes is not None and getattr(boxes, "xyxy", None) is not None
                else np.empty((0, 4), dtype=np.float32)
            )
            confidence = (
                boxes.conf.detach().cpu().numpy()
                if boxes is not None and getattr(boxes, "conf", None) is not None
                else np.ones(len(xyxy), dtype=np.float32)
            )
            if len(xyxy):
                widths = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
                heights = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
                index = int(np.argmax(widths * heights * np.maximum(confidence, 0.05)))
                candidate = _clip_anchor(
                    tuple(float(value) for value in xyxy[index]),
                    width=self._width,
                    height=self._height,
                )
                if self._last is None:
                    self._last = candidate
                else:
                    self._last = tuple(
                        0.72 * previous + 0.28 * current
                        for previous, current in zip(self._last, candidate, strict=True)
                    )
                self._detection_frames += 1
            elif self._last is None:
                self._last = self._fallback()
        if self._last is None:
            self._last = self._fallback()
        if not should_infer or self._detection_frames == 0:
            self._fallback_frames += 1
        return self._last

    def summary(self) -> Mapping[str, Any]:
        return {
            "provider": "unique_yolo_largest_detection_smoothed",
            "device": self._device,
            "confidence": self._confidence,
            "image_size": self._image_size,
            "inference_stride": self._stride,
            "observed_frames": self._observed_frames,
            "inference_frames": self._inference_frames,
            "yolo_detection_frames": self._detection_frames,
            "center_or_last_anchor_frames": self._fallback_frames,
        }


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in _HEX_DIGITS for character in text)


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _is_sha256(text):
        raise A4DatasetValidationError(f"invalid_sha256:{field}:{text or '<missing>'}")
    return text


def assert_no_rebuilt_demo_path(path: str | Path, *, field: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if any(part.casefold() == "rebuilt_demo" for part in resolved.parts):
        raise A4DatasetValidationError(f"rebuilt_demo_path_forbidden:{field}:{resolved}")
    return resolved


def manifest_metadata_path(path: str | Path) -> Path:
    manifest = Path(path).expanduser()
    return manifest.with_suffix(manifest.suffix + ".meta.json")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_seed(*parts: Any, seed: int) -> int:
    payload = "\0".join(str(part) for part in parts) + f"\0{int(seed)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _variant_specs(adv_patch_trajectory_mode: str) -> tuple[A4VariantSpec, ...]:
    if adv_patch_trajectory_mode not in ADV_PATCH_TRAJECTORY_MODES:
        raise A4DatasetValidationError(
            f"unknown_adv_patch_trajectory:{adv_patch_trajectory_mode}"
        )
    specs = [A4VariantSpec("clean", "none", 0, "clean")]
    specs.append(
        A4VariantSpec(
            "adv_patch",
            adv_patch_trajectory_mode,
            1,
            "adv_patch",
        )
    )
    specs.extend(
        (
            A4VariantSpec("glare", "none", 1, "glare"),
            A4VariantSpec("motion_blur", "none", 1, "motion_blur"),
            A4VariantSpec("occlusion", "none", 1, "occlusion"),
            A4VariantSpec(
                "visibility_degradation",
                "none",
                1,
                "visibility_degradation",
            ),
        )
    )
    return tuple(specs)


def load_authoritative_contract(
    manifest_path: str | Path = DEFAULT_AUTHORITATIVE_MANIFEST,
    *,
    expected_video_count: int | None = 36,
) -> dict[str, Any]:
    path = assert_no_rebuilt_demo_path(manifest_path, field="authoritative_manifest")
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A4DatasetValidationError(
            f"authoritative_manifest_invalid:{type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise A4DatasetValidationError("authoritative_manifest_invalid:not_an_object")
    videos = payload.get("videos", [])
    if not isinstance(videos, list):
        raise A4DatasetValidationError("authoritative_manifest_invalid:videos_not_a_list")
    if expected_video_count is not None and len(videos) != int(expected_video_count):
        raise A4DatasetValidationError(
            "authoritative_video_count_mismatch:"
            f"expected={expected_video_count},actual={len(videos)}"
        )
    hashes = tuple(
        _require_sha256(item.get("sha256"), field="authoritative_video.sha256")
        for item in videos
        if isinstance(item, Mapping)
    )
    if len(hashes) != len(videos) or len(set(hashes)) != len(hashes):
        raise A4DatasetValidationError("authoritative_video_hashes_missing_or_duplicate")
    unique_model = payload.get("unique_model", {})
    if not isinstance(unique_model, Mapping):
        raise A4DatasetValidationError("authoritative_unique_model_missing")
    model_hash = _require_sha256(
        unique_model.get("sha256"),
        field="unique_model.sha256",
    )
    if model_hash != UNIQUE_YOLO_SOURCE_SHA256.lower():
        raise A4DatasetValidationError(
            "unique_yolo_source_sha256_mismatch:"
            f"expected={UNIQUE_YOLO_SOURCE_SHA256.lower()},actual={model_hash}"
        )
    model_path = assert_no_rebuilt_demo_path(
        str(unique_model.get("canonical_path", "")),
        field="unique_model.canonical_path",
    )
    return {
        "manifest_path": path,
        "manifest_sha256": sha256_file(path),
        "video_hashes": hashes,
        "model_path": model_path,
        "model_sha256": model_hash,
    }


def load_clean_sources(
    manifest_path: str | Path,
    *,
    authoritative_video_hashes: Sequence[str],
    verify_source_hashes: bool = True,
) -> list[dict[str, Any]]:
    path = assert_no_rebuilt_demo_path(manifest_path, field="clean_source_manifest")
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = set(reader.fieldnames or ())
    required = {
        "clip_id",
        "path",
        "scene_id",
        "split",
        "base_source_sha256",
    }
    missing = sorted(required - fields)
    if missing:
        raise A4DatasetValidationError(f"clean_source_manifest_missing_columns:{missing}")
    if not rows:
        raise A4DatasetValidationError("clean_source_manifest_empty")
    authoritative = {str(value).lower() for value in authoritative_video_hashes}
    seen_ids: set[str] = set()
    seen_scenes: set[str] = set()
    seen_hashes: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in rows:
        clip_id = str(raw.get("clip_id", "")).strip()
        scene_id = str(raw.get("scene_id", "")).strip()
        split = str(raw.get("split", "")).strip()
        if not clip_id or clip_id in seen_ids:
            raise A4DatasetValidationError(f"clean_clip_id_missing_or_duplicate:{clip_id}")
        if not scene_id or scene_id in seen_scenes:
            raise A4DatasetValidationError(
                f"clean_scene_id_missing_or_duplicate:{scene_id or '<missing>'}"
            )
        if split not in {"train", "heldout"}:
            raise A4DatasetValidationError(f"clean_split_invalid:{clip_id}:{split}")
        source_path = assert_no_rebuilt_demo_path(raw.get("path", ""), field=f"{clip_id}.path")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        declared_hash = _require_sha256(
            raw.get("base_source_sha256"),
            field=f"{clip_id}.base_source_sha256",
        )
        if declared_hash in authoritative:
            raise A4DatasetValidationError(
                f"authoritative_video_overlap:{clip_id}:{declared_hash}"
            )
        if declared_hash in seen_hashes:
            raise A4DatasetValidationError(
                f"clean_source_content_duplicate:{clip_id}:{declared_hash}"
            )
        if verify_source_hashes:
            actual_hash = sha256_file(source_path)
            if actual_hash != declared_hash:
                raise A4DatasetValidationError(
                    "clean_source_sha256_mismatch:"
                    f"{clip_id}:expected={declared_hash},actual={actual_hash}"
                )
        seen_ids.add(clip_id)
        seen_scenes.add(scene_id)
        seen_hashes.add(declared_hash)
        row = dict(raw)
        row.update(
            {
                "clip_id": clip_id,
                "scene_id": scene_id,
                "split": split,
                "path": str(source_path),
                "base_source_sha256": declared_hash,
            }
        )
        validated.append(row)
    return validated


def _clip_anchor(
    anchor: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in anchor)
    x1 = min(max(0.0, x1), max(0.0, width - 2.0))
    y1 = min(max(0.0, y1), max(0.0, height - 2.0))
    x2 = min(max(x1 + 2.0, x2), float(width))
    y2 = min(max(y1 + 2.0, y2), float(height))
    return x1, y1, x2, y2


def _trajectory_state(
    mode: str,
    *,
    frame_idx: int,
    anchor: Sequence[float],
    seed: int,
) -> dict[str, float]:
    x1, y1, x2, y2 = (float(value) for value in anchor)
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.48 * y1 + 0.52 * y2
    parameters = _trajectory_parameters(mode, seed=seed)
    patch_size = max(
        12.0,
        float(parameters["base_scale"]) * min(width, height),
    )
    phase = float(parameters["phase_radians"])
    angle = 0.0
    if mode == "target_anchored_static":
        pass
    elif mode == "smooth_drift":
        center_x += float(parameters["x_amplitude"]) * width * math.sin(
            frame_idx / float(parameters["x_period_frames"]) + phase
        )
        center_y += float(parameters["y_amplitude"]) * height * math.sin(
            frame_idx / float(parameters["y_period_frames"]) + 0.7 * phase
        )
    elif mode == "discrete_jump/jitter":
        jump_interval = int(parameters["jump_interval_frames"])
        segment = int(frame_idx) // jump_interval
        rng = np.random.RandomState((int(seed) + 104729 * segment) & 0xFFFFFFFF)
        center_x += float(rng.uniform(-1.0, 1.0)) * float(parameters["jump_x_limit"]) * width
        center_y += float(rng.uniform(-1.0, 1.0)) * float(parameters["jump_y_limit"]) * height
        center_x += float(parameters["jitter_x_amplitude"]) * width * math.sin(
            frame_idx * 1.7 + phase
        )
        center_y += float(parameters["jitter_y_amplitude"]) * height * math.cos(
            frame_idx * 1.3 + phase
        )
    elif mode == "scale_rotation":
        scale_low = float(parameters["scale_low"])
        scale_high = float(parameters["scale_high"])
        patch_size *= scale_low + (scale_high - scale_low) * (
            0.5 + 0.5 * math.sin(frame_idx / float(parameters["scale_period_frames"]) + phase)
        )
        angle = float(parameters["rotation_limit_degrees"]) * math.sin(
            frame_idx / float(parameters["rotation_period_frames"]) + phase
        )
    elif mode == "partial_outside_roi/occlusion":
        center_x = x2 + float(parameters["outside_x_offset"]) * width * math.sin(
            frame_idx / float(parameters["outside_period_frames"]) + phase
        )
        center_y = y1 + float(parameters["outside_y_fraction"]) * height
        angle = float(parameters["rotation_limit_degrees"]) * math.sin(
            frame_idx / float(parameters["rotation_period_frames"]) + phase
        )
        patch_size *= float(parameters["outside_scale"])
    else:
        raise A4DatasetValidationError(f"unknown_adv_patch_trajectory:{mode}")
    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "patch_size": float(patch_size),
        "angle_degrees": float(angle),
    }


def _trajectory_parameters(mode: str, *, seed: int) -> dict[str, float | int]:
    """Select from a fixed broad grid; no authoritative-video statistics are used."""

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    def choose(values: Sequence[float | int]) -> float | int:
        return values[int(rng.randint(0, len(values)))]

    parameters: dict[str, float | int] = {
        "base_scale": choose((0.26, 0.34, 0.42, 0.52, 0.62)),
        "phase_radians": float(rng.uniform(0.0, 2.0 * math.pi)),
    }
    if mode == "target_anchored_static":
        return parameters
    if mode == "smooth_drift":
        parameters.update(
            {
                "x_amplitude": choose((0.10, 0.18, 0.26, 0.34)),
                "y_amplitude": choose((0.08, 0.14, 0.22, 0.30)),
                "x_period_frames": choose((8.0, 13.0, 21.0, 34.0)),
                "y_period_frames": choose((11.0, 17.0, 29.0, 43.0)),
            }
        )
        return parameters
    if mode == "discrete_jump/jitter":
        parameters.update(
            {
                "jump_interval_frames": choose((4, 6, 8, 12, 16)),
                "jump_x_limit": choose((0.18, 0.28, 0.38, 0.48)),
                "jump_y_limit": choose((0.14, 0.24, 0.34, 0.44)),
                "jitter_x_amplitude": choose((0.0, 0.012, 0.025, 0.045)),
                "jitter_y_amplitude": choose((0.0, 0.010, 0.020, 0.040)),
            }
        )
        return parameters
    if mode == "scale_rotation":
        parameters.update(
            {
                "scale_low": choose((0.48, 0.62, 0.76)),
                "scale_high": choose((1.08, 1.28, 1.52, 1.78)),
                "scale_period_frames": choose((6.0, 10.0, 16.0, 24.0)),
                "rotation_limit_degrees": choose((18.0, 32.0, 48.0, 68.0)),
                "rotation_period_frames": choose((7.0, 12.0, 19.0, 31.0)),
            }
        )
        return parameters
    if mode == "partial_outside_roi/occlusion":
        parameters.update(
            {
                "outside_x_offset": choose((0.04, 0.10, 0.18, 0.28)),
                "outside_y_fraction": choose((0.08, 0.18, 0.30, 0.44)),
                "outside_period_frames": choose((8.0, 14.0, 23.0, 37.0)),
                "outside_scale": choose((0.92, 1.12, 1.34, 1.58)),
                "rotation_limit_degrees": choose((0.0, 12.0, 26.0, 42.0)),
                "rotation_period_frames": choose((7.0, 11.0, 18.0, 29.0)),
            }
        )
        return parameters
    raise A4DatasetValidationError(f"unknown_adv_patch_trajectory:{mode}")


def _patch_texture(seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    yy, xx = np.indices((size, size))
    checker = ((xx // 8 + yy // 8) % 2).astype(np.float32)
    stripes = (0.5 + 0.5 * np.sin((xx + 1.7 * yy) / 4.0)).astype(np.float32)
    noise = rng.uniform(0.0, 1.0, (size, size)).astype(np.float32)
    texture = np.empty((size, size, 3), dtype=np.float32)
    texture[..., 0] = 30.0 + 210.0 * checker
    texture[..., 1] = 20.0 + 220.0 * stripes
    texture[..., 2] = 35.0 + 210.0 * noise
    return np.clip(texture, 0.0, 255.0).astype(np.uint8)


def _overlay_patch(
    frame: np.ndarray,
    *,
    texture: np.ndarray,
    state: Mapping[str, float],
    strength: float,
) -> np.ndarray:
    size = max(4, int(round(float(state["patch_size"]))))
    patch = cv2.resize(texture, (size, size), interpolation=cv2.INTER_LINEAR)
    angle = float(state["angle_degrees"])
    if abs(angle) > 1e-6:
        matrix = cv2.getRotationMatrix2D((0.5 * size, 0.5 * size), angle, 1.0)
        patch = cv2.warpAffine(
            patch,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    left = int(round(float(state["center_x"]) - 0.5 * size))
    top = int(round(float(state["center_y"]) - 0.5 * size))
    right = left + size
    bottom = top + size
    frame_height, frame_width = frame.shape[:2]
    dst_left = max(0, left)
    dst_top = max(0, top)
    dst_right = min(frame_width, right)
    dst_bottom = min(frame_height, bottom)
    if dst_left >= dst_right or dst_top >= dst_bottom:
        return frame.copy()
    src_left = dst_left - left
    src_top = dst_top - top
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)
    output = frame.copy()
    alpha = min(1.0, max(0.0, float(strength))) * 0.94
    output[dst_top:dst_bottom, dst_left:dst_right] = cv2.addWeighted(
        frame[dst_top:dst_bottom, dst_left:dst_right],
        1.0 - alpha,
        patch[src_top:src_bottom, src_left:src_right],
        alpha,
        0.0,
    )
    return output


def _glare(frame: np.ndarray, *, anchor: Sequence[float], frame_idx: int, seed: int, strength: float) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in anchor)
    phase = (seed % 997) / 997.0 * 2.0 * math.pi
    center_x = 0.5 * (x1 + x2) + 0.30 * (x2 - x1) * math.sin(frame_idx / 17.0 + phase)
    center_y = 0.5 * (y1 + y2) + 0.22 * (y2 - y1) * math.cos(frame_idx / 21.0 + phase)
    yy, xx = np.ogrid[:height, :width]
    radius = max(18.0, 0.42 * min(x2 - x1, y2 - y1))
    distance = ((xx - center_x) ** 2 + (yy - center_y) ** 2) / max(1.0, radius**2)
    mask = np.exp(-1.8 * distance).astype(np.float32)[..., None]
    alpha = np.clip(mask * (0.92 * strength), 0.0, 0.92)
    glare_color = np.full_like(frame, (210, 245, 255), dtype=np.float32)
    return np.clip(frame.astype(np.float32) * (1.0 - alpha) + glare_color * alpha, 0, 255).astype(np.uint8)


def _motion_blur(frame: np.ndarray, *, frame_idx: int, seed: int, strength: float) -> np.ndarray:
    size = 13
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    phase = (seed % 991) / 991.0 * 2.0 * math.pi
    angle = 55.0 * math.sin(frame_idx / 15.0 + phase)
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    kernel /= max(1e-6, float(kernel.sum()))
    blurred = cv2.filter2D(frame, -1, kernel)
    return cv2.addWeighted(frame, 1.0 - 0.92 * strength, blurred, 0.92 * strength, 0.0)


def _occlusion(frame: np.ndarray, *, anchor: Sequence[float], frame_idx: int, seed: int, strength: float) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in anchor)
    phase = (seed % 983) / 983.0 * 2.0 * math.pi
    width = x2 - x1
    height = y2 - y1
    shift = 0.10 * width * math.sin(frame_idx / 14.0 + phase)
    left = int(round(x1 + 0.14 * width + shift))
    right = int(round(x2 - 0.08 * width + shift))
    top = int(round(y1 + 0.22 * height))
    bottom = int(round(y2 - 0.18 * height))
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (24, 28, 34), thickness=-1)
    cv2.line(overlay, (left, top), (right, bottom), (70, 76, 82), thickness=max(2, int(width / 35)))
    return cv2.addWeighted(frame, 1.0 - 0.96 * strength, overlay, 0.96 * strength, 0.0)


def _visibility_degradation(frame: np.ndarray, *, frame_idx: int, seed: int, strength: float) -> np.ndarray:
    local_seed = (int(seed) + 1_000_003 * int(frame_idx)) & 0xFFFFFFFF
    rng = np.random.RandomState(local_seed)
    noise = rng.normal(0.0, 7.0, frame.shape).astype(np.float32)
    softened = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.2, sigmaY=2.2)
    fog = np.full_like(frame, (188, 196, 202), dtype=np.float32)
    degraded = 0.56 * softened.astype(np.float32) + 0.44 * fog + noise
    return np.clip(
        frame.astype(np.float32) * (1.0 - 0.92 * strength) + degraded * (0.92 * strength),
        0,
        255,
    ).astype(np.uint8)


def _attack_strength(frame_idx: int, *, attack_start_frame: int, attack_ramp_frames: int) -> float:
    if frame_idx < attack_start_frame:
        return 0.0
    if attack_ramp_frames <= 0:
        return 1.0
    return min(1.0, max(0.0, (frame_idx - attack_start_frame + 1) / attack_ramp_frames))


def _render_attack_frame(
    frame: np.ndarray,
    *,
    spec: A4VariantSpec,
    anchor: Sequence[float],
    frame_idx: int,
    seed: int,
    texture: np.ndarray | None,
    attack_start_frame: int,
    attack_ramp_frames: int,
) -> np.ndarray:
    if spec.attack_type == "clean":
        return frame
    strength = _attack_strength(
        frame_idx,
        attack_start_frame=attack_start_frame,
        attack_ramp_frames=attack_ramp_frames,
    )
    if strength <= 0.0:
        return frame
    if spec.attack_type == "adv_patch":
        if texture is None:
            raise RuntimeError("adv_patch_texture_missing")
        state = _trajectory_state(
            spec.trajectory_mode,
            frame_idx=frame_idx,
            anchor=anchor,
            seed=seed,
        )
        frame_height, frame_width = frame.shape[:2]
        frame_area = max(1.0, float(frame_height * frame_width))
        state["patch_size"] = min(
            math.sqrt(frame_area * _ADV_PATCH_FRAME_AREA_RATIO_MAX),
            max(
                math.sqrt(frame_area * _ADV_PATCH_FRAME_AREA_RATIO_MIN),
                float(state["patch_size"]),
            ),
        )
        return _overlay_patch(frame, texture=texture, state=state, strength=strength)
    if spec.attack_type == "glare":
        return _glare(frame, anchor=anchor, frame_idx=frame_idx, seed=seed, strength=strength)
    if spec.attack_type == "motion_blur":
        return _motion_blur(frame, frame_idx=frame_idx, seed=seed, strength=strength)
    if spec.attack_type == "occlusion":
        return _occlusion(frame, anchor=anchor, frame_idx=frame_idx, seed=seed, strength=strength)
    if spec.attack_type == "visibility_degradation":
        return _visibility_degradation(frame, frame_idx=frame_idx, seed=seed, strength=strength)
    raise A4DatasetValidationError(f"unknown_attack_type:{spec.attack_type}")


def _effective_attack_timing(
    frame_count: int,
    *,
    requested_start: int,
    requested_ramp: int,
) -> tuple[int, int]:
    if frame_count < 3:
        raise A4DatasetValidationError(f"source_video_too_short:{frame_count}")
    start = min(max(0, int(requested_start)), max(0, frame_count // 4))
    remaining = max(1, frame_count - start)
    ramp = min(max(0, int(requested_ramp)), max(0, remaining // 3))
    return start, ramp


def _position_capture_for_bounded_window(
    source_path: Path,
    capture: Any,
    *,
    source_start_frame: int,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Position at the exact first carrier frame and prefetch it once.

    Random seek is accepted only when OpenCV reports the requested position
    and the first frame can be decoded. Any uncertainty reopens the source and
    performs an exact sequential discard, never silently substituting frame 0.
    """

    target = int(source_start_frame)
    if target < 0:
        capture.release()
        raise A4DatasetValidationError(f"source_start_frame_invalid:{target}")

    def reported_position_matches(value: Any, expected: int) -> bool:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(parsed) and abs(parsed - float(expected)) <= 0.5

    def valid_frame(ok: Any, frame: Any) -> bool:
        return bool(ok) and isinstance(frame, np.ndarray) and frame.size > 0

    if target == 0:
        ok, first_frame = capture.read()
        if not valid_frame(ok, first_frame):
            capture.release()
            raise A4DatasetValidationError(
                f"source_window_first_frame_decode_failed:{source_path}:offset=0"
            )
        return capture, first_frame, {
            "source_positioning_mode": "sequential_from_zero",
            "source_seek_attempted": False,
            "source_seek_fallback_reason": "none",
        }

    seek_set_succeeded = False
    seek_failure_reason = "seek_set_failed"
    try:
        seek_set_succeeded = bool(
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(target))
        )
        if seek_set_succeeded:
            position_after_set = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if not reported_position_matches(position_after_set, target):
                seek_failure_reason = "seek_position_after_set_mismatch"
            else:
                ok, first_frame = capture.read()
                if not valid_frame(ok, first_frame):
                    seek_failure_reason = "seek_first_frame_decode_failed"
                else:
                    position_after_read = capture.get(cv2.CAP_PROP_POS_FRAMES)
                    if reported_position_matches(position_after_read, target + 1):
                        return capture, first_frame, {
                            "source_positioning_mode": "random_seek_verified",
                            "source_seek_attempted": True,
                            "source_seek_fallback_reason": "none",
                        }
                    seek_failure_reason = "seek_position_after_read_mismatch"
    except Exception as exc:
        seek_failure_reason = f"seek_exception:{type(exc).__name__}"

    capture.release()
    fallback = capture_factory(str(source_path))
    if not fallback.isOpened():
        fallback.release()
        raise A4DatasetValidationError(
            "source_window_fallback_reopen_failed:"
            f"{source_path}:offset={target}:seek_reason={seek_failure_reason}"
        )
    discarded = 0
    try:
        while discarded < target:
            ok, frame = fallback.read()
            if not valid_frame(ok, frame):
                raise A4DatasetValidationError(
                    "source_window_sequential_discard_failed:"
                    f"{source_path}:offset={target}:discarded={discarded}:"
                    f"seek_reason={seek_failure_reason}"
                )
            discarded += 1
        ok, first_frame = fallback.read()
        if not valid_frame(ok, first_frame):
            raise A4DatasetValidationError(
                "source_window_fallback_first_frame_decode_failed:"
                f"{source_path}:offset={target}:discarded={discarded}:"
                f"seek_reason={seek_failure_reason}"
            )
    except Exception:
        fallback.release()
        raise
    return fallback, first_frame, {
        "source_positioning_mode": "sequential_decode_fallback",
        "source_seek_attempted": True,
        "source_seek_set_succeeded": bool(seek_set_succeeded),
        "source_seek_fallback_reason": seek_failure_reason,
        "source_sequential_discarded_frames": int(discarded),
    }


def _render_base_variants(
    row: Mapping[str, Any],
    *,
    output_dir: Path,
    source_manifest_sha256: str,
    generator_seed: int,
    max_frames_per_video: int,
    clip_duration_s: float | None,
    attack_start_frame: int,
    attack_ramp_frames: int,
    codec: str,
    anchor_provider: AnchorProvider,
    adv_patch_trajectory_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = Path(str(row["path"]))
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"unable_to_open_clean_source:{source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if not 0.1 <= fps <= 240.0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise A4DatasetValidationError(f"source_video_dimensions_invalid:{source_path}")
    frame_bounds: list[int] = []
    if max_frames_per_video > 0:
        frame_bounds.append(int(max_frames_per_video))
    if clip_duration_s is not None and float(clip_duration_s) > 0.0:
        frame_bounds.append(max(3, int(round(float(clip_duration_s) * fps))))
    if not frame_bounds:
        raise A4DatasetValidationError(
            "bounded_clip_required:set_max_frames_per_video_or_clip_duration_s"
        )
    planned_frames = min(frame_bounds)
    if source_frames > 0:
        planned_frames = min(source_frames, planned_frames)
    if planned_frames <= 0:
        raise A4DatasetValidationError(f"source_video_frame_count_invalid:{source_path}")
    available_offset = max(0, source_frames - planned_frames) if source_frames > 0 else 0
    offset_seed = _stable_seed(
        row["base_source_sha256"],
        "bounded_source_window",
        seed=generator_seed,
    )
    source_start_frame = (
        int(offset_seed % (available_offset + 1)) if available_offset > 0 else 0
    )
    capture, first_source_frame, positioning = _position_capture_for_bounded_window(
        source_path,
        capture,
        source_start_frame=source_start_frame,
    )
    effective_start, effective_ramp = _effective_attack_timing(
        planned_frames,
        requested_start=attack_start_frame,
        requested_ramp=attack_ramp_frames,
    )
    anchor_provider.reset(width=width, height=height)
    specs = _variant_specs(adv_patch_trajectory_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*str(codec))
    writers: dict[str, cv2.VideoWriter] = {}
    temp_paths: dict[str, Path] = {}
    final_paths: dict[str, Path] = {}
    seeds: dict[str, int] = {}
    textures: dict[str, np.ndarray] = {}
    base_id = str(row["clip_id"])
    render_error: BaseException | None = None
    try:
        for spec in specs:
            final_path = output_dir / f"{base_id}__{spec.suffix}.mp4"
            temp_path = final_path.with_name(f"{final_path.stem}.tmp{final_path.suffix}")
            writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"unable_to_open_video_writer:{temp_path}:codec={codec}")
            final_paths[spec.suffix] = final_path
            temp_paths[spec.suffix] = temp_path
            writers[spec.suffix] = writer
            variant_seed = _stable_seed(
                row["base_source_sha256"],
                spec.attack_type,
                spec.trajectory_mode,
                seed=generator_seed,
            )
            seeds[spec.suffix] = variant_seed
            if spec.attack_type == "adv_patch":
                textures[spec.suffix] = _patch_texture(variant_seed)

        frame_idx = 0
        frame = first_source_frame
        while frame_idx < planned_frames:
            anchor = _clip_anchor(
                anchor_provider.locate(frame, frame_idx=frame_idx),
                width=width,
                height=height,
            )
            for spec in specs:
                rendered = _render_attack_frame(
                    frame,
                    spec=spec,
                    anchor=anchor,
                    frame_idx=frame_idx,
                    seed=seeds[spec.suffix],
                    texture=textures.get(spec.suffix),
                    attack_start_frame=effective_start,
                    attack_ramp_frames=effective_ramp,
                )
                writers[spec.suffix].write(rendered)
            frame_idx += 1
            if frame_idx >= planned_frames:
                break
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
        if source_frames > 0 and frame_idx != planned_frames:
            raise A4DatasetValidationError(
                "source_video_window_decode_incomplete:"
                f"{source_path}:offset={source_start_frame}:"
                f"expected={planned_frames}:actual={frame_idx}:"
                f"positioning={positioning['source_positioning_mode']}"
            )
        if frame_idx < 3:
            raise A4DatasetValidationError(
                f"source_video_decoded_too_few_frames:{source_path}:{frame_idx}"
            )
    except BaseException as exc:
        render_error = exc
    finally:
        capture.release()
        for writer in writers.values():
            writer.release()
    if render_error is not None:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise render_error
    try:
        for spec in specs:
            temp_path = temp_paths[spec.suffix]
            if not temp_path.is_file() or temp_path.stat().st_size <= 0:
                raise RuntimeError(f"rendered_video_missing_or_empty:{temp_path}")
        for spec in specs:
            temp_paths[spec.suffix].replace(final_paths[spec.suffix])
    except Exception:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise

    rows: list[dict[str, Any]] = []
    for spec in specs:
        rendered_path = final_paths[spec.suffix].resolve()
        content_hash = sha256_file(rendered_path)
        provenance_payload = {
            "generator_version": A4_GENERATOR_VERSION,
            "source_manifest_sha256": source_manifest_sha256,
            "base_clip_id": base_id,
            "base_source_sha256": str(row["base_source_sha256"]),
            "scene_id": str(row["scene_id"]),
            "split": str(row["split"]),
            "attack_type": spec.attack_type,
            "trajectory_mode": spec.trajectory_mode,
            "variant_seed": int(seeds[spec.suffix]),
            "attack_start_frame": int(effective_start),
            "attack_ramp_frames": int(effective_ramp),
            "max_frames_per_video": int(max_frames_per_video),
            "clip_duration_s": None if clip_duration_s is None else float(clip_duration_s),
            "source_start_frame": int(source_start_frame),
            "source_end_frame_exclusive": int(source_start_frame + frame_idx),
            "source_frame_count": int(source_frames),
            "source_positioning": positioning,
            "codec": str(codec),
            "trajectory_algorithm": (
                "piecewise_seeded_8_frame_jump_plus_bounded_sinusoidal_jitter"
                if spec.trajectory_mode == "discrete_jump/jitter"
                else spec.trajectory_mode
            ),
            "trajectory_parameters": (
                _trajectory_parameters(spec.trajectory_mode, seed=seeds[spec.suffix])
                if spec.attack_type == "adv_patch"
                else {}
            ),
            "adv_patch_frame_area_ratio_bounds": (
                [
                    _ADV_PATCH_FRAME_AREA_RATIO_MIN,
                    _ADV_PATCH_FRAME_AREA_RATIO_MAX,
                ]
                if spec.attack_type == "adv_patch"
                else []
            ),
        }
        provenance_id = hashlib.sha256(
            _canonical_json(provenance_payload).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "clip_id": f"{base_id}__{spec.suffix}",
                "path": str(rendered_path),
                "base_clip_id": base_id,
                "base_path": str(source_path.resolve()),
                "base_source_sha256": str(row["base_source_sha256"]),
                "scene_id": str(row["scene_id"]),
                "split": str(row["split"]),
                "label": int(spec.label),
                "attack_type": spec.attack_type,
                "trajectory_mode": spec.trajectory_mode,
                "variant_seed": int(seeds[spec.suffix]),
                "attack_start_frame": 0 if spec.label == 0 else int(effective_start),
                "attack_ramp_frames": 0 if spec.label == 0 else int(effective_ramp),
                "frames": int(frame_idx),
                "source_start_frame": int(source_start_frame),
                "source_end_frame_exclusive": int(source_start_frame + frame_idx),
                "source_frame_count": int(source_frames),
                "source_positioning_mode": str(
                    positioning["source_positioning_mode"]
                ),
                "source_seek_fallback_reason": str(
                    positioning["source_seek_fallback_reason"]
                ),
                "width": int(width),
                "height": int(height),
                "fps": float(fps),
                "content_sha256": content_hash,
                "source_manifest_sha256": source_manifest_sha256,
                "generator_version": A4_GENERATOR_VERSION,
                "provenance_id": provenance_id,
                "provenance_json": _canonical_json(provenance_payload),
            }
        )
    return rows, dict(anchor_provider.summary())


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise A4DatasetValidationError("cannot_write_empty_dataset_manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_dataset_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    authoritative_video_hashes: Sequence[str],
    verify_content_hashes: bool,
) -> None:
    if not rows:
        raise A4DatasetValidationError("a4_dataset_manifest_empty")
    required = {
        "clip_id",
        "path",
        "base_clip_id",
        "base_source_sha256",
        "scene_id",
        "split",
        "label",
        "attack_type",
        "trajectory_mode",
        "variant_seed",
        "attack_start_frame",
        "attack_ramp_frames",
        "frames",
        "content_sha256",
        "source_manifest_sha256",
        "generator_version",
        "provenance_id",
        "source_start_frame",
        "source_end_frame_exclusive",
        "source_frame_count",
        "source_positioning_mode",
        "source_seek_fallback_reason",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise A4DatasetValidationError(f"a4_dataset_manifest_missing_columns:{missing}")
    authoritative = {str(value).lower() for value in authoritative_video_hashes}
    seen_clips: set[str] = set()
    content_splits: dict[str, str] = {}
    content_clips: dict[str, str] = {}
    provenance_ids: set[str] = set()
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        clip_id = str(row.get("clip_id", ""))
        if not clip_id or clip_id in seen_clips:
            raise A4DatasetValidationError(f"dataset_clip_id_missing_or_duplicate:{clip_id}")
        seen_clips.add(clip_id)
        path = assert_no_rebuilt_demo_path(row.get("path", ""), field=f"{clip_id}.path")
        if not path.is_file():
            raise FileNotFoundError(path)
        content_hash = _require_sha256(
            row.get("content_sha256"),
            field=f"{clip_id}.content_sha256",
        )
        base_hash = _require_sha256(
            row.get("base_source_sha256"),
            field=f"{clip_id}.base_source_sha256",
        )
        _require_sha256(
            row.get("source_manifest_sha256"),
            field=f"{clip_id}.source_manifest_sha256",
        )
        provenance_id = _require_sha256(
            row.get("provenance_id"),
            field=f"{clip_id}.provenance_id",
        )
        if provenance_id in provenance_ids:
            raise A4DatasetValidationError(f"provenance_id_duplicate:{provenance_id}")
        provenance_ids.add(provenance_id)
        if base_hash in authoritative or content_hash in authoritative:
            raise A4DatasetValidationError(
                f"authoritative_video_overlap:{clip_id}:{base_hash if base_hash in authoritative else content_hash}"
            )
        split = str(row.get("split", ""))
        if split not in {"train", "heldout"}:
            raise A4DatasetValidationError(f"dataset_split_invalid:{clip_id}:{split}")
        source_start = int(row.get("source_start_frame", -1))
        source_end = int(row.get("source_end_frame_exclusive", -1))
        source_count = int(row.get("source_frame_count", -1))
        carrier_frames = int(row.get("frames", -1))
        if source_start < 0 or source_end <= source_start:
            raise A4DatasetValidationError(
                f"source_window_invalid:{clip_id}:{source_start}:{source_end}"
            )
        if source_end - source_start != carrier_frames:
            raise A4DatasetValidationError(
                "source_window_frame_count_mismatch:"
                f"{clip_id}:start={source_start}:end={source_end}:frames={carrier_frames}"
            )
        if source_count > 0 and source_end > source_count:
            raise A4DatasetValidationError(
                f"source_window_exceeds_source:{clip_id}:{source_end}>{source_count}"
            )
        positioning_mode = str(row.get("source_positioning_mode", ""))
        fallback_reason = str(row.get("source_seek_fallback_reason", ""))
        if positioning_mode not in {
            "sequential_from_zero",
            "random_seek_verified",
            "sequential_decode_fallback",
        }:
            raise A4DatasetValidationError(
                f"source_positioning_mode_invalid:{clip_id}:{positioning_mode}"
            )
        if positioning_mode == "sequential_decode_fallback":
            if not fallback_reason or fallback_reason == "none":
                raise A4DatasetValidationError(
                    f"source_seek_fallback_reason_missing:{clip_id}"
                )
        elif fallback_reason != "none":
            raise A4DatasetValidationError(
                f"source_seek_fallback_reason_unexpected:{clip_id}:{fallback_reason}"
            )
        previous_split = content_splits.setdefault(content_hash, split)
        if previous_split != split:
            raise A4DatasetValidationError(
                f"train_heldout_content_overlap:{content_hash}:{previous_split}:{split}"
            )
        previous_clip = content_clips.setdefault(content_hash, clip_id)
        if previous_clip != clip_id:
            raise A4DatasetValidationError(
                f"generated_content_duplicate:{content_hash}:{previous_clip}:{clip_id}"
            )
        if verify_content_hashes:
            actual = sha256_file(path)
            if actual != content_hash:
                raise A4DatasetValidationError(
                    f"rendered_content_sha256_mismatch:{clip_id}:expected={content_hash},actual={actual}"
                )
        groups.setdefault(str(row.get("base_clip_id", "")), []).append(row)

    for base_clip_id, variants in groups.items():
        if not base_clip_id:
            raise A4DatasetValidationError("base_clip_id_missing")
        splits = {str(row["split"]) for row in variants}
        scenes = {str(row["scene_id"]) for row in variants}
        base_hashes = {str(row["base_source_sha256"]).lower() for row in variants}
        source_hashes = {str(row["source_manifest_sha256"]).lower() for row in variants}
        source_windows = {
            (
                str(row["source_start_frame"]),
                str(row["source_end_frame_exclusive"]),
                str(row["source_frame_count"]),
                str(row["source_positioning_mode"]),
                str(row["source_seek_fallback_reason"]),
            )
            for row in variants
        }
        if (
            len(splits) != 1
            or len(scenes) != 1
            or len(base_hashes) != 1
            or len(source_hashes) != 1
            or len(source_windows) != 1
        ):
            raise A4DatasetValidationError(
                f"base_group_split_scene_or_provenance_leakage:{base_clip_id}"
            )
        attack_types = {str(row["attack_type"]) for row in variants}
        expected_types = {"clean", *REQUIRED_ATTACK_TYPES}
        if not expected_types.issubset(attack_types):
            raise A4DatasetValidationError(
                f"base_group_attack_coverage_missing:{base_clip_id}:{sorted(expected_types - attack_types)}"
            )
        trajectories = {
            str(row["trajectory_mode"])
            for row in variants
            if str(row["attack_type"]) == "adv_patch"
        }
        if len(trajectories) != 1 or not trajectories.issubset(
            set(ADV_PATCH_TRAJECTORY_MODES)
        ):
            raise A4DatasetValidationError(
                "base_group_adv_patch_trajectory_invalid:"
                f"{base_clip_id}:{sorted(trajectories)}"
            )

    for split in sorted({str(row["split"]) for row in rows}):
        split_trajectories = {
            str(row["trajectory_mode"])
            for row in rows
            if str(row["split"]) == split and str(row["attack_type"]) == "adv_patch"
        }
        missing_trajectories = set(ADV_PATCH_TRAJECTORY_MODES) - split_trajectories
        if missing_trajectories:
            raise A4DatasetValidationError(
                "split_adv_patch_trajectory_coverage_missing:"
                f"{split}:{sorted(missing_trajectories)}"
            )


def load_a4_dataset_manifest(
    manifest_path: str | Path,
    *,
    verify_content_hashes: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = assert_no_rebuilt_demo_path(manifest_path, field="a4_dataset_manifest")
    metadata_path = manifest_metadata_path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not metadata_path.is_file():
        raise A4DatasetValidationError(f"a4_dataset_metadata_missing:{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise A4DatasetValidationError("a4_dataset_metadata_invalid:not_an_object")
    if int(metadata.get("schema_version", 0) or 0) != A4_DATASET_SCHEMA_VERSION:
        raise A4DatasetValidationError("a4_dataset_schema_version_mismatch")
    if str(metadata.get("generator_version", "")) != A4_GENERATOR_VERSION:
        raise A4DatasetValidationError("a4_dataset_generator_version_mismatch")
    if str(metadata.get("unique_yolo_source_sha256", "")).lower() != UNIQUE_YOLO_SOURCE_SHA256.lower():
        raise A4DatasetValidationError("unique_yolo_source_sha256_mismatch")
    declared_manifest_hash = _require_sha256(
        metadata.get("dataset_manifest_sha256"),
        field="dataset_manifest_sha256",
    )
    actual_manifest_hash = sha256_file(manifest)
    if actual_manifest_hash != declared_manifest_hash:
        raise A4DatasetValidationError(
            "dataset_manifest_sha256_mismatch:"
            f"expected={declared_manifest_hash},actual={actual_manifest_hash}"
        )
    source_manifest_path = assert_no_rebuilt_demo_path(
        metadata.get("source_manifest_path", ""),
        field="source_manifest_path",
    )
    source_manifest_hash = _require_sha256(
        metadata.get("source_manifest_sha256"),
        field="source_manifest_sha256",
    )
    if not source_manifest_path.is_file() or sha256_file(source_manifest_path) != source_manifest_hash:
        raise A4DatasetValidationError("source_manifest_binding_mismatch")
    authoritative_hashes = metadata.get("authoritative_video_sha256", [])
    if not isinstance(authoritative_hashes, list):
        raise A4DatasetValidationError("authoritative_video_sha256_not_a_list")
    authoritative_hashes = [
        _require_sha256(value, field="authoritative_video_sha256")
        for value in authoritative_hashes
    ]
    expected_count = int(metadata.get("authoritative_video_count", -1) or -1)
    if expected_count != 36 or len(authoritative_hashes) != 36 or len(set(authoritative_hashes)) != 36:
        raise A4DatasetValidationError(
            "authoritative_video_hash_contract_mismatch:expected=36:"
            f"declared={expected_count}:actual={len(authoritative_hashes)}"
        )
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    _validate_dataset_rows(
        rows,
        authoritative_video_hashes=authoritative_hashes,
        verify_content_hashes=verify_content_hashes,
    )
    if int(metadata.get("base_count", -1)) != len({row["base_clip_id"] for row in rows}):
        raise A4DatasetValidationError("dataset_base_count_mismatch")
    if int(metadata.get("clip_count", -1)) != len(rows):
        raise A4DatasetValidationError("dataset_clip_count_mismatch")
    return rows, metadata


def build_a4_training_dataset(
    *,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    output_manifest_path: str | Path,
    metadata_path: str | Path | None = None,
    authoritative_manifest_path: str | Path = DEFAULT_AUTHORITATIVE_MANIFEST,
    max_frames_per_video: int = 90,
    clip_duration_s: float | None = 3.0,
    generator_seed: int = 20260716,
    attack_start_frame: int = 12,
    attack_ramp_frames: int = 8,
    codec: str = "mp4v",
    yolo_device: str = "0",
    yolo_confidence: float = 0.25,
    yolo_image_size: int = 640,
    yolo_inference_stride: int = 5,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    _anchor_provider: AnchorProvider | None = None,
) -> dict[str, Any]:
    if len(str(codec)) != 4:
        raise ValueError("codec must be a four-character code")
    output_root = assert_no_rebuilt_demo_path(output_dir, field="output_dir")
    output_manifest = assert_no_rebuilt_demo_path(
        output_manifest_path,
        field="output_manifest_path",
    )
    metadata_destination = assert_no_rebuilt_demo_path(
        metadata_path or manifest_metadata_path(output_manifest),
        field="metadata_path",
    )
    source_manifest = assert_no_rebuilt_demo_path(
        source_manifest_path,
        field="source_manifest_path",
    )
    authoritative = load_authoritative_contract(authoritative_manifest_path)
    model_path = authoritative["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    actual_model_hash = sha256_file(model_path)
    if actual_model_hash != UNIQUE_YOLO_SOURCE_SHA256.lower():
        raise A4DatasetValidationError(
            "unique_yolo_source_file_sha256_mismatch:"
            f"expected={UNIQUE_YOLO_SOURCE_SHA256.lower()},actual={actual_model_hash}"
        )
    source_manifest_hash = sha256_file(source_manifest)
    clean_rows = load_clean_sources(
        source_manifest,
        authoritative_video_hashes=authoritative["video_hashes"],
        verify_source_hashes=True,
    )
    trajectory_by_base: dict[str, str] = {}
    for split in sorted({str(row["split"]) for row in clean_rows}):
        split_rows = sorted(
            (row for row in clean_rows if str(row["split"]) == split),
            key=lambda row: (str(row["base_source_sha256"]), str(row["clip_id"])),
        )
        if len(split_rows) < len(ADV_PATCH_TRAJECTORY_MODES):
            raise A4DatasetValidationError(
                "split_requires_at_least_five_bases_for_adv_patch_trajectory_coverage:"
                f"{split}:{len(split_rows)}"
            )
        rotation = _stable_seed(
            source_manifest_hash,
            split,
            "adv_patch_trajectory_assignment",
            seed=generator_seed,
        ) % len(ADV_PATCH_TRAJECTORY_MODES)
        for index, row in enumerate(split_rows):
            trajectory_by_base[str(row["clip_id"])] = ADV_PATCH_TRAJECTORY_MODES[
                (index + rotation) % len(ADV_PATCH_TRAJECTORY_MODES)
            ]
    anchor_provider: AnchorProvider = _anchor_provider or _YoloAnchorProvider(
        model_path,
        device=yolo_device,
        confidence=yolo_confidence,
        image_size=yolo_image_size,
        inference_stride=yolo_inference_stride,
    )
    generated_rows: list[dict[str, Any]] = []
    anchor_summaries: dict[str, Any] = {}
    for index, clean_row in enumerate(clean_rows, start=1):
        base_rows, anchor_summary = _render_base_variants(
            clean_row,
            output_dir=output_root,
            source_manifest_sha256=source_manifest_hash,
            generator_seed=generator_seed,
            max_frames_per_video=max_frames_per_video,
            clip_duration_s=clip_duration_s,
            attack_start_frame=attack_start_frame,
            attack_ramp_frames=attack_ramp_frames,
            codec=codec,
            anchor_provider=anchor_provider,
            adv_patch_trajectory_mode=trajectory_by_base[str(clean_row["clip_id"])],
        )
        generated_rows.extend(base_rows)
        anchor_summaries[str(clean_row["clip_id"])] = anchor_summary
        if progress:
            progress(
                {
                    "base_index": index,
                    "base_count": len(clean_rows),
                    "base_clip_id": str(clean_row["clip_id"]),
                    "generated_clips": len(base_rows),
                    "generated_total": len(generated_rows),
                }
            )

    _validate_dataset_rows(
        generated_rows,
        authoritative_video_hashes=authoritative["video_hashes"],
        verify_content_hashes=True,
    )
    _write_csv(output_manifest, generated_rows)
    metadata = {
        "schema_version": A4_DATASET_SCHEMA_VERSION,
        "generator_version": A4_GENERATOR_VERSION,
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": source_manifest_hash,
        "dataset_manifest_path": str(output_manifest),
        "dataset_manifest_sha256": sha256_file(output_manifest),
        "output_dir": str(output_root),
        "unique_yolo_source_path": str(model_path),
        "unique_yolo_source_sha256": UNIQUE_YOLO_SOURCE_SHA256.lower(),
        "authoritative_manifest_path": str(authoritative["manifest_path"]),
        "authoritative_manifest_sha256": authoritative["manifest_sha256"],
        "authoritative_video_count": len(authoritative["video_hashes"]),
        "authoritative_video_sha256": list(authoritative["video_hashes"]),
        "base_count": len(clean_rows),
        "clip_count": len(generated_rows),
        "variants_per_base": len(_variant_specs(ADV_PATCH_TRAJECTORY_MODES[0])),
        "required_attack_types": list(REQUIRED_ATTACK_TYPES),
        "adv_patch_trajectory_modes": list(ADV_PATCH_TRAJECTORY_MODES),
        "adv_patch_trajectory_assignment": trajectory_by_base,
        "adv_patch_frame_area_ratio_bounds": [
            _ADV_PATCH_FRAME_AREA_RATIO_MIN,
            _ADV_PATCH_FRAME_AREA_RATIO_MAX,
        ],
        "generator_seed": int(generator_seed),
        "max_frames_per_video": int(max_frames_per_video),
        "clip_duration_s": None if clip_duration_s is None else float(clip_duration_s),
        "requested_attack_start_frame": int(attack_start_frame),
        "requested_attack_ramp_frames": int(attack_ramp_frames),
        "codec": str(codec),
        "anchor_contract": {
            "production_provider": "unique_yolo_largest_detection_smoothed",
            "yolo_device": str(yolo_device),
            "yolo_confidence": float(yolo_confidence),
            "yolo_image_size": int(yolo_image_size),
            "yolo_inference_stride": int(yolo_inference_stride),
            "per_base": anchor_summaries,
        },
        "deterministic_provenance": True,
        "trajectory_parameter_source": (
            "predefined_broad_grid_seeded_by_base_hash_not_authoritative_acceptance"
        ),
        "clean_and_variants_share_base_split_scene": True,
        "content_deduplicated": True,
    }
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_destination.with_suffix(metadata_destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(metadata_destination)
    if metadata_destination != manifest_metadata_path(output_manifest):
        manifest_metadata_path(output_manifest).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    load_a4_dataset_manifest(output_manifest, verify_content_hashes=True)
    return metadata
