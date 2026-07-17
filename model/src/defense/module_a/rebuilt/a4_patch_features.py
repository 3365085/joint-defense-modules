from __future__ import annotations

from typing import Final

import cv2
import numpy as np


A4_PATCH_FRAME_SIZE: Final[tuple[int, int]] = (320, 176)
A4_PATCH_TILE_SIZE: Final[int] = 16
A4_PATCH_MAP_NAMES: Final[tuple[str, ...]] = (
    "saturation",
    "value",
    "chroma",
    "gradient",
    "laplacian",
    "high_saturation_ratio",
    "high_chroma_ratio",
    "edge_ratio",
    "bright_ratio",
    "dark_ratio",
)
A4_PATCH_SUMMARY_NAMES: Final[tuple[str, ...]] = (
    "global_mean",
    "tile_p90",
    "tile_max",
    "max_to_global",
)
A4_PATCH_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f"a4_patch.{map_name}.{summary_name}"
    for map_name in A4_PATCH_MAP_NAMES
    for summary_name in A4_PATCH_SUMMARY_NAMES
)

_ZERO_FEATURES: Final[tuple[float, ...]] = (0.0,) * len(A4_PATCH_FEATURE_NAMES)


def _normalized_bgr(frame: np.ndarray | None) -> np.ndarray | None:
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
        or not (
            np.issubdtype(frame.dtype, np.integer)
            or np.issubdtype(frame.dtype, np.floating)
        )
    ):
        return None
    if frame.dtype == np.uint8:
        bgr = frame
    else:
        values = frame.astype(np.float32)
        if not np.isfinite(values).all():
            return None
        if np.issubdtype(frame.dtype, np.floating) and values.min() >= 0.0 and values.max() <= 1.0:
            values = values * 255.0
        bgr = np.clip(values, 0.0, 255.0).astype(np.uint8)

    try:
        return cv2.resize(bgr, A4_PATCH_FRAME_SIZE, interpolation=cv2.INTER_AREA)
    except cv2.error:
        return None


def _summarize_map(values: np.ndarray) -> tuple[float, ...]:
    tile_rows = A4_PATCH_FRAME_SIZE[1] // A4_PATCH_TILE_SIZE
    tile_columns = A4_PATCH_FRAME_SIZE[0] // A4_PATCH_TILE_SIZE
    tiles = values.reshape(
        tile_rows,
        A4_PATCH_TILE_SIZE,
        tile_columns,
        A4_PATCH_TILE_SIZE,
    ).transpose(0, 2, 1, 3)
    flat_tiles = tiles.reshape(tile_rows * tile_columns, -1)
    tile_means = flat_tiles.mean(axis=1)
    global_mean = float(values.mean())
    tile_max = float(tile_means.max())
    max_to_global = tile_max / global_mean if global_mean > 1e-8 else 0.0
    tile_index = int(0.90 * (tile_means.size - 1))
    tile_p90 = float(np.partition(tile_means, tile_index)[tile_index])
    return (
        global_mean,
        tile_p90,
        tile_max,
        float(max_to_global),
    )


def extract_a4_patch_features(frame: np.ndarray | None) -> tuple[float, ...]:
    """Extract deterministic full-frame tile features from a BGR image."""

    resized = _normalized_bgr(frame)
    if resized is None:
        return _ZERO_FEATURES

    bgr = resized.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    chroma = bgr.max(axis=2) - bgr.min(axis=2)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(
        cv2.magnitude(gradient_x, gradient_y) / np.sqrt(32.0),
        0.0,
        1.0,
    )
    laplacian = np.clip(
        np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=1)) / 4.0,
        0.0,
        1.0,
    )

    maps = (
        saturation,
        value,
        chroma,
        gradient,
        laplacian,
        (saturation >= 0.60).astype(np.float32),
        (chroma >= 0.35).astype(np.float32),
        (gradient >= 0.12).astype(np.float32),
        (value >= 0.85).astype(np.float32),
        (value <= 0.15).astype(np.float32),
    )
    features = tuple(
        feature
        for feature_map in maps
        for feature in _summarize_map(feature_map)
    )
    if len(features) != len(A4_PATCH_FEATURE_NAMES) or not np.isfinite(features).all():
        return _ZERO_FEATURES
    return features


__all__ = ["A4_PATCH_FEATURE_NAMES", "extract_a4_patch_features"]
