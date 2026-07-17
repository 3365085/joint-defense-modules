from __future__ import annotations

import numpy as np
import pytest

from defense.module_a.rebuilt.a4_patch_features import (
    A4_PATCH_FEATURE_NAMES,
    extract_a4_patch_features,
)


def _feature_map(frame: np.ndarray) -> dict[str, float]:
    return dict(zip(A4_PATCH_FEATURE_NAMES, extract_a4_patch_features(frame), strict=True))


def test_feature_names_have_stable_map_major_order() -> None:
    maps = (
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
    summaries = (
        "global_mean",
        "tile_p90",
        "tile_max",
        "max_to_global",
    )

    assert A4_PATCH_FEATURE_NAMES == tuple(
        f"a4_patch.{map_name}.{summary_name}"
        for map_name in maps
        for summary_name in summaries
    )
    assert len(A4_PATCH_FEATURE_NAMES) == 40
    assert len(set(A4_PATCH_FEATURE_NAMES)) == 40


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.empty((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.full((10, 10, 3), np.nan, dtype=np.float32),
        np.full((10, 10, 3), "invalid", dtype=object),
    ],
)
def test_invalid_frames_return_stable_zero_vector(frame: np.ndarray | None) -> None:
    features = extract_a4_patch_features(frame)

    assert features == (0.0,) * 40


def test_valid_frames_return_finite_deterministic_features() -> None:
    frame = np.full((352, 640, 3), 128, dtype=np.uint8)

    first = extract_a4_patch_features(frame)
    second = extract_a4_patch_features(frame.copy())

    assert len(first) == 40
    assert np.isfinite(first).all()
    assert first == second
    assert _feature_map(frame)["a4_patch.value.global_mean"] == pytest.approx(128 / 255)


def test_color_checker_patch_is_distinct_from_plain_clean_frame() -> None:
    clean = np.full((352, 640, 3), 128, dtype=np.uint8)
    attacked = clean.copy()
    colors = np.asarray(
        [
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [0, 255, 255],
        ],
        dtype=np.uint8,
    )
    block_size = 16
    patch_top = 96
    patch_left = 240
    for row in range(10):
        for column in range(10):
            attacked[
                patch_top + row * block_size : patch_top + (row + 1) * block_size,
                patch_left + column * block_size : patch_left + (column + 1) * block_size,
            ] = colors[(row + column) % len(colors)]

    clean_features = np.asarray(extract_a4_patch_features(clean))
    attacked_features = np.asarray(extract_a4_patch_features(attacked))
    clean_by_name = _feature_map(clean)
    attacked_by_name = _feature_map(attacked)

    assert np.abs(attacked_features - clean_features).sum() > 2.0
    assert attacked_by_name["a4_patch.saturation.tile_max"] > 0.50
    assert attacked_by_name["a4_patch.high_chroma_ratio.tile_max"] > 0.50
    assert attacked_by_name["a4_patch.gradient.tile_max"] > 0.10
    assert attacked_by_name["a4_patch.laplacian.tile_max"] > clean_by_name[
        "a4_patch.laplacian.tile_max"
    ]
