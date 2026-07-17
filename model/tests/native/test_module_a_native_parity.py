from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from defense.module_a import native_bridge


def _hist_lbp(
    lbp: np.ndarray,
    box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    if box is not None:
        x1, y1, x2, y2 = box
        patch = lbp[y1:y2, x1:x2]
    else:
        patch = lbp
    if patch.size == 0:
        return np.zeros(32, dtype=np.float32)
    hist = cv2.calcHist(
        [patch.astype(np.uint8)],
        [0],
        None,
        [32],
        [0, 256],
    )
    hist = hist.reshape(-1).astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0.0 else np.zeros(32, dtype=np.float32)


def _hist_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(0.5 * np.abs(left - right).sum())


def _a1_reference(
    lbp: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    baseline: np.ndarray | None,
) -> tuple[object, ...]:
    height, width = lbp.shape
    global_hist = _hist_lbp(lbp)
    baseline_hist = global_hist if baseline is None else baseline
    delta_h_global = _hist_distance(global_hist, baseline_hist)

    grid_scores: list[tuple[float, tuple[int, int, int, int]]] = []
    cell_width = max(16, width // 8)
    cell_height = max(16, height // 8)
    for y in range(0, height, cell_height):
        for x in range(0, width, cell_width):
            box = (x, y, min(width, x + cell_width), min(height, y + cell_height))
            grid_scores.append((_hist_distance(_hist_lbp(lbp, box), global_hist), box))
    grid_scores.sort(key=lambda item: item[0], reverse=True)
    delta_h_local_max = float(grid_scores[0][0]) if grid_scores else 0.0
    local_box = grid_scores[0][1] if grid_scores else (0, 0, width, height)
    local_mean = (
        float(np.mean([item[0] for item in grid_scores])) if grid_scores else 0.0
    )

    delta_h_roi_max = 0.0
    delta_h_target_contrast = 0.0
    delta_h_roi_patch_max = 0.0
    target_box: tuple[int, int, int, int] | None = None
    for roi in rois:
        roi_hist = _hist_lbp(lbp, roi)
        contrast = _hist_distance(roi_hist, global_hist)
        baseline_contrast = _hist_distance(roi_hist, baseline_hist)
        roi_score = max(contrast, baseline_contrast)
        x1, y1, x2, y2 = roi
        sub_width = max(8, (x2 - x1) // 4)
        sub_height = max(8, (y2 - y1) // 4)
        for sub_y in range(y1, y2, sub_height):
            for sub_x in range(x1, x2, sub_width):
                patch_box = (
                    sub_x,
                    sub_y,
                    min(x2, sub_x + sub_width),
                    min(y2, sub_y + sub_height),
                )
                if (
                    patch_box[2] - patch_box[0] < 8
                    or patch_box[3] - patch_box[1] < 8
                ):
                    continue
                patch_hist = _hist_lbp(lbp, patch_box)
                delta_h_roi_patch_max = max(
                    delta_h_roi_patch_max,
                    _hist_distance(patch_hist, roi_hist),
                    _hist_distance(patch_hist, baseline_hist),
                    _hist_distance(patch_hist, global_hist),
                )
        if roi_score > delta_h_roi_max:
            delta_h_roi_max = roi_score
            delta_h_target_contrast = contrast
            target_box = roi

    return (
        delta_h_global,
        delta_h_local_max,
        local_mean,
        local_box,
        delta_h_roi_max,
        delta_h_target_contrast,
        delta_h_roi_patch_max,
        target_box,
    )


def _best_grid_reference(
    value_map: np.ndarray,
    grid: int,
) -> tuple[float, float, tuple[int, int, int, int]]:
    height, width = value_map.shape
    cell_width = max(8, width // max(1, grid))
    cell_height = max(8, height // max(1, grid))
    best = 0.0
    total = 0.0
    count = 0
    best_box = (0, 0, width, height)
    for y in range(0, height, cell_height):
        for x in range(0, width, cell_width):
            x2 = min(width, x + cell_width)
            y2 = min(height, y + cell_height)
            patch = value_map[y:y2, x:x2]
            if patch.size == 0:
                continue
            value = float(np.mean(patch))
            total += value
            count += 1
            if value > best:
                best = value
                best_box = (x, y, x2, y2)
    return best, total / max(1, count), best_box


def _expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx = int(max(1, x2 - x1) * margin)
    dy = int(max(1, y2 - y1) * margin)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(width, x2 + dx),
        min(height, y2 + dy),
    )


def _a2_reference(
    lbp: np.ndarray,
    previous: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    expand_margin: float,
) -> tuple[object, ...]:
    diff = cv2.absdiff(lbp, previous).astype(np.float32) / 255.0
    change_t_global = float(np.mean(diff))
    change_t_local_max, change_t_local_mean, local_box = _best_grid_reference(diff, 8)
    change_t_roi_max = 0.0
    change_t_context_mean = change_t_local_mean
    target_box: tuple[int, int, int, int] | None = None
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_change = (
            float(np.mean(diff[y1:y2, x1:x2])) if x2 > x1 and y2 > y1 else 0.0
        )
        if roi_change > change_t_roi_max:
            change_t_roi_max = roi_change
            target_box = roi
    if target_box is not None:
        x1, y1, x2, y2 = target_box
        outer_x1, outer_y1, outer_x2, outer_y2 = _expand_box(
            target_box,
            lbp.shape[1],
            lbp.shape[0],
            expand_margin,
        )
        ring = diff[outer_y1:outer_y2, outer_x1:outer_x2].copy()
        if ring.size:
            ring[
                y1 - outer_y1 : y2 - outer_y1,
                x1 - outer_x1 : x2 - outer_x1,
            ] = np.nan
            change_t_context_mean = (
                float(np.nanmean(ring))
                if np.isfinite(ring).any()
                else change_t_local_mean
            )
    return (
        change_t_global,
        change_t_local_max,
        change_t_local_mean,
        local_box,
        change_t_roi_max,
        target_box,
        change_t_context_mean,
    )


def _border_values(patch: np.ndarray, border: int) -> np.ndarray:
    if patch.size == 0:
        return np.asarray([], dtype=patch.dtype)
    height, width = patch.shape
    border = max(1, min(border, max(1, height // 2), max(1, width // 2)))
    return np.concatenate(
        [
            patch[:border, :].reshape(-1),
            patch[height - border : height, :].reshape(-1),
            patch[:, :border].reshape(-1),
            patch[:, width - border : width].reshape(-1),
        ]
    )


def _a3b_reference(
    edge_mask: np.ndarray,
    gray: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[float, ...]:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    local_edges = edge_mask[y1:y2, x1:x2]
    edge_density = float(np.mean(local_edges))
    border = max(2, int(min(box_width, box_height) * 0.035))
    border_edges = _border_values(local_edges.astype(np.uint8), border)
    border_edge_density = float(np.mean(border_edges))
    if box_width > 2 * border + 2 and box_height > 2 * border + 2:
        inner_edges = local_edges[
            border : box_height - border,
            border : box_width - border,
        ]
        inner_edge_density = float(np.mean(inner_edges))
    else:
        inner_edge_density = edge_density

    local_gray = gray[y1:y2, x1:x2].astype(np.float32)
    gray_border = _border_values(local_gray, border)
    if box_width > 2 * border + 2 and box_height > 2 * border + 2:
        inner_gray = local_gray[
            border : box_height - border,
            border : box_width - border,
        ]
    else:
        inner_gray = local_gray
    border_mean = float(np.mean(gray_border))
    inner_mean = float(np.mean(inner_gray))
    gray_std = float(np.std(local_gray))
    return (
        edge_density,
        border_edge_density,
        inner_edge_density,
        border_mean,
        inner_mean,
        gray_std,
    )


def _assert_feature_tuple_close(
    actual: tuple[object, ...],
    expected: tuple[object, ...],
    *,
    float_indices: tuple[int, ...],
    exact_indices: tuple[int, ...],
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    for index in float_indices:
        assert float(actual[index]) == pytest.approx(
            float(expected[index]),
            abs=atol,
            rel=rtol,
        )
    for index in exact_indices:
        actual_value = actual[index]
        if isinstance(actual_value, (list, tuple)):
            actual_value = tuple(actual_value)
        expected_value = expected[index]
        if isinstance(expected_value, (list, tuple)):
            expected_value = tuple(expected_value)
        assert actual_value == expected_value


@pytest.fixture(scope="module")
def native():
    if not native_bridge.available:
        pytest.skip(
            "module_a_native unavailable: "
            f"fallback_reason={native_bridge.fallback_reason!r}; "
            f"load_error={native_bridge.load_error!r}"
        )
    return native_bridge.require_native()


def test_bridge_status_is_explicit_and_source_owned() -> None:
    status = native_bridge.status()
    assert status["source_root"].endswith(r"model\native\module_a_native")
    assert status["source_manifest"]
    assert len(status["source_sha256"]) == 64
    assert "rebuilt_demo" not in status["source_root"].casefold()
    if status["available"]:
        assert status["load_error"] is None
        assert status["fallback_reason"] is None
    else:
        assert status["load_error"]
        assert status["fallback_reason"]


def test_bridge_metadata_and_binary_hash(native) -> None:
    status = native_bridge.status()
    assert status["api_version"] == native_bridge.EXPECTED_API_VERSION
    assert status["crate_version"] == native_bridge.EXPECTED_CRATE_VERSION
    assert set(native_bridge.REQUIRED_CAPABILITIES) <= set(status["capabilities"])
    assert status["binary_path"].casefold().endswith(".pyd")
    assert "rebuilt_demo" not in status["binary_path"].casefold()
    assert len(status["binary_sha256"]) == 64
    assert status["build_info"]["crate_version"] == "0.2.0"
    assert status["build_info"]["api_version"] == "1"
    assert status["build_info"]["panic_strategy"] == "unwind"
    assert native.crate_version() == "0.2.0"
    assert native.api_version() == "1"


@pytest.mark.parametrize("seed", [0, 1, 20260715])
@pytest.mark.parametrize("shape", [(1, 1), (17, 31), (64, 96), (127, 191)])
def test_a1_lbp_features_random_and_boundaries(native, seed, shape) -> None:
    rng = np.random.default_rng(seed)
    height, width = shape
    lbp = rng.integers(0, 256, size=shape, dtype=np.uint8)
    rois = [
        (0, 0, width, height),
        (0, 0, max(1, width // 2), max(1, height // 2)),
        (max(0, width - 18), max(0, height - 18), width, height),
    ]
    baseline = None
    if seed % 2:
        baseline = rng.random(32, dtype=np.float32)
        baseline /= baseline.sum(dtype=np.float32)
    actual = native.a1_lbp_features(
        np.ascontiguousarray(lbp),
        rois,
        None if baseline is None else np.ascontiguousarray(baseline),
    )
    expected = _a1_reference(lbp, rois, baseline)
    _assert_feature_tuple_close(
        actual,
        expected,
        float_indices=(0, 1, 2, 4, 5, 6),
        exact_indices=(3, 7),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("seed", [0, 4, 20260715])
@pytest.mark.parametrize("shape", [(8, 9), (63, 95), (128, 192)])
def test_a2_change_features_random_and_boundaries(native, seed, shape) -> None:
    rng = np.random.default_rng(seed)
    height, width = shape
    lbp = rng.integers(0, 256, size=shape, dtype=np.uint8)
    previous = rng.integers(0, 256, size=shape, dtype=np.uint8)
    rois = [
        (0, 0, width, height),
        (0, 0, max(1, width // 3), max(1, height // 3)),
        (max(0, width - 18), max(0, height - 18), width, height),
        (1, 1, 1, 2),
    ]
    margin = 0.45
    actual = native.a2_change_features(lbp, previous, rois, margin)
    expected = _a2_reference(lbp, previous, rois, margin)
    _assert_feature_tuple_close(
        actual,
        expected,
        float_indices=(0, 1, 2, 4, 6),
        exact_indices=(3, 5),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("grid", [-3, 0, 1, 3, 8, 64])
@pytest.mark.parametrize("mode", ["random", "zeros", "negative"])
def test_best_grid_value_f32_parity(native, grid, mode) -> None:
    rng = np.random.default_rng(42)
    if mode == "random":
        values = rng.random((73, 117), dtype=np.float32)
    elif mode == "zeros":
        values = np.zeros((9, 17), dtype=np.float32)
    else:
        values = -rng.random((31, 49), dtype=np.float32)
    actual = native.best_grid_value_f32(values, grid)
    expected = _best_grid_reference(values, grid)
    _assert_feature_tuple_close(
        actual,
        expected,
        float_indices=(0, 1),
        exact_indices=(2,),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("seed", [0, 7, 20260715])
@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 18, 18),
        (0, 0, 96, 64),
        (70, 40, 96, 64),
        (9, 7, 73, 55),
    ],
)
def test_a3b_one_box_stats_parity(native, seed, box) -> None:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=(64, 96), dtype=np.uint8)
    edge_mask = rng.integers(0, 2, size=(64, 96), dtype=np.uint8)
    actual = native.a3b_one_box_stats(edge_mask, gray, *box)
    expected = _a3b_reference(edge_mask, gray, box)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        atol=2e-4,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "gray",
    [
        np.zeros((1, 1), dtype=np.uint8),
        np.full((17, 31), 127, dtype=np.uint8),
        np.arange(63, dtype=np.uint8).reshape(7, 9),
        np.random.default_rng(20260715).integers(
            0,
            256,
            size=(128, 192),
            dtype=np.uint8,
        ),
    ],
)
def test_blinding_laplacian_var_parity(native, gray) -> None:
    expected = float(cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var())
    actual = float(native.blinding_laplacian_var(gray))
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-3, rel=2e-5)


def test_invalid_inputs_raise_python_exceptions_without_abort(native) -> None:
    lbp = np.zeros((32, 48), dtype=np.uint8)
    gray = np.zeros((32, 48), dtype=np.uint8)
    edge_mask = np.zeros((32, 48), dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly 32"):
        native.a1_lbp_features(lbp, [], np.zeros(31, dtype=np.float32))
    with pytest.raises(ValueError, match="supported i32 range"):
        native.a1_lbp_features(lbp, [(0, 0, 2**40, 10)], None)
    with pytest.raises(ValueError, match="same shape"):
        native.a2_change_features(lbp, np.zeros((31, 48), dtype=np.uint8), [], 0.45)
    with pytest.raises(ValueError, match="finite and >= 0"):
        native.a2_change_features(lbp, lbp, [], -0.1)
    with pytest.raises(ValueError, match="finite and >= 0"):
        native.a2_change_features(lbp, lbp, [], float("nan"))
    with pytest.raises(ValueError, match="same shape"):
        native.a3b_one_box_stats(
            edge_mask,
            np.zeros((31, 48), dtype=np.uint8),
            0,
            0,
            18,
            18,
        )

    invalid_boxes = [
        (-1, 0, 18, 18),
        (0, -1, 18, 18),
        (4, 4, 4, 18),
        (4, 4, 18, 4),
        (0, 0, 49, 18),
        (0, 0, 18, 33),
    ]
    for box in invalid_boxes:
        with pytest.raises(ValueError, match="bbox must satisfy"):
            native.a3b_one_box_stats(edge_mask, gray, *box)
