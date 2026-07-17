from __future__ import annotations

import numpy as np
import pytest

from defense.module_a import native_bridge


def _border_values(patch: np.ndarray, border: int) -> np.ndarray:
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


def _python_reference(
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
        inner_mean = float(np.mean(inner_gray))
    else:
        # Preserve the existing native scalar API's small-box semantics.
        inner_mean = float(np.mean(gray_border))
    return (
        edge_density,
        border_edge_density,
        inner_edge_density,
        float(np.mean(gray_border)),
        inner_mean,
        float(np.std(local_gray)),
    )


@pytest.fixture(scope="module")
def native():
    if not native_bridge.available:
        pytest.skip(
            "module_a_native unavailable: "
            f"fallback_reason={native_bridge.fallback_reason!r}; "
            f"load_error={native_bridge.load_error!r}"
        )
    return native_bridge.require_native()


def _boundary_boxes(height: int, width: int) -> list[tuple[int, int, int, int]]:
    return [
        (0, 0, 18, 18),
        (0, 0, width, height),
        (width - 18, height - 18, width, height),
        (0, height - 23, 41, height),
        (width - 37, 0, width, 29),
        (7, 11, width - 9, height - 13),
    ]


def test_bridge_requires_batch_contract_and_reports_hashes(native) -> None:
    status = native_bridge.status()
    assert native_bridge.EXPECTED_API_VERSION == "1"
    assert native_bridge.EXPECTED_A3B_BATCH_API_VERSION == "1"
    assert "a3b_boxes_stats" in native_bridge.REQUIRED_CAPABILITIES
    assert "a3b_boxes_stats" in status["capabilities"]
    assert status["api_version"] == "1"
    assert status["a3b_batch_api_version"] == "1"
    assert status["build_info"]["api_version"] == "1"
    assert status["build_info"]["a3b_batch_api_version"] == "1"
    assert len(status["binary_sha256"]) == 64
    assert len(status["source_sha256"]) == 64
    assert callable(native.a3b_boxes_stats)
    assert native.a3b_batch_api_version() == "1"


@pytest.mark.parametrize("seed", [0, 7, 20260715, 20260716])
def test_batch_matches_scalar_api_and_python_reference(native, seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = (79, 113)
    gray = rng.integers(0, 256, size=shape, dtype=np.uint8)
    edge_mask = rng.integers(0, 2, size=shape, dtype=np.uint8)
    boxes = _boundary_boxes(*shape)

    batch = native.a3b_boxes_stats(edge_mask, gray, boxes)
    scalar = [
        native.a3b_one_box_stats(edge_mask, gray, *box)
        for box in boxes
    ]
    reference = [_python_reference(edge_mask, gray, box) for box in boxes]

    assert batch == scalar
    np.testing.assert_allclose(
        np.asarray(batch, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        atol=2e-4,
        rtol=2e-6,
    )


@pytest.mark.parametrize("seed", [1, 19, 20260716])
def test_batch_accepts_non_contiguous_uint8_views(native, seed: int) -> None:
    rng = np.random.default_rng(seed)
    gray_storage = rng.integers(0, 256, size=(158, 226), dtype=np.uint8)
    edge_storage = rng.integers(0, 2, size=(158, 226), dtype=np.uint8)
    gray = gray_storage[::2, ::2]
    edge_mask = edge_storage[::2, ::2]
    assert not gray.flags.c_contiguous
    assert not edge_mask.flags.c_contiguous
    boxes = _boundary_boxes(*gray.shape)

    batch = native.a3b_boxes_stats(edge_mask, gray, boxes)
    scalar = [
        native.a3b_one_box_stats(edge_mask, gray, *box)
        for box in boxes
    ]
    reference = [_python_reference(edge_mask, gray, box) for box in boxes]

    assert batch == scalar
    np.testing.assert_allclose(
        np.asarray(batch, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        atol=2e-4,
        rtol=2e-6,
    )


def test_batch_empty_boxes_returns_empty_list(native) -> None:
    edge_mask = np.zeros((32, 48), dtype=np.uint8)
    gray = np.zeros((32, 48), dtype=np.uint8)
    assert native.a3b_boxes_stats(edge_mask, gray, []) == []


@pytest.mark.parametrize(
    "box",
    [
        (-1, 0, 18, 18),
        (0, -1, 18, 18),
        (4, 4, 4, 18),
        (4, 4, 18, 4),
        (0, 0, 49, 18),
        (0, 0, 18, 33),
    ],
)
def test_batch_rejects_out_of_bounds_or_invalid_boxes(native, box) -> None:
    edge_mask = np.zeros((32, 48), dtype=np.uint8)
    gray = np.zeros((32, 48), dtype=np.uint8)
    with pytest.raises(ValueError, match="bbox must satisfy"):
        native.a3b_boxes_stats(edge_mask, gray, [(0, 0, 18, 18), box])


def test_batch_rejects_dtype_shape_rank_and_box_shape_errors(native) -> None:
    edge_mask = np.zeros((32, 48), dtype=np.uint8)
    gray = np.zeros((32, 48), dtype=np.uint8)
    boxes = [(0, 0, 18, 18)]

    with pytest.raises(TypeError):
        native.a3b_boxes_stats(edge_mask.astype(np.float32), gray, boxes)
    with pytest.raises(TypeError):
        native.a3b_boxes_stats(edge_mask, gray.astype(np.int16), boxes)
    with pytest.raises(TypeError):
        native.a3b_boxes_stats(edge_mask[:, :, None], gray, boxes)
    with pytest.raises(ValueError, match="same shape"):
        native.a3b_boxes_stats(edge_mask, gray[:-1], boxes)
    with pytest.raises((TypeError, ValueError)):
        native.a3b_boxes_stats(edge_mask, gray, [(0, 0, 18)])
    with pytest.raises((TypeError, ValueError, OverflowError)):
        native.a3b_boxes_stats(edge_mask, gray, [(0, 0, 2**80, 18)])
