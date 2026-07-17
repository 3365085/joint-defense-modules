from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from defense.module_a.rebuilt.detector import (
    ModuleADetector,
    _projection_peak_lines,
)


def _detector(monkeypatch: pytest.MonkeyPatch) -> ModuleADetector:
    monkeypatch.setattr(
        ModuleADetector,
        "_load_classifier",
        lambda self, _path: None,
    )
    monkeypatch.setattr(
        ModuleADetector,
        "_load_flownet",
        lambda self: None,
    )
    return ModuleADetector(
        {
            "module_a": {
                "frame_size": 128,
                "static_image_enabled": False,
                "light_flow_enabled": False,
            }
        }
    )


def _scalar_projection_peak_lines(
    values: np.ndarray,
    limit: int,
    min_gap: int,
) -> list[tuple[int, float]]:
    if values.size == 0:
        return []
    vmax = float(np.max(values))
    if vmax <= 1e-6:
        return []
    norm = values.astype(np.float32) / vmax
    threshold = max(0.18, float(np.percentile(norm, 88)) * 0.82)
    groups: list[tuple[int, float]] = []
    start: int | None = None
    for idx, value in enumerate(norm):
        if value >= threshold and start is None:
            start = idx
        elif value < threshold and start is not None:
            end = idx
            segment = norm[start:end]
            weights = segment + 1e-4
            center = int(
                round(
                    float(
                        np.average(
                            np.arange(start, end),
                            weights=weights,
                        )
                    )
                )
            )
            groups.append((center, float(np.max(segment))))
            start = None
    if start is not None:
        end = len(norm)
        segment = norm[start:end]
        weights = segment + 1e-4
        center = int(
            round(
                float(
                    np.average(
                        np.arange(start, end),
                        weights=weights,
                    )
                )
            )
        )
        groups.append((center, float(np.max(segment))))
    groups.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for center, strength in groups:
        if all(
            abs(center - old_center) >= min_gap
            for old_center, _ in selected
        ):
            selected.append((center, strength))
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: item[0])
    return selected


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([], dtype=np.float32),
        np.zeros(64, dtype=np.float32),
        np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 0.0, 0.5, 0.8, 0.8], dtype=np.float32),
        np.linspace(0.0, 1.0, 127, dtype=np.float32),
        np.random.default_rng(20260716).random(128, dtype=np.float32),
        np.random.default_rng(20260717).random(720, dtype=np.float32),
        np.random.default_rng(20260718).random(1280, dtype=np.float32),
    ],
)
def test_projection_peak_lines_vectorization_preserves_scalar_output(
    values: np.ndarray,
) -> None:
    expected = _scalar_projection_peak_lines(values, limit=9, min_gap=8)

    actual = _projection_peak_lines(values, limit=9, min_gap=8)

    assert actual == expected


def test_a3b_candidate_stats_use_one_batch_native_call_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    gray = np.full((128, 128), 30, dtype=np.uint8)
    for coordinate in range(12, 117, 12):
        cv2.line(gray, (coordinate, 4), (coordinate, 123), 230, 2)
        cv2.line(gray, (4, coordinate), (123, coordinate), 230, 2)

    calls: list[tuple[str, int]] = []
    original_native_call = detector._native_call

    def recorded_native_call(
        stage: str,
        function_name: str,
        *args: Any,
    ) -> Any:
        if stage == "a3b":
            box_count = (
                len(args[2])
                if function_name == "a3b_boxes_stats"
                and len(args) >= 3
                else 1
            )
            calls.append((function_name, box_count))
        return original_native_call(stage, function_name, *args)

    detector._native_call = recorded_native_call  # type: ignore[method-assign]
    detector._extract_media_candidates(
        gray,
        rois=[],
        width=128,
        height=128,
    )

    assert calls
    assert [name for name, _count in calls] == ["a3b_boxes_stats"]
    box_count = calls[0][1]
    assert 1 < box_count <= 64
    assert box_count == 64
    assert detector.native_hit_counts["a3b"] + detector.native_fallback_counts[
        "a3b"
    ] == 1
    if bool(detector.native_status.get("available", False)):
        assert detector.native_hit_counts["a3b"] == 1
        assert detector.native_fallback_counts["a3b"] == 0
        assert detector.native_last_error is None
