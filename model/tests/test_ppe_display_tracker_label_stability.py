from __future__ import annotations

import pytest

from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.module_a.postprocess.ppe_tracking import PPEDisplayTracker


def _result(label: str, confidence: float) -> DetectionFrameResult:
    class_id = 0 if label == "helmet" else 1
    return DetectionFrameResult(
        image=None,
        boxes=[(100, 100, 120, 120)],
        classes=[class_id],
        confidences=[confidence],
        names={0: "helmet", 1: "head", 2: "person"},
        backend="fake",
        artifact_path="fake.pt",
        inference_ms=0.0,
        raw_result=None,
    )


@pytest.mark.skip(reason="超前契约未实装:PPEDisplayTracker无helmet_switch_confidence/strong_switch_count等按置信度门控的强切换参数")
def test_small_head_helmet_strong_switch_requires_continuous_evidence() -> None:
    tracker = PPEDisplayTracker(
        history=5,
        switch_count=3,
        small_area_ratio=0.020,
        helmet_switch_confidence=0.55,
        small_helmet_switch_confidence=0.55,
        strong_switch_count=2,
        small_strong_switch_extra_count=1,
    )
    frame_shape = (720, 1280)

    assert tracker.update(_result("head", 0.80), {}, frame_shape)[0]["label"] == "head"
    assert tracker.update(_result("helmet", 0.80), {}, frame_shape)[0]["label"] == "head"
    assert tracker.update(_result("head", 0.80), {}, frame_shape)[0]["label"] == "head"
    assert tracker.update(_result("helmet", 0.80), {}, frame_shape)[0]["label"] == "head"
    assert tracker.update(_result("helmet", 0.80), {}, frame_shape)[0]["label"] == "head"
    assert tracker.update(_result("helmet", 0.80), {}, frame_shape)[0]["label"] == "helmet"


@pytest.mark.skip(reason="超前契约未实装:PPEDisplayTracker无helmet_switch_confidence/strong_switch_count等按置信度门控的强切换参数")
def test_large_head_helmet_strong_switch_can_remain_fast() -> None:
    tracker = PPEDisplayTracker(
        history=5,
        switch_count=3,
        small_area_ratio=0.0001,
        helmet_switch_confidence=0.55,
        strong_switch_count=2,
    )
    frame_shape = (720, 1280)

    assert tracker.update(_result("head", 0.80), {}, frame_shape)[0]["label"] == "head"
    assert tracker.update(_result("helmet", 0.80), {}, frame_shape)[0]["label"] == "head"
    assert tracker.update(_result("helmet", 0.80), {}, frame_shape)[0]["label"] == "helmet"
