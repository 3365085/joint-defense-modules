from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_monitor_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "module_a_monitor_app.py"
    spec = importlib.util.spec_from_file_location("module_a_monitor_app_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safety_helmet_state_confirms_after_three_recent_candidates():
    monitor = load_monitor_module()
    state = monitor.SafetyHelmetState()

    outputs = [state.update({"candidate": True}) for _ in range(3)]

    assert outputs[0]["warning"] is False
    assert outputs[1]["warning"] is False
    assert outputs[2]["confirmed"] is True
    assert outputs[2]["warning"] is True
    assert outputs[2]["window"] == 6
    assert outputs[2]["trigger_count"] == 3


def test_safety_helmet_state_fast_confirms_strong_head_after_two_frames():
    monitor = load_monitor_module()
    state = monitor.SafetyHelmetState()

    ppe = {
        "candidate": True,
        "head_count": 1,
        "raw_head_count": 1,
        "helmet_count": 0,
        "promoted_head_count": 0,
        "low_conf_temporal_head_count": 0,
        "max_head_confidence": 0.82,
    }
    first = state.update(ppe)
    second = state.update(ppe)

    assert first["confirmed"] is False
    assert second["confirmed"] is True
    assert second["warning"] is True
    assert second["confirmed_source"] == "fast_head"
    assert second["fast_window_positive"] == 2
    assert second["fast_trigger_count"] == 2


def test_safety_helmet_state_does_not_fast_confirm_low_conf_temporal_candidate():
    monitor = load_monitor_module()
    state = monitor.SafetyHelmetState()

    ppe = {
        "candidate": True,
        "head_count": 1,
        "raw_head_count": 1,
        "helmet_count": 0,
        "promoted_head_count": 1,
        "low_conf_temporal_head_count": 1,
        "max_head_confidence": 0.82,
    }
    first = state.update(ppe)
    second = state.update(ppe)

    assert first["confirmed"] is False
    assert second["confirmed"] is False
    assert second["warning"] is False
    assert second["fast_window_positive"] == 0


def test_safety_helmet_state_keeps_confirmed_event_after_visual_warning_expires():
    monitor = load_monitor_module()
    state = monitor.SafetyHelmetState(
        window=2,
        trigger_count=2,
        hold_frames=1,
        event_hold_frames=4,
        fast_trigger_count=2,
    )

    ppe = {
        "candidate": True,
        "head_count": 1,
        "raw_head_count": 1,
        "helmet_count": 0,
        "missing_helmet_count": 1,
        "max_head_confidence": 0.50,
        "reason": "bare_head_without_matched_helmet",
    }
    state.update(ppe)
    confirmed = state.update(ppe)
    first_miss = state.update({"candidate": False})
    second_miss = state.update({"candidate": False})

    assert confirmed["confirmed"] is True
    assert confirmed["warning"] is True
    assert confirmed["event_active"] is True
    assert confirmed["event_last_reason"] == "bare_head_without_matched_helmet"
    assert first_miss["warning"] is False
    assert first_miss["event_active"] is True
    assert second_miss["warning"] is False
    assert second_miss["event_active"] is True
    assert second_miss["event_last_confirmed_source"] == "temporal_window"


def test_safety_helmet_state_event_expires_after_hold_window():
    monitor = load_monitor_module()
    state = monitor.SafetyHelmetState(
        window=2,
        trigger_count=2,
        hold_frames=0,
        event_hold_frames=2,
        fast_trigger_count=2,
    )

    ppe = {
        "candidate": True,
        "head_count": 1,
        "raw_head_count": 1,
        "helmet_count": 0,
        "missing_helmet_count": 1,
        "max_head_confidence": 0.50,
        "reason": "bare_head_without_matched_helmet",
    }
    state.update(ppe)
    confirmed = state.update(ppe)
    first_miss = state.update({"candidate": False})
    second_miss = state.update({"candidate": False})

    assert confirmed["event_active"] is True
    assert first_miss["event_active"] is True
    assert second_miss["event_active"] is False


def test_safety_helmet_state_does_not_confirm_single_spike():
    monitor = load_monitor_module()
    state = monitor.SafetyHelmetState()

    outputs = [
        state.update({"candidate": False}),
        state.update({"candidate": True}),
        state.update({"candidate": False}),
        state.update({"candidate": False}),
    ]

    assert all(not output["confirmed"] for output in outputs)
    assert all(not output["warning"] for output in outputs)


def test_ppe_box_stabilizer_allows_strong_head_switch():
    """Normal-size target with high confidence should still switch promptly."""
    monitor = load_monitor_module()
    stabilizer = monitor.PPEBoxStabilizer(history=5, hold_frames=2, switch_count=2)

    frame_shape = (640, 640)
    # Use a larger box so it is NOT classified as small (area_ratio > 0.018)
    helmet = SimpleNamespace(
        boxes=[[100, 100, 220, 230]],
        classes=[1],
        confidences=[0.86],
        names={0: "head", 1: "helmet"},
    )
    head = SimpleNamespace(
        boxes=[[102, 101, 221, 231]],
        classes=[0],
        confidences=[0.72],
        names={0: "head", 1: "helmet"},
    )

    first = stabilizer.update(helmet, {"helmet_fp_suppression": {}}, frame_shape)
    flipped = stabilizer.update(head, {"helmet_fp_suppression": {}}, frame_shape)

    assert first[0]["label"] == "helmet"
    assert flipped[0]["label"] == "head"


def test_ppe_box_stabilizer_small_target_resists_single_frame_flip():
    """Small/distant targets should NOT flip label on a single frame (Phase 1.1)."""
    monitor = load_monitor_module()
    stabilizer = monitor.PPEBoxStabilizer(history=5, hold_frames=2, switch_count=2)

    frame_shape = (640, 640)
    # Small box: area_ratio ~ 0.017 < 0.018 → is_small=True
    helmet = SimpleNamespace(
        boxes=[[100, 100, 180, 190]],
        classes=[1],
        confidences=[0.86],
        names={0: "head", 1: "helmet"},
    )
    head = SimpleNamespace(
        boxes=[[102, 101, 181, 191]],
        classes=[0],
        confidences=[0.72],
        names={0: "head", 1: "helmet"},
    )

    first = stabilizer.update(helmet, {"helmet_fp_suppression": {}}, frame_shape)
    after_one_head = stabilizer.update(head, {"helmet_fp_suppression": {}}, frame_shape)

    assert first[0]["label"] == "helmet"
    # Small target should stay as helmet after just 1 head frame (conf 0.72 < 0.78)
    assert after_one_head[0]["label"] == "helmet"


def test_ppe_box_stabilizer_hides_suppressed_helmet_box():
    monitor = load_monitor_module()
    stabilizer = monitor.PPEBoxStabilizer()
    detections = SimpleNamespace(
        boxes=[[100, 100, 180, 190], [80, 80, 220, 330]],
        classes=[1, 2],
        confidences=[0.45, 0.90],
        names={1: "helmet", 2: "person"},
    )

    tracks = stabilizer.update(
        detections,
        {"helmet_fp_suppression": {"suppressed_helmet_indices": [0]}},
        (640, 640),
    )

    assert [track["label"] for track in tracks] == ["person"]


def test_ppe_box_stabilizer_enforces_head_helmet_exclusive_cluster():
    monitor = load_monitor_module()
    stabilizer = monitor.PPEBoxStabilizer()
    detections = SimpleNamespace(
        boxes=[[100, 100, 190, 190], [102, 101, 188, 191]],
        classes=[1, 0],
        confidences=[0.55, 0.50],
        names={0: "head", 1: "helmet"},
    )

    tracks = stabilizer.update(detections, {"helmet_fp_suppression": {}}, (640, 640))

    labels = [track["label"] for track in tracks]
    assert len(labels) == 1
    assert not ({"head", "helmet"} <= set(labels))


def test_ppe_box_stabilizer_does_not_expand_multi_person_boxes():
    monitor = load_monitor_module()
    stabilizer = monitor.PPEBoxStabilizer()
    detections = SimpleNamespace(
        boxes=[[10, 10, 60, 80], [120, 12, 170, 82]],
        classes=[0, 0],
        confidences=[0.72, 0.68],
        names={0: "head"},
    )

    tracks = stabilizer.update(detections, {"helmet_fp_suppression": {}}, (640, 640))

    assert len(tracks) == 2
    assert tracks[0]["box"] == [10, 10, 60, 80]
    assert tracks[1]["box"] == [120, 12, 170, 82]
