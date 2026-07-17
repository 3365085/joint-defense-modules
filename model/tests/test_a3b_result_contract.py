from pathlib import Path
from types import SimpleNamespace

import numpy as np

from defense.diagnostics.video_rows import frame_row
from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.module_a.result_contract import adapt_a3b_result
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState
from defense.runtime.frame_processor import FrameProcessor, _static_media_details


def _rebuilt_info(**overrides):
    a3b = {
        "p_media_raw": 0.82,
        "p_media": 0.71,
        "p_media_policy": 0.71,
        "p_media_triggered": True,
        "p_media_confirmed_score": 0.68,
        "media_confirmed": True,
        "p_media_type": "screen_replay",
        "p_media_bbox": (100, 120, 320, 360),
        "p_media_target_related": False,
        "p_media_scores": {
            "edge": 0.56,
            "track": 0.64,
            "yolo_context": 0.21,
        },
        "p_media_strong_evidence": True,
        "media_candidates": [
            {"bbox": [100, 120, 320, 360], "candidate_score": 0.84},
            {"bbox": [20, 20, 80, 90], "candidate_score": 0.52},
        ],
        "stable_count": 9,
        "track_score": 0.64,
        "bbox_jitter": 0.03,
        "scale_jitter": 0.04,
        "candidate_lifetime": 13,
        "plane_score": 0.73,
        "warp_residual": 0.19,
        "flow_gap": 1.12,
        "inside_motion": 0.22,
        "outside_motion": 0.81,
        "inner_outer_motion_ratio": 0.27,
        "homography_inlier_ratio": 0.76,
        "a3b_moire": 0.31,
        "p_media_background_static_suppressed": False,
        "a3b_display_score": 0.69,
        "suppressed_reason": "none",
        "score_cap": 1.0,
        "media_candidate_allowed": True,
        "a3b_state": "confirmed",
        "a3b_result_fresh": True,
    }
    a3b.update(overrides)
    return {"details": {"a3b": a3b}}


def test_legacy_static_media_contract_is_preserved() -> None:
    legacy = {
        "p_media": 0.57,
        "p_media_bbox": [1, 2, 30, 40],
        "p_media_scores": {"edge": 0.4},
        "p_media_candidate_count": 3,
        "p_media_strong_evidence": True,
        "p_media_border_state": {"suppressed": False},
        "triggered": True,
        "state": "confirmed",
        "custom_legacy_diagnostic": {"kept": True},
    }
    info = {"details": {"module_a_features": {"static_media": legacy}}}

    assert adapt_a3b_result(info) == legacy
    assert _static_media_details(info) == legacy
    assert adapt_a3b_result(info) is not legacy


def test_rebuilt_contract_maps_quality_gate_and_keeps_diagnostics() -> None:
    result = adapt_a3b_result(
        _rebuilt_info(
            triggered=False,
            static_image_triggered=False,
        )
    )

    assert result["p_media_scores"]["track"] == 0.64
    assert result["result_contract_source"] == "rebuilt"
    assert result["p_media_candidate_count"] == 2
    assert result["candidate_count"] == 2
    assert result["p_media_strong_evidence"] is True
    assert result["strong_evidence"] is True
    assert result["p_media_bbox"] == [100, 120, 320, 360]
    assert result["bbox"] == [100, 120, 320, 360]
    assert result["p_media_triggered"] is True
    assert result["media_confirmed"] is True
    assert result["triggered"] is True
    assert result["static_image_triggered"] is True
    assert result["state"] == "confirmed"
    assert result["policy"]["p_media_policy"] == 0.71
    assert result["suppression"]["suppressed"] is False
    assert result["stable_count"] == 9
    assert result["candidate_lifetime"] == 13
    assert result["homography_inlier_ratio"] == 0.76

    soft = A3BSoftTriggerState()
    soft_result = soft.update(result)
    assert soft_result["debug"]["candidate_count"] == 2
    assert soft_result["debug"]["quality_gate_passed"] is True


def test_rebuilt_suppression_policy_is_mapped_without_losing_reason() -> None:
    result = adapt_a3b_result(
        _rebuilt_info(
            media_confirmed=False,
            p_media_triggered=False,
            p_media=0.38,
            p_media_policy=0.38,
            suppressed_reason="camera_translation_edge",
            score_cap=0.38,
            media_candidate_allowed=False,
            a3b_state="suppressed",
        )
    )

    assert result["suppression"]["suppressed"] is True
    assert result["suppression"]["reason"] == "camera_translation_edge"
    assert result["policy"]["media_candidate_allowed"] is False
    assert result["p_media_camera_motion_state"]["suppressed"] is True
    assert result["state"] == "suppressed"


def test_diagnostics_uses_rebuilt_contract_instead_of_all_zero_a3b() -> None:
    row = frame_row(
        Path("rebuilt-screen.mp4"),
        17,
        _rebuilt_info(),
    )

    assert row["p_media"] == 0.71
    assert row["a3b_observed_score"] == 0.71
    assert row["a3b_confirmed_score"] == 0.71
    assert row["a3b_display_score"] > 0.0
    assert row["a3b_triggered"] is True
    assert row["track_score"] == 0.64


def test_rebuilt_confirmed_bbox_suppresses_detections_inside_region() -> None:
    pipeline = object.__new__(VideoDefensePipeline)
    pipeline._a3b_suppress_remaining = 0
    pipeline._a3b_suppress_bbox = None
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = DetectionFrameResult(
        image=frame,
        boxes=[
            [140, 160, 220, 260],
            [400, 400, 500, 520],
        ],
        classes=[0, 2],
        confidences=[0.91, 0.88],
        names={0: "person", 2: "helmet"},
        backend="fake",
        artifact_path="",
        inference_ms=0.0,
    )
    info = _rebuilt_info()

    filtered, rois = pipeline._apply_a3b_suppression(
        frame,
        detections,
        [],
        info,
    )

    assert filtered.boxes == [[400, 400, 500, 520]]
    assert filtered.classes == [2]
    assert filtered.confidences == [0.88]
    assert rois == []
    assert info["a3b_suppression_active"] is True
    assert info["a3b_suppression_filtered"] is True
    assert pipeline._a3b_suppress_bbox == (100, 120, 320, 360)
    assert pipeline._a3b_suppress_remaining == 180


def test_rebuilt_unconfirmed_candidate_does_not_start_bbox_suppression() -> None:
    pipeline = object.__new__(VideoDefensePipeline)
    pipeline._a3b_suppress_remaining = 0
    pipeline._a3b_suppress_bbox = None
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = DetectionFrameResult(
        image=frame,
        boxes=[[140, 160, 220, 260]],
        classes=[0],
        confidences=[0.91],
        names={0: "person"},
        backend="fake",
        artifact_path="",
        inference_ms=0.0,
    )
    info = _rebuilt_info(
        media_confirmed=False,
        p_media_confirmed_score=0.0,
        a3b_state="candidate",
    )

    filtered, _ = pipeline._apply_a3b_suppression(
        frame,
        detections,
        [],
        info,
    )

    assert filtered.boxes == [[140, 160, 220, 260]]
    assert pipeline._a3b_suppress_remaining == 0
    assert pipeline._a3b_suppress_bbox is None
    assert "a3b_suppression_active" not in info


def test_frame_processor_labels_authoritative_rebuilt_confirmation_source() -> None:
    class _SoftState:
        def update(self, static_media):
            return {
                "observed_score": static_media["p_media"],
                "confirmed_score": 0.0,
                "confidence": 0.0,
                "display_score": 0.0,
                "state": "normal",
                "triggered": False,
                "triggered_source": "none",
                "reason": "none",
                "debug": {},
            }

    processor = object.__new__(FrameProcessor)
    processor.bundle = SimpleNamespace(
        config={},
        backend="fake",
        model_family="fake",
        artifact_path="",
    )
    processor.a3b_soft = _SoftState()
    processor.ppe_file_realtime_max_render_misses = 2
    processor.ppe_stream_max_render_misses = 2

    status = processor._build_status(
        source_type="file",
        source="rebuilt-screen.mp4",
        profile="test",
        realtime=False,
        frame_idx=17,
        video_time_s=0.5,
        source_fps=30.0,
        fps=20.0,
        dropped_frames=0,
        info=_rebuilt_info(),
        ppe={},
        ppe_tracks=[],
        display_options={},
        feature_options={},
        custom_model={},
        redetect_budget_ok=False,
        redetect_count=0,
        redetect_ms=0.0,
        processing_ms=5.0,
        target_frame_budget_ms=33.0,
        raw_boxes_count=0,
    )

    assert status["a3b_triggered"] is True
    assert status["a3b_state"] == "confirmed"
    assert status["a3b_confirmed_score"] == 0.68
    assert status["a3b_triggered_source"] == "rebuilt_media_confirmed"
