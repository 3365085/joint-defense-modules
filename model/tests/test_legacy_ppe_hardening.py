from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from defense.module_a.alert_state import AlertState
from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.module_a.detector import ModuleADetector
from defense.module_a.ppe_postprocess import (
    PPEPostprocessConfig,
    canonical_ppe_label,
    summarize_ppe_from_detections,
)
from defense.module_a.postprocess.ppe_tracking import canonical_label
from defense.module_a.process_pipeline import (
    _select_fusion_p_adv,
    _select_fusion_suspicious,
    build_classifier_features,
    build_static_media_classifier_features,
)
from defense.module_a.roi_provider import DetectionROIProvider
from defense.module_a.types import ModuleAInput
from defense.runtime.ppe_business import _apply_temporal_helmet_mutex
from defense.runtime.ppe_state import SafetyHelmetState


def _detections(
    *,
    boxes: list[list[int]],
    classes: list[int],
    confidences: list[float],
    names: dict[int, str],
) -> DetectionFrameResult:
    return DetectionFrameResult(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        boxes=boxes,
        classes=classes,
        confidences=confidences,
        names=names,
        backend="fake",
        artifact_path="fake",
        inference_ms=0.0,
    )


@pytest.mark.parametrize(
    ("backend", "expected_score", "expected_suspicious"),
    [
        ("rule", 0.35, False),
        ("classifier", 0.82, True),
        ("rule_or_classifier", 0.82, True),
    ],
)
def test_legacy_fusion_backend_selects_score_and_alert_signal(
    backend: str,
    expected_score: float,
    expected_suspicious: bool,
) -> None:
    classifier_result = {
        "classifier_p_adv": 0.82,
        "classifier_triggered": True,
    }
    selected_score = _select_fusion_p_adv(
        backend,
        rule_p_adv=0.35,
        classifier_result=classifier_result,
    )
    selected_suspicious = _select_fusion_suspicious(
        backend,
        rule_suspicious=False,
        classifier_result=classifier_result,
    )

    assert selected_score == pytest.approx(expected_score)
    assert selected_suspicious is expected_suspicious

    alert = AlertState(window=1, trigger_count=1, hold_frames=0)
    confirmed, active = alert.update(
        selected_suspicious,
        intensity=selected_score,
    )
    assert confirmed is expected_suspicious
    assert active is expected_suspicious


def test_legacy_rule_or_classifier_keeps_rule_trigger() -> None:
    classifier_result = {
        "classifier_p_adv": 0.10,
        "classifier_triggered": False,
    }

    assert _select_fusion_p_adv(
        "rule_or_classifier",
        rule_p_adv=0.73,
        classifier_result=classifier_result,
    ) == pytest.approx(0.73)
    assert _select_fusion_suspicious(
        "rule_or_classifier",
        rule_suspicious=True,
        classifier_result=classifier_result,
    )


def test_legacy_classifier_feature_builders_are_imported_into_process_pipeline() -> None:
    static_features = build_static_media_classifier_features({})
    classifier_features = build_classifier_features(
        overexposure={},
        texture={},
        temporal={},
        motion={},
        blur={},
        track={},
        fusion={},
        roi_count=0,
    )

    assert len(static_features) == 16
    assert classifier_features["rule_p_adv"] == 0.0


def _legacy_classifier_detector(
    tmp_path: Path,
    *,
    fusion_backend: str,
    bias: float,
    device: str,
) -> ModuleADetector:
    artifact_path = tmp_path / f"{fusion_backend}_{bias}.json"
    artifact_path.write_text(
        json.dumps(
            {
                "kind": "torch_logistic_regression",
                "feature_names": ["rule_p_adv"],
                "normalization": {"mean": [0.0], "std": [1.0]},
                "weights": [0.0],
                "bias": bias,
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return ModuleADetector(
        {
            "module_a": {
                "device": device,
                "require_gpu": device.startswith("cuda"),
                "frame_size": 64,
                "keyframe_interval": 1,
                "alert_window": 1,
                "alert_trigger_count": 1,
                "attack_state_hold_frames": 0,
                "light_flow_enabled": False,
                "static_image_enabled": False,
                "fusion_backend": fusion_backend,
                "classifier_artifact": str(artifact_path),
                "use_grid_when_no_roi": True,
                "grid_roi_count": 1,
            }
        }
    )


def test_legacy_classifier_backend_drives_final_result(
    tmp_path: Path,
    cuda_device: str,
) -> None:
    detector = _legacy_classifier_detector(
        tmp_path,
        fusion_backend="classifier",
        bias=10.0,
        device=cuda_device,
    )

    result = detector.process(
        ModuleAInput(
            frame=np.full((64, 64, 3), 128, dtype=np.uint8),
            frame_idx=0,
        )
    )

    assert result.p_adv > 0.99
    assert result.details["module_a"]["p_adv_display"] > 0.30
    assert result.single_frame_suspicious is True
    assert result.alert_confirmed is True
    assert "classifier_fusion" in result.reason_codes


def test_legacy_classifier_backend_can_reject_rule_only_glare(
    tmp_path: Path,
    cuda_device: str,
) -> None:
    detector = _legacy_classifier_detector(
        tmp_path,
        fusion_backend="classifier",
        bias=-10.0,
        device=cuda_device,
    )
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    frame[:32, :, :] = 255

    result = detector.process(ModuleAInput(frame=frame, frame_idx=0))

    assert "overexposure" in result.reason_codes
    assert result.p_adv < 0.01
    assert result.details["module_a"]["p_adv_display"] < 0.01
    assert result.single_frame_suspicious is False
    assert result.alert_confirmed is False


def test_legacy_rule_or_classifier_accepts_classifier_only_signal(
    tmp_path: Path,
    cuda_device: str,
) -> None:
    detector = _legacy_classifier_detector(
        tmp_path,
        fusion_backend="rule_or_classifier",
        bias=10.0,
        device=cuda_device,
    )

    result = detector.process(
        ModuleAInput(
            frame=np.full((64, 64, 3), 128, dtype=np.uint8),
            frame_idx=0,
        )
    )

    assert result.p_adv > 0.99
    assert result.details["module_a"]["p_adv_display"] > 0.30
    assert result.single_frame_suspicious is True
    assert result.alert_confirmed is True


def _head_ppe() -> dict[str, object]:
    return summarize_ppe_from_detections(
        _detections(
            boxes=[[100, 100, 180, 190]],
            classes=[0],
            confidences=[0.90],
            names={0: "head", 1: "helmet"},
        ),
        config=PPEPostprocessConfig(prefer_helmet_on_head_overlap=True),
        frame_shape=(640, 640),
    )


@pytest.mark.parametrize(
    "track",
    [
        {
            "track_id": 1,
            "label": "helmet",
            "box": [100, 100, 180, 190],
            "confidence": 0.01,
            "misses": 1,
            "source": "held",
        },
        {
            "track_id": 1,
            "label": "helmet",
            "box": [100, 100, 180, 190],
            "confidence": 0.90,
            "misses": 1,
            "source": "detected",
        },
        {
            "track_id": 1,
            "label": "helmet",
            "box": [100, 100, 180, 190],
            "confidence": 0.90,
            "misses": 0,
            "source": "held",
        },
        {
            "track_id": 1,
            "label": "helmet",
            "box": [100, 100, 180, 190],
            "confidence": 0.24,
            "misses": 0,
            "source": "detected",
        },
    ],
)
def test_temporal_helmet_mutex_rejects_stale_or_low_confidence_track(
    track: dict[str, object],
) -> None:
    cfg = PPEPostprocessConfig(
        prefer_helmet_on_head_overlap=True,
        head_helmet_mutex_min_helmet_confidence=0.25,
    )

    result = _apply_temporal_helmet_mutex(
        _head_ppe(),
        [track],
        cfg,
        (640, 640),
    )

    assert result["candidate"] is True
    assert result["head_count"] == 1
    assert int(result.get("temporal_helmet_mutex_count", 0) or 0) == 0


def test_temporal_helmet_mutex_accepts_current_confident_helmet() -> None:
    cfg = PPEPostprocessConfig(
        prefer_helmet_on_head_overlap=True,
        head_helmet_mutex_min_helmet_confidence=0.25,
    )
    result = _apply_temporal_helmet_mutex(
        _head_ppe(),
        [
            {
                "track_id": 1,
                "label": "helmet",
                "box": [100, 100, 180, 190],
                "confidence": 0.90,
                "misses": 0,
                "source": "detected",
            }
        ],
        cfg,
        (640, 640),
    )

    assert result["candidate"] is False
    assert result["head_count"] == 0
    assert result["temporal_helmet_mutex_count"] == 1


def test_other_person_helmet_does_not_disable_bare_head_fast_path() -> None:
    state = SafetyHelmetState(
        window=6,
        trigger_count=3,
        fast_window=1,
        fast_trigger_count=1,
        fast_min_confidence=0.65,
    )

    result = state.update(
        {
            "candidate": True,
            "head_count": 1,
            "raw_head_count": 1,
            "helmet_count": 1,
            "missing_helmet_count": 1,
            "promoted_head_count": 0,
            "low_conf_temporal_head_count": 0,
            "max_head_confidence": 0.91,
            "reason": "bare_head_without_matched_helmet",
        }
    )

    assert result["confirmed"] is True
    assert result["confirmed_source"] == "fast_head"


def test_hat_and_without_helmet_share_canonical_rules_across_ppe_layers() -> None:
    assert canonical_ppe_label("hat") == "helmet"
    assert canonical_ppe_label("without-helmet") == "head"
    assert canonical_label("hat") == "helmet"
    assert canonical_label("without helmet") == "head"

    provider = DetectionROIProvider(
        class_names={0: "hat", 1: "without_helmet"},
        min_confidence=0.25,
        margin=0,
        target_labels=("helmet", "head"),
    )
    rois = provider.from_detections(
        boxes=[[20, 20, 70, 70], [300, 300, 360, 360]],
        classes=[0, 1],
        confs=[0.90, 0.92],
    )
    assert [roi.label for roi in rois] == ["helmet", "head"]

    summary = summarize_ppe_from_detections(
        _detections(
            boxes=[[20, 20, 70, 70], [300, 300, 360, 360]],
            classes=[0, 1],
            confidences=[0.90, 0.92],
            names={0: "hat", 1: "without_helmet"},
        ),
        frame_shape=(640, 640),
    )
    assert summary["raw_helmet_count"] == 1
    assert summary["raw_head_count"] == 1
