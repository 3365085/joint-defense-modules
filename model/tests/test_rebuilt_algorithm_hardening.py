from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from defense.module_a.rebuilt import detector as detector_module
from defense.module_a.rebuilt.detector import (
    A4_FEATURE_NAMES,
    A4_FEATURE_SCHEMA_VERSION,
    ModuleADetector,
)
from defense.module_a.types import ModuleAInput, ROI


class _BoundA4Classifier:
    n_features_in_ = len(A4_FEATURE_NAMES)
    feature_importances_ = np.ones(
        len(A4_FEATURE_NAMES), dtype=np.float32
    ) / len(A4_FEATURE_NAMES)

    def predict_proba(
        self,
        _features: object,
    ) -> list[list[float]]:
        return [[0.4, 0.6]]


def _detector(
    monkeypatch: pytest.MonkeyPatch,
    *,
    classifier: Any = None,
    flow_loader: Any = None,
    **module_config: object,
) -> ModuleADetector:
    monkeypatch.setattr(
        ModuleADetector,
        "_load_classifier",
        lambda self, _path: classifier,
    )
    if flow_loader is None:
        monkeypatch.setattr(
            ModuleADetector,
            "_load_flownet",
            lambda self: None,
        )
    else:
        monkeypatch.setattr(
            ModuleADetector,
            "_load_flownet",
            flow_loader,
        )
    config = {
        "frame_size": 64,
        "static_image_enabled": False,
        "light_flow_enabled": False,
        **module_config,
    }
    return ModuleADetector({"module_a": config})


def _item(frame_idx: int, timestamp: float = 1.0) -> ModuleAInput:
    return ModuleAInput(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        frame_idx=frame_idx,
        timestamp=timestamp,
        rois=[],
    )


def _media_payload(detector: ModuleADetector) -> dict[str, Any]:
    payload = detector._empty_a3b()
    payload.update(
        {
            "p_media_raw": 0.82,
            "p_media_raw_triggered": True,
            "p_media": 0.82,
            "p_media_policy": 0.82,
            "p_media_triggered": True,
            "p_media_type": "screen_replay",
            "p_media_bbox": [8, 8, 56, 56],
            "p_media_target_related": False,
            "p_media_strong_evidence": True,
            "media_candidate_allowed": True,
            "suppressed_reason": "none",
            "a3b_state": "candidate",
            "p_media_scores": {
                "candidate_score": 0.80,
                "edge": 0.50,
                "border_contrast": 0.90,
                "display_frame": 0.80,
                "area_ratio": 0.12,
                "boundary": 0.30,
            },
        }
    )
    return payload


def _publish_media_result(
    detector: ModuleADetector,
    *,
    seq: int,
) -> None:
    payload = _media_payload(detector)
    with detector._a3b_bg_lock:
        detector._a3b_result_seq = seq
        detector._a3b_last_success_at = time.time()
        detector._a3b_result_published_at = time.time()
        detector._a3b_result_published_monotonic = time.monotonic()
        detector._a3b_bg_result = payload


def _a4_inputs() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    a1 = {
        "delta_h": 0.2,
        "delta_h_roi_max": 0.3,
        "delta_h_local_max": 0.4,
        "delta_h_target_contrast": 0.1,
        "a1_feature_score": 0.6,
    }
    a2 = {
        "change_t": 0.2,
        "change_t_roi_max": 0.3,
        "change_t_local_max": 0.4,
        "change_t_without_motion_target": 0.1,
        "a2_feature_score": 0.5,
    }
    a3 = {
        "f_flow": 0.2,
        "flow_local_anomaly_ratio": 0.3,
        "flow_residual": 0.4,
        "flow_shape_score": 0.1,
        "flow_target_relation": 0.2,
        "a3_feature_score": 0.7,
    }
    return a1, a2, a3


def test_repeated_a3b_cache_seq_does_not_add_votes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_enabled=True,
        static_image_interval=1000,
        rebuilt_a3b_media_run_floor=1,
    )
    _publish_media_result(detector, seq=1)

    first = detector.process(_item(1))
    for frame_idx in range(2, 8):
        repeated = detector.process(_item(frame_idx))

    first_joint = first.details["joint_decision"]
    repeated_joint = repeated.details["joint_decision"]
    assert first_joint["media_result_consumed"] is True
    assert repeated_joint["media_result_consumed"] is False
    assert repeated_joint["media_run"] == 1
    assert repeated_joint["media_count"] == 1
    assert len(detector.media_hits) == 1
    assert repeated_joint["media_confirmed"] is False

    for seq in (2, 3, 4):
        _publish_media_result(detector, seq=seq)
        confirmed = detector.process(_item(10 + seq))

    confirmed_joint = confirmed.details["joint_decision"]
    assert confirmed_joint["media_run"] == 4
    assert confirmed_joint["media_count"] == 4
    assert confirmed_joint["media_confirmed"] is True
    assert len(detector.media_hits) == 4


def test_expired_a3b_result_cannot_keep_authoritative_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_enabled=True,
        static_image_interval=1000,
        static_image_result_lease_s=10.0,
        rebuilt_a3b_media_run_floor=1,
    )
    for seq in (1, 2, 3, 4):
        _publish_media_result(detector, seq=seq)
        result = detector.process(_item(seq))
    assert result.details["joint_decision"]["media_confirmed"] is True

    with detector._a3b_bg_lock:
        detector._a3b_result_published_monotonic = (
            time.monotonic() - 20.0
        )
    expired = detector.process(_item(20))

    assert expired.details["joint_decision"]["media_result_fresh"] is False
    assert expired.details["joint_decision"]["media_confirmed"] is False


def test_a2_scales_full_frame_roi_into_raft_flow_grid_without_empty_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    monkeypatch.setattr(detector_module, "_NATIVE", None)
    detector.prev_lbp = np.zeros((640, 640), dtype=np.uint8)
    lbp = np.zeros((640, 640), dtype=np.uint8)
    lbp[400:600, 400:600] = 255

    result = detector._compute_a2(
        lbp,
        [
            ROI(
                roi_id="head-1",
                bbox=(400, 400, 600, 600),
                label="head",
                confidence=0.9,
            )
        ],
        640,
        640,
        {
            "exposure_delta": 0.0,
            "overexposure_ratio": 0.0,
            "underexposed_ratio": 0.0,
            "frame_diff_global": 0.1,
        },
        {
            "available": True,
            "mag": np.ones((256, 256), dtype=np.float32),
            "global_motion_weight": 0.0,
        },
    )

    assert result["change_t_motion_aligned"] == pytest.approx(1.0 / 3.0)


def test_reset_clears_a3b_dedup_and_target_anchored_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_enabled=True,
        static_image_interval=1000,
        rebuilt_a3b_media_run_floor=1,
    )
    _publish_media_result(detector, seq=1)
    detector.process(_item(1))
    detector._ta._recent_target_counts.append(3)
    detector._ta.glare_active = True

    detector.reset()

    assert detector._a3b_last_consumed_result_seq == 0
    assert detector._media_run == 0
    assert list(detector.media_hits) == []
    assert list(detector._ta._recent_target_counts) == []
    assert detector._ta.glare_active is False


def test_relative_classifier_path_is_not_resolved_from_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = ModuleADetector._resolve_classifier_path(
        "runtime/artifacts/module_a/a4/a4_classifier.pkl"
    )
    expected = (
        Path(detector_module.__file__).resolve().parents[4]
        / "runtime"
        / "artifacts"
        / "module_a"
        / "a4"
        / "a4_classifier.pkl"
    ).resolve()

    assert resolved == expected
    assert resolved.is_absolute()


def test_missing_classifier_has_visible_degraded_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ModuleADetector,
        "_load_flownet",
        lambda self: None,
    )
    detector = ModuleADetector(
        {
            "module_a": {
                "a4_classifier_path": "missing/a4_classifier.pkl",
                "light_flow_enabled": False,
                "static_image_enabled": False,
            }
        }
    )

    assert detector.a4_classifier_configured is True
    assert detector.a4_classifier_loaded is False
    assert detector.a4_classifier_fallback_reason == "load_failed"
    assert detector.a4_classifier_error is not None
    assert "FileNotFoundError" in detector.a4_classifier_error
    assert Path(detector.a4_classifier_resolved_path).is_absolute()


def test_classifier_schema_mismatch_falls_back_without_predicting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongSchemaClassifier:
        n_features_in_ = 25
        feature_importances_ = np.ones(25, dtype=np.float32) / 25.0

        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: object) -> list[list[float]]:
            self.calls += 1
            return [[0.1, 0.9]]

    classifier = WrongSchemaClassifier()
    detector = _detector(
        monkeypatch,
        classifier=classifier,
        a4_classifier_path="fake.pkl",
    )
    a1, a2, a3 = _a4_inputs()

    result = detector._compute_a4(a1, a2, a3)

    assert classifier.calls == 0
    assert result["a4_classifier_used"] is False
    assert result["a4_classifier_loaded"] is True
    assert result["a4_classifier_fallback_reason"] == (
        "feature_schema_mismatch"
    )
    assert f"expected=25,actual={len(A4_FEATURE_NAMES)}" in result[
        "a4_classifier_error"
    ]


def test_classifier_predict_failure_is_circuit_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClassifier:
        n_features_in_ = len(A4_FEATURE_NAMES)
        feature_importances_ = np.ones(
            len(A4_FEATURE_NAMES), dtype=np.float32
        ) / len(A4_FEATURE_NAMES)

        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: object) -> list[list[float]]:
            self.calls += 1
            raise RuntimeError("predict exploded")

    classifier = FailingClassifier()
    detector = _detector(
        monkeypatch,
        classifier=classifier,
        a4_classifier_path="fake.pkl",
    )
    a1, a2, a3 = _a4_inputs()

    first = detector._compute_a4(a1, a2, a3)
    second = detector._compute_a4(a1, a2, a3)

    assert classifier.calls == 1
    assert first["a4_classifier_used"] is False
    assert second["a4_classifier_used"] is False
    assert detector.a4_classifier_fallback_reason == "predict_failed"
    assert "RuntimeError: predict exploded" == detector.a4_classifier_error


def test_healthy_classifier_runs_once_per_a4_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HealthyClassifier:
        n_features_in_ = len(A4_FEATURE_NAMES)
        feature_importances_ = np.ones(
            len(A4_FEATURE_NAMES), dtype=np.float32
        ) / len(A4_FEATURE_NAMES)

        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: object) -> list[list[float]]:
            self.calls += 1
            return [[0.4, 0.6]]

    classifier = HealthyClassifier()
    detector = _detector(
        monkeypatch,
        classifier=classifier,
        a4_classifier_path="fake.pkl",
    )
    a1, a2, a3 = _a4_inputs()

    results = [
        detector._compute_a4(a1, a2, a3)
        for _ in range(4)
    ]

    assert classifier.calls == 4
    assert all(result["a4_classifier_used"] for result in results)
    assert detector.a4_classifier_fallback_reason == "none"


def test_a4_feature_schema_excludes_async_a3b_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    a1, a2, a3 = _a4_inputs()
    low_a3b = detector._empty_a3b()
    high_a3b = detector._empty_a3b()
    high_a3b.update(
        {
            "p_media_raw": 0.93,
            "p_media_scores": {
                "flow_gap": 0.17,
                "warp_residual": 0.29,
                "display_frame": 0.73,
            },
        }
    )

    low = detector._compute_a4(a1, a2, a3, low_a3b)
    high = detector._compute_a4(a1, a2, a3, high_a3b)

    assert A4_FEATURE_SCHEMA_VERSION == "rebuilt-a4-96-v4"
    assert len(A4_FEATURE_NAMES) == 96
    assert all(
        not name.startswith("a3b.")
        for name in A4_FEATURE_NAMES
    )
    assert len(low["a4_feature_vector"]) == 96
    assert high["a4_feature_vector"] == low["a4_feature_vector"]
    assert high["a4_async_a3b_features_used"] is False


def test_a4_classifier_rejects_incomplete_schema_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ModuleADetector,
        "_load_flownet",
        lambda self: None,
    )
    artifact = tmp_path / "a4_classifier.pkl"
    artifact.write_bytes(
        pickle.dumps(_BoundA4Classifier())
    )
    metadata_path = artifact.with_suffix(
        artifact.suffix + ".meta.json"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_schema_version": (
                    A4_FEATURE_SCHEMA_VERSION
                ),
                "feature_names": list(A4_FEATURE_NAMES),
                "feature_count": len(A4_FEATURE_NAMES),
                "preprocessing": "raw_float32_no_scaler",
                "model_sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    detector = ModuleADetector(
        {
            "module_a": {
                "a4_classifier_path": str(artifact),
                "light_flow_enabled": False,
                "static_image_enabled": False,
            }
        }
    )
    try:
        assert detector.a4_classifier_loaded is False
        assert detector.a4_classifier_fallback_reason == (
            "artifact_contract_version_mismatch"
        )
        assert "artifact_contract_version_mismatch" in str(
            detector.a4_classifier_error
        )
    finally:
        detector.close()


def test_a4_classifier_rejects_unbound_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ModuleADetector,
        "_load_flownet",
        lambda self: None,
    )
    artifact = tmp_path / "a4_classifier.pkl"
    artifact.write_bytes(
        pickle.dumps(_BoundA4Classifier())
    )

    detector = ModuleADetector(
        {
            "module_a": {
                "a4_classifier_path": str(artifact),
                "light_flow_enabled": False,
                "static_image_enabled": False,
            }
        }
    )
    try:
        assert detector.a4_classifier_loaded is False
        assert (
            detector.a4_classifier_fallback_reason
            == "schema_metadata_missing"
        )
        assert "schema_metadata_missing" in str(
            detector.a4_classifier_error
        )
    finally:
        detector.close()


def test_process_details_expose_a4_schema_and_stage_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)

    result = detector.process(_item(1))

    assert result.details["a4_feature_schema"] == {
        "version": A4_FEATURE_SCHEMA_VERSION,
        "names": list(A4_FEATURE_NAMES),
    }
    timing = result.details["timing"]
    expected_stages = {
        "scene_context",
        "lbp",
        "flow",
        "a1",
        "a2",
        "a3",
        "a3b_schedule",
        "a4",
        "blinding",
        "target_anchored",
        "joint",
        "result_build",
        "total",
    }
    assert expected_stages <= timing.keys()
    assert all(timing[name] >= 0.0 for name in expected_stages)
    assert result.timing_ms == pytest.approx(timing["total"])


def test_light_flow_disabled_avoids_flow_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch, light_flow_enabled=False)
    previous = np.zeros((16, 16), dtype=np.uint8)
    current = np.ones((16, 16), dtype=np.uint8)

    result = detector._compute_flow(previous, current)

    assert result["available"] is False
    assert result["flow_skip_reason"] == "disabled_by_config"
    assert detector.flow_requested_device == "cuda:0"
    assert detector.flow_effective_device == "disabled"
    assert detector.flow_backend == "disabled"
    assert detector.flow_fallback_reason == "disabled_by_config"


def test_cpu_flow_device_is_respected_without_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ModuleADetector,
        "_load_classifier",
        lambda self, _path: None,
    )

    detector = ModuleADetector(
        {
            "module_a": {
                "device": "cpu",
                "light_flow_enabled": True,
                "static_image_enabled": False,
            }
        }
    )

    assert detector.flow_requested_device == "cpu"
    assert detector.flow_effective_device == "cpu"
    assert detector.flow_backend == "dis_cpu"
    assert detector.flow_fallback_reason == "requested_cpu"


def test_cuda_unavailable_has_explicit_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(
        ModuleADetector,
        "_load_classifier",
        lambda self, _path: None,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    detector = ModuleADetector(
        {
            "module_a": {
                "device": "cuda:0",
                "light_flow_enabled": True,
                "static_image_enabled": False,
            }
        }
    )

    assert detector.flow_requested_device == "cuda:0"
    assert detector.flow_effective_device == "cpu"
    assert detector.flow_backend == "dis_cpu"
    assert detector.flow_fallback_reason == "cuda_unavailable"


def test_missing_raft_engine_does_not_build_or_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    monkeypatch.setattr(
        ModuleADetector,
        "_load_classifier",
        lambda self, _path: None,
    )
    monkeypatch.setattr(
        detector_module,
        "_resolve_rebuilt_data_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        ModuleADetector,
        "_try_load_raft_trt",
        staticmethod(
            lambda _device: pytest.fail(
                "missing engine must not enter RAFT loader"
            )
        ),
    )
    monkeypatch.setattr(
        ModuleADetector,
        "_build_raft_trt_engine",
        staticmethod(
            lambda *_args: pytest.fail(
                "runtime must not build missing RAFT assets"
            )
        ),
    )
    monkeypatch.setattr(
        ModuleADetector,
        "_load_gpu_lk",
        staticmethod(
            lambda device: {"mode": "gpu_lk", "device": device}
        ),
    )

    detector = ModuleADetector(
        {
            "module_a": {
                "device": "cuda:0",
                "light_flow_enabled": True,
                "static_image_enabled": False,
            }
        }
    )

    assert detector.flow_backend == "gpu_lk"
    assert detector.flow_effective_device == "cuda:0"
    assert detector.flow_fallback_reason == "raft_engine_missing"
    assert list(tmp_path.iterdir()) == []


def test_runtime_flow_failure_is_circuit_broken_to_dis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDis:
        def calc(
            self,
            _previous: np.ndarray,
            current: np.ndarray,
            _initial: object,
        ) -> np.ndarray:
            return np.zeros((*current.shape, 2), dtype=np.float32)

    detector = _detector(
        monkeypatch,
        flow_loader=lambda self: {
            "mode": "raft_trt",
            "device": "cuda:0",
        },
        light_flow_enabled=True,
        light_flow_interval=1,
    )
    calls = {"raft": 0}

    def failing_raft(
        _previous: np.ndarray,
        _current: np.ndarray,
    ) -> tuple[Any, ...]:
        calls["raft"] += 1
        raise RuntimeError("backend failed")

    monkeypatch.setattr(detector, "_raft_flow", failing_raft)
    monkeypatch.setattr(
        detector_module.cv2,
        "DISOpticalFlow_create",
        lambda _preset: FakeDis(),
    )
    previous = np.zeros((16, 16), dtype=np.uint8)
    current = np.ones((16, 16), dtype=np.uint8)

    first = detector._compute_flow(previous, current)
    second = detector._compute_flow(previous, current)

    assert first["available"] is True
    assert second["available"] is True
    assert calls["raft"] == 1
    assert detector._flownet is None
    assert detector.flow_backend == "dis_cpu"
    assert detector.flow_effective_device == "cpu"
    assert detector.flow_fallback_reason == (
        "raft_trt_runtime_failed:RuntimeError"
    )


def test_gpu_lbp_failure_is_not_retried_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        flow_loader=lambda self: {
            "mode": "gpu_lk",
            "device": "cuda:0",
        },
        light_flow_enabled=True,
    )
    calls = {"lbp": 0}

    def failing_gpu_lbp(_gray: np.ndarray) -> np.ndarray:
        calls["lbp"] += 1
        raise RuntimeError("lbp backend failed")

    monkeypatch.setattr(detector, "_compute_lbp_gpu", failing_gpu_lbp)
    gray = np.zeros((16, 16), dtype=np.uint8)

    first = detector._compute_lbp(gray)
    second = detector._compute_lbp(gray)

    assert first.shape == gray.shape
    assert second.shape == gray.shape
    assert calls["lbp"] == 1
    assert detector.lbp_backend == "cpu"
    assert detector.lbp_fallback_reason == (
        "gpu_lbp_failed:RuntimeError"
    )


def test_light_flow_interval_limits_main_path_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        light_flow_enabled=True,
        light_flow_interval=3,
    )
    calls = {"flow": 0}

    def counted_flow(
        _previous: np.ndarray | None,
        gray: np.ndarray,
    ) -> dict[str, Any]:
        calls["flow"] += 1
        return detector._empty_flow_result(
            gray,
            reason="test_sample",
        )

    monkeypatch.setattr(detector, "_compute_flow", counted_flow)

    for frame_idx in range(1, 8):
        result = detector.process(_item(frame_idx))

    assert calls["flow"] == 2
    assert detector.light_flow_interval == 3
    assert result.details["flow_context"]["flow_skip_reason"] == (
        "interval_skip"
    )


def test_a3b_worker_interval_limits_background_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_enabled=True,
        static_image_interval=3,
    )
    calls = {"a3b": 0}
    payload = detector._empty_a3b()

    def counted_a3b(*_args: object) -> dict[str, Any]:
        calls["a3b"] += 1
        return dict(payload)

    monkeypatch.setattr(detector, "_compute_a3b", counted_a3b)

    for frame_idx in range(1, 8):
        detector.process(_item(frame_idx))
        worker = detector._a3b_bg_thread
        if worker is not None:
            worker.join(timeout=2.0)
            assert not worker.is_alive()

    assert calls["a3b"] == 2
    detector.close()


def test_target_anchored_dead_path_is_opt_in_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = _detector(monkeypatch)
    disabled_calls = {"count": 0}

    def disabled_evaluate(**_kwargs: object) -> dict[str, Any]:
        disabled_calls["count"] += 1
        return {"suspicious": False}

    monkeypatch.setattr(disabled._ta, "evaluate", disabled_evaluate)
    for frame_idx in range(1, 4):
        disabled_result = disabled.process(_item(frame_idx))

    enabled = _detector(
        monkeypatch,
        rebuilt_target_anchored_diagnostics=True,
    )
    enabled_calls = {"count": 0}

    def enabled_evaluate(**_kwargs: object) -> dict[str, Any]:
        enabled_calls["count"] += 1
        return {
            "suspicious": False,
            "classifier_bonus": False,
        }

    monkeypatch.setattr(enabled._ta, "evaluate", enabled_evaluate)
    for frame_idx in range(1, 4):
        enabled_result = enabled.process(_item(frame_idx))

    assert disabled_calls["count"] == 0
    assert disabled_result.details["target_anchored"] == {
        "enabled": False,
        "evaluated": False,
        "result": None,
    }
    assert enabled_calls["count"] == 3
    assert enabled_result.details["target_anchored"]["enabled"] is True
    assert enabled_result.details["target_anchored"]["evaluated"] is True


def test_core_a1_a2_a3_a4_stages_run_once_per_processed_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    counts = {"a1": 0, "a2": 0, "a3": 0, "a4": 0}

    for name in tuple(counts):
        attribute = f"_compute_{name}"
        original = getattr(detector, attribute)

        def wrapper(
            *args: object,
            _name: str = name,
            _original: Any = original,
            **kwargs: object,
        ) -> Any:
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(detector, attribute, wrapper)

    for frame_idx in range(1, 5):
        detector.process(_item(frame_idx))

    assert counts == {"a1": 4, "a2": 4, "a3": 4, "a4": 4}


def test_a3_returns_roi_coverage_ratio_for_motion_context_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        residual = np.zeros((16, 16), dtype=np.float32)
        magnitude = np.zeros((16, 16), dtype=np.float32)
        residual[4:8, 4:8] = 1.0
        magnitude[4:8, 4:8] = 1.0
        result = detector._compute_a3(
            {
                "available": True,
                "residual_mag": residual,
                "mag": magnitude,
                "flow_scale": 0.25,
                "global_motion_weight": 0.0,
                "background_coherence": 0.0,
            },
            [ROI("helmet-1", (16, 16, 32, 32), "helmet", 0.9)],
            64,
            64,
            {
                "exposure_delta": 0.0,
                "frame_diff_global": 0.0,
            },
        )

        assert "flow_roi_coverage_ratio" in result
        assert 0.0 <= result["flow_roi_coverage_ratio"] <= 1.0
    finally:
        detector.close()


def test_normal_target_motion_blocks_existing_adv_confirmation_and_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        rebuilt_alert_hold_frames=3,
        rebuilt_sustained_adv_escalation=True,
        rebuilt_sustained_adv_seconds=0.04,
        rebuilt_sustained_adv_require_physical_support=False,
    )
    try:
        baseline = detector.process(_item(0)).details
        detector.reset()
        a1 = copy.deepcopy(baseline["a1"])
        a2 = copy.deepcopy(baseline["a2"])
        a3 = copy.deepcopy(baseline["a3"])
        a4 = copy.deepcopy(baseline["a4"])
        a3b = copy.deepcopy(baseline["a3b"])
        exposure = copy.deepcopy(
            baseline["scene_context"]
        )
        flow = copy.deepcopy(baseline["flow_context"])

        a1.update(
            a1_feature_score=0.42,
            target_related=True,
        )
        a2.update(
            a2_feature_score=0.18,
            target_related=True,
        )
        a3.update(
            a3_feature_score=0.944,
            target_related=True,
            flow_local_anomaly_ratio=0.056,
            flow_roi_coverage_ratio=0.20,
            flow_shape_score=0.944,
            flow_residual_contrast=1.52,
            flow_roi_motion_gap=1.70,
            flow_target_relation=1.0,
        )
        a4.update(
            p_adv=0.95,
            p_adv_triggered=True,
            dominant_adv_input="A3_FLOW_ARTIFACT",
            a4_multi_evidence=0.02,
        )
        a3b.update(
            media_candidate_allowed=False,
            p_media_target_related=False,
            p_media_strong_evidence=False,
            p_media_policy=0.0,
            suppressed_reason="natural_scene_texture_plane",
        )
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_delta=0.005,
            frame_diff_global=0.0107,
        )
        flow.update(global_motion_weight=0.0)
        detector.recent_target_presence.extend([1] * 8)
        detector.adv_hits.extend([1] * 8)
        detector.adv_support_hits.extend([1] * 8)
        detector._alert_hold_channel = "adv"
        detector._alert_hold_remaining = 2

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [
                ROI(
                    "helmet-1",
                    (18, 10, 34, 28),
                    "helmet",
                    0.9,
                )
            ],
            exposure,
            flow,
            blinding={
                "p_blind": 0.0,
                "p_blind_triggered": False,
                "blind_type": "none",
                "sharp_drop": 0.0,
                "glare_blind": 0.0,
                "blind_independent_support": False,
            },
        )

        assert (
            decision["normal_target_motion_exclusion"]
            is True
        )
        assert (
            decision["a3_independent_attack_support"]
            is False
        )
        assert decision["adv_confirmed"] is True
        assert (
            decision["adv_explicitly_suppressed"]
            is True
        )
        assert decision["alert_confirmed"] is False
        assert (
            decision["alert_confirmation_source"]
            == "none"
        )
        assert (
            decision["alert_hold_blocked_reason"]
            == "normal_target_motion_exclusion"
        )
        assert decision["sustained_adv_run"] == 0
        assert decision["sustained_adv_escalated"] is False
        assert (
            decision["confirm_window"][
                "alert_hold_remaining"
            ]
            == 0
        )
    finally:
        detector.close()


def test_normal_motion_gate_blocks_bridge_history_and_hold_even_with_recent_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        rebuilt_alert_hold_frames=3,
        rebuilt_sustained_adv_escalation=False,
    )
    try:
        baseline = detector.process(_item(0)).details
        detector.reset()
        a1 = copy.deepcopy(baseline["a1"])
        a2 = copy.deepcopy(baseline["a2"])
        a3 = copy.deepcopy(baseline["a3"])
        a4 = copy.deepcopy(baseline["a4"])
        a3b = copy.deepcopy(baseline["a3b"])
        exposure = copy.deepcopy(
            baseline["scene_context"]
        )
        flow = copy.deepcopy(baseline["flow_context"])

        a1.update(
            a1_feature_score=0.82,
            target_related=True,
            delta_h_roi_patch_max=0.30,
            delta_h_patch_concentration=1.0,
        )
        a2.update(
            a2_feature_score=0.82,
            target_related=True,
            change_t_global=0.30,
            change_t_local_max=0.40,
            flash_like=False,
        )
        a3.update(
            a3_feature_score=0.20,
            target_related=True,
            flow_local_anomaly_ratio=0.05,
            flow_max_magnitude_norm=0.40,
            flow_roi_coverage_ratio=0.10,
        )
        a4.update(
            p_adv=0.92,
            p_adv_triggered=True,
            dominant_adv_input="A2_LBP_TEMPORAL",
            a4_multi_evidence=0.40,
        )
        a3b.update(
            media_candidate_allowed=False,
            p_media_target_related=False,
            p_media_strong_evidence=False,
            p_media_policy=0.0,
            suppressed_reason="pure_static_background",
        )
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_delta=0.001,
            frame_diff_global=0.025,
        )
        flow.update(global_motion_weight=0.40)
        detector.recent_target_presence.extend([1] * 8)
        detector.adv_hits.extend([1] * 8)
        detector.adv_support_hits.extend([1] * 8)
        detector._adv_cand_bridge_remaining = 2
        detector._adv_cand_bridge_has_physical_support = True
        detector._alert_hold_channel = "adv"
        detector._alert_hold_remaining = 2

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [
                ROI(
                    "helmet-1",
                    (18, 10, 34, 28),
                    "helmet",
                    0.9,
                )
            ],
            exposure,
            flow,
            blinding={
                "p_blind": 0.0,
                "p_blind_triggered": False,
                "blind_type": "none",
                "sharp_drop": 0.0,
                "glare_blind": 0.0,
                "blind_independent_support": False,
            },
        )

        assert decision["normal_motion_texture_change"] is True
        assert (
            decision["adv_candidate_bridge_recent_physical_support"]
            is True
        )
        assert (
            decision["adv_candidate_bridge_explicit_suppression"]
            is True
        )
        assert decision["adv_candidate_bridge_blocked"] is True
        assert decision["adv_candidate_bridged"] is False
        assert decision["adv_confirmed"] is True
        assert decision["adv_explicitly_suppressed"] is True
        assert (
            decision["adv_explicit_suppression_reason"]
            == "normal_motion_texture_change"
        )
        assert decision["alert_confirmed"] is False
        assert (
            decision["alert_hold_blocked_reason"]
            == "normal_motion_texture_change"
        )
    finally:
        detector.close()


@pytest.mark.parametrize(
    (
        "require_physical_support",
        "expected_independent_support",
        "expected_support_requirement",
        "expected_sustained",
        "expected_alert",
    ),
    [
        (True, False, False, False, False),
        (False, False, True, True, True),
    ],
    ids=["physical-support-required", "physical-support-optional"],
)
def test_sustained_adv_physical_support_switch_has_effect(
    monkeypatch: pytest.MonkeyPatch,
    require_physical_support: bool,
    expected_independent_support: bool,
    expected_support_requirement: bool,
    expected_sustained: bool,
    expected_alert: bool,
) -> None:
    detector = _detector(
        monkeypatch,
        rebuilt_sustained_adv_seconds=0.04,
        rebuilt_sustained_adv_require_physical_support=(
            require_physical_support
        ),
    )
    try:
        baseline = detector.process(_item(0)).details
        detector.reset()
        a1 = copy.deepcopy(baseline["a1"])
        a2 = copy.deepcopy(baseline["a2"])
        a3 = copy.deepcopy(baseline["a3"])
        a4 = copy.deepcopy(baseline["a4"])
        a3b = copy.deepcopy(baseline["a3b"])
        exposure = copy.deepcopy(
            baseline["scene_context"]
        )
        flow = copy.deepcopy(baseline["flow_context"])

        a1.update(
            a1_feature_score=0.82,
            target_related=True,
            delta_h_roi_patch_max=0.35,
            delta_h_patch_concentration=1.0,
        )
        a2.update(
            a2_feature_score=0.82,
            target_related=True,
            change_t_global=0.18,
            change_t_local_max=0.70,
            flash_like=False,
        )
        a3.update(
            a3_feature_score=0.10,
            target_related=True,
            flow_local_anomaly_ratio=0.20,
            flow_max_magnitude_norm=0.50,
            flow_roi_coverage_ratio=0.05,
            flow_residual_contrast=0.10,
            a3_residual_hold_active=False,
        )
        a4.update(
            p_adv=0.92,
            p_adv_triggered=True,
            dominant_adv_input="A2_LBP_TEMPORAL",
            a4_multi_evidence=0.50,
        )
        a3b.update(
            media_candidate_allowed=False,
            p_media_target_related=False,
            p_media_strong_evidence=False,
            p_media_policy=0.0,
            suppressed_reason="pure_static_background",
        )
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_delta=0.001,
            frame_diff_global=0.025,
        )
        flow.update(global_motion_weight=0.40)
        detector.recent_target_presence.extend([1] * 8)
        detector.adv_hits.extend([1] * 4)
        detector.adv_support_hits.extend([0] * 4)

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [
                ROI(
                    "helmet-1",
                    (18, 10, 34, 28),
                    "helmet",
                    0.9,
                )
            ],
            exposure,
            flow,
            blinding={
                "p_blind": 0.0,
                "p_blind_triggered": False,
                "blind_type": "none",
                "sharp_drop": 0.0,
                "glare_blind": 0.0,
                "blind_independent_support": False,
            },
        )

        assert decision["adv_candidate_allowed"] is True
        assert decision["adv_single_frame_candidate"] is True
        assert decision["localized_a1_attack_support"] is False
        assert decision["photometric_attack_support"] is False
        assert decision["a3_independent_attack_support"] is False
        assert decision["adv_physical_support"] is False
        assert decision["confirm_window"]["adv_count"] >= 5
        assert decision["confirm_window"]["adv_support_count"] == 0
        assert decision["adv_confirmed"] is False
        assert (
            decision["sustained_adv_has_independent_support"]
            is expected_independent_support
        )
        assert (
            decision[
                "sustained_adv_support_requirement_satisfied"
            ]
            is expected_support_requirement
        )
        assert (
            decision["sustained_adv_escalated"]
            is expected_sustained
        )
        assert decision["alert_confirmed"] is expected_alert
        assert decision["alert_confirmation_source"] == (
            "adv_sustained" if expected_alert else "none"
        )
    finally:
        detector.close()


def test_localized_patch_evidence_supplies_adv_confirmation_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        rebuilt_sustained_adv_escalation=False,
    )
    try:
        baseline = detector.process(_item(0)).details
        detector.reset()
        a1 = copy.deepcopy(baseline["a1"])
        a2 = copy.deepcopy(baseline["a2"])
        a3 = copy.deepcopy(baseline["a3"])
        a4 = copy.deepcopy(baseline["a4"])
        a3b = copy.deepcopy(baseline["a3b"])
        exposure = copy.deepcopy(
            baseline["scene_context"]
        )
        flow = copy.deepcopy(baseline["flow_context"])

        a1.update(
            a1_feature_score=0.88,
            target_related=True,
            delta_h_roi_patch_max=0.72,
            delta_h_patch_concentration=0.90,
        )
        a2.update(
            a2_feature_score=0.30,
            target_related=True,
            change_t_global=0.10,
            change_t_local_max=0.70,
            flash_like=False,
        )
        a3.update(
            a3_feature_score=0.20,
            target_related=True,
            flow_local_anomaly_ratio=0.20,
            flow_max_magnitude_norm=0.50,
            flow_roi_coverage_ratio=0.05,
            flow_residual_contrast=0.10,
            a3_residual_hold_active=False,
        )
        a4.update(
            p_adv=0.92,
            p_adv_triggered=True,
            dominant_adv_input="A1_LBP_SINGLE",
            a4_multi_evidence=0.50,
        )
        a3b.update(
            media_candidate_allowed=False,
            p_media_target_related=True,
            p_media_strong_evidence=False,
            p_media_policy=0.0,
            suppressed_reason=(
                "target_attached_patch_prefers_A1_A2_A3"
            ),
            p_media_scores={
                "boundary": 0.10,
                "display_frame": 0.40,
                "area_ratio": 0.08,
            },
        )
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_delta=0.001,
            frame_diff_global=0.025,
        )
        flow.update(global_motion_weight=0.30)
        detector.recent_target_presence.extend([1] * 8)
        detector.adv_hits.extend([1] * 4)
        detector.adv_support_hits.extend([0] * 4)

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [
                ROI(
                    "helmet-1",
                    (18, 10, 34, 28),
                    "helmet",
                    0.9,
                )
            ],
            exposure,
            flow,
            blinding={
                "p_blind": 0.0,
                "p_blind_triggered": False,
                "blind_type": "none",
                "sharp_drop": 0.0,
                "glare_blind": 0.0,
                "blind_independent_support": False,
            },
        )

        assert decision["localized_a1_attack_support"] is True
        assert decision["a3_independent_attack_support"] is True
        assert decision["adv_physical_support"] is True
        assert decision["confirm_window"]["adv_support_count"] >= 1
        assert decision["adv_confirmed"] is True
        assert decision["alert_confirmed"] is True
        assert decision["primary_channel"] == "adv"
    finally:
        detector.close()


def test_overexposure_with_strong_texture_supplies_glare_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        rebuilt_sustained_adv_escalation=False,
    )
    try:
        baseline = detector.process(_item(0)).details
        detector.reset()
        a1 = copy.deepcopy(baseline["a1"])
        a2 = copy.deepcopy(baseline["a2"])
        a3 = copy.deepcopy(baseline["a3"])
        a4 = copy.deepcopy(baseline["a4"])
        a3b = copy.deepcopy(baseline["a3b"])
        exposure = copy.deepcopy(
            baseline["scene_context"]
        )
        flow = copy.deepcopy(baseline["flow_context"])

        a1.update(
            a1_feature_score=0.90,
            target_related=True,
            delta_h_roi_patch_max=0.20,
            delta_h_patch_concentration=0.30,
        )
        a2.update(
            a2_feature_score=0.82,
            target_related=True,
            change_t_global=0.10,
            change_t_local_max=0.70,
            flash_like=False,
        )
        a3.update(
            a3_feature_score=0.20,
            target_related=True,
            flow_local_anomaly_ratio=0.20,
            flow_max_magnitude_norm=0.50,
            flow_roi_coverage_ratio=0.05,
        )
        a4.update(
            p_adv=0.94,
            p_adv_triggered=True,
            dominant_adv_input="A1_LBP_SINGLE",
            a4_multi_evidence=0.50,
        )
        a3b.update(
            media_candidate_allowed=False,
            p_media_target_related=False,
            p_media_strong_evidence=False,
            p_media_policy=0.0,
            suppressed_reason="pure_static_background",
        )
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.15,
            underexposed_ratio=0.0,
            exposure_delta=0.01,
            frame_diff_global=0.03,
        )
        flow.update(global_motion_weight=0.30)
        detector.recent_target_presence.extend([1] * 8)
        detector.adv_hits.extend([1] * 4)
        detector.adv_support_hits.extend([0] * 4)

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [
                ROI(
                    "helmet-1",
                    (18, 10, 34, 28),
                    "helmet",
                    0.9,
                )
            ],
            exposure,
            flow,
            blinding={
                "p_blind": 0.0,
                "p_blind_triggered": False,
                "blind_type": "none",
                "sharp_drop": 0.0,
                "glare_blind": 0.0,
                "blind_independent_support": False,
            },
        )

        assert decision["glare_attack_support"] is True
        assert decision["photometric_attack_support"] is True
        assert decision["a3_independent_attack_support"] is True
        assert decision["adv_physical_support"] is True
        assert decision["adv_confirmed"] is True
        assert decision["alert_confirmed"] is True
        assert decision["primary_channel"] == "adv"
    finally:
        detector.close()


def test_missing_blind_support_fails_closed_and_cancels_blind_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        rebuilt_alert_hold_frames=3,
        rebuilt_sustained_adv_escalation=False,
    )
    try:
        baseline = detector.process(_item(0)).details
        detector.reset()
        detector.blind_hits.extend([1] * 8)
        detector._alert_hold_channel = "blind"
        detector._alert_hold_remaining = 2

        decision = detector._joint_decision(
            copy.deepcopy(baseline["a1"]),
            copy.deepcopy(baseline["a2"]),
            copy.deepcopy(baseline["a3"]),
            copy.deepcopy(baseline["a4"]),
            copy.deepcopy(baseline["a3b"]),
            [],
            copy.deepcopy(baseline["scene_context"]),
            copy.deepcopy(baseline["flow_context"]),
            blinding={
                "p_blind": 0.90,
                "p_blind_triggered": True,
                "blind_type": "motion_blur",
                "sharp_drop": 0.80,
                "glare_blind": 0.0,
            },
        )

        assert decision["blind_confirmed"] is True
        assert (
            decision["blind_independent_support"]
            is False
        )
        assert (
            decision["blind_explicitly_suppressed"]
            is True
        )
        assert decision["alert_confirmed"] is False
        assert (
            decision["alert_hold_blocked_reason"]
            == "blind_independent_support_missing"
        )
    finally:
        detector.close()


def test_high_motion_target_loss_without_independent_support_is_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        gray = np.indices((64, 64), dtype=np.uint8).sum(axis=0) % 2
        gray = (gray * 255).astype(np.uint8)
        sharpness = float(
            detector_module.cv2.Laplacian(
                gray,
                detector_module.cv2.CV_32F,
            ).var()
        )
        detector._sb_sharp.extend([100.0] * detector._scene_baseline_min)
        detector._sb_contrast.extend([1.0] * detector._scene_baseline_min)
        detector._sb_detstr.extend([1.0] * detector._scene_baseline_min)
        detector.recent_target_presence.extend([1] * 4)
        detector._prev_sharp = sharpness

        def fake_pctl(values: object, _q: float) -> float:
            if values is detector._sb_sharp:
                return sharpness / 0.50
            if values is detector._sb_contrast:
                return 1.0 / 0.90
            return 2.0

        monkeypatch.setattr(detector, "_pctl", fake_pctl)
        result = detector._compute_blinding(
            gray,
            [],
            {
                "brightness_std": 1.0,
                "overexposure_ratio": 0.0,
                "underexposed_ratio": 0.0,
                "frame_diff_global": 0.08,
                "exposure_delta": 0.0,
            },
            {
                "available": True,
                "global_motion_weight": 1.0,
            },
        )

        assert result["blind_type"] == "motion_blur"
        assert (
            result["low_motion_target_loss_support"]
            is False
        )
        assert result["blind_independent_support"] is False
        assert result["sharp_drop_short"] == pytest.approx(0.0)
        assert float(result["p_blind"]) <= 0.40
    finally:
        detector.close()


def test_low_motion_target_loss_promotes_relative_blur_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        gray = np.tile(
            np.linspace(0, 255, 64, dtype=np.uint8),
            (64, 1),
        )
        sharpness = float(
            detector_module.cv2.Laplacian(
                gray,
                detector_module.cv2.CV_32F,
            ).var()
        )
        contrast = float(gray.std())
        detector._sb_sharp.extend(
            [100.0] * detector._scene_baseline_min
        )
        detector._sb_contrast.extend(
            [1.0] * detector._scene_baseline_min
        )
        detector._sb_detstr.extend(
            [1.0] * detector._scene_baseline_min
        )
        detector.recent_target_presence.extend([1] * 4)
        detector._prev_sharp = sharpness

        def fake_pctl(values: object, _q: float) -> float:
            if values is detector._sb_sharp:
                return sharpness / 0.75
            if values is detector._sb_contrast:
                return contrast / 0.95
            return 2.0

        monkeypatch.setattr(detector, "_pctl", fake_pctl)
        result = detector._compute_blinding(
            gray,
            [],
            {
                "brightness_std": contrast,
                "overexposure_ratio": 0.0,
                "underexposed_ratio": 0.0,
                "frame_diff_global": 0.01,
                "exposure_delta": 0.0,
            },
            {
                "available": True,
                "global_motion_weight": 0.10,
            },
        )

        assert result["blind_type"] == "motion_blur"
        assert (
            result["low_motion_target_loss_support"]
            is True
        )
        assert result["blur_detail_ratio"] <= 0.25
        assert result["blind_independent_support"] is True
        assert result["p_blind_triggered"] is True
        assert float(result["p_blind"]) >= detector.theta_blind
    finally:
        detector.close()


def test_low_motion_target_loss_rejects_high_detail_scene_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        gray = np.indices((64, 64), dtype=np.uint8).sum(axis=0) % 2
        gray = (gray * 255).astype(np.uint8)
        sharpness = float(
            detector_module.cv2.Laplacian(
                gray,
                detector_module.cv2.CV_32F,
            ).var()
        )
        contrast = float(gray.std())
        detector._sb_sharp.extend(
            [100.0] * detector._scene_baseline_min
        )
        detector._sb_contrast.extend(
            [1.0] * detector._scene_baseline_min
        )
        detector._sb_detstr.extend(
            [1.0] * detector._scene_baseline_min
        )
        detector.recent_target_presence.extend([1] * 4)
        detector._prev_sharp = sharpness

        def fake_pctl(values: object, _q: float) -> float:
            if values is detector._sb_sharp:
                return sharpness / 0.75
            if values is detector._sb_contrast:
                return contrast / 0.95
            return 2.0

        monkeypatch.setattr(detector, "_pctl", fake_pctl)
        result = detector._compute_blinding(
            gray,
            [],
            {
                "brightness_std": contrast,
                "overexposure_ratio": 0.0,
                "underexposed_ratio": 0.0,
                "frame_diff_global": 0.01,
                "exposure_delta": 0.0,
            },
            {
                "available": True,
                "global_motion_weight": 0.10,
            },
        )

        assert result["blind_type"] == "motion_blur"
        assert result["blur_detail_ratio"] > 0.25
        assert (
            result["low_motion_target_loss_support"]
            is False
        )
        assert result["blind_independent_support"] is False
        assert result["p_blind_triggered"] is False
        assert float(result["p_blind"]) <= 0.40
    finally:
        detector.close()


def test_blind_suspect_freeze_requires_independent_motion_blur_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        detector.recent_target_presence.extend([1, 1, 1, 0])
        a1 = {"a1_feature_score": 0.0}
        a2 = {"a2_feature_score": 0.0}
        a3 = {"a3_feature_score": 0.0}
        joint = {"alert_confirmed": False, "single_frame_candidate": False}
        base = {
            "sharpness": 100.0,
            "contrast": 1.0,
            "det_strength": 0.0,
            "sharp_drop": 0.8,
            "glare_blind": 0.0,
        }

        detector._update_scene_baseline(
            {**base, "blind_independent_support": False},
            a1,
            a2,
            a3,
            joint,
        )
        assert len(detector._sb_sharp) == 1

        detector._sb_sharp.clear()
        detector._sb_contrast.clear()
        detector._sb_detstr.clear()
        detector._sb_maxfeat.clear()
        detector._update_scene_baseline(
            {**base, "blind_independent_support": True},
            a1,
            a2,
            a3,
            joint,
        )
        assert len(detector._sb_sharp) == 0
    finally:
        detector.close()
