from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from defense.diagnostics.module_a_tuning import (
    authoritative_frame_row,
    build_effective_config,
    parse_tuning_patch,
    run_module_a_videos,
    write_module_a_reports,
)


MODEL_ROOT = Path(__file__).resolve().parents[1]
TUNING_TOOL = MODEL_ROOT / "tools" / "module_a_tuning_app.py"
SIGNAL_TOOL = MODEL_ROOT / "tools" / "_diag_signals.py"


def _load_tuning_tool() -> Any:
    spec = importlib.util.spec_from_file_location("module_a_tuning_app_test", TUNING_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _info(
    *,
    frame_idx: int,
    reuse_hit: bool,
    reuse_reason: str,
    backend_predict_count: int,
) -> dict[str, Any]:
    return {
        "alert_confirmed": frame_idx == 1,
        "single_frame_suspicious": frame_idx == 1,
        "attack_state_active": frame_idx == 1,
        "is_attack": frame_idx == 1,
        "p_adv": 0.2 + frame_idx * 0.1,
        "p_adv_display": 0.25 + frame_idx * 0.1,
        "reason_codes": ["AUTHORITATIVE_REASON"] if frame_idx == 1 else [],
        "detector_inference_ms": 0.0 if reuse_hit else 5.0,
        "module_a_timing_ms": 3.0,
        "timing_ms": 8.5,
        "details": {
            "a1": {"a1_feature_score": 0.11},
            "a2": {"a2_feature_score": 0.22},
            "a3": {"a3_feature_score": 0.33, "flow_backend": "fake"},
            "a4": {"p_adv": 0.44},
            "a3b": {
                "p_media_policy": 0.55,
                "media_confirmed": frame_idx == 1,
                "a3b_result_seq": frame_idx,
            },
            "joint_decision": {
                "single_frame_candidate": frame_idx == 1,
                "candidate_source": "adv",
                "primary_channel": "A4",
                "public_reason": "authoritative",
                "suppressed_reason": "none",
            },
            "scene_context": {
                "overexposure_ratio": 0.01,
                "frame_diff_global": 0.02,
            },
            "flow_context": {"backend": "fake", "flow_sampled": True},
            "detections": {"roi_count": 1, "boxes": [[1, 2, 3, 4]]},
            "timing": {
                "pipeline_ms": 8.5,
                "detector_ms": 0.0 if reuse_hit else 5.0,
                "module_a_ms": 3.0,
            },
        },
        "latency_breakdown": {
            "e2e_ms": 8.5,
            "detector_ms": 0.0 if reuse_hit else 5.0,
            "module_a_total_ms": 3.0,
            "frame_resize_ms": 0.5,
            "detector_reuse_hit": reuse_hit,
            "detector_reuse": {"hit": reuse_hit, "reason": reuse_reason},
            "detector_reuse_counters": {
                "backend_predict_count": backend_predict_count,
            },
            "module_a_breakdown": {"a1": 0.5, "a3": 1.5},
        },
    }


class _FakeCapture:
    def __init__(self, _path: str) -> None:
        self.frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8),
        ]
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, _prop: int) -> float:
        return 25.0

    def release(self) -> None:
        self.released = True


class _FakePipeline:
    warmup_frames = 2

    def __init__(self) -> None:
        self.detector_backend = SimpleNamespace(
            backend="fake",
            artifact_path="fake.engine",
            device="cpu",
        )
        self.calls: list[dict[str, Any]] = []
        self.warmups: list[int] = []
        self.reset_count = 0
        self.closed = False

    def warmup(self, frames: int) -> None:
        self.warmups.append(frames)

    def reset(self) -> None:
        self.reset_count += 1

    def process_frame(self, _frame: np.ndarray, **kwargs: Any) -> tuple[None, None, dict[str, Any]]:
        self.calls.append(dict(kwargs))
        frame_idx = int(kwargs["source_frame_idx"])
        return (
            None,
            None,
            _info(
                frame_idx=frame_idx,
                reuse_hit=frame_idx == 1,
                reuse_reason="reused" if frame_idx == 1 else "no_cached_detection",
                backend_predict_count=1,
            ),
        )

    def close(self) -> None:
        self.closed = True


def test_effective_config_uses_deep_merge_and_requires_rebuilt() -> None:
    base, patch = build_effective_config(
        profile="desktop_rtx",
        tuning_patch={"module_a": {"static_image_interval": 9}},
    )

    assert patch == {"module_a": {"static_image_interval": 9}}
    assert base["module_a"]["static_image_interval"] == 9
    assert base["module_a"]["detector_impl"] == "rebuilt"
    assert "a4_classifier_path" in base["module_a"]
    assert "inference" in base

    with pytest.raises(ValueError, match="rebuilt"):
        build_effective_config(
            profile="desktop_rtx",
            tuning_patch={"module_a": {"detector_impl": "legacy"}},
        )


def test_flat_tuning_patch_remains_compatible(tmp_path: Path) -> None:
    assert parse_tuning_patch('{"alert_window": 7}') == {"alert_window": 7}
    patch_path = tmp_path / "patch.json"
    patch_path.write_text('{"module_a": {"alert_window": 8}}', encoding="utf-8")
    assert parse_tuning_patch(patch_path) == {"module_a": {"alert_window": 8}}


def test_frame_row_trusts_authoritative_decisions_without_local_gates() -> None:
    row = authoritative_frame_row(
        video_path=Path("sample.mp4"),
        frame_idx=1,
        source_time_s=0.04,
        info=_info(
            frame_idx=1,
            reuse_hit=True,
            reuse_reason="reused",
            backend_predict_count=1,
        ),
        pipeline_call_wall_ms=9.0,
    )

    assert row["alert_confirmed"] is True
    assert row["single_frame_suspicious"] is True
    assert row["reason_codes"] == ["AUTHORITATIVE_REASON"]
    assert row["candidate_source"] == "adv"
    assert row["detector_reuse_hit"] is True
    assert row["backend_inference_ms"] == 0.0
    assert row["module_a_ms"] == 3.0
    assert row["flow_sampled"] is True
    assert "flow_computed" not in row


def test_frame_row_uses_details_timing_when_latency_breakdown_is_missing() -> None:
    info = _info(
        frame_idx=0,
        reuse_hit=False,
        reuse_reason="no_cached_detection",
        backend_predict_count=1,
    )
    info.pop("latency_breakdown")
    info.pop("module_a_timing_ms")
    info["details"]["timing"] = {
        "scene_context": 0.1,
        "a1": 0.5,
        "result_build": 0.2,
        "total": 3.25,
    }

    row = authoritative_frame_row(
        video_path=Path("sample.mp4"),
        frame_idx=0,
        source_time_s=0.0,
        info=info,
        pipeline_call_wall_ms=9.0,
    )

    assert row["module_a_ms"] == pytest.approx(3.25)
    assert row["module_a_breakdown_ms"] == {
        "scene_context": pytest.approx(0.1),
        "a1": pytest.approx(0.5),
        "result_build": pytest.approx(0.2),
    }


def test_runner_passes_source_context_and_reports_reuse_and_timings(
    tmp_path: Path,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake")
    pipeline = _FakePipeline()

    report = run_module_a_videos(
        [video],
        profile="desktop_rtx",
        tuning_patch={"runtime": {"detector_process_fps_cap": 5}},
        max_frames=2,
        pipeline_factory=lambda _config: pipeline,
        capture_factory=_FakeCapture,
    )

    assert pipeline.warmups == [2]
    assert pipeline.reset_count == 1
    assert pipeline.closed is True
    assert [call["source_frame_idx"] for call in pipeline.calls] == [0, 1]
    assert [call["timestamp"] for call in pipeline.calls] == [0.0, 0.04]
    assert all(call["source_fps"] == 25.0 for call in pipeline.calls)

    video_report = report["videos"][0]
    summary = video_report["summary"]
    assert summary["frames"] == 2
    assert summary["decisions"]["alert_frames"] == 1
    assert summary["detector_reuse"]["hit_frames"] == 1
    assert summary["detector_reuse"]["hit_rate"] == 0.5
    assert summary["detector_reuse"]["backend_predict_count"] == 1
    assert summary["effective_expected_process_fps"] == 5.0
    assert summary["configured_frame_budget_ms"] == 200.0
    assert summary["performance_ms"]["backend_inference_ms"]["mean"] == 2.5
    assert summary["performance_ms"]["module_a_ms"]["mean"] == 3.0
    assert report["configuration"]["detector_impl"] == "rebuilt"

    outputs = write_module_a_reports(report, output_dir=tmp_path)
    report_text = Path(outputs["report"]).read_text(encoding="utf-8")
    frame_lines = Path(outputs["frames"]).read_text(encoding="utf-8").splitlines()
    assert '"frames": [' not in report_text
    assert len(frame_lines) == 2


def test_cli_wrappers_have_no_legacy_detector_or_http_server() -> None:
    tuning_source = TUNING_TOOL.read_text(encoding="utf-8")
    signal_source = SIGNAL_TOOL.read_text(encoding="utf-8")

    forbidden = (
        "from defense.module_a import ModuleADetector",
        "BaseHTTPRequestHandler",
        "HTTPServer",
        "serve_forever",
    )
    for token in forbidden:
        assert token not in tuning_source
        assert token not in signal_source

    # The diagnostic wrapper must not carry the old copied threshold gates.
    for token in ("ev_blur", "main_path", "flow_path", "p_adv>=0.55"):
        assert token not in signal_source


def test_legacy_server_arguments_fail_with_clear_deprecation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tuning_tool()
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake")

    with pytest.raises(SystemExit) as exc:
        tool.main(["--video", str(video), "--server"])

    assert exc.value.code == 2
    assert "legacy HTTP handler" in capsys.readouterr().err
