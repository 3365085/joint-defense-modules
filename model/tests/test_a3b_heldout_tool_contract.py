from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defense.diagnostics import a3b_heldout


class _FakeCapture:
    def __init__(self, _path: str) -> None:
        self.frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8),
        ]
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        self.position += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == a3b_heldout.cv2.CAP_PROP_FPS:
            return 30.0
        if prop == a3b_heldout.cv2.CAP_PROP_POS_MSEC:
            return self.position / 30.0 * 1000.0
        return 0.0

    def release(self) -> None:
        self.released = True


class _FakePipeline:
    def __init__(self) -> None:
        self.reset_count = 0
        self.warmup_calls: list[int] = []
        self.closed = False

    def warmup(self, frames: int) -> None:
        self.warmup_calls.append(frames)

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class _FakeProcessor:
    instances: list["_FakeProcessor"] = []

    def __init__(self, bundle, *, jpeg_quality: int) -> None:
        self.bundle = bundle
        self.jpeg_quality = jpeg_quality
        self.reset_count = 0
        self.calls: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    def reset(self) -> None:
        self.reset_count += 1

    def process(self, _frame, **kwargs):
        self.calls.append(dict(kwargs))
        frame_idx = int(kwargs["frame_idx"])
        triggered = "attack" in str(kwargs["source"]) and frame_idx == 1
        return SimpleNamespace(
            status={
                "a3b_triggered": triggered,
                "a3b_triggered_source": "runtime_soft_trigger" if triggered else "none",
                "a3b_score": 0.7 if triggered else 0.1,
                "a3b_observed_score": 0.6 if triggered else 0.1,
                "a3b_confirmed_score": 0.7 if triggered else 0.0,
                "a3b_error_count": 0,
                "a3b_timed_out_worker_count": 0,
                "a3b_worker_rejected_count": 0,
                "a3b_active_worker_count": 1,
                "a3b_retired_worker_count": 0,
                "a3b_live_worker_count": 1,
                "a3b_global_live_worker_count": 1,
                "a3b_result_expired_count": 0,
                "a3b_schedule_blocked": False,
                "a3b_result_seq": frame_idx + 1,
                "a3b_result_fresh": True,
                "temporal_input": {
                    "strict_source_predecessor": frame_idx > 0,
                },
                "module_a_effective_config": {
                    "detector_impl": "rebuilt",
                },
            },
            info={
                "details": {
                    "a3b": {
                        "result_contract_source": "rebuilt",
                        "media_confirmed": False,
                    }
                }
            },
        )


class _FakeCache:
    instances: list["_FakeCache"] = []

    def __init__(self, *, config_path: Path, root: Path) -> None:
        self.config_path = config_path
        self.root = root
        self.pipeline = _FakePipeline()
        self.cleared = False
        self.__class__.instances.append(self)

    def get(self, **_kwargs):
        return SimpleNamespace(
            pipeline=self.pipeline,
            config={
                "runtime": {
                    "detector_process_fps_cap": 30,
                    "detector_thread_warmup_frames": 2,
                    "jpeg_quality": 82,
                }
            },
            warmup_frames=1,
            backend="fake",
            model_family="fake",
            artifact_path="fake.engine",
            artifact_fingerprint=("path", "fake.engine", "hash"),
            auxiliary_artifact_fingerprint=("empty", ""),
        )

    def clear(self) -> None:
        self.cleared = True
        self.pipeline.close()


def _write_manifest(path: Path, clean: Path, attack: Path) -> None:
    fieldnames = ["clip_id", "path", "label", "attack_type", "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "clean",
                "path": str(clean),
                "label": 0,
                "attack_type": "clean",
                "split": "heldout",
            }
        )
        writer.writerow(
            {
                "clip_id": "attack",
                "path": str(attack),
                "label": 1,
                "attack_type": "glare",
                "split": "heldout",
            }
        )


def test_evaluate_a3b_heldout_uses_frame_processor_and_strict_predecessor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _FakeCache.instances.clear()
    _FakeProcessor.instances.clear()
    clean = tmp_path / "clean.mp4"
    attack = tmp_path / "attack.mp4"
    clean.write_bytes(b"clean")
    attack.write_bytes(b"attack")
    manifest = tmp_path / "dataset_manifest.csv"
    config = tmp_path / "runtime.yaml"
    output = tmp_path / "report.json"
    _write_manifest(manifest, clean, attack)
    config.write_text("runtime: {}", encoding="utf-8")
    monkeypatch.setattr(a3b_heldout, "PipelineCache", _FakeCache)
    monkeypatch.setattr(a3b_heldout, "FrameProcessor", _FakeProcessor)
    monkeypatch.setattr(a3b_heldout.cv2, "VideoCapture", _FakeCapture)

    report = a3b_heldout.evaluate_a3b_heldout(
        manifest=manifest,
        output_json=output,
        config=config,
        repository_root=tmp_path,
        cap_frames=2,
    )

    assert report["summary"]["clips"] == 2
    assert report["summary"]["clean_a3b_fp_videos"] == 0
    assert report["summary"]["clean_module_a_alert_videos"] == 0
    assert report["summary"]["clean_module_a_alert_frames"] == 0
    assert report["summary"]["physical_attack_hit_videos"] == 0
    assert report["summary"]["physical_attack_missed_videos"] == 1
    assert report["summary"]["attack_wrong_channel_videos"] == 1
    assert report["summary"]["clips_with_temporal_predecessor_gaps"] == 0
    assert report["metadata"]["evaluation_path"] == (
        "FrameProcessor/ModuleAResult+A3BSoftTriggerState"
    )
    assert report["metadata"]["evaluation_scope"] == (
        "module_a_alerts_and_a3b_runtime_state"
    )
    assert report["metadata"]["source_identity"]["evaluator_version"] == (
        a3b_heldout.A3B_HELDOUT_EVALUATOR_VERSION
    )
    assert a3b_heldout.heldout_gate_failures(report) == [
        "attack_wrong_channel_videos=1"
    ]
    assert output.is_file()
    processor = _FakeProcessor.instances[0]
    assert processor.reset_count == 2
    assert processor.calls[0]["temporal_previous_frame"] is None
    assert processor.calls[1]["temporal_previous_frame"] is not None
    assert processor.calls[1]["temporal_previous_frame_idx"] == 0
    cache = _FakeCache.instances[0]
    assert cache.cleared is True
    assert cache.pipeline.closed is True


def test_summary_does_not_count_a3b_positive_as_wrong_channel() -> None:
    summary = a3b_heldout._summary(
        profile="desktop_rtx",
        cap_frames=10,
        rows=[
            {
                "label": 2,
                "attack_type": "a3b_replay",
                "a3b_trigger_frames": 4,
            },
            {
                "label": 1,
                "attack_type": "glare",
                "a3b_trigger_frames": 3,
            },
        ],
        elapsed_s=1.0,
    )

    assert summary["a3b_positive_clips"] == 1
    assert summary["a3b_positive_hit_videos"] == 1
    assert summary["a3b_positive_trigger_frames"] == 4
    assert summary["attack_wrong_channel_videos"] == 1
    assert summary["attack_wrong_channel_frames"] == 3


def test_gate_failures_include_background_health_and_warmup_errors() -> None:
    failures = a3b_heldout.heldout_gate_failures(
        {
            "summary": {
                "clips_with_errors": 0,
                "clips_with_backend_errors": 1,
                "clips_with_worker_timeouts": 2,
                "clips_with_temporal_predecessor_gaps": 1,
            },
            "metadata": {
                "thread_warmup_error": "RuntimeError: warmup failed",
            },
        }
    )

    assert failures == [
        "clips_with_backend_errors=1",
        "clips_with_worker_timeouts=2",
        "clips_with_temporal_predecessor_gaps=1",
        "thread_warmup_error=RuntimeError: warmup failed",
    ]


def test_gate_failures_include_clean_module_a_and_a3b_behavior() -> None:
    failures = a3b_heldout.heldout_gate_failures(
        {
            "summary": {
                "clean_module_a_alert_videos": 2,
                "clean_a3b_fp_videos": 1,
                "attack_wrong_channel_videos": 3,
            },
            "metadata": {},
        }
    )

    assert failures == [
        "clean_module_a_alert_videos=2",
        "clean_a3b_fp_videos=1",
        "attack_wrong_channel_videos=3",
    ]


def test_gate_failures_reject_attack_recall_below_current_baseline() -> None:
    failures = a3b_heldout.heldout_gate_failures(
        {
            "summary": {
                "physical_attack_clips": 21,
                "physical_attack_hit_videos": 19,
            },
            "metadata": {},
        }
    )

    assert failures == ["physical_attack_hit_videos=19/21"]


def test_summary_reports_module_a_alerts_separately_from_a3b() -> None:
    summary = a3b_heldout._summary(
        profile="desktop_rtx",
        cap_frames=240,
        rows=[
            {
                "label": 0,
                "attack_type": "clean",
                "module_a_single_frame_suspicious_frames": 4,
                "module_a_attack_detected_frames": 7,
                "module_a_alert_confirmed_frames": 6,
                "module_a_fresh_confirmed_frames": 2,
                "module_a_held_confirmed_frames": 4,
                "module_a_evidence_condition_frames": 6,
                "module_a_primary_channel_counts": {
                    "adv": 2,
                    "blind": 4,
                },
                "a3b_trigger_frames": 0,
            },
            {
                "label": 1,
                "attack_type": "glare",
                "module_a_alert_confirmed_frames": 8,
                "a3b_trigger_frames": 0,
            },
        ],
        elapsed_s=1.0,
    )

    assert summary["clean_module_a_suspicious_videos"] == 1
    assert summary["clean_module_a_suspicious_frames"] == 4
    assert summary["clean_module_a_alert_videos"] == 1
    assert summary["clean_module_a_alert_frames"] == 6
    assert summary["clean_module_a_fresh_confirmed_frames"] == 2
    assert summary["clean_module_a_held_confirmed_frames"] == 4
    assert summary["clean_module_a_evidence_condition_videos"] == 1
    assert summary["clean_module_a_alert_channels"] == {
        "adv": 2,
        "blind": 4,
    }
    assert summary["physical_attack_hit_videos"] == 1
    assert summary["physical_attack_missed_videos"] == 0
    assert summary["clean_a3b_fp_videos"] == 0


def test_cache_is_cleared_when_pipeline_initialization_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean.mp4"
    attack = tmp_path / "attack.mp4"
    clean.write_bytes(b"clean")
    attack.write_bytes(b"attack")
    manifest = tmp_path / "dataset_manifest.csv"
    config = tmp_path / "runtime.yaml"
    _write_manifest(manifest, clean, attack)
    config.write_text("runtime: {}", encoding="utf-8")

    class FailingCache:
        instance = None

        def __init__(self, **_kwargs) -> None:
            self.cleared = False
            self.__class__.instance = self

        def get(self, **_kwargs):
            raise RuntimeError("initialization failed")

        def clear(self) -> None:
            self.cleared = True

    monkeypatch.setattr(a3b_heldout, "PipelineCache", FailingCache)

    with pytest.raises(RuntimeError, match="initialization failed"):
        a3b_heldout.evaluate_a3b_heldout(
            manifest=manifest,
            output_json=tmp_path / "report.json",
            config=config,
            repository_root=tmp_path,
            cap_frames=2,
        )

    assert FailingCache.instance is not None
    assert FailingCache.instance.cleared is True
