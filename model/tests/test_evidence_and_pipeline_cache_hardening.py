from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defense.runtime import evidence as evidence_module
from defense.runtime import pipeline_factory
from defense.runtime.evidence import EvidenceSession


def _frame(value: int = 0) -> np.ndarray:
    return np.full((24, 32, 3), value, dtype=np.uint8)


def _update_a3b(session: EvidenceSession, frame_idx: int, *, active: bool, value: int = 0) -> None:
    session.update(
        frame_idx=frame_idx,
        frame=_frame(value),
        info={},
        ppe={},
        status={
            "a3b_triggered": active,
            "a3b_event_score": 0.8 if active else 0.0,
            "a3b_triggered_source": "test" if active else "",
        },
    )


def test_disabled_evidence_session_performs_zero_filesystem_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "disabled-evidence"
    session = EvidenceSession(
        source_type="camera",
        source="0",
        profile="desktop_rtx",
        root=root,
        enabled=False,
    )

    assert session.session_dir is None
    assert session.manifest_path is None
    assert session.events_jsonl is None
    assert session.update(
        frame_idx=1,
        frame=_frame(),
        info={},
        ppe={},
        status={"a3b_triggered": True},
    ) == []
    assert session.reset(reason="disabled_reset", source_epoch=2) == []
    assert session.close() == []
    assert not root.exists()


def test_evidence_event_summary_contains_source_lineage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_clip_writer(monkeypatch)
    session = EvidenceSession(
        source_type="file",
        source="lineage.mp4",
        profile="desktop_rtx",
        run_id=11,
        source_epoch=7,
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
    )

    session.update(
        frame_idx=30,
        frame=_frame(),
        info={},
        ppe={},
        status={
            "run_id": 0,
            "source_epoch": 0,
            "source_time_s": 1.0,
            "a3b_triggered": True,
        },
    )
    session.update(
        frame_idx=31,
        frame=_frame(),
        info={},
        ppe={},
        status={
            "run_id": 0,
            "source_epoch": 0,
            "source_time_s": 1.25,
            "a3b_triggered": True,
        },
    )
    event = session.close()[0]

    assert event["run_id"] == 0
    assert event["source_epoch"] == 0
    assert event["source_frame_start"] == 30
    assert event["source_frame_end"] == 31
    assert event["source_time_start_s"] == pytest.approx(1.0)
    assert event["source_time_end_s"] == pytest.approx(1.25)
    assert "lineage_conflict" not in event


def _disable_clip_writer(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_module,
        "_write_browser_mp4_from_frames",
        lambda *args, **kwargs: (
            None,
            {
                "evidence_clip_status": "disabled_for_test",
                "evidence_clip_browser_playable": False,
            },
        ),
    )


def test_evidence_session_directory_is_readable_and_unique_for_same_second(tmp_path: Path) -> None:
    first = EvidenceSession(
        source_type="file",
        source="same-source.mp4",
        profile="desktop_rtx",
        root=tmp_path,
    )
    second = EvidenceSession(
        source_type="file",
        source="same-source.mp4",
        profile="desktop_rtx",
        root=tmp_path,
    )

    try:
        assert first.session_dir != second.session_dir
        pattern = re.compile(
            r"^\d{8}_\d{6}_\d{6}_file_same-source_desktop_rtx_[0-9a-f]{6}$"
        )
        assert pattern.fullmatch(first.session_dir.name)
        assert pattern.fullmatch(second.session_dir.name)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("imwrite_result", [False, True])
def test_evidence_write_failure_is_visible_and_not_counted(
    monkeypatch,
    tmp_path: Path,
    caplog,
    imwrite_result: bool,
) -> None:
    monkeypatch.setattr(evidence_module.cv2, "imwrite", lambda *args, **kwargs: imwrite_result)
    caplog.set_level(logging.ERROR, logger=evidence_module.__name__)
    session = EvidenceSession(
        source_type="file",
        source="write-failure.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
    )

    _update_a3b(session, 7, active=True)
    event = session.close()[0]
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))

    assert event["evidence_saved"] is False
    assert event["evidence_saved_frame_count"] == 0
    assert event["evidence_write_attempt_count"] == 1
    assert event["evidence_complete"] is False
    assert event["evidence_partial"] is False
    assert event["evidence_write_failed"] is True
    assert event["evidence_write_error_count"] == 1
    assert event["evidence_representative_path"] == ""
    assert list((Path(event["event_dir"]) / "frames").glob("*.jpg")) == []
    assert manifest["evidence_error_count"] == 1
    assert manifest["recent_errors"][0]["frame_idx"] == 7
    assert "evidence frame write failed" in caplog.text


def test_evidence_prebuffer_writes_prior_frames_once_before_trigger(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_clip_writer(monkeypatch)
    session = EvidenceSession(
        source_type="file",
        source="prebuffer.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=2,
        post_frames=1,
        sample_every=1,
    )

    _update_a3b(session, 1, active=False, value=10)
    _update_a3b(session, 2, active=False, value=20)
    _update_a3b(session, 3, active=True, value=30)
    event = session.close()[0]

    frames_dir = Path(event["evidence_frames_dir"])
    assert [path.name for path in sorted(frames_dir.glob("*.jpg"))] == [
        "frame_000001.jpg",
        "frame_000002.jpg",
        "frame_000003.jpg",
    ]
    assert event["trigger_frame"] == 3
    assert event["evidence_saved_frame_count"] == 3
    assert event["evidence_write_attempt_count"] == 3
    assert event["evidence_write_error_count"] == 0


def test_evidence_partial_write_is_not_reported_as_fully_saved(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_clip_writer(monkeypatch)
    original_imwrite = evidence_module.cv2.imwrite
    write_calls = 0

    def flaky_imwrite(path, frame, params):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return original_imwrite(path, frame, params)
        return False

    monkeypatch.setattr(evidence_module.cv2, "imwrite", flaky_imwrite)
    session = EvidenceSession(
        source_type="file",
        source="partial-write.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=1,
        post_frames=1,
        sample_every=1,
    )

    _update_a3b(session, 1, active=False)
    _update_a3b(session, 2, active=True)
    event = session.close()[0]

    assert event["evidence_saved"] is False
    assert event["evidence_has_saved_frames"] is True
    assert event["evidence_partial"] is True
    assert event["evidence_saved_frame_count"] == 1
    assert event["evidence_write_attempt_count"] == 2
    assert event["evidence_write_error_count"] == 1


def test_evidence_reset_finalizes_active_event_and_clears_prebuffer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_clip_writer(monkeypatch)
    session = EvidenceSession(
        source_type="file",
        source="source-switch.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=2,
        post_frames=1,
        sample_every=1,
    )

    _update_a3b(session, 1, active=False)
    _update_a3b(session, 2, active=True)
    finalized = session.reset(reason="source_epoch_change")
    _update_a3b(session, 10, active=True)
    remaining = session.close()

    assert len(finalized) == 1
    assert finalized[0]["close_reason"] == "source_epoch_change"
    assert finalized[0]["evidence_saved_frame_count"] == 2
    assert len(remaining) == 1
    assert remaining[0]["event_id"] == 2
    assert [
        path.name
        for path in sorted(Path(remaining[0]["evidence_frames_dir"]).glob("*.jpg"))
    ] == ["frame_000010.jpg"]


class _CacheBackend:
    backend = "dummy"
    artifact_path = "dummy://artifact"


class _CachePipeline:
    warmup_frames = 0

    def __init__(self, backend, *, config):
        self.backend = backend
        self.config = config
        self.reset_count = 0
        self.closed = False

    def warmup(self, frames: int) -> None:
        return None

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


def test_pipeline_cache_rebuilds_when_config_content_changes_without_stat_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("first", encoding="utf-8")
    monkeypatch.setattr(pipeline_factory, "configure_runtime_threads", lambda: None)
    monkeypatch.setattr(pipeline_factory, "VideoDefensePipeline", _CachePipeline)
    monkeypatch.setattr(
        pipeline_factory,
        "create_detector_backend",
        lambda config, root: _CacheBackend(),
    )

    def load_config(**kwargs):
        content = Path(kwargs["config_path"]).read_text(encoding="utf-8")
        return {
            "runtime": {},
            "inference": {"backend": "dummy", "model_family": content},
        }

    monkeypatch.setattr(pipeline_factory, "load_runtime_config", load_config)
    cache = pipeline_factory.PipelineCache(config_path=config_path, root=tmp_path)

    first = cache.get(profile="default")
    second = cache.get(profile="default")
    original_stat = config_path.stat()
    config_path.write_text("other", encoding="utf-8")
    os.utime(
        config_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    changed_stat = config_path.stat()
    third = cache.get(profile="default")

    assert first is second
    assert second.cache_hit is True
    assert changed_stat.st_size == original_stat.st_size
    assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns
    assert third is not first
    assert first.pipeline.closed is True
    assert third.cache_hit is False
    assert third.config["inference"]["model_family"] == "other"


def test_pipeline_cache_rebuilds_when_model_content_changes_without_stat_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("stable", encoding="utf-8")
    model_path = tmp_path / "model.engine"
    model_path.write_bytes(b"first!")
    monkeypatch.setattr(pipeline_factory, "configure_runtime_threads", lambda: None)
    monkeypatch.setattr(pipeline_factory, "VideoDefensePipeline", _CachePipeline)
    monkeypatch.setattr(
        pipeline_factory,
        "load_runtime_config",
        lambda **_kwargs: {
            "runtime": {},
            "inference": {"backend": "dummy", "model_family": "dummy"},
        },
    )
    monkeypatch.setattr(
        pipeline_factory,
        "create_detector_backend",
        lambda _config, _root: SimpleNamespace(
            backend="dummy",
            artifact_path=str(model_path),
        ),
    )
    cache = pipeline_factory.PipelineCache(config_path=config_path, root=tmp_path)

    first = cache.get(profile="default")
    second = cache.get(profile="default")
    original_stat = model_path.stat()
    model_path.write_bytes(b"other!")
    os.utime(
        model_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    changed_stat = model_path.stat()
    third = cache.get(profile="default")

    assert first is second
    assert second.cache_hit is True
    assert changed_stat.st_size == original_stat.st_size
    assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns
    assert third is not first
    assert first.pipeline.closed is True
    assert third.cache_hit is False
    assert third.artifact_fingerprint != first.artifact_fingerprint


def test_pipeline_cache_rebuilds_when_a4_classifier_changes_without_stat_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("stable", encoding="utf-8")
    model_path = tmp_path / "model.engine"
    model_path.write_bytes(b"stable")
    a4_path = tmp_path / "rebuilt-a4.pkl"
    a4_path.write_bytes(b"first!")

    class _A4CachePipeline(_CachePipeline):
        def __init__(self, backend, *, config):
            super().__init__(backend, config=config)
            self.detector = SimpleNamespace(
                a4_classifier_resolved_path=str(a4_path),
            )

    monkeypatch.setattr(pipeline_factory, "configure_runtime_threads", lambda: None)
    monkeypatch.setattr(pipeline_factory, "VideoDefensePipeline", _A4CachePipeline)
    monkeypatch.setattr(
        pipeline_factory,
        "load_runtime_config",
        lambda **_kwargs: {
            "runtime": {},
            "inference": {"backend": "dummy", "model_family": "dummy"},
        },
    )
    monkeypatch.setattr(
        pipeline_factory,
        "create_detector_backend",
        lambda _config, _root: SimpleNamespace(
            backend="dummy",
            artifact_path=str(model_path),
        ),
    )
    cache = pipeline_factory.PipelineCache(config_path=config_path, root=tmp_path)

    first = cache.get(profile="default")
    second = cache.get(profile="default")
    original_stat = a4_path.stat()
    a4_path.write_bytes(b"other!")
    os.utime(
        a4_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    changed_stat = a4_path.stat()
    third = cache.get(profile="default")

    assert first is second
    assert second.cache_hit is True
    assert changed_stat.st_size == original_stat.st_size
    assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns
    assert third is not first
    assert first.pipeline.closed is True
    assert third.cache_hit is False
    assert (
        third.auxiliary_artifact_fingerprint
        != first.auxiliary_artifact_fingerprint
    )

    active = cache.get(profile="default")
    assert active is third
    assert active.cache_hit is True


def test_pipeline_cache_auxiliary_artifact_missing_states_are_deterministic(
    tmp_path: Path,
) -> None:
    cache = pipeline_factory.PipelineCache(root=tmp_path)
    pipeline_without_detector = SimpleNamespace()
    pipeline_with_empty_path = SimpleNamespace(
        detector=SimpleNamespace(a4_classifier_resolved_path=""),
    )
    missing_path = tmp_path / "missing-a4.pkl"

    assert cache._pipeline_auxiliary_artifact_path(pipeline_without_detector) == ""
    assert cache._pipeline_auxiliary_artifact_path(pipeline_with_empty_path) == ""
    assert cache._artifact_cache_identity("") == ("empty", "")
    expected_missing = ("path", str(missing_path.resolve()), "missing")
    assert cache._artifact_cache_identity(str(missing_path)) == expected_missing
    assert cache._artifact_cache_identity(str(missing_path)) == expected_missing


def test_pipeline_cache_distinguishes_missing_and_explicit_feature_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("stable", encoding="utf-8")
    monkeypatch.setattr(pipeline_factory, "configure_runtime_threads", lambda: None)
    monkeypatch.setattr(pipeline_factory, "VideoDefensePipeline", _CachePipeline)
    monkeypatch.setattr(
        pipeline_factory,
        "create_detector_backend",
        lambda _config, _root: _CacheBackend(),
    )

    def load_config(**kwargs):
        options = dict(kwargs.get("feature_options") or {})
        return {
            "runtime": {},
            "module_a": {
                "static_image_enabled": options.get(
                    "static_image_enabled",
                    False,
                )
            },
            "inference": {
                "backend": "dummy",
                "model_family": "dummy",
            },
        }

    monkeypatch.setattr(pipeline_factory, "load_runtime_config", load_config)
    cache = pipeline_factory.PipelineCache(config_path=config_path, root=tmp_path)

    inherited = cache.get(
        profile="default",
        feature_options={"a3b_sensitivity": "high"},
    )
    explicit = cache.get(
        profile="default",
        feature_options={
            "a3b_sensitivity": "high",
            "static_image_enabled": True,
        },
    )

    assert inherited is not explicit
    assert inherited.pipeline.closed is True
    assert inherited.config["module_a"]["static_image_enabled"] is False
    assert explicit.config["module_a"]["static_image_enabled"] is True
