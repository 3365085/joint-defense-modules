from __future__ import annotations

from pathlib import Path

import pytest

from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.runtime import pipeline_factory


class WarmupFailingPipeline:
    warmup_frames = 1

    def __init__(self, backend, *, config):
        self.backend = backend
        self.config = config
        self.reset_count = 0

    def warmup(self, frames: int) -> None:
        raise RuntimeError("warmup exploded")

    def reset(self) -> None:
        self.reset_count += 1


class WarmupPipeline:
    warmup_frames = 2

    def __init__(self, backend, *, config):
        self.backend = backend
        self.config = config
        self.warmup_calls = []
        self.reset_count = 0

    def warmup(self, frames: int) -> None:
        self.warmup_calls.append(frames)

    def reset(self) -> None:
        self.reset_count += 1


class DummyBackend:
    backend = "dummy"
    artifact_path = "dummy://artifact"
    names = {}

    def predict(self, image):
        return DetectionFrameResult(
            image=image,
            boxes=[],
            classes=[],
            confidences=[],
            names=self.names,
            backend=self.backend,
            artifact_path=self.artifact_path,
            inference_ms=0.0,
            raw_result=None,
        )


def test_pipeline_cache_exposes_warmup_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_factory, "VideoDefensePipeline", WarmupFailingPipeline)
    monkeypatch.setattr(pipeline_factory, "create_detector_backend", lambda config, root: DummyBackend())
    monkeypatch.setattr(
        pipeline_factory,
        "load_runtime_config",
        lambda **kwargs: {"runtime": {}, "inference": {"backend": "dummy"}},
    )

    bundle = pipeline_factory.PipelineCache(root=tmp_path).get(profile="default")

    assert bundle.backend == "dummy"
    assert bundle.warmup_error == "RuntimeError: warmup exploded"
    assert bundle.pipeline.reset_count == 1


def test_pipeline_cache_exposes_cache_and_init_timings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_factory, "VideoDefensePipeline", WarmupPipeline)
    monkeypatch.setattr(pipeline_factory, "create_detector_backend", lambda config, root: DummyBackend())
    monkeypatch.setattr(
        pipeline_factory,
        "load_runtime_config",
        lambda **kwargs: {"runtime": {}, "inference": {"backend": "dummy"}},
    )

    cache = pipeline_factory.PipelineCache(root=tmp_path)
    first = cache.get(profile="default")
    first_cache_hit = first.cache_hit
    first_warmup_frames = first.warmup_frames
    first_config_load_ms = first.config_load_ms
    first_backend_create_ms = first.backend_create_ms
    first_pipeline_construct_ms = first.pipeline_construct_ms
    first_warmup_ms = first.warmup_ms
    second = cache.get(profile="default")

    assert first is second
    assert first_cache_hit is False
    assert second.cache_hit is True
    assert first_warmup_frames == 2
    assert second.warmup_frames == 0
    assert second.pipeline.warmup_calls == [2]
    assert second.pipeline.reset_count == 2
    assert first_config_load_ms >= 0.0
    assert first_backend_create_ms >= 0.0
    assert first_pipeline_construct_ms >= 0.0
    assert first_warmup_ms >= 0.0
    assert second.cache_get_ms >= 0.0
    assert second.pipeline_reset_ms >= 0.0


def test_pipeline_cache_closes_stale_pipeline_on_key_miss(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_factory, "VideoDefensePipeline", WarmupPipeline)
    monkeypatch.setattr(pipeline_factory, "create_detector_backend", lambda config, root: DummyBackend())

    def load_config(**kwargs):
        return {"runtime": {}, "inference": {"backend": "dummy", "model_family": kwargs["profile"]}}

    monkeypatch.setattr(pipeline_factory, "load_runtime_config", load_config)

    cache = pipeline_factory.PipelineCache(root=tmp_path)
    first = cache.get(profile="default")
    first.pipeline.closed = False
    first.pipeline.close = lambda: setattr(first.pipeline, "closed", True)
    second = cache.get(profile="desktop_rtx")

    assert first is not second
    assert first.pipeline.closed is True
    assert second.cache_hit is False


def test_empty_backend_is_rejected_outside_empty_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        pipeline_factory,
        "load_runtime_config",
        lambda **kwargs: {"runtime": {"allow_empty_backend": True}, "inference": {"backend": "onnx"}},
    )

    try:
        pipeline_factory.PipelineCache(root=tmp_path).get(profile="desktop_rtx")
    except RuntimeError as exc:
        assert "allow_empty_backend" in str(exc)
        assert "empty_smoke" in str(exc)
    else:
        raise AssertionError("empty backend should be rejected outside empty_smoke")


def test_pipeline_constructor_failure_closes_created_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = DummyBackend()
    backend.closed = False
    backend.close = lambda: setattr(backend, "closed", True)

    class ConstructorFailingPipeline:
        def __init__(self, _backend, *, config):
            del config
            raise RuntimeError("pipeline construction exploded")

    monkeypatch.setattr(
        pipeline_factory,
        "VideoDefensePipeline",
        ConstructorFailingPipeline,
    )
    monkeypatch.setattr(
        pipeline_factory,
        "create_detector_backend",
        lambda config, root: backend,
    )
    monkeypatch.setattr(
        pipeline_factory,
        "load_runtime_config",
        lambda **kwargs: {
            "runtime": {},
            "inference": {"backend": "dummy"},
        },
    )

    with pytest.raises(RuntimeError, match="pipeline construction exploded"):
        pipeline_factory.PipelineCache(root=tmp_path).get(profile="default")

    assert backend.closed is True
