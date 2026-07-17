from __future__ import annotations

import json
from pathlib import Path

import pytest

from defense.pipelines.video_decoder_factory import create_video_decoder
from defense.runtime.pipeline_factory import PipelineCache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "configs"
    / "acceptance"
    / "module_a_authoritative_manifest_v1.json"
)


def _normal_source() -> Path:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["videos"]
        if item["asset_id"] == "normal.fixed_camera_1080"
    )
    return Path(record["canonical_path"])


def test_authoritative_tensorrt_gpu_input_matches_host_bgr_path() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for production GPU-input parity")

    source = _normal_source()
    if not source.is_file():
        pytest.skip(f"authoritative source is unavailable: {source}")

    cache = PipelineCache(
        config_path=PROJECT_ROOT / "configs" / "module_a_runtime.yaml"
    )
    decoder = None
    lease = None
    try:
        bundle = cache.get(profile="desktop_rtx")
        backend = bundle.pipeline.detector_backend
        predict_cuda = getattr(backend, "predict_cuda", None)
        if not callable(predict_cuda):
            pytest.fail("production detector backend has no predict_cuda path")

        decoder = create_video_decoder(
            source,
            preference="nvdec",
            allow_cpu_fallback=False,
        )
        lease = decoder.read()
        assert lease is not None
        assert lease.cuda_tensor is not None
        host_bgr = lease.materialize_host_bgr(size=(640, 640))

        host_result = backend.predict(host_bgr)
        gpu_result = predict_cuda(lease.cuda_tensor, image=host_bgr)

        assert gpu_result.input_device.startswith("cuda")
        assert gpu_result.input_format.startswith("rgbp_")
        assert gpu_result.preprocess_ms >= 0.0
        assert gpu_result.classes == host_result.classes
        assert len(gpu_result.boxes) == len(host_result.boxes)
        assert len(gpu_result.confidences) == len(host_result.confidences)

        for gpu_box, host_box in zip(gpu_result.boxes, host_result.boxes):
            assert max(
                abs(int(gpu_value) - int(host_value))
                for gpu_value, host_value in zip(gpu_box, host_box)
            ) <= 1
        for gpu_confidence, host_confidence in zip(
            gpu_result.confidences,
            host_result.confidences,
        ):
            assert abs(float(gpu_confidence) - float(host_confidence)) <= 0.03
    finally:
        if lease is not None:
            lease.release()
        if decoder is not None:
            decoder.close()
        cache.clear()
