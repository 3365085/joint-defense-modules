from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from defense.pipelines._opencv_file_decoder import OpenCVFileDecoder
from defense.pipelines._pynv_file_decoder import PyNvFileDecoder
from defense.pipelines.video_decoder import VideoDecoder, VideoDecoderUnavailable
from defense.pipelines.video_decoder_factory import (
    create_video_decoder,
    normalize_decoder_preference,
)


def _write_test_video(path: Path, *, frame_count: int = 7) -> list[np.ndarray]:
    width, height = 96, 64
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG VideoWriter is unavailable")
    frames: list[np.ndarray] = []
    try:
        for index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 20 + index * 7
            frame[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
            frame[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
            cv2.putText(
                frame,
                str(index),
                (8, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 230, 220),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frames.append(frame)
    finally:
        writer.release()
    return frames


def test_opencv_decoder_contract_seek_eof_and_metrics(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path)
    decoder = OpenCVFileDecoder(video_path)
    assert isinstance(decoder, VideoDecoder)
    assert decoder.info.backend == "opencv"
    assert decoder.info.frame_device == "host"

    first = decoder.read()
    assert first is not None
    first_snapshot = first.host_array.copy()
    assert first.frame_idx == 0
    assert first.pts_s >= 0.0
    assert first.storage == "host"
    second = decoder.read()
    assert second is not None
    assert np.array_equal(first.host_array, first_snapshot)

    resized = first.materialize_host_bgr(
        size=(32, 24),
        roi=(4, 3, 80, 55),
    )
    assert resized.shape == (24, 32, 3)
    assert resized is first.materialize_host_bgr(
        size=(32, 24),
        roi=(4, 3, 80, 55),
    )

    decoder.seek_frame(1)
    sought = decoder.read()
    assert sought is not None
    assert sought.frame_idx == 1
    assert sought.pts_s >= 0.0

    decoder.seek_time(-5.0)
    rewound = decoder.read()
    assert rewound is not None
    assert rewound.frame_idx == 0
    assert rewound.pts_s >= 0.0

    decoder.seek_frame(decoder.info.frame_count)
    assert decoder.read() is None
    status = decoder.status_snapshot()
    assert status["backend"] == "opencv"
    assert status["frames_decoded"] == 4
    assert status["bytes_decoded"] > 0
    assert status["decode_ms_p50"] >= 0.0
    assert status["decode_ms_p95"] >= status["decode_ms_p50"]
    assert status["seek_count"] == 3

    for lease in (first, second, sought, rewound):
        lease.release()
    decoder.close()
    assert decoder.status_snapshot()["closed"] is True


def test_factory_auto_fallback_is_explicit_and_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "fallback.avi"
    _write_test_video(video_path, frame_count=3)

    import defense.pipelines._pynv_file_decoder as pynv_module

    class FailingNvdec:
        def __init__(self, *_args, **_kwargs) -> None:
            raise VideoDecoderUnavailable("synthetic_nvdec_failure")

    monkeypatch.setattr(pynv_module, "PyNvFileDecoder", FailingNvdec)
    decoder = create_video_decoder(
        video_path,
        preference="auto",
        allow_cpu_fallback=True,
    )
    status = decoder.status_snapshot()
    assert status["requested_backend"] == "auto"
    assert status["effective_backend"] == "opencv"
    assert status["fallback_count"] == 1
    assert "synthetic_nvdec_failure" in status["fallback_reason"]
    decoder.close()


def test_factory_can_forbid_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "no_fallback.avi"
    _write_test_video(video_path, frame_count=2)

    import defense.pipelines._pynv_file_decoder as pynv_module

    class FailingNvdec:
        def __init__(self, *_args, **_kwargs) -> None:
            raise VideoDecoderUnavailable("nvdec_required")

    monkeypatch.setattr(pynv_module, "PyNvFileDecoder", FailingNvdec)
    with pytest.raises(VideoDecoderUnavailable, match="nvdec_required"):
        create_video_decoder(
            video_path,
            preference="nvdec",
            allow_cpu_fallback=False,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", "auto"),
        ("NVDEC", "nvdec"),
        ("pynv", "nvdec"),
        ("opencv", "opencv"),
        ("cpu", "opencv"),
    ],
)
def test_decoder_preference_normalization(raw: str, expected: str) -> None:
    assert normalize_decoder_preference(raw) == expected


def test_decoder_preference_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="expected=auto\\|nvdec\\|opencv"):
        normalize_decoder_preference("mystery")


@pytest.mark.parametrize(
    ("asset_id", "expected_codec"),
    [
        ("normal.fixed_camera_1080", "h264"),
        ("a3b.authoritative_target", "hevc"),
    ],
)
def test_nvdec_authoritative_h264_and_hevc_smoke(
    pkg_root: Path,
    asset_id: str,
    expected_codec: str,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("PyNvVideoCodec")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    manifest = json.loads(
        (
            pkg_root
            / "configs"
            / "acceptance"
            / "module_a_authoritative_manifest_v1.json"
        ).read_text(encoding="utf-8")
    )
    source = Path(
        next(
            row["canonical_path"]
            for row in manifest["videos"]
            if row["asset_id"] == asset_id
        )
    )
    if not source.is_file():
        pytest.skip(f"authoritative source is unavailable: {source}")

    decoder = PyNvFileDecoder(source)
    leases = []
    try:
        assert decoder.info.codec == expected_codec
        assert decoder.info.output_format == "rgbp"
        for expected_idx in range(2):
            lease = decoder.read()
            assert lease is not None
            leases.append(lease)
            assert lease.frame_idx == expected_idx
            assert lease.pts_s >= 0.0
            assert lease.cuda_tensor.is_cuda
            assert lease.metadata["surface_cloned"] is True
        status = decoder.status_snapshot()
        assert status["source_alias_is_ascii"] is True
        assert "project_runtime_junction" in status["source_alias_mode"]
        assert status["fallback_count"] == 0
    finally:
        for lease in leases:
            lease.release()
        decoder.close()
