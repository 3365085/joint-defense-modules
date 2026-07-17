from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import defense.pipelines.video_decoder_factory as factory_module
from defense.pipelines.derived_video_cache import DerivedVideoResolution
from defense.pipelines.video_decoder import VideoStreamInfo
from defense.pipelines.video_decoder_factory import create_video_decoder


def _write_video(path: Path, *, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG VideoWriter is unavailable")
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), 20 + index * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class _ZeroFrameNvdec:
    opened_paths: list[Path] = []
    instances: list["_ZeroFrameNvdec"] = []

    def __init__(
        self,
        source: str | Path,
        *,
        requested_backend: str = "nvdec",
        **_kwargs: Any,
    ) -> None:
        self.source = Path(source).resolve()
        self.requested_backend = requested_backend
        self.closed = False
        self.eof = False
        self.info = VideoStreamInfo(
            source=str(self.source),
            backend="nvdec",
            codec="h264",
            width=64,
            height=48,
            fps=12.0,
            frame_count=3,
            duration_s=0.25,
            gpu_device="cuda:0",
            output_format="rgbp",
            frame_device="cuda:0",
        )
        type(self).opened_paths.append(self.source)
        type(self).instances.append(self)

    def read(self) -> None:
        self.eof = True
        return None

    def seek_time(self, _seconds: float) -> None:
        self.eof = False

    def seek_frame(self, _frame_idx: int) -> None:
        self.eof = False

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "backend": "nvdec",
            "effective_backend": "nvdec",
            "source": str(self.source),
            "codec": "h264",
            "gpu_device": "cuda:0",
            "output_format": "rgbp",
            "frame_device": "cuda:0",
            "demux_mode": "synthetic_zero_frame",
            "demux_bytes_read": self.source.stat().st_size,
            "frames_decoded": 0,
            "bytes_decoded": 0,
            "fallback_count": 0,
            "fallback_reason": "",
            "fallback_reasons": [],
            "next_frame_idx": 0,
            "eof": self.eof,
            "closed": self.closed,
            "close_error": "",
        }

    def close(self) -> None:
        self.closed = True


def _install_zero_frame_nvdec(monkeypatch: pytest.MonkeyPatch) -> None:
    import defense.pipelines._pynv_file_decoder as pynv_module

    _ZeroFrameNvdec.opened_paths.clear()
    _ZeroFrameNvdec.instances.clear()
    monkeypatch.setattr(pynv_module, "PyNvFileDecoder", _ZeroFrameNvdec)


def test_zero_frame_eof_switches_to_visible_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "zero-frame.avi"
    _write_video(source)
    _install_zero_frame_nvdec(monkeypatch)

    decoder = create_video_decoder(
        source,
        preference="nvdec",
        allow_cpu_fallback=True,
    )
    lease = None
    try:
        before = decoder.status_snapshot()
        assert before["effective_backend"] == "nvdec"
        assert before["fallback_count"] == 0

        lease = decoder.read()
        assert lease is not None
        assert lease.frame_idx == 0
        assert lease.storage == "host"

        status = decoder.status_snapshot()
        assert status["effective_backend"] == "opencv"
        assert status["frames_decoded"] == 1
        assert status["fallback_count"] == 1
        assert status["fallback_reason"].startswith(
            "nvdec_runtime_zero_frame_eof:"
        )
        assert "demux_mode=synthetic_zero_frame" in status["fallback_reason"]
        assert status["runtime_fallback_attempted"] is True
        assert status["runtime_fallback_succeeded"] is True
        assert status["runtime_fallback_error"] == ""
        assert status["nvdec_attempt_frames_decoded"] == 0
        assert status["nvdec_attempt_eof"] is True
        assert status["derived_cache_used"] is False
        assert status["derived_cache_validation"] == "not_used"
        assert _ZeroFrameNvdec.instances[0].closed is True
    finally:
        if lease is not None:
            lease.release()
        decoder.close()


def test_verified_derived_zero_frame_falls_back_to_original_source_with_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.avi"
    derived = tmp_path / "derived.mp4"
    metadata_path = tmp_path / "metadata.json"
    _write_video(source)
    derived.write_bytes(b"synthetic-derived")
    metadata_path.write_text("{}", encoding="utf-8")
    _install_zero_frame_nvdec(monkeypatch)

    source_sha256 = "A" * 64
    derived_sha256 = "B" * 64
    resolution = DerivedVideoResolution(
        source_path=source.resolve(),
        decode_path=derived.resolve(),
        metadata_path=metadata_path.resolve(),
        source_sha256=source_sha256,
        derived_sha256=derived_sha256,
        metadata_sha256="C" * 64,
        metadata={
            "source": {
                "asset_id": "normal.synthetic",
                "role": "normal",
                "label": "normal",
                "attack_type": None,
                "codec": "mjpeg",
            },
            "derived": {
                "codec": "h264",
                "frame_count": 3,
                "duration_s": 0.25,
            },
            "profile": {
                "id": "synthetic_nvdec_profile",
                "sha256": "D" * 64,
            },
            "transcode": {
                "decode_backend": "ffmpeg_software_mjpeg",
                "encode_backend": "h264_nvenc",
            },
            "verification": {
                "decoded_framemd5_match": True,
                "frame_count_match": True,
                "nvdec_frame_count_match": True,
                "r_frame_rate_match": True,
                "avg_frame_rate_match": True,
            },
        },
    )
    monkeypatch.setattr(
        factory_module,
        "resolve_derived_video_source",
        lambda _source: resolution,
    )

    decoder = create_video_decoder(
        source,
        preference="nvdec",
        allow_cpu_fallback=True,
    )
    lease = None
    try:
        assert _ZeroFrameNvdec.opened_paths == [derived.resolve()]
        lease = decoder.read()
        assert lease is not None
        assert lease.storage == "host"

        status = decoder.status_snapshot()
        assert status["effective_backend"] == "opencv"
        assert status["fallback_count"] == 1
        assert status["derived_cache_attempted"] is True
        assert status["derived_cache_runtime_fallback"] is True
        assert status["derived_cache_used"] is False
        assert status["derived_cache_validation"] == "verified"
        assert Path(status["source"]) == source.resolve()
        assert Path(status["decode_source"]) == source.resolve()
        assert Path(status["derived_decode_source"]) == derived.resolve()
        assert status["source_sha256"] == source_sha256
        assert status["decode_source_sha256"] == source_sha256
        assert status["derived_metadata_path"] == str(
            metadata_path.resolve()
        )
        assert status["derived_metadata_sha256"] == "C" * 64
    finally:
        if lease is not None:
            lease.release()
        decoder.close()
