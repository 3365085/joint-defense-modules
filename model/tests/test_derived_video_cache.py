from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import defense.pipelines._pynv_file_decoder as pynv_module
from defense.pipelines.derived_video_cache import (
    canonical_payload_sha256,
    DerivedVideoCacheError,
    DerivedVideoDecoder,
    resolve_derived_video_source,
    sha256_file,
)
from defense.pipelines.video_decoder import (
    VideoDecoderUnavailable,
    VideoStreamInfo,
)
from defense.pipelines.video_decoder_factory import create_video_decoder


def _write_cache_metadata(
    *,
    cache_root: Path,
    source: Path,
    derived: Path,
    derived_sha256: str | None = None,
) -> Path:
    source_sha256 = sha256_file(source)
    cache_dir = cache_root / source_sha256.lower()
    cache_dir.mkdir(parents=True)
    cached_derived = cache_dir / derived.name
    cached_derived.write_bytes(derived.read_bytes())
    profile_payload = {
        "container": "mp4",
        "video_codec": "h264",
        "pixel_format": "yuv420p",
        "encoder": "h264_nvenc",
        "preset": "p4",
        "tune": "lossless",
    }
    metadata = {
        "schema_version": 1,
        "artifact_type": "nvdec_derived_video",
        "source": {
            "asset_id": "attack.synthetic",
            "path": str(source.resolve()),
            "sha256": source_sha256,
            "size_bytes": source.stat().st_size,
            "role": "physical_attack",
            "label": "attack",
            "attack_type": "synthetic",
            "codec": "mpeg4",
            "width": 1920,
            "height": 1080,
            "fps": "2997/50",
            "frame_count": 120,
            "duration_s": 2.002,
        },
        "derived": {
            "path": str(cached_derived.resolve()),
            "relative_path": cached_derived.name,
            "sha256": derived_sha256 or sha256_file(cached_derived),
            "size_bytes": cached_derived.stat().st_size,
            "codec": "h264",
            "profile": "High 4:4:4 Predictive",
            "pixel_format": "yuv420p",
            "width": 1920,
            "height": 1080,
            "fps": "2997/50",
            "frame_count": 120,
            "duration_s": 2.002,
        },
        "profile": {
            "id": "h264_nvenc_lossless_yuv420p_v1",
            "sha256": canonical_payload_sha256(profile_payload),
            "payload": profile_payload,
        },
        "transcode": {
            "tool": "ffmpeg",
            "decode_backend": "ffmpeg_software_mpeg4",
            "encode_backend": "h264_nvenc",
            "preset": "p4",
            "tune": "lossless",
        },
        "toolchain": {
            "ffmpeg_path": "C:/tools/ffmpeg.exe",
            "ffmpeg_version": "ffmpeg synthetic",
            "ffmpeg_sha256": "1" * 64,
            "builder_path": "D:/project/tools/build_nvdec_derived_video.py",
            "builder_sha256": "2" * 64,
        },
        "verification": {
            "width_match": True,
            "height_match": True,
            "r_frame_rate_match": True,
            "avg_frame_rate_match": True,
            "frame_count_match": True,
            "duration_within_one_frame": True,
            "decoded_framemd5_match": True,
            "nvdec_frame_count_match": True,
            "nvdec_effective_backend": True,
            "nvdec_fallback_zero": True,
        },
    }
    metadata_path = cache_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def test_resolver_returns_none_when_source_has_no_declared_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "direct.mp4"
    source.write_bytes(b"direct-source")

    assert (
        resolve_derived_video_source(
            source,
            cache_root=tmp_path / "cache",
        )
        is None
    )


def test_resolver_rejects_declared_cache_with_wrong_derived_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    derived = tmp_path / "derived.mp4"
    source.write_bytes(b"source-video")
    derived.write_bytes(b"derived-video")
    cache_root = tmp_path / "cache"
    _write_cache_metadata(
        cache_root=cache_root,
        source=source,
        derived=derived,
        derived_sha256="0" * 64,
    )

    with pytest.raises(
        DerivedVideoCacheError,
        match="derived_cache_sha256_mismatch",
    ):
        resolve_derived_video_source(source, cache_root=cache_root)


def test_factory_decodes_verified_derived_file_but_preserves_source_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    derived = tmp_path / "derived.mp4"
    source.write_bytes(b"source-video")
    derived.write_bytes(b"derived-video")
    cache_root = tmp_path / "cache"
    metadata_path = _write_cache_metadata(
        cache_root=cache_root,
        source=source,
        derived=derived,
    )
    cached_derived = metadata_path.parent / derived.name
    opened_paths: list[Path] = []

    class FakeNvdec:
        def __init__(
            self,
            path: str | Path,
            *,
            gpu_id: int,
            requested_backend: str,
            **_kwargs: Any,
        ) -> None:
            opened_paths.append(Path(path).resolve())
            self.info = VideoStreamInfo(
                source=str(Path(path).resolve()),
                backend="nvdec",
                codec="h264",
                width=1920,
                height=1080,
                fps=2997 / 50,
                frame_count=120,
                duration_s=2.002,
                gpu_device=f"cuda:{gpu_id}",
                output_format="rgbp",
                frame_device=f"cuda:{gpu_id}",
            )
            self.requested_backend = requested_backend
            self.closed = False
            self.cancelled = False

        def read(self) -> None:
            return None

        def seek_time(self, _seconds: float) -> None:
            return None

        def seek_frame(self, _frame_idx: int) -> None:
            return None

        def request_cancel(self) -> None:
            self.cancelled = True

        def status_snapshot(self) -> dict[str, Any]:
            return {
                "requested_backend": self.requested_backend,
                "backend": "nvdec",
                "effective_backend": "nvdec",
                "source": self.info.source,
                "codec": "h264",
                "gpu_device": self.info.gpu_device,
                "output_format": "rgbp",
                "frame_device": self.info.frame_device,
                "fallback_count": 0,
                "fallback_reason": "",
                "closed": self.closed,
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(pynv_module, "PyNvFileDecoder", FakeNvdec)
    monkeypatch.setenv(
        "MODULE_A_DERIVED_VIDEO_CACHE_ROOT",
        str(cache_root),
    )

    decoder = create_video_decoder(
        source,
        preference="nvdec",
        allow_cpu_fallback=False,
    )

    assert isinstance(decoder, DerivedVideoDecoder)
    assert opened_paths == [cached_derived.resolve()]
    assert Path(decoder.info.source) == source.resolve()
    status = decoder.status_snapshot()
    assert Path(status["source"]) == source.resolve()
    assert Path(status["decode_source"]) == cached_derived.resolve()
    assert status["derived_cache_used"] is True
    assert status["derived_cache_validation"] == "verified"
    assert status["source_sha256"] == sha256_file(source)
    assert status["decode_source_sha256"] == sha256_file(cached_derived)
    assert status["derived_metadata_sha256"] == sha256_file(metadata_path)
    assert status["source_asset_id"] == "attack.synthetic"
    assert status["source_role"] == "physical_attack"
    assert status["source_label"] == "attack"
    assert status["source_attack_type"] == "synthetic"
    assert status["source_codec"] == "mpeg4"
    assert status["derived_codec"] == "h264"
    assert status["derived_profile_id"] == (
        "h264_nvenc_lossless_yuv420p_v1"
    )
    assert len(status["derived_profile_sha256"]) == 64
    assert status["derived_expected_frame_count"] == 120
    assert status["derived_expected_duration_s"] == pytest.approx(2.002)
    assert status["transcode_decode_backend"] == "ffmpeg_software_mpeg4"
    assert status["transcode_encode_backend"] == "h264_nvenc"
    assert status["derived_frame_parity"] is True
    assert status["derived_frame_count_match"] is True
    assert status["derived_fps_match"] is True
    decoder.close()
    assert decoder.status_snapshot()["closed"] is True


def test_declared_derived_cache_never_falls_back_to_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    derived = tmp_path / "derived.mp4"
    source.write_bytes(b"source-video")
    derived.write_bytes(b"derived-video")
    cache_root = tmp_path / "cache"
    _write_cache_metadata(
        cache_root=cache_root,
        source=source,
        derived=derived,
        derived_sha256="0" * 64,
    )
    monkeypatch.setenv(
        "MODULE_A_DERIVED_VIDEO_CACHE_ROOT",
        str(cache_root),
    )

    with pytest.raises(
        VideoDecoderUnavailable,
        match="derived_cache_required:derived_cache_sha256_mismatch",
    ):
        create_video_decoder(
            source,
            preference="nvdec",
            allow_cpu_fallback=True,
        )
