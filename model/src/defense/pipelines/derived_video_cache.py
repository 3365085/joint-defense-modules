from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from defense.pipelines.video_decoder import (
    DecodedFrameLease,
    VideoDecoder,
    VideoDecoderError,
    VideoStreamInfo,
)


class DerivedVideoCacheError(VideoDecoderError):
    """A declared source-bound derived video failed provenance validation."""


@dataclass(frozen=True, slots=True)
class DerivedVideoResolution:
    source_path: Path
    decode_path: Path
    metadata_path: Path
    source_sha256: str
    derived_sha256: str
    metadata_sha256: str
    metadata: dict[str, Any]


_HASH_CACHE_LOCK = threading.Lock()
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def default_derived_video_cache_root() -> Path:
    configured = str(
        os.environ.get("MODULE_A_DERIVED_VIDEO_CACHE_ROOT") or ""
    ).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[3]
        / "runtime"
        / "artifacts"
        / "video_decode"
    )


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    key = (os.path.normcase(str(resolved)), int(stat.st_size), int(stat.st_mtime_ns))
    with _HASH_CACHE_LOCK:
        cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest().upper()
    with _HASH_CACHE_LOCK:
        _HASH_CACHE[key] = value
    return value


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest().upper()


def _same_path(left: Any, right: Path) -> bool:
    try:
        candidate = Path(str(left)).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return os.path.normcase(str(candidate)) == os.path.normcase(str(right))


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DerivedVideoCacheError(
            "derived_cache_metadata_unreadable:"
            f"{type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(data, dict):
        raise DerivedVideoCacheError("derived_cache_metadata_not_object")
    return data


def _candidate_metadata_paths(source_path: Path, cache_root: Path) -> list[Path]:
    if not cache_root.is_dir():
        return []
    source_stat = source_path.stat()
    candidates: list[Path] = []
    for metadata_path in sorted(cache_root.glob("*/metadata.json")):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = data.get("source", {})
            if not isinstance(source, dict):
                continue
            if int(source.get("size_bytes") or -1) != int(source_stat.st_size):
                continue
            if not _same_path(source.get("path"), source_path):
                continue
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            continue
        candidates.append(metadata_path)
    return candidates


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise DerivedVideoCacheError(code)


def resolve_derived_video_source(
    source: str | Path,
    *,
    cache_root: str | Path | None = None,
) -> DerivedVideoResolution | None:
    source_path = Path(source).expanduser().resolve(strict=True)
    root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else default_derived_video_cache_root()
    )
    candidates = _candidate_metadata_paths(source_path, root)
    if not candidates:
        return None

    source_sha256 = sha256_file(source_path)
    metadata_path = root / source_sha256.lower() / "metadata.json"
    _require(
        metadata_path in candidates and metadata_path.is_file(),
        "derived_cache_source_hash_directory_mismatch",
    )
    data = _load_metadata(metadata_path)
    source_meta = data.get("source", {})
    derived_meta = data.get("derived", {})
    transcode = data.get("transcode", {})
    profile = data.get("profile", {})
    toolchain = data.get("toolchain", {})
    verification = data.get("verification", {})
    _require(data.get("schema_version") == 1, "derived_cache_schema_version")
    _require(
        data.get("artifact_type") == "nvdec_derived_video",
        "derived_cache_artifact_type",
    )
    _require(isinstance(source_meta, dict), "derived_cache_source_metadata")
    _require(isinstance(derived_meta, dict), "derived_cache_derived_metadata")
    _require(isinstance(transcode, dict), "derived_cache_transcode_metadata")
    _require(isinstance(profile, dict), "derived_cache_profile_metadata")
    _require(isinstance(toolchain, dict), "derived_cache_toolchain_metadata")
    _require(
        isinstance(verification, dict),
        "derived_cache_verification_metadata",
    )
    _require(
        _same_path(source_meta.get("path"), source_path),
        "derived_cache_source_path_mismatch",
    )
    _require(
        str(source_meta.get("sha256") or "").upper() == source_sha256,
        "derived_cache_source_sha256_mismatch",
    )
    _require(
        int(source_meta.get("size_bytes") or -1) == source_path.stat().st_size,
        "derived_cache_source_size_mismatch",
    )
    _require(
        metadata_path.parent.name.upper() == source_sha256,
        "derived_cache_directory_hash_mismatch",
    )

    relative_path = str(derived_meta.get("relative_path") or "").strip()
    _require(bool(relative_path), "derived_cache_relative_path_missing")
    decode_path = (metadata_path.parent / relative_path).resolve(strict=True)
    try:
        decode_path.relative_to(metadata_path.parent.resolve())
    except ValueError as exc:
        raise DerivedVideoCacheError(
            "derived_cache_path_escapes_source_directory"
        ) from exc
    derived_sha256 = str(derived_meta.get("sha256") or "").upper()
    _require(bool(derived_sha256), "derived_cache_sha256_missing")
    _require(
        int(derived_meta.get("size_bytes") or -1) == decode_path.stat().st_size,
        "derived_cache_size_mismatch",
    )
    _require(
        sha256_file(decode_path) == derived_sha256,
        "derived_cache_sha256_mismatch",
    )
    _require(
        str(derived_meta.get("codec") or "").lower() in {"h264", "hevc"},
        "derived_cache_codec_not_nvdec_production",
    )
    _require(
        str(transcode.get("encode_backend") or "").lower()
        in {"h264_nvenc", "hevc_nvenc"},
        "derived_cache_encoder_not_nvenc",
    )
    _require(
        bool(str(transcode.get("decode_backend") or "").strip()),
        "derived_cache_transcode_decode_backend_missing",
    )
    profile_id = str(profile.get("id") or "").strip()
    profile_sha256 = str(profile.get("sha256") or "").upper()
    profile_payload = profile.get("payload", {})
    _require(bool(profile_id), "derived_cache_profile_id_missing")
    _require(
        isinstance(profile_payload, dict),
        "derived_cache_profile_payload_invalid",
    )
    _require(
        profile_sha256 == canonical_payload_sha256(profile_payload),
        "derived_cache_profile_sha256_mismatch",
    )
    for field in (
        "ffmpeg_path",
        "ffmpeg_version",
        "ffmpeg_sha256",
        "builder_path",
        "builder_sha256",
    ):
        _require(
            bool(str(toolchain.get(field) or "").strip()),
            f"derived_cache_toolchain_{field}_missing",
        )

    parity_keys = (
        "width_match",
        "height_match",
        "r_frame_rate_match",
        "avg_frame_rate_match",
        "frame_count_match",
        "duration_within_one_frame",
        "decoded_framemd5_match",
        "nvdec_frame_count_match",
        "nvdec_effective_backend",
        "nvdec_fallback_zero",
    )
    _require(
        all(verification.get(key) is True for key in parity_keys),
        "derived_cache_verification_failed",
    )
    _require(
        int(source_meta.get("width") or -1)
        == int(derived_meta.get("width") or -2),
        "derived_cache_width_mismatch",
    )
    _require(
        int(source_meta.get("height") or -1)
        == int(derived_meta.get("height") or -2),
        "derived_cache_height_mismatch",
    )
    _require(
        int(source_meta.get("frame_count") or -1)
        == int(derived_meta.get("frame_count") or -2),
        "derived_cache_frame_count_mismatch",
    )
    _require(
        str(source_meta.get("fps") or "")
        == str(derived_meta.get("fps") or ""),
        "derived_cache_fps_mismatch",
    )
    return DerivedVideoResolution(
        source_path=source_path,
        decode_path=decode_path,
        metadata_path=metadata_path,
        source_sha256=source_sha256,
        derived_sha256=derived_sha256,
        metadata_sha256=sha256_file(metadata_path),
        metadata=data,
    )


class DerivedVideoDecoder:
    """Expose original-source lineage while decoding a verified derived file."""

    def __init__(
        self,
        decoder: VideoDecoder,
        resolution: DerivedVideoResolution,
    ) -> None:
        self._decoder = decoder
        self._resolution = resolution

    @property
    def info(self) -> VideoStreamInfo:
        return replace(
            self._decoder.info,
            source=str(self._resolution.source_path),
        )

    def read(self) -> DecodedFrameLease | None:
        return self._decoder.read()

    def seek_time(self, seconds: float) -> None:
        self._decoder.seek_time(seconds)

    def seek_frame(self, frame_idx: int) -> None:
        self._decoder.seek_frame(frame_idx)

    def status_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._decoder.status_snapshot())
        metadata = self._resolution.metadata
        transcode = metadata.get("transcode", {})
        verification = metadata.get("verification", {})
        source_meta = metadata.get("source", {})
        derived_meta = metadata.get("derived", {})
        profile = metadata.get("profile", {})
        return {
            **snapshot,
            "source": str(self._resolution.source_path),
            "decode_source": str(self._resolution.decode_path),
            "derived_cache_used": True,
            "derived_cache_validation": "verified",
            "source_sha256": self._resolution.source_sha256,
            "decode_source_sha256": self._resolution.derived_sha256,
            "derived_metadata_path": str(self._resolution.metadata_path),
            "derived_metadata_sha256": self._resolution.metadata_sha256,
            "source_asset_id": str(
                source_meta.get("asset_id") or "unknown"
            ),
            "source_role": str(source_meta.get("role") or "unknown"),
            "source_label": str(source_meta.get("label") or "unknown"),
            "source_attack_type": source_meta.get("attack_type"),
            "source_codec": str(source_meta.get("codec") or "unknown"),
            "derived_codec": str(derived_meta.get("codec") or "unknown"),
            "derived_profile_id": str(profile.get("id") or "unknown"),
            "derived_profile_sha256": str(
                profile.get("sha256") or ""
            ).upper(),
            "derived_expected_frame_count": int(
                derived_meta.get("frame_count") or 0
            ),
            "derived_expected_duration_s": float(
                derived_meta.get("duration_s") or 0.0
            ),
            "transcode_decode_backend": str(
                transcode.get("decode_backend") or "unknown"
            ),
            "transcode_encode_backend": str(
                transcode.get("encode_backend") or "unknown"
            ),
            "derived_frame_parity": bool(
                verification.get("decoded_framemd5_match")
            ),
            "derived_frame_count_match": bool(
                verification.get("frame_count_match")
                and verification.get("nvdec_frame_count_match")
            ),
            "derived_fps_match": bool(
                verification.get("r_frame_rate_match")
                and verification.get("avg_frame_rate_match")
            ),
        }

    def request_cancel(self) -> None:
        callback = getattr(self._decoder, "request_cancel", None)
        if callable(callback):
            callback()

    def close(self) -> None:
        self._decoder.close()
