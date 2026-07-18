from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from defense.pipelines._decoder_metrics import DecoderMetrics
from defense.pipelines.video_decoder import (
    DecodedFrameLease,
    VideoDecoderError,
    VideoDecoderUnavailable,
    VideoStreamInfo,
)
from defense.runtime_paths import runtime_data_root


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_video_decode_alias_root() -> Path:
    return runtime_data_root() / "video_decode_alias"


def _source_identity(path: Path) -> tuple[str, dict[str, int]]:
    stat = path.stat()
    payload = "|".join(
        (
            str(path.resolve(strict=True)).casefold(),
            str(int(stat.st_dev)),
            str(int(stat.st_ino)),
            str(int(stat.st_size)),
            str(int(stat.st_mtime_ns)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), {
        "source_device_id": int(stat.st_dev),
        "source_file_id": int(stat.st_ino),
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_volume(left: Path, right: Path) -> bool:
    left_drive = os.path.splitdrive(str(left.resolve(strict=False)))[0].casefold()
    right_drive = os.path.splitdrive(str(right.resolve(strict=False)))[0].casefold()
    return bool(left_drive and left_drive == right_drive)


def _create_identity_bound_alias(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return "hardlink_reuse"
        except OSError:
            pass
        if (
            source.stat().st_size == destination.stat().st_size
            and _sha256(source) == _sha256(destination)
        ):
            return "copy_reuse"
        destination.unlink()
    try:
        os.link(source, destination)
        if not os.path.samefile(source, destination):
            raise OSError("hardlink_samefile_verification_failed")
        return "hardlink"
    except OSError as hardlink_error:
        try:
            shutil.copy2(source, destination)
            if source.stat().st_size != destination.stat().st_size:
                raise OSError("copy_size_verification_failed")
            if _sha256(source) != _sha256(destination):
                raise OSError("copy_sha256_verification_failed")
            return f"copy_after_hardlink_failure:{type(hardlink_error).__name__}"
        except Exception:
            destination.unlink(missing_ok=True)
            raise


def _ascii_access_root(physical_root: Path) -> Path:
    resolved = physical_root.resolve(strict=False)
    drive, _tail = os.path.splitdrive(str(resolved))
    if not drive:
        raise OSError(f"alias_drive_unavailable:{physical_root}")
    target_id = hashlib.sha256(str(resolved).casefold().encode("utf-8")).hexdigest()[:16]
    return Path(f"{drive}\\module_a_runtime_alias\\{target_id}")


def _ensure_windows_junction(physical_root: Path) -> Path:
    access_root = _ascii_access_root(physical_root)
    access_root.parent.mkdir(parents=True, exist_ok=True)
    if access_root.exists():
        if os.path.samefile(access_root, physical_root):
            return access_root
        raise OSError(f"ascii_alias_root_collision:{access_root}")
    if os.name != "nt":
        raise OSError("ascii_junction_requires_windows")
    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(access_root),
            str(physical_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if result.returncode != 0 or not access_root.exists():
        detail = (result.stderr or result.stdout or "").strip()
        raise OSError(
            f"ascii_junction_create_failed:exit={result.returncode}:{detail}"
        )
    if not os.path.samefile(access_root, physical_root):
        raise OSError(f"ascii_junction_identity_mismatch:{access_root}")
    return access_root


@dataclass(slots=True)
class _AsciiSourceAlias:
    source_path: Path
    decoder_path: Path
    storage_paths: tuple[Path, ...]
    identity: str
    identity_metadata: dict[str, int]
    mode: str
    cleanup_enabled: bool
    created: bool
    cleaned: bool = False
    cleanup_error: str = ""
    _samefile_verified: bool = field(default=False, repr=False)

    @classmethod
    def acquire(
        cls,
        source: str | Path,
        *,
        alias_root: str | Path | None = None,
        cleanup: bool = True,
    ) -> "_AsciiSourceAlias":
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        identity, identity_metadata = _source_identity(source_path)
        if str(source_path).isascii():
            return cls(
                source_path=source_path,
                decoder_path=source_path,
                storage_paths=(),
                identity=identity,
                identity_metadata=identity_metadata,
                mode="direct_ascii",
                cleanup_enabled=False,
                created=False,
                cleaned=True,
                _samefile_verified=True,
            )

        physical_root = (
            Path(alias_root).expanduser().resolve(strict=False)
            if alias_root is not None
            else default_video_decode_alias_root()
        )
        if alias_root is None and not _same_volume(source_path, physical_root):
            physical_root = (
                _ascii_access_root(source_path.parent)
                / "runtime"
                / "video_decode_alias"
            )
        physical_root.mkdir(parents=True, exist_ok=True)
        suffix = source_path.suffix.lower()
        if not suffix or not suffix[1:].isalnum():
            suffix = ".bin"
        filename = f"src_{identity[:32]}{suffix}"
        storage_path = physical_root / filename
        mode = _create_identity_bound_alias(source_path, storage_path)
        storage_paths: list[Path] = [storage_path]
        decoder_path = storage_path

        if not str(storage_path).isascii():
            try:
                access_root = _ensure_windows_junction(physical_root)
                decoder_path = access_root / filename
                mode = f"{mode}+project_runtime_junction"
            except OSError as junction_error:
                access_root = _ascii_access_root(physical_root) / "files"
                access_root.mkdir(parents=True, exist_ok=True)
                decoder_path = access_root / filename
                secondary_mode = _create_identity_bound_alias(
                    source_path,
                    decoder_path,
                )
                storage_paths.append(decoder_path)
                mode = (
                    f"{mode}+drive_ascii_fallback:{secondary_mode}:"
                    f"{type(junction_error).__name__}"
                )

        if not str(decoder_path).isascii():
            for path in reversed(storage_paths):
                path.unlink(missing_ok=True)
            raise OSError(f"decoder_alias_not_ascii:{decoder_path}")
        samefile_verified = os.path.samefile(source_path, decoder_path)
        if not samefile_verified and "copy_" not in mode:
            for path in reversed(storage_paths):
                path.unlink(missing_ok=True)
            raise OSError(f"decoder_alias_identity_mismatch:{decoder_path}")
        return cls(
            source_path=source_path,
            decoder_path=decoder_path,
            storage_paths=tuple(storage_paths),
            identity=identity,
            identity_metadata=identity_metadata,
            mode=mode,
            cleanup_enabled=bool(cleanup),
            created=True,
            _samefile_verified=samefile_verified,
        )

    def cleanup(self) -> bool:
        if self.cleaned:
            return True
        if not self.cleanup_enabled:
            return False
        errors: list[str] = []
        for path in reversed(self.storage_paths):
            for attempt in range(4):
                try:
                    path.unlink(missing_ok=True)
                    break
                except OSError as exc:
                    if attempt >= 3:
                        errors.append(f"{path}:{type(exc).__name__}:{exc}")
                    else:
                        gc.collect()
                        time.sleep(0.05 * (attempt + 1))
        self.cleanup_error = "; ".join(errors)
        self.cleaned = not errors and all(not path.exists() for path in self.storage_paths)
        return self.cleaned

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "source_identity": self.identity,
            **self.identity_metadata,
            "source_alias_mode": self.mode,
            "source_alias_created": bool(self.created),
            "source_alias_path": str(self.decoder_path),
            "source_alias_storage_paths": [str(path) for path in self.storage_paths],
            "source_alias_is_ascii": str(self.decoder_path).isascii(),
            "source_alias_samefile_verified": bool(self._samefile_verified),
            "source_alias_cleanup_enabled": bool(self.cleanup_enabled),
            "source_alias_cleaned": bool(self.cleaned),
            "source_alias_cleanup_deferred": bool(
                self.cleanup_enabled and self.cleanup_error
            ),
            "source_alias_cleanup_error": self.cleanup_error,
        }


def _validate_size(size: tuple[int, int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if len(size) != 2:
        raise VideoDecoderError(f"invalid_size:{size!r}")
    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise VideoDecoderError(f"invalid_size:{size!r}")
    return width, height


def _clamp_roi(
    roi: tuple[int, int, int, int] | None,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    if len(roi) != 4:
        raise VideoDecoderError(f"invalid_roi:{roi!r}")
    x1, y1, x2, y2 = (int(value) for value in roi)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise VideoDecoderError(f"empty_roi:{roi!r}")
    return x1, y1, x2, y2


class _FileChunkFeeder:
    """Own the only file handle used by callback-based PyNv demuxing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("rb")
        self.bytes_read = 0
        self.closed = False

    def feed_chunk(self, demuxer_buffer: Any) -> int:
        if self.closed or self._handle is None:
            return 0
        data = self._handle.read(len(demuxer_buffer))
        if not data:
            return 0
        demuxer_buffer[: len(data)] = data
        self.bytes_read += len(data)
        return len(data)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.close()


def _probe_file_metadata(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {}
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps < 0.0:
            fps = 0.0
        frame_count = max(
            0,
            int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)),
        )
        return {
            "width": max(
                0,
                int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)),
            ),
            "height": max(
                0,
                int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)),
            ),
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": frame_count / fps if frame_count and fps else 0.0,
        }
    finally:
        capture.release()


def _codec_name(codec: Any) -> str:
    text = str(codec or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower() or "unknown"


class PyNvFileDecoder:
    """NVDEC file decoder returning owned RGBP CUDA tensors."""

    def __init__(
        self,
        source: str | Path,
        *,
        gpu_id: int = 0,
        requested_backend: str = "nvdec",
        alias_root: str | Path | None = None,
        cleanup_alias: bool = True,
        metrics_max_samples: int = 2048,
        decoder_cache_size: int = 4,
    ) -> None:
        del decoder_cache_size  # Low-level decoder has no persistent file cache.
        self._requested_backend = str(requested_backend or "nvdec")
        self._gpu_id = max(0, int(gpu_id))
        self._metrics = DecoderMetrics(max_samples=metrics_max_samples)
        self._closed = False
        self._cancel_requested = threading.Event()
        self._eof = False
        self._stream_eof = False
        self._next_frame_idx = 0
        self._close_error = ""
        self._decoder: Any = None
        self._demuxer: Any = None
        self._feeder: _FileChunkFeeder | None = None
        self._surface_queue: deque[tuple[Any, float]] = deque()
        self._demux_bytes_read = 0
        self._demux_mode = "filename_ascii_alias"

        try:
            import PyNvVideoCodec as nvc
            import torch
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.v2 import functional as vision_functional
        except Exception as exc:
            raise VideoDecoderUnavailable(
                f"pynv_import_failed:{type(exc).__name__}:{exc}"
            ) from exc

        if not torch.cuda.is_available():
            raise VideoDecoderUnavailable("pynv_cuda_unavailable")
        if self._gpu_id >= torch.cuda.device_count():
            raise VideoDecoderUnavailable(
                f"pynv_gpu_id_out_of_range:{self._gpu_id}:"
                f"device_count={torch.cuda.device_count()}"
            )

        self._torch = torch
        self._resize_interpolation = InterpolationMode.BILINEAR
        self._vision_functional = vision_functional
        self._nvc = nvc
        self._pynv_version = str(getattr(nvc, "__version__", "unknown"))
        try:
            alias = _AsciiSourceAlias.acquire(
                source,
                alias_root=alias_root,
                cleanup=cleanup_alias,
            )
        except Exception as exc:
            raise VideoDecoderUnavailable(
                f"pynv_ascii_alias_failed:{type(exc).__name__}:{exc}"
            ) from exc
        self._alias = alias

        try:
            file_metadata = _probe_file_metadata(alias.source_path)
            # The callback-based PyNv demuxer can consume an MP4 with an edit
            # list as one opaque packet and then report EOF without yielding a
            # decodable surface.  The source alias is already ASCII-safe, so
            # use PyNv's filename demuxer which preserves the container index
            # and key-frame lookup semantics.
            demuxer = nvc.CreateDemuxer(str(alias.decoder_path))
            codec_id = demuxer.GetNvCodecId()
            decoder = nvc.CreateDecoder(
                gpuid=self._gpu_id,
                codec=codec_id,
                usedevicememory=True,
                outputColorType=nvc.OutputColorType.RGBP,
                enableDecodeStats=False,
            )
            width = max(
                0,
                int(demuxer.Width() or file_metadata.get("width", 0) or 0),
            )
            height = max(
                0,
                int(demuxer.Height() or file_metadata.get("height", 0) or 0),
            )
            fps = float(
                demuxer.FrameRate() or file_metadata.get("fps", 0.0) or 0.0
            )
            if not math.isfinite(fps) or fps < 0.0:
                fps = 0.0
            frame_count = max(0, int(file_metadata.get("frame_count", 0) or 0))
            duration_s = max(
                0.0,
                float(file_metadata.get("duration_s", 0.0) or 0.0),
            )
            codec = _codec_name(codec_id)
        except Exception as exc:
            alias.cleanup()
            raise VideoDecoderUnavailable(
                f"pynv_open_failed:{type(exc).__name__}:{exc}"
            ) from exc

        self._feeder = None
        self._demuxer = demuxer
        self._decoder = decoder
        self._info = VideoStreamInfo(
            source=str(alias.source_path),
            backend="nvdec",
            codec=codec,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration_s,
            gpu_device=f"cuda:{self._gpu_id}",
            output_format="rgbp",
            frame_device=f"cuda:{self._gpu_id}",
        )

    @property
    def info(self) -> VideoStreamInfo:
        return self._info

    def _require_decoder(self) -> Any:
        if (
            self._closed
            or self._decoder is None
            or self._demuxer is None
        ):
            raise VideoDecoderError("video_decoder_closed:nvdec")
        return self._decoder

    def _decode_next_surface(self) -> tuple[Any, float] | None:
        if self._cancel_requested.is_set():
            raise VideoDecoderError("video_decoder_cancelled:nvdec")
        self._require_decoder()
        if self._surface_queue:
            return self._surface_queue.popleft()
        if self._stream_eof:
            return None

        accumulated_ms = 0.0
        while not self._surface_queue:
            if self._cancel_requested.is_set():
                raise VideoDecoderError("video_decoder_cancelled:nvdec")
            started = time.perf_counter()
            try:
                packet = next(self._demuxer)
            except StopIteration:
                self._stream_eof = True
                self._demux_bytes_read = max(
                    self._demux_bytes_read,
                    int(self._alias.identity_metadata["source_size_bytes"]),
                )
                return None
            try:
                surfaces = self._decoder.Decode(packet)
            except Exception as exc:
                raise VideoDecoderError(
                    f"pynv_decode_packet_failed:{type(exc).__name__}:{exc}"
                ) from exc
            accumulated_ms += (time.perf_counter() - started) * 1000.0
            if surfaces:
                per_frame_ms = accumulated_ms / len(surfaces)
                self._surface_queue.extend(
                    (surface, per_frame_ms) for surface in surfaces
                )
        return self._surface_queue.popleft()

    def _destroy_pipeline(self) -> None:
        feeder = self._feeder
        decoder = self._decoder
        demuxer = self._demuxer
        self._feeder = None
        self._decoder = None
        self._demuxer = None
        self._surface_queue.clear()
        del decoder
        del demuxer
        gc.collect()
        if feeder is not None:
            self._demux_bytes_read = max(
                self._demux_bytes_read,
                int(feeder.bytes_read),
            )
            feeder.close()

    def _rebuild_pipeline(self) -> None:
        self._destroy_pipeline()
        try:
            demuxer = self._nvc.CreateDemuxer(str(self._alias.decoder_path))
            decoder = self._nvc.CreateDecoder(
                gpuid=self._gpu_id,
                codec=demuxer.GetNvCodecId(),
                usedevicememory=True,
                outputColorType=self._nvc.OutputColorType.RGBP,
                enableDecodeStats=False,
            )
        except Exception:
            raise
        self._feeder = None
        self._demuxer = demuxer
        self._decoder = decoder
        self._stream_eof = False
        self._surface_queue.clear()

    def read(self) -> DecodedFrameLease | None:
        self._require_decoder()
        if self._eof:
            return None
        frame_idx = self._next_frame_idx
        if self._info.frame_count > 0 and frame_idx >= self._info.frame_count:
            self._eof = True
            self._metrics.record_eof()
            return None

        decoded = self._decode_next_surface()
        if decoded is None:
            self._eof = True
            self._metrics.record_eof()
            return None
        decoded_surface, packet_decode_ms = decoded

        torch = self._torch
        sync_started = time.perf_counter()
        shared_tensor = torch.from_dlpack(decoded_surface)
        torch.cuda.synchronize(self._gpu_id)
        decode_ms = packet_decode_ms + (
            time.perf_counter() - sync_started
        ) * 1000.0
        if shared_tensor.ndim != 3 or int(shared_tensor.shape[0]) != 3:
            raise VideoDecoderError(
                f"pynv_invalid_rgbp_shape:{tuple(shared_tensor.shape)!r}"
            )
        if not shared_tensor.is_cuda:
            raise VideoDecoderError(
                f"pynv_unexpected_frame_device:{shared_tensor.device}"
            )

        with torch.cuda.device(self._gpu_id):
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            copy_start.record()
            owned_tensor = shared_tensor.clone(
                memory_format=torch.contiguous_format
            )
            copy_end.record()
            copy_end.synchronize()
            d2d_copy_ms = float(copy_start.elapsed_time(copy_end))

        height = int(owned_tensor.shape[1])
        width = int(owned_tensor.shape[2])
        raw_timestamp = max(
            0,
            int(getattr(decoded_surface, "timestamp", 0) or 0),
        )
        del shared_tensor
        del decoded_surface

        pts_s = frame_idx / self._info.fps if self._info.fps > 0.0 else 0.0
        pts_s = max(0.0, float(pts_s))
        self._next_frame_idx = frame_idx + 1
        self._metrics.record_decode(decode_ms)
        self._metrics.record_d2d_copy(d2d_copy_ms)
        self._metrics.record_frame(
            int(owned_tensor.numel() * owned_tensor.element_size())
        )

        def materialize(
            size: tuple[int, int] | None,
            roi: tuple[int, int, int, int] | None,
        ) -> Any:
            started = time.perf_counter()
            output = owned_tensor
            bounded_roi = _clamp_roi(roi, width=width, height=height)
            if bounded_roi is not None:
                x1, y1, x2, y2 = bounded_roi
                output = output[:, y1:y2, x1:x2]
            output_size = _validate_size(size)
            if output_size is not None and (
                int(output.shape[2]) != output_size[0]
                or int(output.shape[1]) != output_size[1]
            ):
                # torchvision's CUDA uint8 resize avoids allocating a full
                # 4K float32 intermediate (~100 MiB per frame).  The previous
                # path could stall strict predecessor materialization for
                # hundreds of milliseconds when preview and detection resized
                # different NVDEC surfaces concurrently.
                output = self._vision_functional.resize(
                    output,
                    [output_size[1], output_size[0]],
                    interpolation=self._resize_interpolation,
                    antialias=False,
                )
            bgr_hwc = output[[2, 1, 0]].permute(1, 2, 0).contiguous()
            host_array = bgr_hwc.cpu().numpy()
            self._metrics.record_d2h_copy(
                (time.perf_counter() - started) * 1000.0
            )
            return host_array

        return DecodedFrameLease(
            frame_idx=frame_idx,
            pts_s=pts_s,
            width=width,
            height=height,
            pixel_format="rgbp",
            storage=f"cuda:{self._gpu_id}",
            decode_ms=decode_ms,
            d2d_copy_ms=d2d_copy_ms,
            owner=owned_tensor,
            cuda_tensor=owned_tensor,
            metadata={
                "backend": "nvdec",
                "requested_backend": self._requested_backend,
                "source": str(self._alias.source_path),
                "decoder_source": str(self._alias.decoder_path),
                "decoder_timestamp_raw": raw_timestamp,
                "surface_cloned": True,
                "pynv_version": self._pynv_version,
            },
            _host_materializer=materialize,
        )

    def seek_time(self, seconds: float) -> None:
        value = max(0.0, float(seconds))
        if self._info.fps > 0.0:
            target = round(value * self._info.fps)
        elif self._info.duration_s > 0.0 and self._info.frame_count > 0:
            target = round(value / self._info.duration_s * self._info.frame_count)
        else:
            target = 0
        self.seek_frame(target)

    def seek_frame(self, frame_idx: int) -> None:
        self._require_decoder()
        target = max(0, int(frame_idx))
        if self._info.frame_count > 0:
            target = min(target, self._info.frame_count)
        try:
            self._rebuild_pipeline()
            skipped = 0
            while skipped < target:
                if self._cancel_requested.is_set():
                    raise VideoDecoderError("pynv_seek_cancelled")
                decoded = self._decode_next_surface()
                if decoded is None:
                    break
                surface, _decode_ms = decoded
                del surface
                skipped += 1
        except Exception as exc:
            self._eof = True
            raise VideoDecoderError(
                f"pynv_seek_frame_failed:{target}:"
                f"{type(exc).__name__}:{exc}"
            ) from exc
        self._next_frame_idx = skipped
        self._eof = skipped < target or target >= self._info.frame_count > 0
        self._metrics.record_seek()

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "requested_backend": self._requested_backend,
            "backend": "nvdec",
            "effective_backend": "nvdec",
            "source": str(self._alias.source_path),
            "codec": self._info.codec,
            "gpu_device": self._info.gpu_device,
            "output_format": self._info.output_format,
            "frame_device": self._info.frame_device,
            "pynv_version": self._pynv_version,
            "surface_clone_policy": "synchronous_d2d_owned_tensor",
            "demux_mode": self._demux_mode,
            "demux_bytes_read": int(self._demux_bytes_read),
            "closed": bool(self._closed),
            "cancel_requested": self._cancel_requested.is_set(),
            "eof": bool(self._eof),
            "next_frame_idx": int(self._next_frame_idx),
            "close_error": self._close_error,
            **self._alias.status_snapshot(),
            **self._metrics.snapshot(),
        }

    def request_cancel(self) -> None:
        """Request cancellation without destroying decoder state cross-thread."""
        self._cancel_requested.set()

    def close(self) -> None:
        self._cancel_requested.set()
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        try:
            self._destroy_pipeline()
        except Exception as exc:
            errors.append(f"decoder_close:{type(exc).__name__}:{exc}")
        if not self._alias.cleanup() and self._alias.cleanup_enabled:
            errors.append(
                "alias_cleanup:"
                + (self._alias.cleanup_error or "alias_still_present")
            )
        self._close_error = "; ".join(errors)
