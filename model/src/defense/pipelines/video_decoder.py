from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


class VideoDecoderError(RuntimeError):
    """Base error for the unified file decoder contract."""


class VideoDecoderUnavailable(VideoDecoderError):
    """Requested decoder backend cannot be initialized for this source."""


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    source: str
    backend: str
    codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    gpu_device: str | None = None
    output_format: str = "bgr24"
    frame_device: str = "host"


HostMaterializer = Callable[
    [tuple[int, int] | None, tuple[int, int, int, int] | None],
    Any,
]
ReleaseCallback = Callable[[], None]


@dataclass(slots=True)
class DecodedFrameLease:
    """Stable decoded-frame ownership boundary.

    Backends must guarantee that ``cuda_tensor``/``host_array`` remain unchanged
    until ``release()``. PyNv decoder-owned surfaces therefore cannot be exposed
    directly when subsequent decode calls can recycle them; the backend must
    lock the surface or perform a GPU-to-GPU copy into owned storage first.

    GPU-to-CPU conversion is explicit through ``materialize_host_bgr`` so every
    download can be measured and surfaced in runtime status.
    """

    frame_idx: int
    pts_s: float
    width: int
    height: int
    pixel_format: str
    storage: str
    decode_ms: float
    d2d_copy_ms: float = 0.0
    owner: Any = None
    cuda_tensor: Any = None
    host_array: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _host_materializer: HostMaterializer | None = field(
        default=None,
        repr=False,
    )
    _release_callback: ReleaseCallback | None = field(
        default=None,
        repr=False,
    )
    _released: bool = field(default=False, init=False, repr=False)
    _host_cache: dict[
        tuple[tuple[int, int] | None, tuple[int, int, int, int] | None],
        Any,
    ] = field(default_factory=dict, init=False, repr=False)
    _state_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    @property
    def released(self) -> bool:
        return self._released

    def materialize_host_bgr(
        self,
        *,
        size: tuple[int, int] | None = None,
        roi: tuple[int, int, int, int] | None = None,
    ) -> Any:
        key = (size, roi)
        with self._state_lock:
            if self._released:
                raise VideoDecoderError("decoded_frame_lease_released")
            cached = self._host_cache.get(key)
            if cached is not None:
                return cached
            host_array = self.host_array
            materializer = self._host_materializer

        if size is None and roi is None and host_array is not None:
            value = host_array
        elif materializer is not None:
            # Different consumers (preview and detection) may materialize
            # different sizes concurrently. The owned CUDA tensor is stable, so
            # avoid holding the cache lock across GPU work and D2H transfer.
            value = materializer(size, roi)
        else:
            raise VideoDecoderError(
                "host_bgr_materialization_unavailable:"
                f"storage={self.storage}:pixel_format={self.pixel_format}"
            )

        with self._state_lock:
            if self._released:
                raise VideoDecoderError("decoded_frame_lease_released")
            existing = self._host_cache.get(key)
            if existing is not None:
                return existing
            self._host_cache[key] = value
            return value

    def release(self) -> None:
        with self._state_lock:
            if self._released:
                return
            self._released = True
            callback = self._release_callback
            self._release_callback = None
            self._host_materializer = None
            self._host_cache.clear()
        if callback is not None:
            callback()

    def __enter__(self) -> "DecodedFrameLease":
        if self._released:
            raise VideoDecoderError("decoded_frame_lease_released")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


@runtime_checkable
class VideoDecoder(Protocol):
    @property
    def info(self) -> VideoStreamInfo: ...

    def read(self) -> DecodedFrameLease | None: ...

    def seek_time(self, seconds: float) -> None: ...

    def seek_frame(self, frame_idx: int) -> None: ...

    def status_snapshot(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
