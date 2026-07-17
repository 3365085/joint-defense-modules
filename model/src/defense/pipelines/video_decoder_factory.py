from __future__ import annotations

from pathlib import Path
from typing import Any

from defense.pipelines.derived_video_cache import (
    DerivedVideoCacheError,
    DerivedVideoDecoder,
    resolve_derived_video_source,
)
from defense.pipelines._opencv_file_decoder import OpenCVFileDecoder
from defense.pipelines._runtime_fallback_decoder import (
    RuntimeFallbackVideoDecoder,
)
from defense.pipelines.video_decoder import (
    VideoDecoder,
    VideoDecoderUnavailable,
)


_PREFERENCES = {
    "auto": "auto",
    "nvdec": "nvdec",
    "pynv": "nvdec",
    "pynvvideocodec": "nvdec",
    "opencv": "opencv",
    "cpu": "opencv",
}


def normalize_decoder_preference(preference: str) -> str:
    normalized = str(preference or "auto").strip().lower()
    try:
        return _PREFERENCES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported_video_decoder_preference:{preference!r}:"
            "expected=auto|nvdec|opencv"
        ) from exc


def _fallback_reason(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > 800:
        text = text[:797] + "..."
    return f"nvdec_init_failed:{type(exc).__name__}:{text}"


def create_video_decoder(
    source: str | Path,
    *,
    preference: str = "auto",
    allow_cpu_fallback: bool = True,
    gpu_id: int = 0,
    nvdec_options: dict[str, Any] | None = None,
    opencv_options: dict[str, Any] | None = None,
) -> VideoDecoder:
    """Create a file decoder with explicit, observable NVDEC fallback."""

    selected = normalize_decoder_preference(preference)
    opencv_kwargs = dict(opencv_options or {})
    if selected == "opencv":
        return OpenCVFileDecoder(
            source,
            requested_backend=selected,
            **opencv_kwargs,
        )

    derived_resolution = None
    try:
        from defense.pipelines._pynv_file_decoder import PyNvFileDecoder

        derived_resolution = resolve_derived_video_source(source)
        decoder: VideoDecoder = PyNvFileDecoder(
            (
                derived_resolution.decode_path
                if derived_resolution is not None
                else source
            ),
            gpu_id=gpu_id,
            requested_backend=selected,
            **dict(nvdec_options or {}),
        )
        if derived_resolution is not None:
            decoder = DerivedVideoDecoder(decoder, derived_resolution)
        if allow_cpu_fallback:
            return RuntimeFallbackVideoDecoder(
                decoder,
                source,
                requested_backend=selected,
                opencv_options=opencv_kwargs,
            )
        return decoder
    except Exception as exc:
        if isinstance(exc, DerivedVideoCacheError):
            raise VideoDecoderUnavailable(
                f"derived_cache_required:{exc}"
            ) from exc
        if derived_resolution is not None:
            raise VideoDecoderUnavailable(
                "derived_nvdec_required:"
                f"{type(exc).__name__}:{exc}"
            ) from exc
        if not allow_cpu_fallback:
            if isinstance(exc, VideoDecoderUnavailable):
                raise
            raise VideoDecoderUnavailable(_fallback_reason(exc)) from exc
        return OpenCVFileDecoder(
            source,
            requested_backend=selected,
            fallback_reason=_fallback_reason(exc),
            **opencv_kwargs,
        )


def available_video_decoder_backends() -> tuple[str, ...]:
    return "nvdec", "opencv"
