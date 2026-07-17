from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.pipelines._decoder_metrics import DecoderMetrics
from defense.pipelines.video_decoder import (
    DecodedFrameLease,
    VideoDecoderError,
    VideoDecoderUnavailable,
    VideoStreamInfo,
)


def _fourcc_name(value: float) -> str:
    parsed = int(value or 0)
    chars = "".join(chr((parsed >> (8 * index)) & 0xFF) for index in range(4))
    return chars.replace("\x00", "").strip() or "unknown"


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


def _validate_size(size: tuple[int, int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if len(size) != 2:
        raise VideoDecoderError(f"invalid_size:{size!r}")
    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise VideoDecoderError(f"invalid_size:{size!r}")
    return width, height


class OpenCVFileDecoder:
    """CPU file decoder implementing the unified stable-frame lease contract."""

    def __init__(
        self,
        source: str | Path,
        *,
        requested_backend: str = "opencv",
        fallback_reason: str = "",
        metrics_max_samples: int = 2048,
    ) -> None:
        path = Path(source).expanduser().resolve(strict=False)
        if not path.is_file():
            raise VideoDecoderUnavailable(f"opencv_source_not_found:{path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise VideoDecoderUnavailable(f"opencv_open_failed:{path}")

        self._source_path = path
        self._capture: cv2.VideoCapture | None = capture
        self._requested_backend = str(requested_backend or "opencv")
        self._metrics = DecoderMetrics(max_samples=metrics_max_samples)
        if fallback_reason:
            self._metrics.record_fallback(fallback_reason)
        self._closed = False
        self._eof = False
        self._next_frame_idx = 0

        width = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
        height = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps < 0.0:
            fps = 0.0
        frame_count = max(
            0,
            int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)),
        )
        duration_s = frame_count / fps if frame_count > 0 and fps > 0.0 else 0.0
        self._info = VideoStreamInfo(
            source=str(path),
            backend="opencv",
            codec=_fourcc_name(capture.get(cv2.CAP_PROP_FOURCC)),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=max(0.0, duration_s),
            gpu_device=None,
            output_format="bgr24",
            frame_device="host",
        )

    @property
    def info(self) -> VideoStreamInfo:
        return self._info

    def _require_capture(self) -> cv2.VideoCapture:
        if self._closed or self._capture is None:
            raise VideoDecoderError("video_decoder_closed:opencv")
        return self._capture

    def read(self) -> DecodedFrameLease | None:
        capture = self._require_capture()
        if self._eof:
            return None

        frame_idx = self._next_frame_idx
        started = time.perf_counter()
        ok, frame = capture.read()
        decode_ms = (time.perf_counter() - started) * 1000.0
        if not ok or frame is None:
            self._eof = True
            self._metrics.record_eof()
            return None

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise VideoDecoderError(
                f"opencv_invalid_frame_shape:{getattr(frame, 'shape', None)!r}"
            )
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if not math.isfinite(position_ms) or position_ms < 0.0:
            position_ms = 0.0
        pts_s = position_ms / 1000.0
        if pts_s <= 0.0 and self._info.fps > 0.0:
            pts_s = frame_idx / self._info.fps
        pts_s = max(0.0, float(pts_s))
        self._next_frame_idx = frame_idx + 1
        self._metrics.record_decode(decode_ms)
        self._metrics.record_frame(frame.nbytes)

        def materialize(
            size: tuple[int, int] | None,
            roi: tuple[int, int, int, int] | None,
        ) -> np.ndarray:
            output = frame
            bounded_roi = _clamp_roi(roi, width=width, height=height)
            if bounded_roi is not None:
                x1, y1, x2, y2 = bounded_roi
                output = output[y1:y2, x1:x2]
            output_size = _validate_size(size)
            if output_size is not None and (
                output.shape[1] != output_size[0]
                or output.shape[0] != output_size[1]
            ):
                interpolation = (
                    cv2.INTER_AREA
                    if output_size[0] < output.shape[1]
                    or output_size[1] < output.shape[0]
                    else cv2.INTER_LINEAR
                )
                output = cv2.resize(output, output_size, interpolation=interpolation)
            return np.ascontiguousarray(output)

        return DecodedFrameLease(
            frame_idx=frame_idx,
            pts_s=pts_s,
            width=width,
            height=height,
            pixel_format="bgr24",
            storage="host",
            decode_ms=decode_ms,
            d2d_copy_ms=0.0,
            owner=frame,
            host_array=frame,
            metadata={
                "backend": "opencv",
                "requested_backend": self._requested_backend,
                "source": str(self._source_path),
            },
            _host_materializer=materialize,
        )

    def seek_time(self, seconds: float) -> None:
        capture = self._require_capture()
        value = max(0.0, float(seconds))
        if not capture.set(cv2.CAP_PROP_POS_MSEC, value * 1000.0):
            frame_idx = round(value * self._info.fps) if self._info.fps > 0.0 else 0
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
                raise VideoDecoderError(f"opencv_seek_time_failed:{value}")
        position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0.0))
        self._next_frame_idx = max(0, position)
        self._eof = False
        self._metrics.record_seek()

    def seek_frame(self, frame_idx: int) -> None:
        capture = self._require_capture()
        target = max(0, int(frame_idx))
        if self._info.frame_count > 0:
            target = min(target, self._info.frame_count)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
            raise VideoDecoderError(f"opencv_seek_frame_failed:{target}")
        self._next_frame_idx = target
        self._eof = target >= self._info.frame_count > 0
        self._metrics.record_seek()

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "requested_backend": self._requested_backend,
            "backend": "opencv",
            "effective_backend": "opencv",
            "source": str(self._source_path),
            "codec": self._info.codec,
            "gpu_device": None,
            "output_format": self._info.output_format,
            "frame_device": self._info.frame_device,
            "closed": bool(self._closed),
            "eof": bool(self._eof),
            "next_frame_idx": int(self._next_frame_idx),
            **self._metrics.snapshot(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
