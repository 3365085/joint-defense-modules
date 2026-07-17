from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from defense.pipelines._opencv_file_decoder import OpenCVFileDecoder
from defense.pipelines.video_decoder import (
    DecodedFrameLease,
    VideoDecoder,
    VideoDecoderError,
    VideoDecoderUnavailable,
    VideoStreamInfo,
)


def _safe_snapshot(decoder: VideoDecoder) -> dict[str, Any]:
    try:
        return dict(decoder.status_snapshot())
    except Exception as exc:
        return {
            "status_error": f"{type(exc).__name__}:{exc}",
        }


def _runtime_reason(
    code: str,
    snapshot: dict[str, Any],
    exc: BaseException | None = None,
) -> str:
    details = [
        code,
        f"backend={snapshot.get('effective_backend') or snapshot.get('backend') or 'nvdec'}",
        f"codec={snapshot.get('codec') or 'unknown'}",
        f"frames_decoded={int(snapshot.get('frames_decoded') or 0)}",
        f"demux_mode={snapshot.get('demux_mode') or 'unknown'}",
        f"demux_bytes_read={int(snapshot.get('demux_bytes_read') or 0)}",
    ]
    if exc is not None:
        detail = " ".join(str(exc).split())
        details.append(f"error={type(exc).__name__}:{detail}")
    rendered = ":".join(details)
    if len(rendered) > 800:
        rendered = rendered[:797] + "..."
    return rendered


class RuntimeFallbackVideoDecoder:
    """Switch an initialized NVDEC decoder to OpenCV on read-time failure.

    Decoder construction fallback alone cannot catch a backend that opens the
    container successfully but returns EOF before its first frame.  This adapter
    keeps that failure observable and retries the original source through the
    configured CPU fallback path.
    """

    def __init__(
        self,
        primary: VideoDecoder,
        source: str | Path,
        *,
        requested_backend: str,
        opencv_options: dict[str, Any] | None = None,
    ) -> None:
        self._source_path = Path(source).expanduser().resolve(strict=False)
        self._requested_backend = str(requested_backend or "nvdec")
        self._opencv_options = dict(opencv_options or {})
        self._primary = primary
        self._current = primary
        self._lock = threading.RLock()
        self._cancel_requested = threading.Event()
        self._closed = False
        self._frames_delivered = 0
        self._fallback_attempted = False
        self._fallback_succeeded = False
        self._fallback_trigger = ""
        self._fallback_error = ""
        self._primary_snapshot: dict[str, Any] = {}
        self._close_errors: list[str] = []

    @property
    def info(self) -> VideoStreamInfo:
        with self._lock:
            return self._current.info

    def _require_open(self) -> None:
        if self._closed:
            raise VideoDecoderError("video_decoder_closed:runtime_fallback")

    def _activate_fallback(
        self,
        reason: str,
        *,
        resume_frame: int,
    ) -> VideoDecoder:
        with self._lock:
            self._require_open()
            if self._fallback_succeeded:
                return self._current
            if self._cancel_requested.is_set():
                raise VideoDecoderError(
                    "video_decoder_cancelled:runtime_fallback"
                )
            primary = self._primary
            self._primary_snapshot = _safe_snapshot(primary)
            self._fallback_attempted = True
            self._fallback_trigger = reason

        try:
            fallback = OpenCVFileDecoder(
                self._source_path,
                requested_backend=self._requested_backend,
                fallback_reason=reason,
                **self._opencv_options,
            )
            if resume_frame > 0:
                fallback.seek_frame(resume_frame)
        except Exception as exc:
            with self._lock:
                self._fallback_error = (
                    f"{type(exc).__name__}:{' '.join(str(exc).split())}"
                )
            raise VideoDecoderUnavailable(
                "nvdec_runtime_fallback_failed:"
                f"trigger={reason}:"
                f"fallback={type(exc).__name__}:{exc}"
            ) from exc

        close_error = ""
        try:
            primary.close()
        except Exception as exc:
            close_error = f"primary_close:{type(exc).__name__}:{exc}"

        with self._lock:
            self._current = fallback
            self._fallback_succeeded = True
            self._fallback_error = ""
            if close_error:
                self._close_errors.append(close_error)
            closed_primary = _safe_snapshot(primary)
            if closed_primary:
                self._primary_snapshot = closed_primary
        return fallback

    def read(self) -> DecodedFrameLease | None:
        with self._lock:
            self._require_open()
            current = self._current
            is_primary = not self._fallback_succeeded

        try:
            lease = current.read()
        except Exception as exc:
            if not is_primary or self._cancel_requested.is_set():
                raise
            snapshot = _safe_snapshot(current)
            reason = _runtime_reason(
                "nvdec_runtime_read_failed",
                snapshot,
                exc,
            )
            resume_frame = max(
                self._frames_delivered,
                int(snapshot.get("next_frame_idx") or 0),
            )
            fallback = self._activate_fallback(
                reason,
                resume_frame=resume_frame,
            )
            lease = fallback.read()
        else:
            if lease is None and is_primary:
                snapshot = _safe_snapshot(current)
                decoded_count = max(
                    self._frames_delivered,
                    int(snapshot.get("frames_decoded") or 0),
                )
                if decoded_count == 0:
                    reason = _runtime_reason(
                        "nvdec_runtime_zero_frame_eof",
                        snapshot,
                    )
                    fallback = self._activate_fallback(
                        reason,
                        resume_frame=0,
                    )
                    lease = fallback.read()

        if lease is not None:
            with self._lock:
                self._frames_delivered = max(
                    self._frames_delivered,
                    int(lease.frame_idx) + 1,
                )
        return lease

    def seek_time(self, seconds: float) -> None:
        with self._lock:
            self._require_open()
            current = self._current
        current.seek_time(seconds)

    def seek_frame(self, frame_idx: int) -> None:
        with self._lock:
            self._require_open()
            current = self._current
        current.seek_frame(frame_idx)

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            current = self._current
            primary_snapshot = dict(self._primary_snapshot)
            fallback_attempted = self._fallback_attempted
            fallback_succeeded = self._fallback_succeeded
            fallback_trigger = self._fallback_trigger
            fallback_error = self._fallback_error
            closed = self._closed
            close_errors = list(self._close_errors)

        snapshot = _safe_snapshot(current)
        if not primary_snapshot:
            primary_snapshot = dict(snapshot)

        primary_reasons = list(primary_snapshot.get("fallback_reasons") or [])
        current_reasons = (
            list(snapshot.get("fallback_reasons") or [])
            if fallback_succeeded
            else []
        )
        fallback_reasons = [*primary_reasons, *current_reasons]
        if fallback_attempted and fallback_trigger not in fallback_reasons:
            fallback_reasons.append(fallback_trigger)

        if fallback_succeeded:
            primary_frames = int(primary_snapshot.get("frames_decoded") or 0)
            primary_bytes = int(primary_snapshot.get("bytes_decoded") or 0)
            snapshot["frames_decoded"] = primary_frames + int(
                snapshot.get("frames_decoded") or 0
            )
            snapshot["bytes_decoded"] = primary_bytes + int(
                snapshot.get("bytes_decoded") or 0
            )
            snapshot["source"] = str(self._source_path)
            snapshot["decode_source"] = str(self._source_path)

        snapshot.update(
            {
                "requested_backend": self._requested_backend,
                "fallback_count": len(fallback_reasons),
                "fallback_reason": (
                    fallback_reasons[-1] if fallback_reasons else ""
                ),
                "fallback_reasons": fallback_reasons,
                "runtime_fallback_attempted": fallback_attempted,
                "runtime_fallback_succeeded": fallback_succeeded,
                "runtime_fallback_trigger": fallback_trigger,
                "runtime_fallback_error": fallback_error,
                "nvdec_attempt_backend": (
                    primary_snapshot.get("effective_backend")
                    or primary_snapshot.get("backend")
                ),
                "nvdec_attempt_codec": primary_snapshot.get("codec"),
                "nvdec_attempt_frames_decoded": int(
                    primary_snapshot.get("frames_decoded") or 0
                ),
                "nvdec_attempt_eof": bool(primary_snapshot.get("eof", False)),
                "closed": bool(closed or snapshot.get("closed", False)),
            }
        )

        primary_close_error = str(primary_snapshot.get("close_error") or "").strip()
        current_close_error = str(snapshot.get("close_error") or "").strip()
        all_close_errors = [
            value
            for value in [*close_errors, primary_close_error, current_close_error]
            if value and value != "none"
        ]
        snapshot["close_error"] = "; ".join(dict.fromkeys(all_close_errors))

        if bool(primary_snapshot.get("derived_cache_used", False)):
            for key, value in primary_snapshot.items():
                if key.startswith("derived_") or key in {
                    "source_sha256",
                    "derived_metadata_path",
                    "derived_metadata_sha256",
                    "source_asset_id",
                    "source_role",
                    "source_label",
                    "source_attack_type",
                    "source_codec",
                    "transcode_decode_backend",
                    "transcode_encode_backend",
                }:
                    snapshot[key] = value
            snapshot["derived_cache_attempted"] = True
            snapshot["derived_cache_runtime_fallback"] = fallback_succeeded
            snapshot["derived_decode_source"] = primary_snapshot.get(
                "decode_source"
            )
            if fallback_succeeded:
                snapshot["derived_cache_used"] = False
                snapshot["decode_source"] = str(self._source_path)
                snapshot["decode_source_sha256"] = primary_snapshot.get(
                    "source_sha256"
                )
        else:
            snapshot.setdefault("derived_cache_used", False)
            snapshot.setdefault("derived_cache_validation", "not_used")
            snapshot["derived_cache_attempted"] = False
            snapshot["derived_cache_runtime_fallback"] = False
            snapshot["derived_decode_source"] = None
        return snapshot

    def request_cancel(self) -> None:
        self._cancel_requested.set()
        with self._lock:
            current = self._current
        callback = getattr(current, "request_cancel", None)
        if callable(callback):
            callback()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            current = self._current
            primary = self._primary

        errors: list[str] = []
        seen: set[int] = set()
        for label, decoder in (("current", current), ("primary", primary)):
            if id(decoder) in seen:
                continue
            seen.add(id(decoder))
            try:
                decoder.close()
            except Exception as exc:
                errors.append(f"{label}_close:{type(exc).__name__}:{exc}")
        if errors:
            with self._lock:
                self._close_errors.extend(errors)
            raise VideoDecoderError("; ".join(errors))
