from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any, Iterable


def _ordered_values(values: Iterable[float]) -> list[float]:
    return sorted(
        float(value)
        for value in values
        if math.isfinite(float(value))
    )


def _percentile(ordered: list[float], percentile: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(100.0, float(percentile))) / 100.0 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class DecoderMetrics:
    """Bounded, thread-safe latency and fallback accounting for file decoders."""

    def __init__(self, *, max_samples: int = 2048) -> None:
        sample_count = max(16, int(max_samples))
        self._decode_ms: deque[float] = deque(maxlen=sample_count)
        self._d2d_copy_ms: deque[float] = deque(maxlen=sample_count)
        self._d2h_copy_ms: deque[float] = deque(maxlen=sample_count)
        self._lock = threading.Lock()
        self._frames_decoded = 0
        self._bytes_decoded = 0
        self._eof_count = 0
        self._seek_count = 0
        self._fallback_reasons: list[str] = []
        self._version = 0
        self._cached_version = -1
        self._cached_snapshot: dict[str, Any] | None = None

    def _mark_dirty(self) -> None:
        self._version += 1

    @staticmethod
    def _valid_ms(value: float) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            return 0.0
        return max(0.0, parsed)

    def record_decode(self, milliseconds: float) -> None:
        with self._lock:
            self._decode_ms.append(self._valid_ms(milliseconds))
            self._mark_dirty()

    def record_d2d_copy(self, milliseconds: float) -> None:
        with self._lock:
            self._d2d_copy_ms.append(self._valid_ms(milliseconds))
            self._mark_dirty()

    def record_d2h_copy(self, milliseconds: float) -> None:
        with self._lock:
            self._d2h_copy_ms.append(self._valid_ms(milliseconds))
            self._mark_dirty()

    def record_frame(self, byte_count: int) -> None:
        with self._lock:
            self._frames_decoded += 1
            self._bytes_decoded += max(0, int(byte_count))
            self._mark_dirty()

    def record_eof(self) -> None:
        with self._lock:
            self._eof_count += 1
            self._mark_dirty()

    def record_seek(self) -> None:
        with self._lock:
            self._seek_count += 1
            self._mark_dirty()

    def record_fallback(self, reason: str) -> None:
        normalized = str(reason or "").strip()
        if not normalized:
            return
        with self._lock:
            self._fallback_reasons.append(normalized)
            self._mark_dirty()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if (
                self._cached_snapshot is not None
                and self._cached_version == self._version
            ):
                return dict(self._cached_snapshot)
            decode_ms = tuple(self._decode_ms)
            d2d_copy_ms = tuple(self._d2d_copy_ms)
            d2h_copy_ms = tuple(self._d2h_copy_ms)
            fallback_reasons = tuple(self._fallback_reasons)
            frames_decoded = self._frames_decoded
            bytes_decoded = self._bytes_decoded
            eof_count = self._eof_count
            seek_count = self._seek_count
            version = self._version

        decode_ordered = _ordered_values(decode_ms)
        d2d_ordered = _ordered_values(d2d_copy_ms)
        d2h_ordered = _ordered_values(d2h_copy_ms)
        snapshot = {
            "frames_decoded": int(frames_decoded),
            "bytes_decoded": int(bytes_decoded),
            "decode_sample_count": len(decode_ms),
            "decode_ms_p50": round(
                _percentile(decode_ordered, 50.0),
                6,
            ),
            "decode_ms_p95": round(
                _percentile(decode_ordered, 95.0),
                6,
            ),
            "d2d_copy_sample_count": len(d2d_copy_ms),
            "d2d_copy_ms_p50": round(
                _percentile(d2d_ordered, 50.0),
                6,
            ),
            "d2d_copy_ms_p95": round(
                _percentile(d2d_ordered, 95.0),
                6,
            ),
            "d2h_copy_sample_count": len(d2h_copy_ms),
            "d2h_copy_ms_p50": round(
                _percentile(d2h_ordered, 50.0),
                6,
            ),
            "d2h_copy_ms_p95": round(
                _percentile(d2h_ordered, 95.0),
                6,
            ),
            "eof_count": int(eof_count),
            "seek_count": int(seek_count),
            "fallback_count": len(fallback_reasons),
            "fallback_reason": fallback_reasons[-1] if fallback_reasons else "",
            "fallback_reasons": list(fallback_reasons),
        }
        with self._lock:
            if version == self._version:
                self._cached_version = version
                self._cached_snapshot = dict(snapshot)
        return snapshot
