from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class FramePacket:
    seq: int
    frame_idx: int
    source_time_s: float
    wall_time_ms: float
    epoch: int
    frame: np.ndarray
    width: int
    height: int
    fps: float
    flags: dict[str, Any]


class PreviewBus:
    """Latest frame holder for display.

    Preview is intentionally independent from detection. The renderer reads the
    newest source frame at its own cadence and must never wait for inference.
    """

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.latest: FramePacket | None = None
        self.latest_seq = 0
        self.closed = False

    def publish(self, packet: FramePacket) -> None:
        with self.condition:
            if self.closed:
                return
            self.latest = packet
            self.latest_seq = packet.seq
            self.condition.notify_all()

    def latest_packet(self) -> FramePacket | None:
        with self.condition:
            return self.latest

    def latest_packet_if_open(self) -> FramePacket | None:
        with self.condition:
            if self.closed:
                return None
            return self.latest

    def wait_for_frame(self, last_seq: int, timeout: float = 0.2) -> FramePacket | None:
        deadline = time.perf_counter() + max(0.0, timeout)
        with self.condition:
            while self.latest_seq <= last_seq and not self.closed:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(0.05, remaining))
            if self.closed and self.latest_seq <= last_seq:
                return None
            return self.latest

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class DetectionBus:
    """Latest-only detector queue.

    If inference is slower than the source clock, old pending frames are
    replaced. This is the backpressure point that prevents 0.3x playback.
    """

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.latest: FramePacket | None = None
        self.latest_seq = 0
        self.consumed_seq = 0
        self.closed = False
        self.dropped = 0

    def push(self, packet: FramePacket) -> None:
        with self.condition:
            if self.closed:
                return
            if self.latest is not None and self.latest.seq > self.consumed_seq:
                self.dropped += 1
            self.latest = packet
            self.latest_seq = packet.seq
            self.condition.notify_all()

    def pop_latest(self, last_seq: int, timeout: float = 0.2) -> FramePacket | None:
        deadline = time.perf_counter() + max(0.0, timeout)
        with self.condition:
            while self.latest_seq <= last_seq and not self.closed:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.condition.wait(timeout=min(0.05, remaining))
            if self.latest is None or self.latest_seq <= last_seq:
                return None
            self.consumed_seq = self.latest.seq
            return self.latest

    def clear(self) -> None:
        with self.condition:
            self.latest = None
            self.consumed_seq = self.latest_seq
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
