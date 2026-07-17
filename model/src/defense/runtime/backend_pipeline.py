from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from defense.pipelines.video_decoder import DecodedFrameLease


class SharedFrameLease:
    """Reference-counted ownership for one stable decoded-frame lease."""

    def __init__(self, lease: DecodedFrameLease) -> None:
        self.lease = lease
        self._lock = threading.Lock()
        self._references = 0
        self._released = False

    def acquire(self) -> None:
        with self._lock:
            if self._released:
                raise RuntimeError("shared_frame_lease_already_released")
            self._references += 1

    def release(self) -> None:
        release_lease = False
        with self._lock:
            if self._references <= 0:
                return
            self._references -= 1
            if self._references == 0 and not self._released:
                self._released = True
                release_lease = True
        if release_lease:
            self.lease.release()

    @property
    def references(self) -> int:
        with self._lock:
            return self._references


@dataclass(slots=True)
class FramePacket:
    seq: int
    frame_idx: int
    source_time_s: float
    wall_time_ms: float
    epoch: int
    frame: np.ndarray | None
    width: int
    height: int
    fps: float
    flags: dict[str, Any]
    previous_frame: np.ndarray | None = None
    previous_frame_idx: int | None = None
    previous_source_time_s: float | None = None
    decoder_lease: DecodedFrameLease | None = None
    previous_decoder_lease: DecodedFrameLease | None = None
    decoder_lease_owner: SharedFrameLease | None = None
    previous_decoder_lease_owner: SharedFrameLease | None = None

    def __post_init__(self) -> None:
        if self.decoder_lease is not None and self.decoder_lease_owner is None:
            self.decoder_lease_owner = SharedFrameLease(self.decoder_lease)
        if (
            self.previous_decoder_lease is not None
            and self.previous_decoder_lease_owner is None
        ):
            self.previous_decoder_lease_owner = SharedFrameLease(
                self.previous_decoder_lease
            )

    def acquire_lease_refs(self) -> None:
        owners = [
            self.decoder_lease_owner,
            self.previous_decoder_lease_owner,
        ]
        acquired: list[SharedFrameLease] = []
        try:
            for owner in owners:
                if owner is None or owner in acquired:
                    continue
                owner.acquire()
                acquired.append(owner)
        except Exception:
            for owner in reversed(acquired):
                owner.release()
            raise

    def release_lease_refs(self) -> None:
        released: list[SharedFrameLease] = []
        for owner in (
            self.decoder_lease_owner,
            self.previous_decoder_lease_owner,
        ):
            if owner is None or owner in released:
                continue
            owner.release()
            released.append(owner)


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
        self._latest_ref_held = False

    def publish(self, packet: FramePacket) -> None:
        with self.condition:
            if self.closed:
                return
            packet.acquire_lease_refs()
            previous = self.latest
            previous_ref_held = self._latest_ref_held
            self.latest = packet
            self._latest_ref_held = True
            self.latest_seq = packet.seq
            self.condition.notify_all()
        if previous is not None and previous_ref_held:
            previous.release_lease_refs()

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
            if (
                self.closed
                or self.latest is None
                or self.latest_seq <= last_seq
            ):
                return None
            packet = self.latest
            packet.acquire_lease_refs()
            return packet

    def clear(self) -> None:
        """Discard the buffered preview frame without changing sequence order."""
        with self.condition:
            previous = self.latest
            previous_ref_held = self._latest_ref_held
            self.latest = None
            self._latest_ref_held = False
            self.condition.notify_all()
        if previous is not None and previous_ref_held:
            previous.release_lease_refs()

    def close(self) -> None:
        with self.condition:
            if self.closed:
                self.condition.notify_all()
                return
            self.closed = True
            previous = self.latest
            previous_ref_held = self._latest_ref_held
            self._latest_ref_held = False
            self.condition.notify_all()
        if previous is not None and previous_ref_held:
            previous.release_lease_refs()


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
        self._latest_ref_held = False

    def push(self, packet: FramePacket) -> None:
        with self.condition:
            if self.closed:
                return
            if self.latest is not None and self.latest.seq > self.consumed_seq:
                self.dropped += 1
            packet.acquire_lease_refs()
            previous = self.latest
            previous_ref_held = self._latest_ref_held
            self.latest = packet
            self._latest_ref_held = True
            self.latest_seq = packet.seq
            self.condition.notify_all()
        if previous is not None and previous_ref_held:
            previous.release_lease_refs()

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
            self.condition.notify_all()
            packet = self.latest
            packet.acquire_lease_refs()
            release_storage = bool(self.closed)
            if release_storage:
                self.latest = None
                self._latest_ref_held = False
        if release_storage:
            packet.release_lease_refs()
        return packet

    def wait_until_consumed(self, seq: int, timeout: float = 0.2) -> bool:
        deadline = time.perf_counter() + max(0.0, timeout)
        with self.condition:
            while self.consumed_seq < int(seq) and not self.closed:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(0.05, remaining))
            return self.consumed_seq >= int(seq)

    def clear(self) -> None:
        with self.condition:
            previous = self.latest
            previous_ref_held = self._latest_ref_held
            self.latest = None
            self._latest_ref_held = False
            self.consumed_seq = self.latest_seq
            self.condition.notify_all()
        if previous is not None and previous_ref_held:
            previous.release_lease_refs()

    def close(self) -> None:
        with self.condition:
            if self.closed:
                self.condition.notify_all()
                return
            self.closed = True
            previous = self.latest
            previous_ref_held = self._latest_ref_held
            release_storage = bool(
                previous is None
                or previous.seq <= self.consumed_seq
            )
            if release_storage:
                self.latest = None
                self._latest_ref_held = False
            self.condition.notify_all()
        if (
            previous is not None
            and release_storage
            and previous_ref_held
        ):
            previous.release_lease_refs()
