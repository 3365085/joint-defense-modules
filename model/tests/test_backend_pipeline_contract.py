from __future__ import annotations

import threading
import time

import numpy as np

from defense.runtime.backend_pipeline import DetectionBus, FramePacket, PreviewBus


def _packet(seq: int, t: float) -> FramePacket:
    frame = np.full((8, 12, 3), seq, dtype=np.uint8)
    return FramePacket(
        seq=seq,
        frame_idx=seq,
        source_time_s=t,
        wall_time_ms=t * 1000.0,
        epoch=1,
        frame=frame,
        width=12,
        height=8,
        fps=25.0,
        flags={},
    )


def test_detection_bus_keeps_only_latest_pending_frame() -> None:
    bus = DetectionBus()
    bus.push(_packet(1, 0.0))
    bus.push(_packet(2, 0.04))
    bus.push(_packet(3, 0.08))

    item = bus.pop_latest(0, timeout=0.01)

    assert item is not None
    assert item.seq == 3
    assert bus.dropped == 2


def test_detection_bus_does_not_replay_consumed_frame_after_close() -> None:
    bus = DetectionBus()
    bus.push(_packet(1, 0.0))

    item = bus.pop_latest(0, timeout=0.01)
    assert item is not None
    assert item.seq == 1

    bus.close()

    assert bus.pop_latest(item.seq, timeout=0.01) is None


def test_detection_bus_drains_unconsumed_frame_once_after_close() -> None:
    bus = DetectionBus()
    bus.push(_packet(1, 0.0))
    bus.close()

    item = bus.pop_latest(0, timeout=0.01)

    assert item is not None
    assert item.seq == 1
    assert bus.pop_latest(item.seq, timeout=0.01) is None


def test_detection_bus_wait_until_consumed_unblocks_on_pop() -> None:
    bus = DetectionBus()
    bus.push(_packet(1, 0.0))
    results: list[bool] = []

    thread = threading.Thread(
        target=lambda: results.append(bus.wait_until_consumed(1, timeout=0.5))
    )
    thread.start()
    time.sleep(0.02)

    item = bus.pop_latest(0, timeout=0.01)
    thread.join(timeout=1.0)

    assert item is not None
    assert item.seq == 1
    assert results == [True]


def test_preview_bus_continues_while_detection_consumer_is_slow() -> None:
    preview = PreviewBus()
    detection = DetectionBus()
    produced_preview: list[int] = []

    def slow_detector() -> None:
        item = detection.pop_latest(0, timeout=0.1)
        assert item is not None
        time.sleep(0.25)
        detection.pop_latest(item.seq, timeout=0.1)

    thread = threading.Thread(target=slow_detector)
    thread.start()
    for seq in range(1, 7):
        packet = _packet(seq, seq / 25.0)
        preview.publish(packet)
        detection.push(packet)
        latest = preview.latest_packet()
        if latest is not None:
            produced_preview.append(latest.seq)
        time.sleep(0.02)
    thread.join(timeout=1.0)

    assert produced_preview == [1, 2, 3, 4, 5, 6]
    assert detection.dropped >= 3


def test_preview_bus_does_not_replay_cached_frame_after_close() -> None:
    preview = PreviewBus()
    preview.publish(_packet(1, 0.0))

    assert preview.latest_packet_if_open() is not None

    preview.close()

    assert preview.latest_packet_if_open() is None
    assert preview.wait_for_frame(1, timeout=0.01) is None
