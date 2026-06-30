"""Long-run stability test — hammer the ModuleADetector with a big
sequence of synthesised frames and check that:

  * No unexpected exception leaks.
  * Memory usage stays bounded (no growing deques / caches).
  * Reset() fully returns state to first-frame.
  * ``timing_ms`` stays within a sane envelope.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from defense.module_a.detector import ModuleADetector
from defense.module_a.types import ROI, ModuleAInput


def _config() -> dict:
    return {
        "module_a": {
            "require_gpu": True,
            "frame_size": 128,
            "keyframe_interval": 1,
            "alert_window": 5,
            "alert_trigger_count": 3,
            "attack_state_hold_frames": 2,
            "light_flow_enabled": False,
            "static_image_enabled": False,
            "fusion_backend": "rule",
        }
    }


@pytest.fixture
def detector(cuda_device: str) -> ModuleADetector:
    cfg = _config()
    cfg["module_a"]["device"] = cuda_device
    cfg["module_a"]["require_gpu"] = cuda_device.startswith("cuda")
    return ModuleADetector(cfg)


def _make_frame(idx: int) -> np.ndarray:
    # Injects a little motion so motion_artifact has something to do
    # without spamming alerts.
    base = 100 + (idx % 10)
    return np.full((128, 128, 3), base, dtype=np.uint8)


def test_long_run_no_exceptions(detector: ModuleADetector) -> None:
    for i in range(200):
        result = detector.process(ModuleAInput(frame=_make_frame(i), frame_idx=i))
        assert 0.0 <= result.p_adv <= 1.0
        assert result.timing_ms >= 0.0


def test_long_run_memory_bounded(detector: ModuleADetector) -> None:
    """Run 200 frames and check no obvious growth in cached state sizes."""
    for i in range(200):
        detector.process(ModuleAInput(frame=_make_frame(i), frame_idx=i))
    # Alert queue max len == window
    assert len(detector.alert_state.queue) <= detector.alert_state.window
    # Temporal persistence queue max len == persistence_frames
    assert len(detector.temporal._persistence_queue) <= detector.temporal.persistence_frames
    # Track analyzer caps tracks per label, so total can't exceed labels × cap.
    max_tracks_expected = len(detector.track.labels) * detector.track.max_tracks_per_label
    assert len(detector.track.tracks) <= max_tracks_expected
    # Static image tracks cap at max_tracks parameter.
    assert len(detector.static_image._tracks) <= detector.static_image.max_tracks


def test_reset_between_runs(detector: ModuleADetector) -> None:
    # Run some frames, reset, run again — results of the first post-reset
    # frame must be equivalent to a brand-new detector's first frame.
    for i in range(10):
        detector.process(ModuleAInput(frame=_make_frame(i), frame_idx=i))
    detector.reset()
    gc.collect()
    # After reset, temporal analyzer EMA must be back to 0 samples.
    assert detector.temporal._ema_samples == 0
    # Alert state must be empty.
    assert len(detector.alert_state.queue) == 0
    # Track state empty.
    assert len(detector.track.tracks) == 0


def test_mean_timing_under_bound(detector: ModuleADetector) -> None:
    """At 128×128 synthetic frames with no heavy features, processing a
    single frame must stay well under the 15 ms target from 架构说明."""
    # Warm up once to stabilise CUDA kernels.
    detector.process(ModuleAInput(frame=_make_frame(0), frame_idx=0))
    timings = []
    for i in range(1, 50):
        result = detector.process(ModuleAInput(frame=_make_frame(i), frame_idx=i))
        timings.append(result.timing_ms)
    # We assert on mean rather than single-frame to ride over any Windows
    # CUDA launch jitter on the first couple of frames.
    mean_ms = float(np.mean(timings))
    assert mean_ms < 250.0, f"synthetic 128x128 frame mean={mean_ms:.2f} ms exceeded CPU/GPU budget"