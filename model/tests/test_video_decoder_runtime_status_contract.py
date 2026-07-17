from __future__ import annotations

import math
from typing import Any

from defense.runtime.config import load_runtime_config
from defense.runtime.runner import MonitorEngine, _decoder_status_contract


DECODER_STATUS_KEYS = {
    "requested_backend",
    "backend",
    "effective_backend",
    "codec",
    "gpu_device",
    "output_format",
    "frame_device",
    "decode_p50_ms",
    "decode_p95_ms",
    "d2d_copy_p50_ms",
    "d2d_copy_p95_ms",
    "gpu_to_cpu_copy_p50_ms",
    "gpu_to_cpu_copy_p95_ms",
    "frames_decoded",
    "bytes_decoded",
    "fallback_count",
    "fallback_reason",
    "close_error",
    "decode_source",
    "derived_cache_used",
    "derived_cache_validation",
    "source_sha256",
    "decode_source_sha256",
    "derived_metadata_path",
    "derived_metadata_sha256",
    "source_asset_id",
    "source_role",
    "source_label",
    "source_attack_type",
    "source_codec",
    "derived_codec",
    "derived_profile_id",
    "derived_profile_sha256",
    "derived_expected_frame_count",
    "derived_expected_duration_s",
    "transcode_decode_backend",
    "transcode_encode_backend",
    "derived_frame_parity",
    "derived_frame_count_match",
    "derived_fps_match",
}

DECODER_NONNEGATIVE_NUMERIC_KEYS = {
    "decode_p50_ms",
    "decode_p95_ms",
    "d2d_copy_p50_ms",
    "d2d_copy_p95_ms",
    "gpu_to_cpu_copy_p50_ms",
    "gpu_to_cpu_copy_p95_ms",
    "frames_decoded",
    "bytes_decoded",
    "fallback_count",
    "derived_expected_frame_count",
    "derived_expected_duration_s",
}


class _Cache:
    def clear(self) -> None:
        return None


def _runtime_decoder_settings(runtime: dict[str, Any]) -> tuple[Any, Any]:
    nested = runtime.get("video_decoder")
    if not isinstance(nested, dict):
        nested = runtime.get("decoder")
    if not isinstance(nested, dict):
        nested = {}

    requested_backend = nested.get(
        "requested_backend",
        nested.get(
            "backend",
            nested.get(
                "preference",
                runtime.get(
                    "video_decoder_backend",
                    runtime.get("video_decoder_preference"),
                ),
            ),
        ),
    )
    allow_cpu_fallback = nested.get(
        "allow_cpu_fallback",
        runtime.get("video_decoder_allow_cpu_fallback"),
    )
    return requested_backend, allow_cpu_fallback


def test_initial_monitor_status_exposes_stable_decoder_contract() -> None:
    status = MonitorEngine(_Cache()).get_status()  # type: ignore[arg-type]

    assert isinstance(status["detector_compute_fps"], (int, float))
    assert math.isfinite(float(status["detector_compute_fps"]))
    assert float(status["detector_compute_fps"]) >= 0.0
    assert isinstance(status["decoder"], dict)
    decoder = status["decoder"]
    assert DECODER_STATUS_KEYS <= set(decoder)
    assert decoder["requested_backend"] in {None, "", "nvdec"}
    assert decoder["backend"] in {None, "", "not_started"}
    assert decoder["effective_backend"] in {None, "", "not_started"}
    assert decoder["fallback_count"] == 0
    assert decoder["fallback_reason"] in {"", "not_started", "none"}
    assert decoder["close_error"] in {"", "not_started", "none"}
    assert decoder["derived_cache_used"] is False
    assert decoder["derived_cache_validation"] == "not_used"
    assert decoder["derived_frame_parity"] is False
    assert decoder["derived_frame_count_match"] is False
    assert decoder["derived_fps_match"] is False

    for key in DECODER_NONNEGATIVE_NUMERIC_KEYS:
        value = decoder[key]
        assert isinstance(value, (int, float)), f"{key} must be numeric"
        assert math.isfinite(float(value)), f"{key} must be finite"
        assert float(value) >= 0.0, f"{key} must be non-negative"

    assert decoder["decode_p95_ms"] >= decoder["decode_p50_ms"]
    assert decoder["d2d_copy_p95_ms"] >= decoder["d2d_copy_p50_ms"]
    assert (
        decoder["gpu_to_cpu_copy_p95_ms"]
        >= decoder["gpu_to_cpu_copy_p50_ms"]
    )


def test_default_runtime_config_explicitly_requests_nvdec_with_visible_fallback() -> None:
    config = load_runtime_config(profile="default")
    runtime = config["runtime"]

    requested_backend, allow_cpu_fallback = _runtime_decoder_settings(runtime)

    assert str(requested_backend or "").strip().lower() == "nvdec", (
        "production default must explicitly request NVDEC; 'auto' or an "
        "implicit Python default is not an auditable production setting"
    )
    assert allow_cpu_fallback is True, (
        "CPU fallback may remain enabled, but the default config must state it "
        "explicitly so status can report the effective backend and reason"
    )


def test_decoder_fallback_contract_never_allows_silent_backend_change() -> None:
    decoder = _decoder_status_contract(
        {
            "requested_backend": "nvdec",
            "backend": "opencv",
            "effective_backend": "opencv",
            "codec": "h264",
            "frame_device": "host",
            "decode_ms_p50": 1.0,
            "decode_ms_p95": 2.0,
            "d2d_copy_ms_p50": 0.0,
            "d2d_copy_ms_p95": 0.0,
            "d2h_copy_ms_p50": 0.5,
            "d2h_copy_ms_p95": 0.8,
            "fallback_count": 1,
            "fallback_reason": "nvdec_init_failed:synthetic",
        }
    )

    assert decoder["effective_backend"] != decoder["requested_backend"]
    assert decoder["fallback_count"] > 0
    assert str(decoder["fallback_reason"]).strip()
    assert decoder["decode_p50_ms"] == 1.0
    assert decoder["decode_p95_ms"] == 2.0
    assert decoder["gpu_to_cpu_copy_p50_ms"] == 0.5
    assert decoder["gpu_to_cpu_copy_p95_ms"] == 0.8
