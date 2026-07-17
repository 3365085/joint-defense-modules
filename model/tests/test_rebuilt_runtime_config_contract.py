from __future__ import annotations

from defense.runtime.config import load_runtime_config


def test_desktop_rtx_exposes_effective_rebuilt_a3b_configuration() -> None:
    config = load_runtime_config(profile="desktop_rtx")
    module_a = config["module_a"]

    assert module_a["detector_impl"] == "rebuilt"
    assert module_a["static_image_enabled"] is True
    assert module_a["static_image_interval"] == 3
    assert module_a["static_image_worker_timeout_s"] == 3.0
    assert module_a["static_image_result_lease_s"] == 5.0
    assert module_a["static_image_max_retired_workers"] == 2
    assert module_a["static_image_global_worker_limit"] == 2
    assert module_a["rebuilt_theta_media"] == 0.55
    assert module_a["rebuilt_theta_media_raw"] == 0.50
    assert module_a["rebuilt_a3b_independent_trigger"] is True
    assert module_a["rebuilt_a3b_tighten_gate"] is True
    assert module_a["rebuilt_a3b_gate_candidate_min"] == 0.70
    assert module_a["rebuilt_a3b_gate_edge_min"] == 0.45
    assert module_a["rebuilt_a3b_gate_edge_max"] == 0.58
    assert module_a["rebuilt_a3b_gate_border_contrast_min"] == 0.80
    assert module_a["rebuilt_a3b_soft_gate_candidate_tolerance"] == 0.001
    assert module_a["rebuilt_a3b_soft_gate_aspect_ratio_min"] == 0.40
    assert module_a["rebuilt_a3b_soft_gate_aspect_ratio_max"] == 2.50
    assert module_a["rebuilt_a3b_media_run_floor"] == 15
    assert module_a["rebuilt_a3b_media_run_gap_tol"] == 3
