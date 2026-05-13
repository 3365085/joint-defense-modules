"""Regression guard for the 7-clip sample smoke report.

This test does NOT re-run the smoke (too expensive for CI); it just
validates the last run's JSON against the baseline-after-打磨 numbers.
Regenerate the report with::

    python tests/run_samples_smoke.py

and this test tolerates ±10% around the headline metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPORT_PATH = Path(__file__).resolve().parent / "samples_smoke_report.json"

# Baseline numbers captured after P1-A-4/5/6/7 打磨 on 2026-05-13.
# Each entry:  (min_alert_frames, max_timing_mean_ms)
BASELINE = {
    "adv_patch_attacked.mp4": (400, 25.0),
    "clean_baseline.mp4": (None, 22.0),  # None → must be 0 alerts.
    "glare_attacked.mp4": (100, 22.0),
    "motion_blur_attacked.mp4": (100, 22.0),
    "occlusion_attacked.mp4": (100, 22.0),
    "screen_spoof_attacked.mp4": (400, 25.0),
    "visibility_degradation_attacked.mp4": (100, 22.0),
}


@pytest.fixture(scope="module")
def smoke_report():
    if not REPORT_PATH.exists():
        pytest.skip(
            "tests/samples_smoke_report.json missing; run tests/run_samples_smoke.py first"
        )
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_smoke_report_summary_ok(smoke_report) -> None:
    assert smoke_report["summary"]["ok"] is True, smoke_report["summary"]["verdicts"]


def test_per_clip_minimum_detection_rate(smoke_report) -> None:
    by_clip = {r["clip"]: r for r in smoke_report["results"]}
    for clip, (min_alert, _timing_budget) in BASELINE.items():
        if clip not in by_clip:
            pytest.skip(f"{clip} not present in smoke report")
        alert = by_clip[clip]["alert_frames"]
        if min_alert is None:
            assert alert == 0, f"{clip} must stay calm: alert_frames={alert}"
        else:
            assert alert >= min_alert, (
                f"{clip} detection regression: {alert} < {min_alert}"
            )


def test_per_clip_mean_timing_under_budget(smoke_report) -> None:
    by_clip = {r["clip"]: r for r in smoke_report["results"]}
    for clip, (_alert, budget_ms) in BASELINE.items():
        if clip not in by_clip:
            pytest.skip(f"{clip} not present in smoke report")
        mean_ms = by_clip[clip]["timing_mean_ms"]
        assert mean_ms <= budget_ms, (
            f"{clip} mean timing regression: {mean_ms:.2f} ms > {budget_ms} budget"
        )