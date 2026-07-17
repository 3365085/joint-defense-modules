from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from defense.diagnostics.final_acceptance_matrix import (
    AcceptanceFrame,
    CATEGORY_ORDER,
    ManifestValidationError,
    load_acceptance_manifest,
    report_exit_code,
    run_acceptance_matrix,
)


class _FakePipeline:
    def __init__(self, alerts: dict[str, set[int]]) -> None:
        self.alerts = alerts
        self.current_clip = ""
        self.frame_idx = 0
        self.reset_calls = 0
        self.started_clips: list[str] = []
        self.closed = False

    def reset(self) -> None:
        self.reset_calls += 1
        self.frame_idx = 0

    def start_clip(self, clip: Any) -> None:
        self.current_clip = clip.clip_id
        self.started_clips.append(clip.clip_id)

    def process_frame(self, frame: Any) -> tuple[None, None, dict[str, bool]]:
        del frame
        confirmed = self.frame_idx in self.alerts.get(self.current_clip, set())
        self.frame_idx += 1
        return None, None, {"alert_confirmed": confirmed}

    def close(self) -> None:
        self.closed = True


class _RuntimeFakePipeline(_FakePipeline):
    def __init__(self, alerts: dict[str, set[int]]) -> None:
        super().__init__(alerts)
        self.runtime_calls: list[dict[str, Any]] = []

    def process_runtime_frame(
        self,
        frame: Any,
        *,
        timestamp: float,
        previous_frame: Any | None,
        current_source_frame_idx: int | None,
        previous_source_frame_idx: int | None,
        previous_source_time_s: float | None,
    ) -> tuple[None, None, dict[str, bool]]:
        self.runtime_calls.append(
            {
                "frame": frame,
                "timestamp": timestamp,
                "previous_frame": previous_frame,
                "current_source_frame_idx": current_source_frame_idx,
                "previous_source_frame_idx": previous_source_frame_idx,
                "previous_source_time_s": previous_source_time_s,
            }
        )
        _, _, info = super().process_frame(frame)
        gap = (
            current_source_frame_idx - previous_source_frame_idx
            if current_source_frame_idx is not None
            and previous_source_frame_idx is not None
            else None
        )
        info["temporal_input"] = {
            "previous_frame_applied": previous_frame is not None,
            "source_gap_frames": gap,
            "strict_source_predecessor": previous_frame is not None and gap == 1,
        }
        return None, None, info


def _matrix_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        positive = category.startswith("P")
        rows.append(
            {
                "clip_id": category.lower(),
                "path": f"clips/{category.lower()}.mp4",
                "category": category,
                "label": "positive" if positive else "negative",
                "attack_start_frame": 3 if positive else None,
                "attack_end_frame": 5 if positive else None,
                "source_id": f"source-{category}",
            }
        )
    return rows


def test_load_acceptance_manifest_supports_json_and_csv(tmp_path: Path) -> None:
    rows = _matrix_rows()[:1] + _matrix_rows()[3:4]
    json_path = tmp_path / "matrix.json"
    json_path.write_text(json.dumps({"clips": rows}), encoding="utf-8")

    json_clips = load_acceptance_manifest(json_path)

    assert [clip.category for clip in json_clips] == ["P1", "N1"]
    assert json_clips[0].path == (tmp_path / "clips" / "p1.mp4").resolve()
    assert json_clips[0].attack_start_frame == 3
    assert json_clips[1].attack_start_frame is None

    csv_path = tmp_path / "matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    csv_clips = load_acceptance_manifest(csv_path)

    assert [clip.clip_id for clip in csv_clips] == ["p1", "n1"]
    assert csv_clips[1].is_positive is False


def test_run_acceptance_matrix_reports_event_metrics_and_summary(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps(_matrix_rows()), encoding="utf-8")
    output_path = tmp_path / "report.json"
    pipeline = _FakePipeline(
        {
            "p1": {1, 3, 4, 6, 7},
            "p2": set(),
            "p3": {6},
            "n1": set(),
            "n2": {2},
            "n3": set(),
            "n4": set(),
        }
    )

    report = run_acceptance_matrix(
        manifest_path,
        pipeline=pipeline,
        frame_source_factory=lambda clip: range(8),
        output_path=output_path,
    )

    clips = {row["clip_id"]: row for row in report["clips"]}
    p1 = clips["p1"]
    assert p1["pre_attack_false_positive"] is True
    assert p1["pre_attack_confirmed_frames"] == 1
    assert p1["hit_after_onset"] is True
    assert p1["first_hit_frame"] == 3
    assert p1["first_delay_frames"] == 0
    assert p1["attack_window_confirmed_frames"] == 2
    assert p1["post_attack_confirmed_frames"] == 2
    assert p1["post_attack_linger_frames"] == 2
    assert p1["confirmed_frame_count"] == 5

    assert clips["p2"]["positive_missed"] is True
    assert clips["p3"]["positive_missed"] is True
    assert clips["p3"]["post_attack_linger_frames"] == 1
    assert clips["n1"]["negative_false_positive"] is False
    assert clips["n2"]["negative_false_positive"] is True

    summary = report["summary"]
    assert summary["positive_clips"] == 3
    assert summary["positive_missed_clips"] == 2
    assert summary["positive_miss_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert summary["negative_clips"] == 4
    assert summary["negative_false_positive_clips"] == 1
    assert summary["negative_false_positive_rate"] == 0.25
    assert summary["by_category"]["P2"]["positive_missed_clips"] == 1
    assert summary["by_category"]["N2"]["negative_false_positive_clips"] == 1
    assert pipeline.reset_calls == 7
    assert pipeline.started_clips == [
        "p1",
        "p2",
        "p3",
        "n1",
        "n2",
        "n3",
        "n4",
    ]
    assert pipeline.closed is False
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"] == summary
    assert report_exit_code(report) == 1


def test_pipeline_factory_is_owned_and_truncated_positive_is_reported(
    tmp_path: Path,
) -> None:
    rows = [_matrix_rows()[0]]
    rows[0]["attack_end_frame"] = 10
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps(rows), encoding="utf-8")
    pipeline = _FakePipeline({"p1": {3}})

    report = run_acceptance_matrix(
        manifest_path,
        pipeline_factory=lambda: pipeline,
        frame_source_factory=lambda clip: range(8),
    )

    assert report["clips"][0]["status"] == "error"
    assert "before attack_end_frame 10" in report["clips"][0]["error"]
    assert report["summary"]["failed_clips"] == 1
    assert report["summary"]["positive_missed_clips"] == 0
    assert pipeline.closed is True
    assert report_exit_code(report) == 1


def test_report_exit_code_passes_only_clean_acceptance_matrix(tmp_path: Path) -> None:
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps(_matrix_rows()), encoding="utf-8")
    pipeline = _FakePipeline(
        {
            "p1": {3, 4, 5},
            "p2": {3, 4, 5},
            "p3": {3, 4, 5},
            "n1": set(),
            "n2": set(),
            "n3": set(),
            "n4": set(),
        }
    )

    report = run_acceptance_matrix(
        manifest_path,
        pipeline=pipeline,
        frame_source_factory=lambda clip: range(6),
    )

    assert report["summary"]["failed_clips"] == 0
    assert report["summary"]["positive_missed_clips"] == 0
    assert report["summary"]["negative_false_positive_clips"] == 0
    assert report["summary"]["positive_pre_attack_false_positive_clips"] == 0
    assert report["summary"]["positive_post_attack_linger_clips"] == 0
    assert report_exit_code(report) == 0


def test_report_exit_code_allows_bounded_post_attack_linger() -> None:
    report = {
        "summary": {
            "failed_clips": 0,
            "positive_missed_clips": 0,
            "negative_false_positive_clips": 0,
            "positive_pre_attack_false_positive_clips": 0,
            "positive_post_attack_linger_clips": 1,
        },
        "clips": [
            {
                "is_positive": True,
                "post_attack_confirmed_frames": 20,
                "post_attack_linger_frames": 20,
                "temporal_strict_predecessor_complete": True,
            }
        ],
    }

    assert report_exit_code(report) == 0


def test_report_exit_code_fails_on_excessive_post_attack_linger() -> None:
    report = {
        "summary": {
            "failed_clips": 0,
            "positive_missed_clips": 0,
            "negative_false_positive_clips": 0,
            "positive_pre_attack_false_positive_clips": 0,
            "positive_post_attack_linger_clips": 1,
        },
        "clips": [
            {
                "is_positive": True,
                "post_attack_confirmed_frames": 21,
                "post_attack_linger_frames": 21,
                "temporal_strict_predecessor_complete": True,
            }
        ],
    }

    assert report_exit_code(report) == 1


def test_report_exit_code_fails_on_late_post_attack_realert() -> None:
    report = {
        "summary": {
            "failed_clips": 0,
            "positive_missed_clips": 0,
            "negative_false_positive_clips": 0,
            "positive_pre_attack_false_positive_clips": 0,
            "positive_post_attack_linger_clips": 1,
        },
        "clips": [
            {
                "is_positive": True,
                "post_attack_confirmed_frames": 8,
                "post_attack_linger_frames": 3,
                "temporal_strict_predecessor_complete": True,
            }
        ],
    }

    assert report_exit_code(report) == 1


def test_report_exit_code_allows_confirmation_at_delay_limit() -> None:
    report = {
        "summary": {
            "failed_clips": 0,
            "positive_missed_clips": 0,
            "negative_false_positive_clips": 0,
            "positive_pre_attack_false_positive_clips": 0,
        },
        "runtime": {"max_first_confirmation_delay_s": 2.0},
        "clips": [
            {
                "is_positive": True,
                "first_delay_s": 2.0,
                "post_attack_confirmed_frames": 0,
                "post_attack_linger_frames": 0,
                "temporal_strict_predecessor_complete": True,
            }
        ],
    }

    assert report_exit_code(report) == 0


def test_report_exit_code_fails_on_confirmation_after_delay_limit() -> None:
    report = {
        "summary": {
            "failed_clips": 0,
            "positive_missed_clips": 0,
            "negative_false_positive_clips": 0,
            "positive_pre_attack_false_positive_clips": 0,
        },
        "runtime": {"max_first_confirmation_delay_s": 2.0},
        "clips": [
            {
                "is_positive": True,
                "first_delay_s": 2.001,
                "post_attack_confirmed_frames": 0,
                "post_attack_linger_frames": 0,
                "temporal_strict_predecessor_complete": True,
            }
        ],
    }

    assert report_exit_code(report) == 1


def test_report_exit_code_fails_on_temporal_gap_violation(tmp_path: Path) -> None:
    row = _matrix_rows()[3]
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps([row]), encoding="utf-8")
    pipeline = _RuntimeFakePipeline({"n1": set()})
    frames = [
        AcceptanceFrame("frame-10", source_frame_idx=10, source_time_s=0.4),
        AcceptanceFrame("frame-12", source_frame_idx=12, source_time_s=0.48),
    ]

    report = run_acceptance_matrix(
        manifest_path,
        pipeline=pipeline,
        frame_source_factory=lambda clip: frames,
    )

    assert report["clips"][0]["temporal_strict_predecessor_complete"] is False
    assert report_exit_code(report) == 1


def test_runtime_pipeline_receives_source_time_and_strict_previous_frame(
    tmp_path: Path,
) -> None:
    row = _matrix_rows()[3]
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps([row]), encoding="utf-8")
    pipeline = _RuntimeFakePipeline({"n1": set()})
    frames = [
        AcceptanceFrame("frame-10", source_frame_idx=10, source_time_s=0.4),
        AcceptanceFrame("frame-11", source_frame_idx=11, source_time_s=0.44),
        AcceptanceFrame("frame-12", source_frame_idx=12, source_time_s=0.48),
    ]

    report = run_acceptance_matrix(
        manifest_path,
        pipeline=pipeline,
        frame_source_factory=lambda clip: frames,
    )

    assert report["clips"][0]["status"] == "completed"
    assert report["clips"][0]["temporal_input_frames"] == 3
    assert report["clips"][0]["temporal_previous_applied_frames"] == 2
    assert report["clips"][0]["temporal_strict_predecessor_frames"] == 2
    assert report["clips"][0]["temporal_gap_violation_frames"] == 0
    assert report["clips"][0]["temporal_strict_predecessor_complete"] is True
    assert pipeline.runtime_calls == [
        {
            "frame": "frame-10",
            "timestamp": 0.4,
            "previous_frame": None,
            "current_source_frame_idx": 10,
            "previous_source_frame_idx": None,
            "previous_source_time_s": None,
        },
        {
            "frame": "frame-11",
            "timestamp": 0.44,
            "previous_frame": "frame-10",
            "current_source_frame_idx": 11,
            "previous_source_frame_idx": 10,
            "previous_source_time_s": 0.4,
        },
        {
            "frame": "frame-12",
            "timestamp": 0.48,
            "previous_frame": "frame-11",
            "current_source_frame_idx": 12,
            "previous_source_frame_idx": 11,
            "previous_source_time_s": 0.44,
        },
    ]


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"category": "N1", "label": "positive"}, "conflicts with category"),
        ({"attack_start_frame": None}, "require attack frame bounds"),
        ({"attack_end_frame": 2}, "must be >="),
    ],
)
def test_manifest_rejects_inconsistent_positive_schema(
    tmp_path: Path, patch: dict[str, Any], message: str
) -> None:
    row = _matrix_rows()[0]
    row.update(patch)
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text(json.dumps([row]), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match=message):
        load_acceptance_manifest(manifest_path)


def test_cli_only_delegates_parsed_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_final_acceptance_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("run_final_acceptance_matrix", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict[str, Any] = {}

    def fake_run(
        manifest_path: Path,
        *,
        output_path: Path,
        config_path: Path,
        profile: str,
    ) -> dict[str, Any]:
        captured.update(
            {
                "manifest_path": manifest_path,
                "output_path": output_path,
                "config_path": config_path,
                "profile": profile,
            }
        )
        return {"summary": {"failed_clips": 0}}

    monkeypatch.setattr(module, "run_runtime_acceptance_matrix", fake_run)
    manifest_path = tmp_path / "matrix.json"
    output_path = tmp_path / "report.json"
    config_path = tmp_path / "runtime.yaml"

    exit_code = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--profile",
            "test-profile",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "manifest_path": manifest_path,
        "output_path": output_path,
        "config_path": config_path,
        "profile": "test-profile",
    }
