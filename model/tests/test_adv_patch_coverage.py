from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from defense.diagnostics.adv_patch_coverage import analyze_adv_patch_coverage


MODEL_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = MODEL_ROOT / "tools" / "analyze_adv_patch_coverage.py"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(frame: int, p_adv: float, **updates: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": frame,
        "p_adv": p_adv,
        "adv_candidate_allowed": False,
        "adv_physical_support": False,
        "alert": False,
        "adv_confirmed": False,
        "adv_explicit_suppression_reason": "none",
        "joint_suppressed_reason": "none",
        "gate_scene_baseline": False,
        "gate_normal_motion": False,
        "normal_target_motion_exclusion": False,
        "normal_roi_flow_target_motion": False,
        "raw_person": 0,
        "raw_head": 0,
        "raw_helmet": 0,
    }
    row.update(updates)
    return row


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("analyze_adv_patch_coverage_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reports_rates_blockers_segments_and_anchor_presence(tmp_path: Path) -> None:
    source = tmp_path / "frames.jsonl"
    _write_rows(
        source,
        [
            _row(0, 0.20),
            _row(
                1,
                0.70,
                adv_explicit_suppression_reason="scene_baseline_normal",
                joint_suppressed_reason="scene_baseline_normal",
                gate_scene_baseline=True,
                raw_helmet=1,
            ),
            _row(
                2,
                0.80,
                adv_candidate_allowed=True,
                adv_physical_support=True,
                alert=True,
                adv_confirmed=True,
                raw_helmet=1,
            ),
            _row(
                3,
                0.90,
                adv_candidate_allowed=True,
                alert=True,
                raw_person=1,
            ),
            _row(4, 0.40, normal_target_motion_exclusion=True),
            _row(5, 0.95, adv_explicit_suppression_reason="unsupported_a3_motion"),
        ],
    )

    report = analyze_adv_patch_coverage(source, p_adv_threshold=0.65, fps=25.0)

    assert report["total_frames"] == 6
    assert report["coverage_evaluable"] is True
    assert "pass" not in report
    assert report["p_adv"]["above_threshold_frames"] == 4
    assert report["p_adv"]["above_threshold_rate"] == pytest.approx(4 / 6)
    assert report["p_adv"]["max"] == pytest.approx(0.95)
    assert report["decisions"]["adv_candidate_allowed"]["rate"] == pytest.approx(2 / 6)
    assert report["decisions"]["adv_physical_support"]["rate"] == pytest.approx(1 / 6)
    assert report["decisions"]["alert"]["rate"] == pytest.approx(2 / 6)
    assert report["decisions"]["adv_confirmed"]["rate"] == pytest.approx(1 / 6)
    assert report["first_alarm"] == {
        "frame": 2,
        "time_s": pytest.approx(0.08),
        "time_source": "frame+fps",
    }
    assert report["alarm_segments"]["count"] == 1
    assert report["alarm_segments"]["longest"]["frame_count"] == 2
    assert report["alarm_segments"]["longest"]["duration_s"] == pytest.approx(0.08)
    blocked = report["above_threshold_blocked"]
    assert blocked["frames"] == 2
    assert blocked["rate_of_above_threshold"] == pytest.approx(0.5)
    assert blocked["adv_explicit_suppression_reasons"] == {
        "scene_baseline_normal": 1,
        "unsupported_a3_motion": 1,
    }
    assert report["gates"]["gate_scene_baseline"]["rate"] == pytest.approx(1 / 6)
    assert report["gates"]["normal_target_motion_exclusion"]["rate"] == pytest.approx(1 / 6)
    assert report["yolo_anchor_presence"]["raw_helmet"]["rate"] == pytest.approx(2 / 6)
    assert report["yolo_anchor_presence"]["any_anchor"]["rate"] == pytest.approx(3 / 6)


def test_uses_source_time_and_breaks_segment_on_frame_gap(tmp_path: Path) -> None:
    source = tmp_path / "frames.jsonl"
    _write_rows(
        source,
        [
            _row(10, 0.8, source_time_s=1.0, alert=True),
            _row(11, 0.8, source_time_s=1.1, alert=True),
            _row(13, 0.8, source_time_s=1.3, alert=True),
        ],
    )

    report = analyze_adv_patch_coverage(source)

    assert report["first_alarm"] == {
        "frame": 10,
        "time_s": 1.0,
        "time_source": "source_time_s",
    }
    assert report["alarm_segments"]["count"] == 2
    assert [segment["frame_count"] for segment in report["alarm_segments"]["segments"]] == [2, 1]


@pytest.mark.parametrize(
    "content,match",
    [
        ("not-json\n", "invalid JSON"),
        (json.dumps({"frame": 0}) + "\n", "missing required field p_adv"),
        (
            json.dumps(_row(1, 0.2)) + "\n" + json.dumps(_row(1, 0.3)) + "\n",
            "strictly increasing",
        ),
    ],
)
def test_invalid_input_fails_closed(tmp_path: Path, content: str, match: str) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        analyze_adv_patch_coverage(source)


def test_cli_writes_optional_output_and_never_fabricates_pass(tmp_path: Path, capsys) -> None:
    source = tmp_path / "frames.jsonl"
    output = tmp_path / "report.json"
    _write_rows(source, [_row(0, 0.7, alert=True)])
    tool = _load_tool()

    assert tool.main([str(source), "--output", str(output)]) == 0

    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == saved
    assert saved["coverage_evaluable"] is True
    assert "pass" not in saved
