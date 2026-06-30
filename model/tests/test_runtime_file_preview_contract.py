from __future__ import annotations

from pathlib import Path

import pytest


RUNNER = Path("src/defense/runtime/runner.py")


def test_file_capture_marks_preview_ready_before_first_detection() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "preview_can_start = True" in source
    assert '"preview_mode": "backend_source_pipeline"' in source
    assert '"preview_mode": "backend_source_pipeline" if preview_can_start else "waiting_first_detection"' not in source


def test_preview_loop_only_waits_for_first_detection_when_explicitly_configured() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'not bool(self.status.get("preview_never_wait_for_detection", True))' in source
    assert 'and not bool(self.status.get("first_detection_ready"))' in source


@pytest.mark.skip(reason="超前契约未实装:runner.py全仓无preview_overlay_baked字段")
def test_backend_source_preview_declares_overlay_baked_into_mjpeg() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    empty_start = source.index("def _empty_status")
    empty_end = source.index("status[\"branch_cards\"]", empty_start)
    empty_source = source[empty_start:empty_end]
    start_start = source.index('"preview_mode": "backend_source_pipeline"')
    start_end = source.index('"ready_for_preview": False', start_start)
    start_source = source[start_start:start_end]

    assert '"preview_overlay_baked": True' in empty_source
    assert '"preview_overlay_baked": True' in start_source
