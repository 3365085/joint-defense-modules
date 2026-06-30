from __future__ import annotations

from pathlib import Path


SCRIPT = Path("tools/validate_sync_delay_smoke.py")


def test_validate_sync_delay_smoke_uses_current_monitor_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "process_browser_frame" not in text
    assert "begin_preview" not in text
    assert "browser_frame_queue_policy" not in text
    assert "wait_ready_for_preview" in text
    assert "get_overlay" in text
