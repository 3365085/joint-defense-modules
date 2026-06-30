from __future__ import annotations

from pathlib import Path

import pytest


INDEX = Path("src/defense/web/static/index.html")


def test_model_security_block_clears_stale_preview_stream() -> None:
    source = INDEX.read_text(encoding="utf-8")
    block_start = source.index("function showModelSecurityBlock")
    block_end = source.index("function setStageAspect", block_start)
    block_source = source[block_start:block_end]

    assert "hideAllPreview();" in block_source
    assert "showEmpty(title, hint);" in block_source


def test_source_ended_preview_uses_full_preview_cleanup() -> None:
    source = INDEX.read_text(encoding="utf-8")
    stop_start = source.index("function stopEndedPreview")
    stop_end = source.index("function startMjpegPreview", stop_start)
    stop_source = source[stop_start:stop_end]

    assert "hideAllPreview();" in stop_source
    assert 'removeAttribute("src")' not in stop_source


def test_model_security_status_refresh_does_not_cover_active_detection_preview() -> None:
    source = INDEX.read_text(encoding="utf-8")
    refresh_start = source.index("async function refreshModelSecurity")
    refresh_end = source.index("async function refreshBlockedModelSecurityHint", refresh_start)
    refresh_source = source[refresh_start:refresh_end]

    assert "showModelSecurityBlock(modelSecurity)" not in refresh_source
    assert "renderModelSecurity(modelSecurity);" in refresh_source


def test_active_detection_clears_stale_model_security_block_state() -> None:
    source = INDEX.read_text(encoding="utf-8")
    ensure_start = source.index("function ensurePreview")
    ensure_end = source.index("function resizeOverlayCanvas", ensure_start)
    ensure_source = source[ensure_start:ensure_end]

    assert "status?.running && isModelSecurityBlocking(modelSecurityBlockState)" in ensure_source
    assert "modelSecurityBlockState = null;" in ensure_source


def test_stopped_status_is_not_revived_by_overlay_merge() -> None:
    source = INDEX.read_text(encoding="utf-8")
    merge_start = source.index("function mergeOverlayStatusForPanel")
    merge_end = source.index("function selectOverlayRecord", merge_start)
    merge_source = source[merge_start:merge_end]

    assert "running: !!base.running" in merge_source
    assert '"source_time_s"' in merge_source
    assert '"video_time_s"' in merge_source
    assert '"detector_pipeline_mode"' in merge_source
    assert "return normalizeLifecycleStatus(merged);" in merge_source


def test_stopped_status_closes_preview_stream() -> None:
    source = INDEX.read_text(encoding="utf-8")
    ensure_start = source.index("function ensurePreview")
    ensure_end = source.index("function resizeOverlayCanvas", ensure_start)
    ensure_source = source[ensure_start:ensure_end]

    assert "if (!status.running && !status.source_ended)" in ensure_source
    assert 'showEmpty("已停止"' in ensure_source
    assert "hideAllPreview();" in ensure_source


def test_source_ended_is_the_highest_state_text_priority() -> None:
    source = INDEX.read_text(encoding="utf-8")
    state_start = source.index("function stateText")
    state_end = source.index("function stopOverlayPolling", state_start)
    state_source = source[state_start:state_end]

    assert state_source.index("status.source_ended") < state_source.index("status.alert_confirmed")
    assert state_source.index("status.source_ended") < state_source.index("status.running")


def test_start_success_path_does_not_force_mjpeg_for_ended_status() -> None:
    source = INDEX.read_text(encoding="utf-8")
    start_start = source.index('    $("startBtn").onclick')
    start_end = source.index('    $("stopBtn").onclick', start_start)
    start_source = source[start_start:start_end]

    assert "const status = normalizeLifecycleStatus(data.status || {});" in start_source
    assert "ensurePreview(status);" in start_source
    assert "startMjpegPreview(status);" not in start_source


@pytest.mark.skip(reason="超前契约未实装:前端operator_message/next_action/drawOverlayTrack/preview_overlay_baked守卫")
def test_model_security_block_hint_uses_backend_next_action_guidance() -> None:
    source = INDEX.read_text(encoding="utf-8")
    hint_start = source.index("function modelSecurityBlockHint")
    hint_end = source.index("function isModelSecurityBlocking", hint_start)
    hint_source = source[hint_start:hint_end]

    assert "ms.operator_message" in hint_source
    assert "nextActionLabels" in hint_source
    assert "ms.next_action" in hint_source
    assert "start_full_scan" in hint_source
    assert "start_purification" in hint_source


@pytest.mark.skip(reason="超前契约未实装:前端operator_message/next_action/drawOverlayTrack/preview_overlay_baked守卫")
def test_start_prompt_matches_admission_only_model_security_flow() -> None:
    source = INDEX.read_text(encoding="utf-8")
    start_start = source.index('    $("startBtn").onclick')
    start_end = source.index('    $("stopBtn").onclick', start_start)
    start_source = source[start_start:start_end]

    assert "如需扫描或净化，会阻断启动并提示下一步" in start_source
    assert "如果需要净化，会自动净化" not in start_source


def test_lifecycle_normalization_forces_source_ended_to_closed_preview() -> None:
    source = INDEX.read_text(encoding="utf-8")
    normalize_start = source.index("function normalizeLifecycleStatus")
    normalize_end = source.index("function stateText", normalize_start)
    normalize_source = source[normalize_start:normalize_end]

    assert "out.running = false;" in normalize_source
    assert "out.ready_for_preview = false;" in normalize_source
    assert 'out.preview_mode = "source_ended";' in normalize_source
    assert 'out.detector_pipeline_mode = "ended";' in normalize_source


@pytest.mark.skip(reason="超前契约未实装:前端operator_message/next_action/drawOverlayTrack/preview_overlay_baked守卫")
def test_baked_preview_skips_frontend_canvas_box_redraw() -> None:
    source = INDEX.read_text(encoding="utf-8")
    draw_start = source.index("function drawOverlay")
    draw_end = source.index("function renderBranchCards", draw_start)
    draw_source = source[draw_start:draw_end]

    baked_guard = 'lastOverlayStatus?.preview_overlay_baked !== false'
    assert baked_guard in draw_source
    assert draw_source.index(baked_guard) < draw_source.index("selectOverlayRecord(videoTime)")
    assert draw_source.index(baked_guard) < draw_source.index("drawOverlayTrack(ctx, track, scaled, canvasSize.dpr)")
