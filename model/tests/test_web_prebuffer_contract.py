from __future__ import annotations

from pathlib import Path


HTML = Path("src/defense/web/static/index.html")
MODEL_SECURITY_HTML = Path("src/defense/web/static/model_security.html")


def test_frontend_does_not_call_browser_frame_detection() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "/api/detect-frame" not in html
    assert "detectCurrentVideoFrame" not in html
    assert "captureVideoJpeg(video)" not in html


def test_mp4_uses_backend_preview_stream_not_native_video_pipeline() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert 'startMjpegPreview(status);' in html
    assert '/api/runs/${runId}/preview.mjpg' in html
    assert 'function updateStartButtonState(status)' in html
    assert 'id="playPauseBtn"' in html
    assert 'id="seekSlider"' in html
    assert 'id="speedSelect"' in html
    assert 'function controlRun(action' in html
    assert '/api/runs/${runId}/control' in html
    assert "nativeVideo" not in html
    assert "analysisVideo" not in html
    assert "/api/begin-preview" not in html


def test_backend_source_pipeline_debug_state_is_reported() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "controlRun(" in html
    assert "backend_latest_only" in html or "pipelineText" in html
    assert "检测输出${detectFps}fps" in html
    assert "检测算力${computeFps}fps" in html
    assert "丢弃旧检测帧 ${dropped}" in html


def test_stop_still_clears_preview_and_overlay_polling() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "function stopOverlayPolling()" in html
    assert "stopOverlayPolling();" in html
    assert 'await api("/api/stop", {})' in html
    assert "activePreviewRunId = 0;" in html
    assert 'removeAttribute("data-run-id")' in html


def test_frontend_does_not_reference_removed_realtime_control() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert '$("realtime")' not in html


def test_progress_controls_are_only_shown_for_local_mp4_runs() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "const showProgressControls = isFile && duration > 0 && (running || !!status?.source_ended);" in html
    assert '$("runControls").style.display = showProgressControls ? "grid" : "none";' in html
    assert '$("seekSlider").disabled = !running || !!status?.source_ended;' in html


def test_camera_source_uses_camera_selector_value() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert 'return $("cameraSelect").value || "0";' in html
    assert 'return $("cameraSelect").value || $("sourceValue").value || "0";' not in html


def test_status_refresh_uses_adaptive_timeout() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "setInterval(refresh, 200)" not in html
    assert "refreshIntervals" in html
    assert "scheduleRefresh(nextRefreshMs)" in html


def test_monitor_page_does_not_expose_runtime_model_controls() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert 'id="profile"' not in html
    assert 'id="enableCustomModel"' not in html
    assert 'id="customModelPath"' not in html
    assert 'id="customModelBackend"' not in html
    assert 'id="customModelFamily"' not in html
    assert '$("profile")' not in html
    assert '$("enableCustomModel")' not in html
    assert "customModelDraftDirty" not in html


def test_monitor_page_routes_stored_custom_model_by_bypass_switch() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert 'function getStored(key, fallback = "")' in html
    assert 'const DEFAULT_RUNTIME_PROFILE = "desktop_rtx";' in html
    assert "function currentRuntimeProfile()" in html
    assert 'localStorage.getItem("moduleA.lastProfile")' in html

    custom_model_start = html.index("function customModelOptions()")
    custom_model_end = html.index("function runtimeModelSelectionSummary", custom_model_start)
    custom_model_function = html[custom_model_start:custom_model_end]
    test_custom_model_keys = (
        "moduleA.lastCustomModelEnabled",
        "moduleA.lastModelPath",
        "moduleA.lastCustomModelFamily",
        "moduleA.lastCustomClassNames",
    )
    for key in test_custom_model_keys:
        assert f'"{key}"' in custom_model_function
        assert f'localStorage.getItem("{key}")' in custom_model_function
    assert "if (!modelSecurityBypassEnabled())" not in custom_model_function
    assert 'const startEndpoint = bypassModelSecurity ? "/api/test/start" : "/api/start";' in html
    assert "body.test_bypass_model_security = true;" in html
    assert 'getStored("moduleA.testBypassModelSecurity", "false") === "true"' in html
    assert "profile: currentRuntimeProfile()" in html
    assert "custom_model: customModel" in html
    assert "backendLabel(status)" in html


def test_security_center_checkbox_controls_main_monitor_bypass() -> None:
    monitor_html = HTML.read_text(encoding="utf-8")
    security_html = MODEL_SECURITY_HTML.read_text(encoding="utf-8")

    assert '"/api/test/start"' in monitor_html
    assert 'id="testBypassModelSecurity"' in security_html
    assert 'id="testBypassStartBtn"' not in security_html
    assert '"/api/test/start"' not in security_html
    assert "test_bypass_model_security" in monitor_html
    assert 'getStored("moduleA.testBypassModelSecurity", "false") === "true"' in monitor_html
    assert '$("testBypassModelSecurity").checked = getStored("moduleA.testBypassModelSecurity", "false") === "true";' in security_html
    assert "开启后，返回主监控台点击“开始检测”" in security_html


def test_file_preview_no_longer_waits_for_first_detection() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "不会先播放没有检测框的视频" not in html
    assert "正在加载模型/引擎并准备后端预览。" in html


def test_file_metrics_remain_visible_after_source_end() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "running || !!status?.source_ended" in html
    assert "检测算力${computeFps}fps" in html
    assert "检测覆盖${coverage}%" in html
    assert "丢弃旧检测帧 ${dropped}" in html


def test_security_center_exposes_runtime_model_controls() -> None:
    html = Path("src/defense/web/static/model_security.html").read_text(encoding="utf-8")
    assert 'id="profileSelect"' not in html
    assert 'id="enableCustomModel"' in html
    assert 'id="customModelPath"' in html
    assert 'id="sourcePtPath"' not in html
    assert 'id="customModelBackend"' not in html
    assert 'id="customModelFamily"' in html
    assert 'id="browseModelBtn"' in html
    assert 'id="browseSourcePtBtn"' not in html
    assert 'id="saveModelConfigBtn"' not in html
    assert 'id="resetModelConfigBtn"' not in html
    assert 'id="resolvedBackend"' in html
    assert 'id="resolvedFamily"' in html
    assert "const DEFAULT_RUNTIME_PROFILE = \"desktop_rtx\";" in html
    assert 'mode: "model"' in html
    assert "moduleA.lastProfile" in html
    assert "moduleA.lastCustomModelEnabled" in html
    assert "moduleA.lastModelPath" in html
    assert "moduleA.lastSourcePtPath" not in html
    assert "profile: currentRuntimeProfile()" in html
    assert "custom_model: customModelOptions()" in html
    assert "后端：" in html
    assert "模型族：" in html
    assert "配置变更会立即同步到主监控台" in html
    assert "主监控台会将其作为正常运行模型并由B模块执行安全准入" in html
    assert "按运行模型哈希自动核对源 PT 绑定" in html
    assert 'const backend = "auto";' in html
    assert '$("testBypassModelSecurity").disabled = !enabled;' in html
    assert 'setStored("moduleA.testBypassModelSecurity", "false")' in html
    assert 'production_authoritative: "生产权威模型"' in html


def test_monitor_page_displays_actual_purified_or_bypassed_runtime() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "普通启动将由B模块替换为净化可信模型" in html
    assert "测试绕过B模块 · 实际直接运行原始模型" in html
    assert "B模块已替换为净化运行模型" in html


def test_main_page_keeps_model_security_as_entry_only() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert 'id="openModelSecurityCenterBtn"' in html
    assert 'id="modelSecurityRisk"' not in html
    assert 'id="modelSecurityScanTime"' not in html
    assert 'id="refreshModelSecurityBtn"' not in html
    assert 'id="testBypassModelSecurity"' not in html
    assert 'if (!customModelOptions().enabled) return false;' in html
    assert 'production_authoritative: "生产权威模型"' in html
    assert "默认生产运行使用唯一权威 PT 及其哈希绑定 TensorRT engine" in html
    assert "详细扫描、净化、报告和白名单管理请打开安全中心" in html


def test_security_logs_page_is_manual_refresh_and_clearable() -> None:
    html = Path("src/defense/web/static/model_security_logs.html").read_text(encoding="utf-8")
    assert "清空日志" in html
    assert "clearLogs" in html
    assert "/api/model-security/logs/clear" in html
    assert "pauseBtn" not in html
    assert "setTimeout" not in html
    assert "setInterval" not in html
    assert 'id="refreshBtn"' in html
    assert "/api/model-security/logs" in html


def test_status_panel_can_use_latest_overlay_record_as_live_fallback() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "function mergeOverlayStatusForPanel(status)" in html
    assert "status = mergeOverlayStatusForPanel(status);" in html
    assert "latestOverlayStatusRecord()" in html
    assert "ppe_head_count: Number(status.ppe_head_count || 0)" in html


def test_frontend_stops_preview_polling_after_source_ended() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "function stopEndedPreview()" in html
    assert "if (status.source_ended) {" in html
    assert "stopOverlayPolling();" in html
    assert 'if (!status.source_ended) drawOverlay(status);' in html
