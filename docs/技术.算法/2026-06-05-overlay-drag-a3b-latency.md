# Overlay 拖框与 A3b 响应时间记录

## 背景

用户在 Web 监控台指出检测框仍有拖框现象，并补充希望 A3b 响应更快，当前体感约 3 秒。该问题涉及预览帧、后端 overlay 选帧、A3b 时序窗口和 PPE 框渲染，属于检测效果和时序展示问题。

## 当前判断

- 拖框根因已定位为后端文件实时预览与前端 overlay 时间线策略不一致：前端在文件实时预览中不保留未匹配旧轨迹，但后端将 overlay 烘进 MJPEG 时仍对插值记录使用 `keep_unmatched_tracks=True`。
- 当后一条检测记录已经因 A3b 媒体 ROI 抑制而变为空框时，后端仍可能在两条记录之间短暂带上前一条 person/head/helmet 旧框，表现为框拖在画面上。
- 本轮未调整模型权重、类别语义、PPE 语义、A3b 阈值或确认策略。

## 代码链路依据

- 后端预览渲染链路：`src/defense/runtime/runner.py::_preview_render_loop` -> `_select_preview_overlay` -> `_render_backend_preview` -> `render_preview`。
- 前端策略链路：`src/defense/web/static/index.html::selectOverlayRecord` 在文件实时预览下设置 `keepUnmatchedTracks: false`。
- 共用插值语义：`src/defense/web/overlay_timeline.py::interpolate_overlay` 在 `keep_unmatched_tracks=False` 时会丢弃下一条记录中已经不存在的轨迹。
- 修复后，后端 `_select_preview_overlay` 在文件实时预览下也使用 `keep_unmatched_tracks=not file_realtime_preview`，与前端策略对齐。

## A3b 响应时间证据

对目标视频 `5e145bf778577e75118502e263d00c41.mp4` 使用同一 Web API 重跑：

- 第一条 A3b ROI bbox：`source_time_s=1.0667`，状态 `observing`。
- 第一条 `suspect/triggered`：`source_time_s=1.2333`，来源 `observed_window`。
- 观察窗口证据：`observed_only_window_hits=3` 后进入疑似状态。

因此该样本上后端 A3b 首次观察和疑似触发均明显早于 3 秒。用户体感的慢更可能来自视觉拖框、状态刷新/展示、或只把 `suspect` 后的持续提示当作有效响应。继续缩短可见响应应优先优化展示语义，例如让 `observing` 状态更明确，而不是直接降低 A3b 阈值。

## 验证记录

- 回归测试：`pixi run cmd /C "cd /D model && set PYTHONPATH=src&& python -m pytest -q tests/test_model_security_bypass_and_metrics.py"`，结果 `26 passed`。
- Smoke：`pixi run smoke`，结果 `306 passed, 3 skipped`。
- Web API 复跑：A3b bbox 在 `1.0667s` 出现，`suspect` 在 `1.2333s` 出现，bbox 后记录 `ppe_tracks=[]` 且 `ppe_source_auth_media_suppressed=true`。
- MJPEG 视觉抽检目录：`D:\联合防御模块\runtime\verification\dragbox_20260605_030757`。
- 抽检帧：`4.033s`、`6.100s`、`8.133s` 三帧均未见 person/head/helmet 旧框拖尾，底部 PPE 计数为 0，并显示 `source_auth_media_roi_suppressed`。

## 影响范围

- 影响文件实时预览中后端烘焙 overlay 的显示策略。
- 不影响检测结果记录、PPE 业务判定、A3b 时序状态机和模型推理。
- 对 RTSP/摄像头等非文件实时预览路径保持原策略。

## 后续建议

- 若用户仍觉得 A3b 慢，下一步应单独做展示层优化：把 `observing` 与 `suspect` 的 UI 文案、颜色和 HUD 状态分离，明确“已观察到翻拍迹象”和“已形成疑似告警”。
- 若要进一步压低 A3b 算法确认时间，应作为显式行为调参任务处理，并用负样本视频验证误报率，不能只针对当前样本降阈值。
