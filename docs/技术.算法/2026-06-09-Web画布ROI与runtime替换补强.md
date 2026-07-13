# Web画布ROI与runtime替换补强

记录时间：2026-06-09

## 问题背景

前一轮修复已经让后端 preview、overlay record、timeline lineage 和检测框坐标元数据更完整，但并行审计发现三个工程缺口：

1. Web canvas overlay 层存在函数和 timeline，但 `drawOverlay()` 实际只清空 canvas，没有绘制 `ppe_tracks`。
2. ROI redetect merge 诊断已经进入 overlay record，但 summary/CSV 没聚合，后续很难看出 ROI 框最终被保留还是被 NMS 压掉。
3. runtime replacement 的 start response/status 能从 `model_security` 旁路读到 selected 净化模型身份，但 `model_security_runtime_replacement` 本身不够自洽。

## 当前判断

后端烧录预览框和 Web canvas overlay 是两条不同链路。后端预览框缩放正确，并不自动证明 Web canvas overlay 坐标闭环。Web canvas 必须显式读取 overlay record、按 box space 缩放并绘制。

ROI redetect 诊断必须同时进入 frame/status/overlay/summary/CSV，才能支撑检测框问题复盘。

runtime replacement 需要在一个对象内同时说明 source blocked 与 selected runtime identity，避免用户或前端需要跨多个对象手动拼接。

## 代码链路依据

1. `model/src/defense/web/static/index.html`
   - `drawOverlay()`
   - `selectOverlayRecord()`
   - `overlayBoxSpace()`
   - `scaledOverlayBox()`
   - `drawOverlayTrack()`
2. `model/src/defense/diagnostics/ppe_overlay_summary.py`
   - `summarize_ppe_overlay_records()`
   - `ppe_overlay_row()`
   - `_roi_redetect_merge_summary()`
3. `model/src/defense/model_security/service.py`
   - `_runtime_replacement_target()`
   - `prepare_runtime_for_start()`
   - `trusted_purified_runtime_model()`
4. `model/src/defense/web/fastapi_app.py`
   - `_runtime_replacement_target()`
   - `_normalize_runtime_replacement()`
   - `_resolve_model_security_start()`
5. `model/src/defense/diagnostics/visual_acceptance_frames.py`
   - `export_visual_acceptance_frames()`

## 影响范围

1. Web 监控台在使用 overlay polling/timeline 时，可以实际把 `ppe_tracks` 绘制到 canvas 上。
2. ROI 复检是否命中、候选是否被丢弃、最终来源是 full frame 还是 ROI、NMS 压制来源，都能进入 summary/CSV。
3. 最终视觉验收目录默认不能复用非空目录，降低旧帧混入风险。
4. `/api/start` 返回的 `model_security_runtime_replacement.selected_runtime` 能直接表达 selected 净化模型的路径、hash、fingerprint 和报告 identity。

## 结论

本轮已完成第一轮工程闭环，但视觉结论仍不能宣布通过。Web canvas 绘制逻辑需要在真实页面中用浏览器截图和用户验收确认；检测框最终效果仍以用户亲自查看结果视频和连续 PNG 帧为准。

## 后续建议

1. 启动 Web 服务后，用浏览器实际查看 canvas overlay 是否与 MJPEG 画面同步。
2. 对用户指出的异常帧，结合 overlay JSON、ROI summary、CSV 行和连续 PNG 帧定位。
3. 如果 Web canvas 仍出现偏移，优先检查 canvas CSS object-fit、naturalWidth/naturalHeight、DPR 与 `overlay_coordinate_space.box_space_shape`。
4. 用户未验收前禁止提交；提交成功后再运行 `codegraph init -i`。
