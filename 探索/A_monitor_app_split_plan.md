# Monitor_App (3382 行) 拆分方案（仅设计，暂不执行）

## 为什么要拆

- `tools/module_a_monitor_app.py` 目前 3382 行，单文件包含：
  - HTTP server + handler
  - HTML / CSS / JS 字符串常量（约 1800 行 INDEX_HTML）
  - MonitorState + Pipeline lifecycle
  - EvidenceRecorder / EvidenceSession
  - PPE 业务规则
  - 输入源探测工具
  - Tkinter 文件选择器

- 单文件合并后维护/阅读困难，且 A/B 合并时希望剥离 PPE 业务层以接到 B 的
  helmet 能力。

## 切分建议（按优先级）

1. **`ui/index_html.py`**：把 HTML/CSS/JS 字符串搬出去。零语义风险，纯字符串。
2. **`evidence/recorder.py`**：`ChannelEvidenceRecorder` + `MonitorEvidenceSession`。
3. **`state/monitor_state.py`**：`MonitorState` + `PipelineCache`。
4. **`ppe.py`**：`SafetyHelmetState`、`summarize_ppe_from_detections`、`draw_ppe_hud`。
   这一层**合并后会被 B 接管或替换**，所以先独立出来。
5. **`sources/` subpackage**：`open_probe_capture`、`configure_capture_runtime`、
   `test_source_connectivity`、`scan_camera_devices`、`pick_local_file`。
6. **`server.py`**：`MonitorRequestHandler` + `create_server` + `main`。

`module_a_monitor_app.py` 将只保留 `main()` 入口 + wire up。

## 拆分前必须

- 手动冒烟一次（当前 probe_web_start.py 已覆盖）。
- 保存一份全文 diff 基线。
- CI 里 `python -c "import tools.module_a_monitor_app; print('ok')"` 走一道
  smoke import（目前没有，合并后要加）。

## 拆分后收益

- 每个文件 < 500 行，可单独 review。
- PPE 层剥离后，合并时直接换成 B 的 helmet adapter。
- 接 joint_decision hook 时只需要改 `state/monitor_state.py`。

## 不在本次打磨范围

这个是合并后的重构任务（P2-A-11），今晚只留设计文档，不动代码。原因：
拆分会引入较大 diff；合并前必须确保 Web 端零功能回退，手动测试成本高。
