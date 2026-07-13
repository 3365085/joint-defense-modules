# PPE 断框/托框 A4-render-plus1 行为候选复验记录

日期：2026-06-06

## 背景

A3.1/A4 dry-run 显示 `max_render_misses_plus1` 可覆盖 16 个 abrupt event，候选规模 19 个，理论风险低于 plus2。本轮将它作为第一行为候选复验：只把文件 realtime 下的 `runtime.ppe_file_realtime_max_render_misses` 显式设为 3，其余模型权重、类别语义、置信阈值、A3b 策略、shadow removal、business filter 均不改变。

## 代码链路

- `model/configs/module_a_runtime.yaml`
  - 行为复验阶段临时加入 `runtime.ppe_file_realtime_max_render_misses: 3`
- `model/src/defense/runtime/frame_processor.py`
  - 既有 `FrameProcessor.__init__()` 从 `runtime` 配置读取该字段
  - 既有 `FrameProcessor.process()` 将该值传入 `_ppe_max_render_misses(...)`
  - 既有 status 字段输出 `ppe_file_realtime_max_render_misses`
- `model/tests/test_frame_processor_status_contract.py`
  - 新增状态契约测试，确认配置值 3 能进入运行时 status

该候选只改变文件 realtime 的 held track 显示保留上限；不改变 detector 输出、PPE 类别语义、业务确认规则或 Web API 字段名。复验后该候选因指标不达标，不应留在默认运行配置中。

## 运行条件

- 目标视频：`D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频.mp4`
- profile：`desktop_rtx`
- realtime：`true`
- feature options：`static_image_enabled=true`，`a3b_sensitivity=high`
- display options：`show_boxes=true`，`show_person_boxes=false`，`show_module_hud=true`，`show_ppe_hud=true`
- B 模块：`test_bypass_model_security=true`，仅用于 PPE 行为候选复验，不写白名单，不作为生产准入结论

证据目录：

- 首轮：`runtime_evidence/ppe_box_tuning_20260605_215454/a4_render_plus1_candidate_current`
- 复验：`runtime_evidence/ppe_box_tuning_20260605_215454/a4_render_plus1_candidate_repeat1`

主要产物：

- `a4_candidate_start_response.json`
- `a4_candidate_status_polls.json`
- `a4_candidate_overlay.json`
- `a4_candidate_summary.json`
- `a4_candidate_abrupt_events.csv`
- `a4_candidate_drops.csv`
- `a4_candidate_held_tracks.csv`

## 回归验证

Pixi 聚焦回归通过：

`tests/test_model_security_bypass_and_metrics.py tests/test_ppe_postprocess.py tests/test_ppe_alert_state.py tests/test_web_detection_readiness_contract.py tests/test_web_index_preview_state_contract.py tests/test_web_prebuffer_contract.py tests/test_web_overlay_timeline_contract.py tests/test_ppe_tracking_aliases.py tests/test_frame_processor_status_contract.py`

结果：`119 passed`

## 关键指标

dry-run 基线来自 `a3_1_a4_dry_run_current`，其真实运行仍为 `max_render_misses=2`。

| 指标 | dry-run 基线 | A4-plus1 首轮 | A4-plus1 复验 |
| --- | ---: | ---: | ---: |
| overlay records | 319 | 320 | 320 |
| diagnostic max_render_misses | 2 | 3 | 3 |
| `misses_exceed_render_cap` | 99 | 88 | 84 |
| `shadow_track_removed` | 132 | 113 | 120 |
| `held_track_not_eligible` | 16 | 17 | 11 |
| abrupt event total | 43 | 45 | 46 |
| abrupt drop sum | 46 | 52 | 52 |
| abrupt with render-cap drop | 24 | 23 | 24 |
| held track instances | 69 | 81 | 87 |
| held misses >= 3 | 0 | 21 | 20 |
| max consecutive records with held tracks | 4 | 6 | 6 |

## 当前判断

1. **候选已生效**。两次 API 运行中 status 与诊断记录均确认 `ppe_file_realtime_max_render_misses=3`，不存在“配置没接进去”的问题。
2. **render cap drop 确实下降**。`misses_exceed_render_cap` 从 99 降到 88/84，说明 plus1 对 render cap 层有直接作用。
3. **但 abrupt 指标未改善，且复验更差**。abrupt event 从 43 增到 45/46，abrupt drop sum 从 46 增到 52。首轮和复验方向一致，不能视为单次采样噪声。
4. **托框/旧框风险上升**。held track instances 从 69 增到 81/87，且出现 20+ 个 misses=3 的 held 实例；max consecutive held records 从 4 增到 6。
5. **A4-plus1 暂不具备提交条件**。它降低了 render cap 删除次数，但把一部分断框延后一帧后集中掉，存在把“断框”换成“短托框后再断”的风险。

## 结论

A4-render-plus1 当前应标记为“行为候选已复验但暂不采纳”。它不应进入默认运行配置、不应进入提交，也不应进入提交前视觉验收闭环；若要继续，应转向更窄的 display-only 桥接方案，或单独进入 A3.1 strict-shadow 多人重叠专项。

后续建议：

1. 不继续扩大 render cap 到 plus2；dry-run 已显示 plus2 会保留更多旧框，当前 plus1 的 stale proxy 已经上升。
2. 若继续 A4，应设计“显示层短桥接、业务计数不变、且不累计 held=3+ track”的候选，避免改变 `ppe_class_counts`。
3. 多人重叠/共框问题应进入 A3.1 strict-shadow 行为专项，重点验证是否能减少 same-head/same-helmet 误删，同时控制重复框。
4. 远距离 person 有框但 head/helmet 无框的问题仍需独立实验，不能用本轮 render cap 结果覆盖。

## 验收状态

本轮完成了 Web API text-only 行为复验和 Pixi 回归，但由于候选指标未达到可采纳标准，未生成提交候选结果视频，也未进入连续 3 秒逐帧视觉验收。当前结论是阻断提交，而不是等待视觉验收补票。
