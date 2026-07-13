# PPE 断框/托框 A3.1 strict-shadow 行为候选复验记录

日期：2026-06-06

## 背景

A3.1 dry-run 显示 stricter same-label shadow profile 会少删一部分 same-head/same-helmet 候选，理论上可能改善多人重叠、两人共框和同类误删。本轮把 dry-run profile 临时落到实际行为复验：只对 same-head/same-helmet shadow removal 使用 `same_label_stricter_v1`，跨类 head/helmet 互斥仍保持原逻辑，模型权重、类别语义、置信阈值、A3b、render cap、business filter 均不改变。

## 代码链路

- `model/src/defense/module_a/postprocess/ppe_tracking.py`
  - `PPEDisplayTracker` 新增 `shadow_overlap_profile`
  - 默认值为 `legacy`，默认行为不变
  - `same_label_stricter_v1` 只对 `same_head_overlap` / `same_helmet_overlap` 使用严格几何删除条件
  - 新增 `shadow_profile_kept_decisions` 记录“legacy 会删、当前 profile 保留”的同类重叠候选
- `model/src/defense/runtime/frame_processor.py`
  - 从 `ppe_tracking.shadow_overlap_profile` 读取配置
  - status 输出 `ppe_shadow_overlap_profile`
- `model/src/defense/runtime/overlay_records.py`
  - overlay record 透出 `ppe_shadow_overlap_profile`
- `model/tests/test_model_security_bypass_and_metrics.py`
  - 保持 legacy shadow 决策诊断测试
  - 新增 strict profile 会保留 ambiguous same-head overlap 的测试

## 运行条件

- 目标视频：`D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频.mp4`
- 临时配置：`ppe_tracking.shadow_overlap_profile=same_label_stricter_v1`
- profile：`desktop_rtx`
- realtime：`true`
- feature options：`static_image_enabled=true`，`a3b_sensitivity=high`
- display options：`show_boxes=true`，`show_person_boxes=false`，`show_module_hud=true`，`show_ppe_hud=true`
- B 模块：`test_bypass_model_security=true`，仅用于 PPE 复验，不写白名单，不作为生产准入结论

证据目录：

`runtime_evidence/ppe_box_tuning_20260605_215454/a3_1_strict_shadow_candidate_current`

主要产物：

- `a3_1_strict_overlay.json`
- `a3_1_strict_status_polls.json`
- `a3_1_strict_summary.json`
- `a3_1_strict_abrupt_events.csv`
- `a3_1_strict_drops.csv`
- `a3_1_strict_shadow_profile_kept.csv`

## 回归验证

Pixi 聚焦回归通过：

`tests/test_model_security_bypass_and_metrics.py tests/test_ppe_postprocess.py tests/test_ppe_alert_state.py tests/test_web_detection_readiness_contract.py tests/test_web_index_preview_state_contract.py tests/test_web_prebuffer_contract.py tests/test_web_overlay_timeline_contract.py tests/test_ppe_tracking_aliases.py`

结果：`116 passed`

## 关键指标

对比基线来自 `a3_1_a4_dry_run_current`。

| 指标 | dry-run 基线 | A3.1 strict-shadow 候选 |
| --- | ---: | ---: |
| overlay records | 319 | 319 |
| profile | legacy | same_label_stricter_v1 |
| `shadow_track_removed` | 132 | 135 |
| `misses_exceed_render_cap` | 99 | 119 |
| `held_track_not_eligible` | 16 | 35 |
| `shadow_profile_kept_decision_total` | dry-run would keep 30 | actual kept 50 |
| abrupt event total | 43 | 47 |
| abrupt drop sum | 46 | 52 |
| held track instances | 69 | 62 |

## 当前判断

1. **候选已生效**。status 与 diagnostics 均确认 `same_label_stricter_v1` 全程生效，且实际记录了 50 个 `shadow_profile_kept_decisions`。
2. **它没有改善断框/托框主指标**。abrupt event 从 43 升到 47，abrupt drop sum 从 46 升到 52。
3. **重复/不稳定轨迹风险上升**。虽然 strict profile 少删了一批同类重叠，但后续 render cap 与 held eligibility 压力更高，`misses_exceed_render_cap` 从 99 升到 119，`held_track_not_eligible` 从 16 升到 35。
4. **不能把 dry-run would-keep 直接等价为收益**。实际行为中，保留更多同类重叠候选会改变后续分配、低上下文过滤与 render gate，收益没有直接转化为更稳定显示。

## 结论

A3.1 `same_label_stricter_v1` 当前应标记为“行为候选已复验但暂不采纳”。它适合作为后续多人密集专项中的诊断 profile，但不应直接作为默认行为提交，也不应进入提交前结果视频与 3 秒逐帧视觉验收。

后续如果继续多人/共框专项，应改为更窄的策略：

1. 只在存在明确双人空间证据时放松 same-head/same-helmet 删除，例如 person upper-body 分离、中心距随目标尺寸归一后足够大、或 track 历史轨迹分离。
2. 将 `shadow_profile_kept_decisions` 与后续 `business_filter/render_gate` 关联，筛掉“保留后马上被 render/business 吃掉”的无效候选。
3. 不再单纯全局放宽 same-label shadow removal。

## 验收状态

本轮完成了 Web API text-only 复验和 Pixi 回归。由于指标未达到可采纳标准，临时配置已撤回，服务已重启回默认行为；未生成提交候选结果视频，也未进入连续 3 秒逐帧视觉验收。
