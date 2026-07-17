# Module A 正常视频确认误报修复与最终复核记录

## 问题背景

当前生产 rebuilt 链路曾在正常视频上真实产生确认告警，不能以 “A3b clean FP=0”
代表完整 Module A 正常。修复前重跑 27 段 clean（5997 帧）的结果为：

- `alert_confirmed` 视频：3/27；
- `alert_confirmed` 帧：168；
- `single_frame_suspicious` 视频：9/27；
- `single_frame_suspicious` 帧：134。

真实 warehouse 正常视频的 Web/latest-only 20 秒复现中，曾产生 229 条
`attack_detected`、216 条 `alert_confirmed` 和 2 个 Module A evidence event，
输出视频实际显示红色“告警确认”，不是统计口径误会。

原始报告：

- `runtime/diagnostics/module_a_20260715_final/module_a_clean27_postfix_report.json`
- `runtime/diagnostics/module_a_20260715_final/module_a_clean27_postfix_frames.jsonl`
- `runtime/diagnostics/module_a_20260715_final/module_a_clean27_postfix_metrics.json`
- `runtime/diagnostics/module_a_20260715_final/clean_warehouse_runtime_preview_overlay.json`
- `runtime/diagnostics/module_a_20260715_final/clean_warehouse_joint_first240.json`

同期 21 段物理攻击基线命中 20/21、2555 个 alert frames。本轮属于明确授权的
behavior-tuning 与确定性程序修复；未修改模型权重、类别语义、PPE 语义，也未通过
隐藏事件、降低颜色级别或改 UI 文案规避确认误报。

## 当前判断与代码链路

### sustained escalation 与 alert hold

Pexels 正常视频已证明旧 sustained escalation 可在以下状态下仅凭 raw `p_adv`
强制确认：

- `candidate_source=NONE`；
- `adv_candidate_allowed=false`；
- `scene_baseline_normal=true`；
- `adv_physical_support=false`。

这与逐帧门控“当前场景正常、应抑制”的结论冲突。修复后 sustained escalation
必须仍有允许的 candidate 或独立物理支持；`scene_baseline_normal=true` 且没有
独立物理支持时不得升级。

旧 alert hold 可只凭 `p_adv >= theta_adv` 跨通道刷新，在有效 candidate 消失后
继续延长确认。修复后 hold 只能由原确认通道当前仍有效的 candidate 刷新：

- adv hold 只接受 adv candidate；
- blind hold 只接受 blind candidate；
- media hold 只接受 media candidate；
- candidate 消失后按固定 hold 递减退出。

公开诊断保留：

- `sustained_adv_has_independent_support`
- `sustained_adv_scene_allowed`
- `alert_hold_refresh_signal`
- `alert_hold_refresh_source`

### adv candidate bridge 旁路复审

独立 reviewer 发现旧 candidate bridge 在近期出现过合法 candidate 后，只要 raw
`p_adv_triggered=true` 就可把 `adv_candidate_allowed=false` 的当前帧重新标为
candidate。这会旁路 `scene_baseline_normal`、`normal_target_motion_exclusion`
等明确抑制，并可能继续刷新 adv hold。

最终修复不是直接关闭 bridge，因为强制要求“当前帧 physical support”会把攻击召回从
20/21 降至 15/21。当前实现改为保留最近有效 candidate 的 physical-support 血统：

- `normal_target_motion_exclusion` 始终禁止桥接，即使历史有 support；
- 其他明确抑制只有在当前和最近有效 candidate 都没有 physical support 时禁止桥接；
- 桥接仍要求 raw trigger，并要求当前特征/目标上下文或近期独立 physical support；
- reset 时清除 bridge support 血统；
- diagnostics 暴露 `adv_candidate_bridged`、`bridge_eligible`、`bridge_blocked`、
  `recent_physical_support` 和 remaining。

因此最终 bridge 不能仅凭 raw `p_adv` 推翻正常人员运动抑制，同时保留了
adv_patch/occlusion 的短时证据连续性。

### A3 正常目标运动

仓库误报关键帧的 A3 特征为：

- `flow_local_anomaly_ratio` 约 0.056；
- `flow_shape_score` 约 0.944；
- ROI residual contrast / motion gap 约 1.52 / 1.70；
- YOLO 只有 helmet/head 小框，没有 person 人体框。

`_compute_a3` 原先只在 residual contrast、motion gap 同时较低，或异常像素
ROI coverage 大于等于 0.50 时，将目标运动压到 0.22。小头部框无法覆盖人体运动，
导致正常行走在低全局变化下被解释为 A3 攻击。

处理链路：

1. `_compute_a3` 输出已有内部计算量 `flow_roi_coverage_ratio`。
2. `_joint_decision` 增加 `normal_target_motion_exclusion`：
   - 目标相关；
   - ROI coverage 大于等于 0.15；
   - `max(A1, A2) < 0.55`；
   - `frame_diff_global < 0.018`；
   - `exposure_delta < 0.006`。
3. 命中该排除时，A3 不进入 `adv_candidate_allowed`，也不作为 sustained escalation
   的 `adv_physical_support`，避免被持续升级重新拉回告警。

这不是把所有目标相关光流当成正常，而是要求低全局变化、低曝光变化、A1/A2 无强支持、
且异常光流确实覆盖检测 ROI 的联合条件。攻击侧是否覆盖更多真实人员运动形态，仍需独立
锁定集确认。

### Branch B motion blur

安全帽摘戴 clean 中，正常动作造成清晰度基线下降和检测置信度丢失，
`p_blind` 可达约 0.81；而部分真实攻击的 `p_blind` p95 仅约 0.31 至 0.35，
二者重叠，不能通过全局抬高 `theta_blind` 分离。

motion blur 现要求至少一个独立支持：

- 非 `motion_blur` 类型；或
- `sharp_drop >= 0.85`；或
- `sharpness <= 80`；或
- `contrast_drop >= 0.18`；或
- `sharp_drop_short >= 0.05`；或
- `frame_diff_global >= 0.018`；或
- `exposure_delta >= 0.010`；或
- `overexposure_ratio >= 0.10`。

无独立支持的 motion blur 将 `p_blind` 限制到 0.40。该
`blind_independent_support` 同时门控：

- 单帧 blind candidate；
- blind sustained degradation 累计；
- `_update_scene_baseline` 的 blind suspect baseline freeze。

这避免正常摘戴/转头先冻结基线，再依靠目标丢失持续自强化为致盲告警。

## 影响范围

- 核心判定修复位于：
  - `model/src/defense/module_a/rebuilt/detector.py`
- 真实状态与显示归因修复位于：
  - `model/src/defense/runtime/frame_processor.py`
  - `model/src/defense/runtime/overlay_records.py`
  - `model/src/defense/visualization/overlay.py`
- 诊断门禁扩展位于：
  - `model/src/defense/diagnostics/a3b_heldout.py`
- 新增/强化测试包括：
  - `model/tests/test_module_a_alert_policy_hardening.py`
  - `model/tests/test_rebuilt_algorithm_hardening.py`
  - `model/tests/test_module_a_alert_display_contract.py`
  - `model/tests/test_a3b_heldout_tool_contract.py`
  - `model/tests/test_runtime_config_invariants.py`

显示层没有改变 evidence 活跃条件，也没有隐藏 raw `p_adv`、`p_blind`、疑似状态、
失败门控或事件状态；只修正 detector hold 冒充新鲜确认、blind 通道误显示为 `p_adv`
等归因错误。

## 验证结果

聚焦与相关回归测试：

- 算法 scope：29 项聚焦测试通过；
- 相关 Module A/A3b 合同回归：86 项通过；
- 主线程新增策略与显示/配置聚焦测试：62 项通过；
- 首次最终 smoke 暴露了 preframe effective-config schema 未同步的真实契约缺口：
  `FrameProcessor` 已公开 12 个新增策略有效值，但 `MonitorEngine` 启动前空状态仍缺字段。
  已同步 `model/src/defense/runtime/runner.py` 与两处精确 schema 测试；
- 修复后最终 `pixi run smoke`：626 passed、34 skipped；
- 当前改动 Python 文件 Ruff、内联 JavaScript `node --check`、`git diff --check`
  均通过。全仓 Ruff 仍有 451 项既有/第三方/范围外问题，不属于本轮新增。

### 完整 48 段生产口径

最终报告：

`runtime/diagnostics/module_a_20260715_final/module_a_a3b_heldout_bridge_guard_v4_final.json`

最终代码、配置与该行为报告的补充哈希绑定：

`runtime/diagnostics/module_a_20260715_final/module_a_a3b_heldout_bridge_guard_v4_final_binding.json`

结果：

- 27 clean：
  - Module A `alert_confirmed` 视频 0/27；
  - `alert_confirmed` 帧 0；
  - Module A evidence-condition 视频/帧 0/0；
  - `single_frame_suspicious` 视频 6/27、35 帧。
- 21 physical attacks：
  - 命中 20/21；
  - alert frames 2058；
  - 唯一 miss 仍为既有 `clip_0041 adv_patch`。
- A3b：
  - clean FP 0；
  - 非 A3b wrong-channel 0；
  - backend/worker/temporal health error 0；
  - `gate_failures=[]`。

诊断 gate 已补充攻击召回门：完整 21 段物理攻击若低于 20 段命中会直接失败，
避免再次出现 clean 为 0、但攻击召回已降至 15/21 时仍显示 `gate_failures=[]`。

相对修复前攻击基线，视频级命中保持 20/21。alert frames 从 2555 降至 2058，
主要来自 raw `p_adv` 不再无限刷新 hold；不能只按帧数下降解释为视频级召回下降。

### A3b 目标视频

目标视频：

`素材/视频中出现干扰视频/5e145bf778577e75118502e263d00c41.mp4`

确定性 FrameProcessor 报告：

`runtime/diagnostics/module_a_20260715_final/a3b_target_frameprocessor_alertfix_final.json`

结果：

- first A3b trigger：frame 30 / 1.0 s；
- trigger source：`confirmed_track`；
- A3b trigger frames：42；
- Module A confirmed frames：0；
- Module A evidence condition：0；
- health errors：0；
- `gate_failures=[]`。

最终又使用同一目标视频实跑生产 `MonitorEngine`、`realtime=True`、
`backend_latest_only/latest_only`：

`runtime/diagnostics/a3b_latest_only_bridge_guard_v4_final/`

- 首次 A3b 逻辑确认：source frame 33 / 1.100 s；
- 画面从 source frame 30 / 1.000 s 附近已显示稳定 A3b 框；
- trigger source：`confirmed_track`；
- A3b triggered overlay records：27；
- Module A confirmed/evidence：0/0；
- backend/A3b health error：0；
- dropped detection frames：5，未造成触发失败；
- `stale_overlay_dropped=0`，正常到达 `source_ended=true`。

相对确定性全帧 frame 30 / 1.0 s，真实 latest-only 晚 3 个源帧 / 0.100 s，
未出现此前“难触发”或数秒级延迟。候选框和确认框前后关键帧均已保存在该目录。

### warehouse 真实 Web/latest-only

最终真实链路报告：

- `runtime/diagnostics/module_a_20260715_final/clean_warehouse_runtime_preview_bridge_guard_v4_desktop_final.mp4`
- `runtime/diagnostics/module_a_20260715_final/clean_warehouse_runtime_preview_bridge_guard_v4_desktop_final_overlay.json`
- `runtime/diagnostics/module_a_20260715_final/clean_warehouse_runtime_preview_bridge_guard_v4_desktop_final_summary.json`

20.12 秒、425 条 overlay 的结果：

- `attack_detected=0`；
- `alert_confirmed=0`；
- `attack_state_active=0`；
- Module A evidence saved/recent events：0/0；
- A3b triggered：0；
- PPE warning：0；
- backend/A3b health errors：0；
- dropped detection/preview frames：0/0；
- production mode：`backend_latest_only` + `latest_only`。

raw `p_adv` 没有被隐藏：该正常视频 `p_adv max≈0.987`、`p95≈0.975`，但由于
candidate/scene/physical-support gate 不成立，没有再错误升级为确认。原误报附近
source frame 200 和 760 的最终关键帧抽检也没有红色“告警确认”：

- `runtime/diagnostics/module_a_20260715_final/clean_warehouse_bridge_guard_v4_keyframes_src200.jpg`
- `runtime/diagnostics/module_a_20260715_final/clean_warehouse_bridge_guard_v4_keyframes_src760.jpg`

## A4/XGBoost 与过拟合风险

当前 `a4_classifier.pkl` 仍能加载，但尚未充分绑定：

- 训练 feature names/order；
- schema version；
- 第 20 维真实训练语义；
- 预处理版本；
- detector/ROI 来源；
- 训练数据 identity。

正常 warehouse 的 `p_adv` 长期高于 `theta_adv=0.65`，说明概率校准、生产分布漂移、
正常样本覆盖或 schema 匹配仍有风险。强制 classifier fallback 的实验反而得到
clean confirmed 16/27、attack hit 19/21，因此本轮没有禁用 A4，也没有在已参与策略
选择的同一 heldout 上盲目重训。

该风险不能由当前 0/27 dev 结果消除。后续若改 A1/A2/A3 特征、flow 帧间语义、
ROI/detector 来源、A4 特征顺序/含义或归一化，必须重新建立训练资产绑定，并在新的
source-lineage 独立锁定集上验证或重训。

## 结论与后续建议

本轮确定性程序修复已在当前 27 clean dev 矩阵达到 `alert_confirmed 0/27`、
Module A evidence 0，并保持 21 attack 视频级命中 20/21；warehouse 真实
latest-only 和 A3b 目标视频也没有出现相互回退。

但当前 heldout 已参与策略选择，且存在重复、lineage、编码器和固定攻击 onset 风险，
因此只能说明“当前程序逻辑冲突已关闭并通过现有工程门”，不能宣称已证明真实场景泛化，
也不能宣称不存在过拟合。最终交付仍需用户本人完成 Web 验收并明确授权；在此之前不得
提交。若新独立 clean 再出现确认误报，应保留 raw 分数、candidate、失败门控和事件状态，
继续定位根因，不得改用 UI 掩盖、禁用证据保存或简单全局抬阈值的方式绕过。

## Web 帧数口径补充

用户 Web 实跑目标视频后看到事件卡片中的“帧 2 - 12”，容易误解为整段视频只处理了
12 帧。运行状态复核表明该次实际 source frame 为 278/280，视频正常到达末尾；
“2 - 12”是某个 PPE/A3b **事件源帧区间**，证据视频又会按 evidence FPS 抽样保存，
都不等于视频总帧或检测总帧。

前端已将相关文字明确改为：

- `事件源帧 X - Y（不是视频总帧）`
- `证据抽样关键帧 N 张`

该修改只澄清观测口径，不改变检测、确认、事件保存或颜色语义。
