# PPE 断框/托框 A3 shadow-removal dry-run 记录

日期：2026-06-06

## 背景

A2b 分阶段诊断显示 `shadow_track_removed` 是 PPE display tracker 内部主要 drop 来源之一，但旧诊断只知道“被 shadow removal 删除”，不知道删除对之间的 IoU、中心距、containment、标签关系和保留/删除理由。A3 的目标是补足这部分 text-only 证据，先判断 shadow removal 是否主要在清理重复候选，还是在多人重叠时把不同目标误合并。

本轮不改模型权重、不改类别语义、不改阈值、不改绘制行为、不改实际 shadow removal 结果，只记录当前逻辑已经做出的删除决策。

## 代码链路

入口仍为：

`FrameProcessor.process()` -> `evaluate_ppe_business()` -> `PPEDisplayTracker.update()` -> `_remove_shadow_tracks()` -> `build_overlay_record()`

A3 新增字段位于 `ppe_tracking_diagnostics`：

- `shadow_decision_count`
- `shadow_decisions`

每条 `shadow_decisions` 包含：

- `same_target_reason`
- `selection_reason`
- `iou`
- `center_distance_ratio`
- `containment`
- `kept`
- `dropped`

其中 `kept` 与 `dropped` 只记录 track id、label、source、misses、age、confidence、box、hold 状态等文本字段，不包含图片或视频内容。

## 运行条件

- 目标视频：`D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频.mp4`
- profile：`desktop_rtx`
- realtime：`true`
- feature options：`static_image_enabled=true`，`a3b_sensitivity=high`
- display options：`show_boxes=true`，`show_person_boxes=false`，`show_module_hud=true`，`show_ppe_hud=true`
- B 模块：`test_bypass_model_security=true`，仅用于 PPE 诊断，不写入白名单，不作为生产准入结论

证据目录：

`runtime_evidence/ppe_box_tuning_20260605_215454/a3_shadow_dry_run_current`

主要产物：

- `a3_shadow_overlay.json`
- `a3_shadow_status_polls.json`
- `a3_shadow_decision_summary.json`
- `a3_shadow_decisions.csv`
- `a3_shadow_abrupt_events.csv`

## 关键结果

- overlay records：318
- diagnostic records：318
- frame range：0..1550
- records with shadow decisions：92
- shadow decision total：133

same-target reason：

| reason | count |
| --- | ---: |
| same_head_overlap | 119 |
| same_helmet_overlap | 8 |
| head_helmet_overlap | 6 |

selection reason：

| reason | count |
| --- | ---: |
| score_confidence_minus_misses | 127 |
| helmet_not_confident_enough | 5 |
| helmet_confidently_covers_head | 1 |

被删除标签与来源：

| item | count |
| --- | ---: |
| dropped head | 120 |
| dropped helmet | 13 |
| dropped detected | 127 |
| dropped held | 6 |

保留到删除的标签关系：

| kept -> dropped | count |
| --- | ---: |
| head -> head | 119 |
| helmet -> helmet | 8 |
| head -> helmet | 5 |
| helmet -> head | 1 |

几何统计：

- IoU：min 0.057，max 0.818，avg 0.473
- center distance ratio：min 0.0011，max 0.0289，avg 0.00585
- containment：min 0.186，max 1.000，avg 0.710

abrupt drop 关联：

- abrupt events：37
- abrupt events with shadow decision：15
- abrupt drop sum：41
- abrupt drop sum with shadow：15
- abrupt labels：head 29，helmet 8

## 当前判断

1. 本视频中 shadow removal 的主要形态是 `same_head_overlap`，不是大量 head/helmet 互斥。`head_helmet_overlap` 只有 6 次。
2. 绝大多数被删 track 是 `detected`，不是 held 旧框；因此这一路径更像“清理同帧/近邻重复候选”，不是简单的旧框托尾。
3. 但 `same_head_overlap` 的几何条件可能偏激进：存在 IoU 很低但中心距极近、containment 达标而被视为同目标的情况。多人重叠、小目标密集时，这可能把不同人的 head 候选当成 shadow 删除。该判断仍需下一轮 dry-run 对比确认。
4. 最大的 abrupt drop 事件并非都伴随 shadow decision；A3 统计中最大几项仍指向 `misses_exceed_render_cap` 与 `held_track_not_eligible`。因此 shadow removal 是重要分支，但不是唯一主因。
5. 目前仍不能直接放宽 shadow removal。直接放宽可能保留重复框、错类框，导致画面更乱；应先做 A3.1 “候选严格化 dry-run”，只模拟不同 shadow 判定条件下哪些决策会改变。

## 下一步建议

A3.1 继续保持 dry-run，不改实际行为：

- 对 `same_head_overlap` 单独计算 stricter alternative：
  - 当前逻辑是否命中
  - 如果要求更高 IoU 或更高 containment 是否仍命中
  - 如果两框中心距很近但 IoU 极低，标记为 `ambiguous_dense_heads`
- 输出“若采用候选规则，将少删多少 head、可能多留多少重复框”的文本指标。
- 只有 A3.1 明确显示某个候选规则能减少 abrupt drop 且重复框风险可控，才进入行为候选实验。

## 验收状态

A3 当前仅是 text-only 诊断，不是可提交的行为调参方案，也不满足提交前视觉验收条件。若后续产生行为候选，仍必须生成目标视频检测结果，并抽连续 3 秒逐帧肉眼检查拖框、断框、旧框滞留和画面同步问题。
