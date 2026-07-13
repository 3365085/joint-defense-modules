# PPE 检测框重合恢复慢与 head/helmet 反转记录

## 问题背景

用户指出个别多人重合场景仍会断框，且人分离后框恢复偏慢；另有个别人会触发 `head/helmet` 显示反转或颜色闪烁。

本轮对比对象为已认可的三类 YOLOv8 reference，权重为 `model/baseline_training/runs/baseline_yolov8_three_put/best.pt`，类别顺序为 `helmet, head, person`。

## 当前判断

1. 框恢复慢主要来自显示 tracker 的历史平滑、遮挡期间 held track、以及 shadow 去重共同作用。此前较高 `smooth_alpha` 会让分离后的新框位置爬得慢；本轮保留质量模式下较快的坐标响应。
2. `head/helmet` 反转主要不是模型完全失效，而是同一头部区域常同时存在 `head` 与低置信 `helmet` 候选；项目 tracker 旧逻辑允许单帧强证据快速切换稳定标签，容易把候选竞争放大成颜色/标签闪烁。
3. 质量模式下加入“强标签切换连续确认”后，能减少反转，同时保持比 v1 更多的 helmet 显示量。

## 代码链路依据

- 检测输入与显示坐标：`model/src/defense/runtime/frame_processor.py`
  - 质量模式保持 4K 源视频等比缩放到 `1280x720`，不再压成 `1280x1280`。
  - `smooth_alpha=0.58` 保留较快位置响应。
- PPE 显示 tracker：`model/src/defense/module_a/postprocess/ppe_tracking.py`
  - `_merge_same_target_items()` 会把同簇 `head/helmet` 候选合并为代表。
  - `_update_stable_label()` 负责稳定标签切换。
  - 本轮新增 `strong_switch_count`、`mature_strong_switch_extra_count`、`small_strong_switch_extra_count`，让强切换也需要连续同向证据。
- 回归测试：`model/tests/test_ppe_display_tracker_label_stability.py`
  - 覆盖小目标单帧高置信 `helmet` 不应立刻把稳定 `head` 切掉。
  - 覆盖大目标仍可在连续证据下快速切换。

## 实验指标

全段结果对比：

- v1：空间相邻标签反转 `105`，同轨迹标签反转 `33`，可见 helmet 总数 `616`。
- v4：空间相邻标签反转 `133`，同轨迹标签反转 `42`，可见 helmet 总数 `663`。
- v6：空间相邻标签反转 `112`，同轨迹标签反转 `36`，可见 helmet 总数 `646`。

重点窗口 `675-810`：

- v1：空间相邻标签反转 `57`，同轨迹标签反转 `14`，可见 helmet 总数 `96`。
- v4：空间相邻标签反转 `83`，同轨迹标签反转 `22`，可见 helmet 总数 `130`。
- v6：空间相邻标签反转 `63`，同轨迹标签反转 `17`，可见 helmet 总数 `118`。

## 影响范围

- 本轮修改只影响 PPE 显示 tracker 的稳定标签切换，不修改模型权重、类别语义，也不增加额外 GPU 推理。
- 框坐标响应仍由质量模式的检测输入尺寸、等比预处理、较低平滑参数和 held/extrapolation 逻辑共同决定。
- `person` 仍作为上下文和抑制误检辅助，不作为 PPE 告警主证据。

## 结论

本轮 v6 是一个折中版本：相对 v4 明显降低 `head/helmet` 反转，相对 v1 仍保留更多 helmet 显示。它不能证明最终通过，仍需要用户观看完整结果视频和重点 3 秒 PNG 验收帧。

## 后续建议

1. 用户优先观看完整 v6 结果视频，并重点检查 `690-750`、`1130-1160`、`450-510`、`910-960`。
2. 若仍觉得反转明显，可继续提高 `strong_switch_count` 或只对小目标提高 `small_strong_switch_extra_count`，但可能降低 helmet 的即时显示量。
3. 若仍觉得多人分离后框慢，下一步应重点调 shadow 去重与同人分离判据，而不是继续降低 `smooth_alpha`。
4. 若需要精确解释 raw 同簇竞争，建议在 overlay JSON 中增加完整 raw box 明细；当前只能通过 tracker diagnostics 和 reference JSON 间接判断。
