# hand/head hard-negative 微调候选清单与失败判断

## 问题背景

用户检查完整项目检测视频后确认，本轮为了压制“手被识别为 head”的微调候选整体效果不行，部分模型甚至在视频开头人物检出上明显退化。该现象说明问题已超出单个手部误检，属于微调候选破坏基础检出能力或类别稳定性的失败。

## 当前模型产出清单

`purification_lab/models/finetuned` 下实际存在：

- 8 个训练 run，每个通常有 `best.pt` 和 `last.pt`。
- 4 个 blend/interpolation 权重，不是独立训练 run。
- 1 个空目录 `hand_head_hardneg_yolov8n_20260610_e18_img1280`，没有 `.pt` 权重，不计入模型。

训练 run：

1. `hand_head_hardneg_yolov8n_20260610_debug`
2. `hand_head_hardneg_yolov8n_20260610_run3_e18_img1280`
3. `hand_head_hardneg_helmetpos_yolov8n_20260610_e10_img1280`
4. `hand_head_hardonly_frozen_yolov8n_20260611_e8_img1280`
5. `hand_head_balanced_yolov8n_20260611_e8_img1280`
6. `hand_head_balanced_tail_yolov8n_20260611_e6_img1280`
7. `hand_head_balanced_tail_strong_yolov8n_20260611_e4_img1280`
8. `hand_head_tail_strong_helmet_yolov8n_20260611_e3_img1280`

Blend 权重：

1. `blend_run3_e10_alpha15.pt`
2. `blend_run3_e10_alpha25.pt`
3. `blend_run3_e10_alpha35.pt`
4. `blend_run3_e10_alpha50.pt`

## 已确认判断

这些候选不应进入生产替换。用户指出“四个核心模型都不行，最开始的人都检测不出来”，这意味着当前微调路线不是简单地没有解决手部误检，而是造成了基础 `person/head/helmet` 检出能力退化。

## 原因判断

当前路线主要基于 baseline pseudo label、局部 hard-negative 过滤和少量同视频正例补充。该方法的问题是：

- baseline 错误会被继承进训练集；
- hard-negative 只压制局部手部误检，不能学习真实复杂遮挡分布；
- 同视频正例覆盖不足，容易损害基础检出泛化；
- `head/helmet/person` 是强耦合类别，少量局部微调会造成类别边界漂移；
- 如果开头人物都检不出来，说明模型本体已经失败，后处理无法补救。

## 结论

本轮微调候选应整体标记为失败，不应继续用这些模型调 tracking 或 overlay。下一步如果继续做模型，应停止 pseudo-label-only 微调，改成人工标注 hard-negative + 足量公开三类数据联合训练，并先用 YOLO reference 对开头基础检出、overlap 手部误检、helmet 正例和无帽负例四段同时验收。
