# reference 同帧重复框抑制记录

## 背景

人工 hard-negative 训练得到的候选模型在完整 reference 视频中，肉眼准确度有改善迹象，但同一帧内 `head/helmet` 重复框过多，影响验收观看和后续显示链路判断。

## 当前判断

这是模型输出后的同帧候选冗余问题，不应再通过改变模型权重、类别语义、全局置信度阈值或 person/head-zone 强约束来处理。本次仅在 reference 诊断渲染层增加同类重复框抑制，用于评估“保留最高置信框后的视频观感”。

## 代码链路

新增诊断脚本：

`model/src/defense/diagnostics/dedup_reference_video.py`

输入为已有 YOLO reference JSON，输出去重后的 JSON、summary、report 和重渲染视频。该脚本不重新跑模型，不接入项目 tracking/overlay 主链路。

## 抑制策略

只处理同一帧内同类别重复框，默认类别为 `head,helmet`：

- 按置信度从高到低遍历。
- 若同类框与已保留框 `IoU >= 0.55`，删除低置信框。
- 若同类框被已保留框高度包含或高度覆盖，包含率 `>= 0.90`，删除低置信框。
- 不做 person 过滤。
- 不做 head-zone 过滤。
- 不做 head/helmet 跨类互斥。

## 本次结果

输入：

`D:/联合防御模块/model/runs/yolo_reference/2026-06-13-manual-hardneg-best-full-img1280-conf005-hide-person/reference_detections_0_1555.json`

输出：

`D:/联合防御模块/model/runs/yolo_reference/2026-06-13-manual-hardneg-best-full-img1280-conf005-hide-person-dedup-iou055-contain090/dedup_reference_result_0_1555.mp4`

全段统计：

- 原始：`head=24206, helmet=1233, person=14190`
- 去重后：`head=7378, helmet=475, person=14190`
- 删除原因：`same_label_iou=12534, same_label_containment=5052`

完整顺序 reference 的 235-305 帧统计：

- 原始：`head=1273, helmet=198, person=645`，可见 `head/helmet` 最大每帧 33 个。
- 去重后：`head=354, helmet=83, person=645`，可见 `head/helmet` 最大每帧 9 个。

## 结论

同帧同类去重可以显著改善 reference 视频观感，并且比 person/head-zone 强约束更少改变模型语义。该策略可作为候选显示层后处理继续人工验收；是否进入项目主链路，需要用户先确认去重版视频没有误删真实相邻目标。
