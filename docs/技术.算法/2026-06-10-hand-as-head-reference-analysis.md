# 手部被识别为 head 的模型与项目链路判断

## 问题背景

固定镜头室外视频在人物重叠、手经过头部附近时，出现手被识别并显示为 `head` 的问题。用户已确认 YOLO reference 视频中也存在手部被识别为 `head` 的现象，因此需要区分模型本体误检与项目后处理、tracking、overlay 对误检的放大。

## 当前判断

这是“模型本体误检为主，项目链路可能放大显示为辅”的问题。

YOLO reference 使用三类模型 `helmet, head, person`，在 `225-406` 帧区间以 `imgsz=1280`、`conf=0.05` 运行。reference 报告显示该区间 `182/182` 帧有检测，类别计数为 `head=1358, helmet=477, person=2600`。在用户目标窗口附近，source frame `260` 命中 `head`，其中一个 `head` 框置信度约 `0.8516`，中心落在目标区域内。这说明模型本体已经把遮挡/手部形态当作 `head`，不是单纯的项目 overlay 坐标或 tracking 显示错误。

项目 v4 结果已经引入 person-state / helmet trust / head-anchor 后处理，`head_not_supported_by_person_head_anchor` 全段触发 `858` 次，说明项目侧已经压制了一批离开真实头部区域的 head 候选。但 `overlap_225_406` 统计仍为 `visible_head_frames=182/182`，说明当前项目显示链路仍会保留一部分模型误检，特别是当误检框落在人物头部候选区域、置信度高、且能被 person context 支撑时，纯后处理很难可靠区分“真实头部”和“经过头部附近的手”。

## 代码链路依据

- reference 证据：
  - `model/runs/yolo_reference/2026-06-10-overlap-225-406-img1280-conf005-hide-person/reference_report_225_406.md`
  - `model/runs/yolo_reference/2026-06-10-overlap-225-406-img1280-conf005-hide-person/reference_summary_225_406.json`
- 项目 v4 证据：
  - `model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v4-head-anchor-full-video/head_anchor_check_summary.json`
  - `model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v4-head-anchor-full-video/overlay_summary_full.json`
- 相关生产链路：
  - `model/src/defense/module_a/postprocess/ppe_tracking.py`
  - `model/src/defense/runtime/frame_processor.py`
  - `model/src/defense/runtime/ppe_business.py`

## 影响范围

后处理可以降低误显示，但不应承担根治模型语义错误的责任。若继续只靠规则压制，需要非常小心：

- 不能把 `head` 全局阈值抬高，否则远处真实无帽头部会漏检。
- 不能把 `person` 当 PPE 告警主证据，只能作为上下文和误检抑制辅助。
- 不能增加额外 GPU 推理。
- 必须保留外卖小哥 helmet 正例和最后 5 秒无帽负例。

## 结论

建议训练更好的三类模型，尤其补充 hard-negative 数据，把“手、胳膊、遮挡在头部附近但不是头部”的样本明确教给模型。项目侧可以继续做小范围后处理，目标是减少误检被 tracking/overlay 放大，但不应期待规则完全消除 reference 中已经高置信命中的手部 head。

## 最小 hard-negative 数据方案

1. 从固定镜头室外视频抽取重叠问题段，优先覆盖 source frame `235-305`，再补充 `282-305` 进入右侧和最大误检附近帧。
2. 抽帧不要只抽误检最明显的单帧，应每隔 2-5 帧取一张，并保留手部经过头部前、中、后的连续变化。
3. 标注时只标真实 `head`、`helmet`、`person`；手、胳膊、遮挡物不标为任何类，使其成为背景负样本。
4. 补充相邻正常场景作为正例，避免模型学成“手靠近头就全部压掉”，导致真实 head 漏检。
5. 用现有三类 YOLOv8 权重微调，不改类别顺序和 PPE 语义。
6. 验收必须重新生成隐藏 `person` 的 YOLO reference 视频，再生成项目结果视频，对比 overlap 段、外卖小哥 helmet 段和最后 5 秒无帽段。

## 后续建议

短期：检查项目 v4 在 `225-406` 的具体可见 head 框是否比 reference 更持久、更大或位置更误导；若是，只对 head-anchor/person-state 的显示门控做小范围收紧。

中期：建立 hard-negative 小集并微调三类模型。训练后优先看 reference 是否仍在 frame `260` 附近把手高置信报为 `head`。只有 reference 先改善，项目侧视觉效果才有稳定上限。

长期：把“手经过头部、人物重叠、拿取头盔、远处小目标、无帽末段”固定为 PPE 视觉验收集，每次模型或后处理变更都跑同一套 reference 与项目结果视频。
