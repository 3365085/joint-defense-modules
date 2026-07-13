# hard-negative 微调后 reference 断框和 head/helmet 反转问题

## 问题背景

使用 `hand_head_tail_strong_helmet_yolov8n_20260611_e3_img1280` 跑完整固定镜头室外视频后，用户反馈 reference 视频本身仍存在断框，以及同一目标在 `head` 与 `helmet` 之间状态反转的问题。

Reference 是裸模型输出，只隐藏 `person` 渲染，不包含项目侧 tracking、person-state、overlay hold 等显示稳定逻辑。因此如果 reference 已经不稳定，说明模型本体还不能作为最终替换模型。

## 当前判断

当前候选模型不能上线替换 baseline。它确实降低了 `235-266` 和 `282-305` 中手部大面积高置信 `head` 误检，但引入或暴露了更严重的类别稳定性问题：同一空间位置附近的 `head/helmet` 判断会跨帧反转，且不是只由 `conf=0.05` 的低置信噪声造成。

完整 reference 输出：

- `model/runs/yolo_reference/2026-06-11-hand-head-tail-strong-helmet-e3-full-img1280-conf005-hide-person/reference_result_0_1555.mp4`
- `model/runs/yolo_reference/2026-06-11-hand-head-tail-strong-helmet-e3-full-img1280-conf005-hide-person/reference_detections_0_1555.json`

完整视频统计：

- `conf>=0.05`: `head=5400`, `helmet=2905`
- `conf>=0.25`: `head=4056`, `helmet=892`
- `conf>=0.25` 时仍有 `601` 帧同时出现 `head` 和 `helmet`
- 粗空间网格分析显示多个目标区域在 `conf>=0.25` 下仍有多次 `head/helmet` 翻转，说明不是简单提高显示阈值就能根治

## 原因分析

本轮训练主要依赖 reference pseudo label 和局部 hard-negative 过滤。pseudo-label 的问题是：

- baseline 自身在复杂重叠场景里就有错误标签，错误会被继承到训练集；
- 手部 hard-negative 可以压掉一类误检，但没有教会模型稳定区分真实 `head` 与 `helmet`；
- helmet 正例来自同一视频局部片段，能够保留数量，但覆盖的外观、遮挡和角度变化不够；
- `head` 与 `helmet` 是细粒度相邻类别，局部小样本微调容易在两个类别之间摇摆。

因此，继续只靠 pseudo-label 数据加 repeat 调比例，风险是修掉一段手部误检，又制造另一段类别反转。

## 影响范围

项目侧 tracking、head-anchor、helmet trust 可以降低短时显示抖动，但不能把不稳定的裸模型输出变成可靠模型。若强行接入当前候选模型，可能出现：

- 同一目标状态在 `head/helmet` 间反复切换；
- 真实 helmet 框断续出现；
- overlay hold 放大错误状态；
- 业务状态和视觉框不同步。

## 结论

当前候选模型应标记为实验失败或仅作为中间数据参考，不能作为生产替换模型。

下一步应停止 pseudo-label 比例调参，改为最小人工标注集：

1. 从问题视频抽连续帧，而不是零散帧：`225-315`、`470-825`、最后 `1250-1555`。
2. 每 2-3 帧标注一帧，重点覆盖手过头、人物重叠、右边缘离场、拿取头盔、无帽末段。
3. 只标真实 `helmet/head/person`；手、胳膊、遮挡物不标。
4. 补充真实 helmet 与 bare head 的相邻正例，避免模型学成二分类摇摆。
5. 训练验收必须同时检查：
   - overlap 段大 `head` 误检；
   - 同一目标 `head/helmet` 状态连续性；
   - helmet 正例连续性；
   - 最后无帽负例；
   - 项目 overlay 是否放大裸模型抖动。

在人工标注集完成前，不建议继续用当前 hard-negative 微调模型替换 runtime 权重。
