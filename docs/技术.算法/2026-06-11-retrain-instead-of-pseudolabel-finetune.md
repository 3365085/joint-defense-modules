# 是否应放弃伪标签微调并重新训练

## 问题背景

当前多轮 hand/head hard-negative 微调候选在 YOLO reference 中仍表现不稳定。用户观察到：第一版模型相对正常，但后续模型会把视频开头裸头识别为 `helmet`，甚至基础人物/头部检出能力退化。

## 当前判断

应该放弃当前 pseudo-label 微调路线，改为重新训练更干净的数据集。

这里的“重新训练”不是从随机权重开始，也不是继续堆同一视频的 reference pseudo label，而是使用公开三类数据作为基础，再加入本项目人工标注的 hard-negative/positive 小集，重新训练或从可信预训练权重微调。

## 为什么当前路线应停止

当前路线失败的核心原因：

- 训练标签大量来自 baseline/reference pseudo label，baseline 错误会被继承；
- helmet 正例段使用低阈值 reference 标签，会把 `head/helmet` 反转噪声写进训练集；
- 基础数据中 `helmet` 数量明显高于 `head`，后续重复采样进一步推高 helmet 先验；
- 手过头、人物重叠、开头裸头、拿头盔、无帽末段都不是干净人工标注；
- YOLO reference 已经失败，说明不是项目 overlay/tracking 能解决的问题。

## 建议重训数据方案

最小可执行数据集应包含：

1. 公开基础数据：
   - 当前 Kaggle AndrewMvd Hard Hat Detection 可保留；
   - 如能下载 Roboflow Hard Hat Workers 数据集，可作为新增基础数据；
   - 统一类别顺序为 `helmet=0, head=1, person=2`。

2. 本项目人工标注 hard-negative：
   - 开头裸头段：用于防止裸头被吸到 `helmet`；
   - `225-315`：手经过头部、人物重叠、手臂遮挡；
   - `470-825`：外卖小哥真实 helmet 正例，但必须人工确认；
   - `1250-1555`：无帽负例和末段稳定性。

3. 标注规则：
   - 只标真实 `helmet/head/person`；
   - 手、手臂、遮挡物不标任何类别；
   - 拿在手里的头盔是否标注必须单独定规则，不能混入“佩戴 helmet”语义；
   - train/val 必须按时间段隔离，不能让同一连续视频帧同时支配训练和验收。

## 结论

当前候选模型应整体废弃。继续调权重、repeat、低阈值 pseudo label 或后处理都不是可靠路线。下一步应重新组织人工标注数据，先训练能通过 YOLO reference 的模型本体，再考虑项目 runtime/overlay 验收。
