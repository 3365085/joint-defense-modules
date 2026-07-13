# YOLOv8 三类 head/helmet/person 外部模型与数据源判断

## 问题背景

当前固定镜头室外视频中仍存在手、手臂或遮挡区域被模型识别为 `head` 的情况。此前 hard-negative 微调候选虽然压低了部分手部误检，但完整 YOLO reference 仍出现断框，以及同一目标在 `head` 与 `helmet` 之间反复切换的问题，因此不能作为生产替换模型。

用户进一步询问是否存在更可信、训练数据更多、且直接输出三类 `head/helmet/person` 的 YOLOv8 模型。

## 当前判断

未发现可以直接替换本项目三类权重的可信现成 YOLOv8 模型。

较接近的公开来源主要是数据集或 Roboflow 托管模型，而不是已经经过本项目视频验收的离线 `.pt` 权重。即使页面报告较高 mAP，也只能说明其在原数据分布上表现较好，不能证明它能解决当前视频里的手部遮挡、人物重叠、侧边缘离场、拿取头盔等难例。

## 候选来源

1. Roboflow / Northeastern University Hard Hat Workers
   - 类别为 `head, helmet, person`。
   - 数据量约 7035 张图片。
   - 页面显示有预训练模型/API和较高指标。
   - 风险：类别顺序与本项目 `helmet, head, person` 不同；托管模型不等同于可直接纳入本项目的离线生产权重；必须下载数据或权重后跑本项目 YOLO reference 验收。

2. Kaggle / Andrew Mvd Hard Hat Detection
   - 数据量约 5000 张图片。
   - PASCAL VOC 标注，类别为 `Helmet, Person, Head`。
   - 更适合作为重新训练 YOLOv8 的基础数据，而不是直接替换模型。

3. SHWD Safety Helmet Wearing Dataset
   - 数据量约 7581 张图片。
   - 主要覆盖 helmet 与 normal head，包含大量 head 负例。
   - 缺点是并非完整三类 `head/helmet/person` 数据集，不能单独作为本项目三类模型训练集。

4. Hugging Face / keremberke YOLOv8 hard-hat model
   - 是 YOLOv8 hard-hat 相关现成模型。
   - 公开类别为 `Hardhat` 与 `NO-Hardhat`，不是本项目需要的三类 `head/helmet/person`。
   - 不建议作为替换权重。

## 建议路线

可信方案不是直接找一个“别人训练好的三类模型”替换，而是做联合训练：

1. 以 Roboflow Hard Hat Workers 或 Kaggle Hard Hat Detection 作为大数据基础。
2. 统一映射到本项目类别顺序：`helmet=0, head=1, person=2`。
3. 补充人工标注的本项目难例：
   - `225-315`：手经过头部、人物重叠、手被误识别为 head。
   - `470-825`：外卖小哥真实 helmet 正例。
   - `1250-1555`：无帽负例和末段稳定性。
4. 标注规则：只标真实 `helmet/head/person`，手、手臂、遮挡物不标为任何类别。
5. 训练后先跑 YOLO reference，不直接接入项目 overlay。

## 验收标准

候选模型必须至少通过以下 reference 验收后，才值得进入项目链路：

- overlap 段不再把手部高置信显示为 `head`。
- 真实 helmet 段连续稳定，不频繁断框。
- 同一目标不在 `head` 与 `helmet` 间高频反转。
- 末段无帽负例不误报 helmet。
- 完整视频 reference 稳定后，再跑项目 overlay/tracking 验收。

## 结论

目前没有足够可信的现成三类 YOLOv8 权重可直接替换。本项目应把公开数据源当作训练基础，再叠加本视频人工 hard-negative 标注，训练新的三类模型。否则仅靠后处理或继续小样本 pseudo-label 微调，会继续在“压掉手部误检”和“保住 helmet 稳定性”之间摇摆。
