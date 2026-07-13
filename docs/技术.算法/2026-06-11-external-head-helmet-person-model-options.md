# 外部 head/helmet/person 三类模型可用性判断

## 问题背景

当前自训练 hard-negative 候选虽然压低了部分手部误识别为 `head` 的问题，但完整 reference 视频仍存在断框和同一目标 `head/helmet` 状态反转。用户询问是否存在更可信、训练数据更多、输出三类 `head/helmet/person` 的 YOLOv8 模型。

## 当前判断

没有发现可以直接作为生产替换的“可信现成权重”。公开资料里最接近的是 Hard Hat Workers / Hard Hat Detection 系列数据集和模型，类别确实接近或等于 `head, helmet, person`，数据量在 5000 到 7000 张级别。但这些数据集主要来自工地/安全帽场景，不保证覆盖当前固定镜头室外视频里的手部遮挡、人物重叠、右边缘离场、拿取头盔等难例。

外部模型可以作为候选基线或预训练来源，但必须先跑本仓库的 YOLO reference 验收，不能直接替换 runtime 权重。

## 候选来源

1. Roboflow Universe / Northeastern University Hard Hat Workers：
   - 类别：`head, helmet, person`
   - 数据量约 7035 images
   - 页面报告模型指标较高，但需要在本项目视频上重新验证。

2. Kaggle / Andrew Mvd Hard Hat Detection：
   - 数据量约 5000 images
   - PASCAL VOC 标注，类别为 `Helmet, Person, Head`
   - 适合重新训练 YOLOv8，而不是直接拿来当现成模型。

3. SHWD Safety Helmet Wearing Dataset：
   - 数据量约 7581 images
   - 包含 helmet 与 head/normal head，但不是完整三类 `head/helmet/person`
   - 可作为补充 head/helmet 区分的数据源，不适合作为本项目三类模型的唯一数据。

## 建议方案

优先方案不是继续微调当前小样本伪标签模型，而是做“公开三类数据 + 本项目人工 hard-negative”的联合训练：

- 基础数据：Hard Hat Workers 或 Kaggle Hard Hat Detection。
- 本项目补充：`225-315` overlap 手部遮挡段、`470-825` helmet 正例段、`1250-1555` 无帽负例段。
- 标注规则：只标真实 `head/helmet/person`，手、胳膊、遮挡物不标。
- 验收规则：先看 YOLO reference，只有 reference 本体稳定后，再进入项目 overlay/tracking 验收。

## 结论

可以补充外部数据训练更好的模型，但不建议直接下载某个三类模型替换。所谓“可信”必须以本项目 reference 视频为准：手部不再高置信命中 `head`，同一目标 `head/helmet` 不频繁反转，helmet 正例不断框，无帽负例不误报。
