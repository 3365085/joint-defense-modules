# 后续微调候选把裸头识别成 helmet 的原因判断

## 问题背景

用户观看主线候选的 YOLO reference 视频后发现：第一个 `run3_e18` 相对正常，而后续模型会把视频开头第一个人的 `head` 识别为 `helmet`。该现象发生在 YOLO reference 视频中，因此应优先判断为模型本体问题，而不是项目 runtime、tracking 或 overlay 问题。

## 当前判断

主要原因不是 `person` 类，也不是项目显示链路，而是后续微调数据策略把模型的 `head/helmet` 类别边界推歪了。

第一个模型主要做 hand-as-head hard-negative，后续模型开始补 `470-825` 的 helmet 正例、tail 段、strong repeat 等。补数据的方式依赖 reference pseudo label，并且 `positive-label=helmet` 只用于选择“包含 helmet 的帧”，实际写标签时会把该帧所有满足阈值的 detection 写入训练标签。因此如果 reference 在这些帧里已经存在 `head/helmet` 反转或弱 helmet 噪声，噪声就会被加入训练集。

## 代码链路依据

`purification_lab/scripts/build_hand_head_hardneg_dataset.py` 中：

- `positive-frames` 默认为 `470-825`。
- `positive-label` 默认为 `helmet`。
- `positive-label-conf` 默认为 `0.05`。
- `positive-repeat` 默认为 `3`。
- 选择 positive frame 时只要求该帧存在一个 `helmet` detection。
- 写入 positive split 时调用 `_extract_split`，该函数会写入该帧里所有通过阈值的 detection，而不是只写真实 helmet。

这意味着 positive 数据不是人工干净标签，而是低阈值 reference pseudo label。

## 实际数据分布证据

按 `train.txt` 重复项计数，几轮训练集的标签分布为：

| 数据集 | train images | helmet | head | person |
| --- | ---: | ---: | ---: | ---: |
| `hand_head_hardneg_20260610` | 4884 | 17567 | 6914 | 19036 |
| `hand_head_hardneg_helmetpos_20260610` | 5175 | 18083 | 7916 | 21352 |
| `hand_head_balanced_20260611` | 4853 | 17611 | 6696 | 18728 |
| `hand_head_balanced_tail_20260611` | 4877 | 17695 | 6832 | 18828 |
| `hand_head_balanced_tail_strong_20260611` | 5717 | 18715 | 10552 | 25608 |
| `hand_head_tail_strong_helmet_20260611` | 5911 | 19059 | 11220 | 27152 |

基础 clean 数据本身 `helmet` 数量就显著多于 `head`。后续模型又叠加同一视频的 pseudo-labeled helmet 正例，导致模型更容易把模糊头部、浅色头部、遮挡头部或局部圆形区域推向 `helmet`。

YOLO reference full-video 统计也显示后续模型本体输出发生明显漂移：

| 候选 | head | helmet | person |
| --- | ---: | ---: | ---: |
| `run3_e18` | 6271 | 1496 | 11038 |
| `helmetpos_e10` | 11148 | 4370 | 33611 |
| `balanced_tail_e6` | 5890 | 3402 | 14833 |
| `tail_strong_helmet_e3` | 5400 | 2905 | 15904 |

`helmetpos_e10` 的 `helmet` 和 `person` 都明显膨胀，说明它不是简单增强了真实 helmet，而是让模型在本视频分布上更容易触发 PPE 类。

## 影响范围

如果 YOLO reference 已经把开头裸头识别为 `helmet`，项目后处理最多只能抑制显示或稳定状态，不能把错误模型本体变成可靠模型。强行接入这些候选会造成：

- 开头裸头误判为 helmet；
- 手部误识别和 head/helmet 反转继续存在；
- 真实 helmet 段断框；
- 项目 overlay 可能进一步放大错误状态。

## 结论

后续模型离谱的原因是微调数据策略失败：使用低阈值 pseudo label 和重复采样去补 helmet 正例，在 helmet/head 本来就不平衡的数据上继续推高 helmet 先验，导致裸头被吸到 helmet 类。当前候选应整体废弃，不应继续在这些权重上调后处理。

下一步应停止 pseudo-label-only 微调。若继续训练，必须改为人工标注：

1. 开头裸头段必须作为 bare head 正例；
2. `225-315` 手过头与重叠段只标真实 head/helmet/person，手不标；
3. `470-825` helmet 正例必须人工确认，不能直接全信 reference；
4. 保留独立 validation，不能让同一视频重复帧同时支配 train 和验收。
