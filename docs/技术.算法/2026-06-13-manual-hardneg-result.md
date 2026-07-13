# manual hard-negative 训练验收记录

## 背景

本轮目标是用人工标注的固定镜头室外难例，修正“手被识别为 head”的模型本体误检。训练数据使用基础干净三类数据集与人工 hard-negative 训练子集混合，人工 holdout 不进入训练。

## 训练状态

训练目录：`D:/联合防御模块/purification_lab/models/manual_hardneg/clean_manual_hardneg_1280_e80_20260613`

训练原计划为 `epochs=80, patience=20, imgsz=1280, batch=8`。实际训练在第 36 轮中途异常中断，未生成 `lab_train_manifest.json`；`err.log` 为空。Ultralytics 原生 resume 对该中断 checkpoint 没有稳定进入 epoch，因此本轮验收使用已生成的 `weights/best.pt`。

`best.pt` 对应第 34 轮，是基础 clean val 上本轮最高 `mAP50-95`：

- `mAP50=0.93964`
- `mAP50-95=0.65546`

## holdout 验证

人工 holdout：`D:/defense_purification_data/three_class_clean_manual_hardneg_20260613/manual_holdout.yaml`

输出：`D:/联合防御模块/purification_lab/models/manual_hardneg/clean_manual_hardneg_1280_e80_20260613/manual_holdout_eval.json`

聚合指标：

- `precision=0.63035`
- `recall=0.70507`
- `mAP50=0.66972`
- `mAP50-95=0.21528`

结论：人工难例上的定位质量仍然偏低，说明这批样本确实覆盖了当前模型困难区域；不能仅凭基础 clean val 指标判断模型已可替换。

## reference 视频验收

新模型 reference 输出：

- `D:/联合防御模块/model/runs/yolo_reference/2026-06-13-manual-hardneg-best-225-406-img1280-conf005-hide-person/reference_result_225_406.mp4`
- `D:/联合防御模块/model/runs/yolo_reference/2026-06-13-manual-hardneg-best-225-406-img1280-conf005-hide-person/reference_detections_225_406.json`
- `D:/联合防御模块/model/runs/yolo_reference/2026-06-13-manual-hardneg-best-225-406-img1280-conf005-hide-person/reference_summary_225_406.json`

与 baseline reference 在 235-305 帧的 JSON 对比：

- baseline：`head` 总框数 546，最大 head 面积 141732，面积大于等于 50000 的帧数 48。
- manual hard-negative best：`head` 总框数 1734，最大 head 面积 95400，面积大于等于 50000 的帧数 39。

按置信度过滤后：

- `conf>=0.25`：新模型仍有 413 个 head，面积大于等于 50000 的帧仍为 39。
- `conf>=0.5`：新模型 head 总数降至 62，但大面积误检仍覆盖 38 帧。
- `conf>=0.7`：大面积 head 被压掉，但 helmet/person 也明显不足，不是可直接使用的阈值策略。

## 结论

本轮训练没有可靠解决“手被识别为 head”。它降低了部分超大误检框面积，但引入了明显的 head 过报和置信度校准问题；在 reference 视频上仍保留高置信大面积 head 误检。因此该 `best.pt` 只能作为实验候选，不建议替换当前 baseline。

后续更合理的方向是继续补充更严格的专场景人工标注，并加入足量“手、胳膊、遮挡头部但非 head”的负样本上下文；仅靠这 244 张训练难例混入 4500 张基础数据，容易让模型学成更激进的 head 检测器。
