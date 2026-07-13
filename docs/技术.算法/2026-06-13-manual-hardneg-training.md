# 手工困难样本增强训练记录

## 背景

固定镜头室外视频中存在“手被识别为 head”的模型本体误检。已完成 331 张人工困难样本标注，其中 `train_candidate` 用于训练，`holdout_candidate` 用于后续验收。

## 数据选择

本次训练使用：

- 基础干净三类数据：`D:/defense_purification_data/three_class_clean`
- 人工困难样本包：`D:/defense_purification_data/manual_hardneg_fixed_outdoor_20260612_seq`
- 混合训练集：`D:/defense_purification_data/three_class_clean_manual_hardneg_20260613`

训练集包含基础 `train` 4500 张和人工 `train_candidate` 244 张，共 4744 张。训练过程的 `val` 只使用基础干净 `val` 500 张。人工 `holdout_candidate` 87 张单独保留在 `holdout`，不进入训练数据配置。

未使用的数据：

- 未使用旧模型/reference 自动检测结果作为训练标签。
- 未使用 `D:/defense_purification_data/yolo_training` 中的投毒/净化中间数据。
- 未使用此前漂移的 fine-tuned 候选模型继续训练。
- 本地未确认存在可直接纳入的 7k 外部三类干净数据集，因此本轮未混入该数据源。

## 训练设置

起点模型：`D:/联合防御模块/model/baseline_training/runs/baseline_yolov8_three_put/best.pt`

输出目录：`D:/联合防御模块/purification_lab/models/manual_hardneg/clean_manual_hardneg_1280_e80_20260613`

主要参数：

- `imgsz=1280`
- `batch=8`
- `epochs=80`
- `patience=20`
- `optimizer=AdamW`
- `lr0=0.001`
- `mosaic=0.2`
- `mixup=0`
- `copy_paste=0`
- `erasing=0`

## 当前状态

训练已启动并进入正常 epoch。第一轮训练完成后基础验证集指标为 `mAP50=0.88179`、`mAP50-95=0.55236`。最终是否改善“手误识别为 head”必须以后续 reference 视频和人工 holdout 验收为准。
