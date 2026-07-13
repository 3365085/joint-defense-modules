# 手部误识别为 head 的 hard-negative 训练记录

## 问题背景

固定镜头室外视频在人物重叠、手臂经过头部附近时，baseline 三类 YOLOv8 模型会把手或手臂识别成 `head`。已通过 YOLO reference 确认这是模型本体误检，不是单纯 overlay 或 tracking 显示错误。

本轮目标是在不改变类别顺序 `helmet, head, person`、不改 PPE 语义、不提高全局阈值、不增加额外 GPU 推理的前提下，补充 hard-negative 数据并微调模型，降低手部被识别为 `head` 的概率，同时保留外卖小哥 helmet 正例。

## 数据与训练

新增数据构建脚本：

- `purification_lab/scripts/build_hand_head_hardneg_dataset.py`

核心策略：

- 以 baseline reference 检测结果作为 pseudo label 来源。
- 对 source frame `235-266` 的中右区域疑似手部 `head` 框做 hard-negative：只保留真实 `helmet/head/person` 标注，手和手臂不标注。
- 扩展 source frame `282-305` 的右边缘大面积疑似手部 `head` 框过滤。
- 加入 `470-825` 的 helmet 正例帧，避免 hard-negative 训练压坏 helmet 检出。

最终候选数据集：

- `D:\defense_purification_data\hand_head_tail_strong_helmet_20260611\data.yaml`
- hard-negative repeat: `20`
- helmet positive repeat: `3`

最终候选模型：

- `D:\联合防御模块\purification_lab\models\finetuned\hand_head_tail_strong_helmet_yolov8n_20260611_e3_img1280\weights\best.pt`
- SHA256: `89857a34964479550f6da015733a920eaa8ffca4e2ea9b4b6cd78600e1f1fe9b`
- input model: `hand_head_balanced_tail_strong_yolov8n_20260611_e4_img1280`
- training: `epochs=3, imgsz=1280, batch=8, freeze=10, lr0=0.00005, optimizer=AdamW`

## Reference 验收结果

模型 reference 视频：

- overlap: `model/runs/yolo_reference/2026-06-11-hand-head-tail-strong-helmet-e3-225-406-img1280-conf005-hide-person/reference_result_225_406.mp4`
- helmet positive: `model/runs/yolo_reference/2026-06-11-hand-head-tail-strong-helmet-e3-470-825-img1280-conf005-hide-person/reference_result_470_825.mp4`

关键 JSON：

- `model/runs/yolo_reference/2026-06-11-hand-head-tail-strong-helmet-e3-225-406-img1280-conf005-hide-person/reference_detections_225_406.json`
- `model/runs/yolo_reference/2026-06-11-hand-head-tail-strong-helmet-e3-470-825-img1280-conf005-hide-person/reference_detections_470_825.json`

对比指标：

| 模型 | 235-266 大 head >50k / 高置信 | 282-305 大 head >50k / 高置信 | 470-825 helmet |
| --- | ---: | ---: | ---: |
| baseline | 9 帧 / 9 个 conf>=0.25 | 24 帧 / 24 个 conf>=0.25 | 172 |
| balanced_tail_e6 | 1 帧 / 0 个 conf>=0.25 | 8 帧 / 4 个 conf>=0.25 | 371 |
| tail_strong_e4 | 1 帧 / 0 个 conf>=0.25 | 6 帧 / 1 个 conf>=0.25 | 101 |
| tail_strong_helmet_e3 | 0 帧 / 0 个 conf>=0.25 | 5 帧 / 0 个 conf>=0.25 | 178 |

当前最终候选 `tail_strong_helmet_e3` 的判断：

- `235-266` 中高风险手部大 `head` 已清除。
- `282-305` 仍有 5 帧低置信大 `head` 框，但没有 `conf>=0.25` 的高置信残留。
- `470-825` helmet 检出为 `178`，接近 baseline `172`，没有出现 run3 的 helmet 崩塌，也没有 helmetpos 的过量误检。

## 当前判断

这是当前实验中最均衡的模型候选。它没有完全让所有低置信手部大框消失，但已经把高置信误检压到 0，同时保住 helmet 正例。后续如果项目显示链路仍放大低置信残留，应优先在项目后处理中结合置信度、person head anchor 和时序稳定性做小范围显示抑制，而不是继续提高全局模型阈值。

最终是否替换生产模型仍需要人工观看 reference 视频与项目结果视频确认；本轮未提交、未改 runtime 默认权重。
