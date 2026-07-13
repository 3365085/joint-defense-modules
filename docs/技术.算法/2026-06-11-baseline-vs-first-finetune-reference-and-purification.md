# baseline 与第一个微调模型 reference 对比及净化流程使用判断

## 问题背景

用户希望只比较 baseline 模型和第一个微调模型的 YOLO reference 视频，从二者中保留一个。同时用户询问：后续如果按不同数据集重新训练模型，是否也可以使用当前项目的投毒扫描与净化方法。

## Reference 设置

两段 reference 均使用完整固定镜头室外视频：

```text
source_video = D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频.mp4
start_frame = 0
end_frame = 1555
imgsz = 1280
conf = 0.05
model_family = yolov8
hide_labels = person
```

## 产物

baseline：

```text
model/runs/yolo_reference/2026-06-11-baseline-three-put-full-img1280-conf005-hide-person/reference_result_0_1555.mp4
model/runs/yolo_reference/2026-06-11-baseline-three-put-full-img1280-conf005-hide-person/reference_detections_0_1555.json
model/runs/yolo_reference/2026-06-11-baseline-three-put-full-img1280-conf005-hide-person/reference_summary_0_1555.json
model/runs/yolo_reference/2026-06-11-baseline-three-put-full-img1280-conf005-hide-person/reference_report_0_1555.md
```

第一个微调模型 `run3_e18`：

```text
model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/01-run3-e18-img1280-conf005-hide-person/reference_result_0_1555.mp4
model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/01-run3-e18-img1280-conf005-hide-person/reference_detections_0_1555.json
model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/01-run3-e18-img1280-conf005-hide-person/reference_summary_0_1555.json
model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/01-run3-e18-img1280-conf005-hide-person/reference_report_0_1555.md
```

## 模型身份

baseline:

```text
path = model/baseline_training/runs/baseline_yolov8_three_put/best.pt
sha256 = D18748492C3819AF7788E4C15D9983EB9CBA0731D3090EF64AE1DA75E7E57C1B
```

run3_e18:

```text
path = purification_lab/models/finetuned/hand_head_hardneg_yolov8n_20260610_run3_e18_img1280/weights/best.pt
sha256 = 7194B392D6AA1BCCBEAFAE7942AB68663ED398B869EA7218DC2FCBD5C650A8DD
```

## Reference 统计

baseline:

```text
head = 6906
helmet = 3016
person = 16123
```

run3_e18:

```text
head = 6271
helmet = 1496
person = 11038
```

统计只能辅助判断，最终是否保留应以用户观看 reference 视频为准，重点看开头人物裸头、overlap 手部误检、helmet 正例连续性和末段无帽负例。

## 不同数据集模型与投毒净化流程

后续如果使用不同数据集重新训练模型，仍应使用当前项目的模型安全流程，但不能绕过准入：

1. 每个新模型都必须作为独立 runtime artifact 记录。
2. 信任记录必须绑定 source PT、runtime artifact、class names、PPE mapping、scanner version 和报告 hash。
3. 新模型进入生产前必须做 full scan，结果 clean/trusted 后才能作为 runtime replacement。
4. suspicious 模型才进入净化；净化候选必须复扫 clean/trusted 后才能替换。
5. 不同数据集训练的模型可以作为对照模型、teacher 或候选模型，但必须保持类别顺序 `helmet, head, person`。
6. 视觉验收仍必须以 YOLO reference 为第一关，项目 overlay/tracking 是第二关。

## 结论

baseline 和 `run3_e18` 已生成可直接观看的完整 reference。是否保留其中一个，应先由用户视觉确认模型本体。后续重新训练不同数据集模型时，投毒检测和净化流程仍然适用，但这些模型必须按新模型进入扫描、净化、准入和 reference 验收链路，不能直接替换生产权重。
