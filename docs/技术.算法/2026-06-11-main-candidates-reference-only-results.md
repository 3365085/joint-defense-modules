# 主线微调候选 full YOLO reference 结果

## 问题背景

用户怀疑项目检测视频链路可能存在问题，因此要求对主线微调候选只跑 YOLO reference 视频，不跑项目 runtime、tracking、person-state 或 overlay。该试验用于隔离判断模型本体效果。

## 试验设置

固定设置：

```text
source_video = D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频.mp4
start_frame = 0
end_frame = 1555
imgsz = 1280
conf = 0.05
model_family = yolov8
hide_labels = person
```

所有命令均通过 Pixi 执行：

```text
pixi run python -m defense.diagnostics.yolo_reference_video ...
```

## 产物目录

1. `model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/01-run3-e18-img1280-conf005-hide-person`
2. `model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/02-helmetpos-e10-img1280-conf005-hide-person`
3. `model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/03-balanced-e8-img1280-conf005-hide-person`
4. `model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/04-balanced-tail-e6-img1280-conf005-hide-person`
5. `model/runs/yolo_reference/2026-06-11-main-candidates-full-reference/05-balanced-tail-strong-e4-img1280-conf005-hide-person`
6. `model/runs/yolo_reference/2026-06-11-hand-head-tail-strong-helmet-e3-full-img1280-conf005-hide-person`

每个目录均包含：

```text
reference_result_0_1555.mp4
reference_detections_0_1555.json
reference_summary_0_1555.json
reference_report_0_1555.md
```

## 运行统计

`conf=0.05` 下完整视频 class count：

| 候选 | head | helmet | person |
| --- | ---: | ---: | ---: |
| run3_e18 | 6271 | 1496 | 11038 |
| helmetpos_e10 | 11148 | 4370 | 33611 |
| balanced_e8 | 6128 | 2082 | 16970 |
| balanced_tail_e6 | 5890 | 3402 | 14833 |
| balanced_tail_strong_e4 | 5598 | 2544 | 12780 |
| tail_strong_helmet_e3 | 5400 | 2905 | 15904 |

## 当前判断

这批 reference 视频完全绕开项目检测显示链路，因此如果 reference 中已经存在开头人物漏检、断框、手部误识别为 head 或同一目标 head/helmet 状态反转，就应归因为模型本体问题，而不是项目 overlay/tracking 问题。

当前仅记录产物与统计，最终视觉判断以用户观看 reference 视频为准。
