# YOLO reference 检测视频工具记录

## 问题背景

当前检测框优化的核心疑问是：用户指出的第 30 帧近景白衣人没有框，究竟是模型本身检不出，还是项目实时链路、后处理、tracking 或 overlay 显示导致断框。

OpenCV 启发式风险扫描可以发现可疑帧段，但需要反复调阈值，不能直接证明模型本体是否有检测能力。因此本轮改为建立高质量离线 YOLO reference 视频工具。

## 当前判断

以 `baseline_training/runs/baseline_yolov8_three_put/best.pt` 三类权重作为 reference，在原始视频 `素材/手机随意录制的视频/固定镜头室外视频.mp4` 的 230-411 源帧区间执行离线推理。

关键参数：

- `imgsz = 1280`
- `conf = 0.05`
- `device = cuda:0`
- 类别顺序：`helmet, head, person`
- 目标源帧：`260`
- 对应验收 local frame：`30`
- 目标区域：`[1728, 832, 2496, 2008]`

工具结果显示：源帧 260 的目标区域被 reference 命中，且 257-263 连续 7 帧均有命中。因此当前结论是：模型本体能检出该目标；项目结果视频中断框/漏框更可能来自实时链路、推理分辨率、帧率限制、latest-only、后处理、tracking、overlay 匹配或显示参数。

## 代码链路依据

新增工具：

- `model/src/defense/diagnostics/yolo_reference_video.py`
- Pixi 任务：`pixi run yolo-reference-video`
- 默认权重：`baseline_training/runs/baseline_yolov8_three_put/best.pt`

工具输出：

- `reference_result_*.mp4`
- `reference_detections_*.json`
- `reference_summary_*.json`
- `reference_report_*.md`

本轮输出目录：

- `model/runs/yolo_reference/2026-06-09-three-put-230-411-img1280-conf005/`

关键结果：

- 结果视频可打开；
- 视频分辨率：`3840x2160`
- 帧数：`182`
- 帧率：`60.49`
- 源帧范围：`230-411`
- 源帧 260 / local frame 30：`exact_frame_hit = true`
- 目标窗口命中：`257, 258, 259, 260, 261, 262, 263`

## 影响范围

YOLO reference 工具只用于诊断模型本体检测能力，不改变生产检测链路，不替代最终用户验收。

如果 reference 能检出而项目结果视频漏检，后续重点应排查：

1. runtime 当前使用的权重是否仍是旧 YOLOv5 或 TensorRT artifact；
2. runtime `image_size = 640` 与 reference `imgsz = 1280` 的分辨率差异；
3. `process_fps_cap`、`detector_process_fps_cap`、`latest_only` 是否导致跳帧；
4. PPE 后处理是否过滤、互斥或降级了有效框；
5. tracking 是否错误持有旧框、误合并或丢失新检测；
6. overlay frame matching 是否把检测框匹配到了错误帧；
7. 页面/预览渲染是否和后端 overlay JSON 不一致。

## 结论和建议

结论：用户指出的第 30 帧问题，三类 YOLOv8 reference 能检出目标；当前更应把问题定位为项目实时链路和显示链路问题，而不是模型权重完全不可用。

后续建议：

1. 用同一个三类 YOLOv8 权重对项目实时链路生成结果视频，确认 runtime 没有继续使用旧 YOLOv5/TensorRT artifact；
2. 对比 `reference_detections_*.json` 与项目 `overlay_*.json`，定位目标框是在 detector 输出阶段丢失，还是在 PPE 后处理/tracking/overlay 阶段丢失；
3. 做一轮 `image_size=1280` 的高质量非实时项目检测模式，关闭或放宽实时跳帧限制，用于和 reference 对齐；
4. 用户最终事件效果验收前，仍需生成独立 3 秒完整 PNG 验收证据。
