# Web 预览人物框纵向偏移诊断记录

## 问题背景

用户在 Web 监控台暂停画面中指出人物框有偏移。画面使用 `img#stream` 显示后端 MJPEG 预览，当前视频为 `固定镜头室外视频.mp4`，自定义模型为三类 YOLOv8 PPE 权重。

## 当前判断

这次偏移不是模型本体检测框偏移，而是 Web 后端预览渲染时的坐标缩放错误。当前运行状态中：

- 预览画面尺寸为 `960x540`；
- PPE overlay 记录中的框坐标空间为 detector frame：`box_space_shape=[360,640]`；
- 后端 MJPEG 渲染曾调用 `scale_ppe_tracks(..., target_shape=(540,960))`，但没有传入实际 `source_shape`，导致默认按 `640x640` 源坐标缩放。

因此 x 轴比例刚好接近正确，但 y 轴从 `360 -> 540` 应乘 `1.5`，却被按 `640 -> 540` 乘 `0.84375`，人物框会明显向上压缩/漂移。

## 代码链路依据

- `model/src/defense/runtime/frame_processor.py` 写入 `overlay_coordinate_space`，声明 PPE boxes 存在 detector-frame 坐标中。
- `model/src/defense/runtime/overlay_records.py` 把 `runtime_source_frame_shape`、`detector_frame_shape` 和 `overlay_coordinate_space` 带入 Web overlay 记录。
- `model/src/defense/runtime/runner.py` 的 `_render_backend_preview()` 负责把后端预览帧和 overlay 画成 MJPEG。
- `model/src/defense/visualization/overlay.py` 的 `scale_ppe_tracks()` 默认 `source_shape=(640,640)`，适合方形默认输入，但不适合当前 `640x360` detector-frame 坐标。

## 修复

`_render_backend_preview()` 现在优先读取：

1. `overlay["overlay_coordinate_space"]["box_space_shape"]`
2. 其次回退 `overlay["detector_frame_shape"]`
3. 最后才回退 `(640,640)`

然后把该 source shape 传给 `scale_ppe_tracks()`。这样当前 `640x360 -> 960x540` 的后端预览框会按 x/y 分别正确缩放。

## 验证

已新增并运行聚焦测试：

`pixi run python -m pytest -q model/tests/test_overlay_coordinate_space_contract.py`

结果：`3 passed`。

## 影响范围

该修复只影响后端 MJPEG 预览中烘焙到图片里的框坐标缩放；不改变模型推理、tracking、PPE 业务判断、B 模块准入和 overlay 记录原始坐标。前端 canvas overlay 本身已经按 `box_space_shape` 缩放，本次主要修后端预览 JPEG 的框。
