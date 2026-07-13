# Roboflow Hard Hat Workers reference 试验记录

## 问题背景

当前本项目三类模型来自 Kaggle AndrewMvd Hard Hat Detection 数据集，但在固定镜头室外视频里仍会把手、手臂或遮挡区域识别成 `head`，且此前 small hard-negative 微调候选在完整 reference 上仍出现断框和 `head/helmet` 状态反转。

用户指出当前模型训练源已经是 Kaggle 数据，因此要求尝试 Roboflow Universe 的 `joseph-nelson/hard-hat-workers`，不接入原项目程序，只改检测试验流程并跑 reference 视频查看模型本体效果。

## 当前判断

这次要验证的是 Roboflow 托管模型本体，而不是项目 runtime、tracking 或 overlay。项目原有 `pixi run yolo-reference-video` 只接受本地 Ultralytics / YOLOv5 权重文件，不能直接调用 Roboflow Hosted API。

因此新增了独立试验脚本：

```text
purification_lab/scripts/run_roboflow_reference_video.py
```

该脚本仅用于 `purification_lab` 试验，不接入 `model/src/defense/runtime`，也不修改生产检测链路。

关于“是否因为加了 `person` 才导致效果差”：当前判断为不是单点原因。Kaggle AndrewMvd 数据本身就包含 `Helmet / Person / Head`，加入 `person` 后可能改变类别损失分配和小目标学习难度，但手被识别成 `head` 的根因更可能是训练分布里缺少手贴脸、手过头、人物重叠、遮挡边缘等 hard-negative 场景。模型在这些局部纹理和形状上把手或手臂学成了裸头特征。

## Roboflow 来源信息

待验证模型：

- Project: `hard-hat-workers`
- Version: `13`
- 页面来源：`https://universe.roboflow.com/joseph-nelson/hard-hat-workers`
- 类别：`head / helmet / person`
- 页面显示训练数据规模约 `16867` images
- 页面显示 mAP@50 约 `96.9%`

这些指标只代表其原始数据分布上的结果，仍需用本项目固定镜头视频的 reference 结果验收。

## 本地执行情况

本机没有 `ROBOFLOW_API_KEY` 环境变量，也没有本地导出的 Roboflow `.pt` 权重。

已执行的探测：

```text
https://api.roboflow.com/joseph-nelson/hard-hat-workers/13 -> HTTP 401
https://api.roboflow.com/joseph-nelson/hard-hat-workers -> HTTP 401
https://api.roboflow.com/dataset/hard-hat-workers/13 -> HTTP 401
```

返回信息显示该方法要求 `api_key`。

直接请求推理端点：

```text
https://detect.roboflow.com/hard-hat-workers/13 -> HTTP 403 / Cloudflare
https://serverless.roboflow.com/hard-hat-workers/13 -> HTTP 403 / Cloudflare
```

因此本轮未能真实跑出 Roboflow reference 视频；阻塞原因为缺少 Roboflow API key 或离线导出的权重。

## 已生成阻塞证据

已用 Pixi 跑了脚本的缺少密钥路径，生成试验目录：

```text
purification_lab/runs/roboflow_reference/2026-06-11-hard-hat-workers-v13-overlap-225-406-conf005-hide-person/
```

其中包含：

```text
roboflow_reference_summary_225_406.json
roboflow_reference_detections_225_406.json
roboflow_reference_report_225_406.md
```

状态为：

```text
blocked_missing_api_key
```

## 拿到 API key 后的复跑命令

先在当前 shell 设置环境变量：

```powershell
$env:ROBOFLOW_API_KEY="<your_key>"
```

然后运行 overlap 段：

```powershell
pixi run python "purification_lab\scripts\run_roboflow_reference_video.py" --source-video "D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频.mp4" --output-dir "runs\roboflow_reference\2026-06-11-hard-hat-workers-v13-overlap-225-406-conf005-hide-person" --start-frame 225 --end-frame 406 --confidence 0.05 --hide-labels person
```

如果 overlap 段通过，再跑：

```powershell
pixi run python "purification_lab\scripts\run_roboflow_reference_video.py" --source-video "D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频.mp4" --output-dir "runs\roboflow_reference\2026-06-11-hard-hat-workers-v13-helmet-470-825-conf005-hide-person" --start-frame 470 --end-frame 825 --confidence 0.05 --hide-labels person
```

以及末段无帽负例：

```powershell
pixi run python "purification_lab\scripts\run_roboflow_reference_video.py" --source-video "D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频.mp4" --output-dir "runs\roboflow_reference\2026-06-11-hard-hat-workers-v13-tail-1250-1555-conf005-hide-person" --start-frame 1250 --end-frame 1555 --confidence 0.05 --hide-labels person
```

## 验收重点

1. `225-406` overlap 段：手经过头部时不应高置信显示为 `head`。
2. `470-825` helmet 正例段：外卖小哥真实 helmet 应连续稳定。
3. `1250-1555` 无帽负例段：不应误报 helmet。
4. 同一目标不应在 `head` 和 `helmet` 之间高频反转。

## 结论

当前不能判断 Roboflow Hard Hat Workers 是否优于现有模型，因为缺少 API key 或可离线运行的导出权重。已完成的是可复跑试验脚本和阻塞记录。拿到 Roboflow API key 或导出 `.pt` 权重后，应先跑上述 reference 段，不应直接接入项目 runtime。
