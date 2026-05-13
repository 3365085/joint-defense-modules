# 模块A实时视频防御系统交付包

这是可独立交付的运行目录。它只包含监控台运行所需的模块代码、模型文件、配置文件、示例视频、启动脚本和依赖清单，不包含训练缓存、旧实验目录和开发态中间产物。

## 快速开始（3 步复现）

1. **装环境**（首次）：双击 `安装或更新依赖_首次运行.bat`（需要 CUDA 12.x 显卡）
2. **验证 GPU**：双击 `验证CUDA环境.bat`，应该看到 `cuda= True` 和显卡型号
3. **启动**：双击 `启动_模块A监控台.bat`，浏览器自动打开 `http://127.0.0.1:7860/`

在界面里：
- 选"MP4 文件路径"，点"浏览"选 `samples/` 下任一视频
- 勾选"静态媒介欺骗检测"
- 点"开始检测"，观察右侧三路告警（p_adv / 翻拍A3b / p_synth）

## 示例视频（samples/）

| 文件 | 场景 | 预期行为 |
|---|---|---|
| `clean_baseline.mp4` | 真实仓库巡检 | 零误报，p_adv ≤ 阈值 |
| `glare_attacked.mp4` | 强光眩光攻击 | p_adv 确认告警 |
| `visibility_degradation_attacked.mp4` | 可见性退化 | p_adv 确认告警 |
| `motion_blur_attacked.mp4` | 运动模糊 | p_adv 确认告警 |
| `occlusion_attacked.mp4` | 遮挡攻击 | p_adv 确认告警 |
| `adv_patch_attacked.mp4` | 对抗补丁（胸口贴附，平滑跟踪） | p_adv + A3b 双重告警 |
| `screen_spoof_attacked.mp4` | 手机屏幕翻拍 | A3b 翻拍检测触发 |

## 目录说明

- `tools/module_a_monitor_app.py`：Web 监控台主入口
- `defense/module_a/`：A1-A4 特征检测 + 融合层 + 告警状态机
- `defense/pipelines/`：视频流输入到告警输出的完整管线
- `experiments/configs/`：运行配置、算力档位、A4 通用校准分类器
- `baseline_training/runs/*/weights/`：YOLOv5 v1 与 YOLOv8 v2 的推理权重
- `samples/`：示例视频，覆盖 clean + 5 类物理扰动攻击 + 翻拍
- `异常记录/`：运行时自动生成，保存异常事件视频、代表帧和 `events.json`
- `架构说明.md`：项目技术路线与功能检测路线说明

## 启动脚本

| 脚本 | 用途 |
|---|---|
| `安装或更新依赖_首次运行.bat` | 首次部署时安装 PyTorch CUDA/TensorRT/ONNXRuntime/Ultralytics |
| `验证CUDA环境.bat` | 检查 CUDA/PyTorch 是否可用 |
| `启动_模块A监控台.bat` | 启动本地 Web 监控台 |
| `使用大项目环境_本机快速配置.bat` | 仅开发机使用：把本目录 .pixi 联接到大项目环境，不需要重新下载依赖 |

## PowerShell 运行

```powershell
pixi run app
```

## 运行边界

- 默认配置使用 YOLOv5 v1 TensorRT，本机 CUDA 路径性能最好
- `best.onnx` 保留为跨设备推理工件；边缘设备可以用 ONNX Runtime 或设备厂商 NPU 工具链继续转换
- TensorRT `.engine` 和具体 GPU/驱动/TensorRT 版本有关；换机器后如不能加载，应使用包内 `best.onnx` 重新构建 engine
- **伪造视频流告警（p_synth）在 Web 端已强制关闭**，仍在开发中，切勿开启
- 当前交付包是运行产品包，不包含训练脚本、素材下载脚本和历史实验归档
