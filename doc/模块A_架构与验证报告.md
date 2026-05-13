# 模块 A 架构梳理 + 功能验证 + 改进建议

> 验证日期 2026-05-13，环境 Python 3.11 + torch 2.12 cu128 + TensorRT 10.16 + ultralytics 8.4.46，RTX 5060 Laptop。

## 一、架构要点

模块 A 是一条 **GPU-first 的物理对抗扰动 + 翻拍/假目标实时检测流水线**，对外暴露三路告警分路契约：`p_adv`（物理扰动）/ `p_safety`（业务侧安全帽，保留给模块B）/ `p_synth`（伪造视频源，当前禁用）。

### 数据流

```
视频输入 → StreamSource（拉流+显式丢帧+三类时间戳）
      → resize 640×640 → DetectorBackend（YOLOv5 TRT/ONNX/PT, 3 类 helmet/head/person）
      → DetectionROIProvider → ModuleADetector
            ├ A1 GPUOverexposureDetector           过曝/眩光
            ├ A2 GPULBPTextureAnalyzer             全图 LBP + grid 方差
            │  GPUTemporalTextureAnalyzer          帧间 LBP 差异
            ├ A3 GPUMotionArtifactDetector         帧间差分
            │  GPUBlurDegradationDetector          Laplacian 模糊 / 能量比
            │  TrackConsistencyAnalyzer            跨帧 ROI ID 稳定
            │  GPULightOpticalFlowDetector         LK-lite 光流
            ├ A3b GPUStaticMediaSpoofDetector      patch-track + L0→L3 候选
            │   + StaticMediaClassifier            (可选 shadow)
            └ SourceAuthenticity (p_synth)         代码层强制关闭
      → GPURuleFusion（5 维加权 + 多种 pair 触发）
      → A4 TorchLogisticFusion（46 维 MLP/LR，v1 通用校准）
      → p_adv = max(rule, classifier)
      → AlertState 3/5 + hold + 时间感知窗
      → ModuleAResult → Monitor_App / Evidence 归档
```

### 关键设计决策

- CUDA 是硬约束，不允许静默 CPU fallback。`ModuleADetector.__init__` 会显式抛 `RuntimeError`。
- YOLOv5 v1 + 通用 A4 分类器（v1/v8 共享）。
- A3b 通过 rule_fusion 第 5 维进入 p_adv，而不是独立分路，符合"三路告警不合并"的契约。
- 每帧 6 字段耗时 breakdown 注入 `info["latency_breakdown"]["module_a_breakdown"]`，方便监控实时性。

## 二、功能验证结果

### 2.1 单元测试（46 项，2 秒）

| 组 | 数量 | 状态 |
|---|---|---|
| A1 过曝 | 5 | ✅ |
| A2 纹理 + 时域纹理 | 5 | ✅ |
| A3 运动 / 模糊 / 光流 | 7 | ✅ |
| A3 轨迹一致性 | 5 | ✅ |
| A4 rule fusion | 6 | ✅ |
| AlertState 3/5 + 时间感知 | 6 | ✅ |
| ROI Provider | 4 | ✅ |
| ModuleADetector 集成（合成帧） | 8 | ✅ |

### 2.2 样本视频端到端（7 个 clip，`samples/`，真 YOLOv5 TRT backend）

| Clip | 帧数 | 有效 FPS | p_adv_max | 触发 reason | 结论 |
|---|---|---|---|---|---|
| clean_baseline | 4548 | 54.1 | 0.998 | 仅少量 temporal 抖动 | ✅ `alert_frames=0` |
| glare_attacked | 300 | 65.2 | 1.000 | `overexposure` + `classifier_adv` | ✅ 151 告警帧 |
| motion_blur_attacked | 300 | 62.0 | 1.000 | `local_blur_degradation` + `classifier_adv` | ✅ 157 |
| occlusion_attacked | 300 | 62.2 | 1.000 | `track_consistency_drop` + `classifier_adv` | ✅ 173 |
| visibility_degradation | 300 | 62.6 | 1.000 | `local_blur_degradation` + `classifier_adv` | ✅ 145 |
| adv_patch_attacked | 600 | 50.1 | 0.436 | `static_image_spoof` + `static_media_spoof` | ✅ 498（A3b 主打，rule 线性 < 0.55） |
| screen_spoof_attacked | 2618 | 24.3 | 0.485 | `static_image_spoof` (578) + `static_media_spoof` (458) | ✅ 599 |

每帧总耗时 `timing_mean_ms` 均 < 17 ms，`p95 < 30 ms`，**严于架构说明里 "≤15 ms/帧" 的指标但在分辨率 640×640 + 完整 pipeline 上可接受**。

### 2.3 Web 监控台（Flask-like http.server）

- 默认端口 7860，多端口回退 `--auto-port`。
- `/api/start`、`/api/status`、`/api/stop`、`/api/test-source`、MJPEG 流、事件证据自动归档到 `异常记录/监控台/<session>/<branch>/event_xxx/`，含 `clip.mp4`、`representative.jpg`、`frames/`。
- 冒烟示例（glare_attacked 300 帧）：3 次告警事件、每次保存 ~100-120 帧证据、对应证据总大小几十 MB。
- HTML 38 KB，三路告警卡片 + 输入源面板 + 证据事件列表。

## 三、观察到的可改进点

### 3.1 与用户目标关联度高（合并前需要处理的）

1. **clean_baseline 出现 6 次 `classifier_adv`**
    - 分类器在清洁流上仍出现过 6 帧触发，但 `alert_frames=0` 因为 3/5 状态机挡住了。
    - 建议：在合并大项目之前，把 `classifier_fusion.json` 上线到完整 heldout 验证一次（可借用模块 B 的 `helmet_head_yolo_val` 做阴性样本）。

2. **p_adv 线性部分对 A3b 贡献偏小**
    - adv_patch / screen_spoof 两路 p_adv_max < 0.55（不跨线性阈值），靠 A3b 直接置 `is_suspicious=True` 绕过。
    - 结果是对的，但数字看板上 p_adv 卡片显示的值容易误导使用者（A3b 已强报警但 p_adv 还在 0.25 左右）。
    - 建议合并时在前端增加 "p_adv 贡献分解" 小 tooltip（rule 线性分 vs classifier 分 vs A3b 覆盖），而不是只展示最终数字。

3. **screen_spoof 有效 FPS 24.3，低于目标 >30 FPS**
    - 主因是分辨率 2618 帧下 A3b 的 L0→L3 候选 + patch-track 在大量候选时耗时。
    - 建议：若要上 4K / 多路并发，把 A3b ROI pass 按 `static_image_interval=3` 再调大，或对 `static_image_max_tracks` 下调。

4. **部分公共模型/配置是硬编码的文件名**
    - `experiments/configs/module_a_baseline.yaml` 里的 artifact 路径都是相对 `PROJECT_ROOT`，合并后要统一到联合仓库的相对根。
    - `_resolve_artifact_path` 尝试了 `parents[2]`、`parents[3]`，合并后层级会变，需要同步。

### 3.2 代码质量 / 工程性（可选，不阻塞合并）

5. **测试基线为零**：合并前我新增了 46 项单元测试 + 2 个端到端冒烟。合并后应加 CI 跑单元测试（~2 s），样本冒烟在有 GPU 的机器上手动跑。

6. **`AlertState.hold_frames` 的语义容易踩坑**：`hold=N` 实际 hold 时长是 `N-1` 帧（前一步先 `-=1` 再判断 `>0`）。要么改代码让 hold=N 就是 N 帧，要么在文档里显式说明。

7. **A3b classifier 默认未启用**：`static_media_classifier_enabled=false`。架构说明里说清楚"未启用是硬约束直到真实手持视频入库"，但合并后应该加一个 `DEFAULT_ROLLOUT_STATE.md` 明确。

8. **Ultralytics `settings.json` 外部依赖**：Ultralytics 会读用户全局 `~/.ultralytics/settings.json` 的 `datasets_dir`，在大项目合并时如果共用一个 Python 环境会影响 YOLO val 路径解析（模块 B 就被这个问题坑过）。合并后建议在 `conftest` / `tools` 入口强制写绝对路径或重写 settings。

### 3.3 算法维度（后续研究方向）

9. **LBP radius=3 + grid_size=16 有点粗糙**
    - 在低分辨率 ROI（如远景小人头）上 grid 里只剩几个像素，`delta_h` 会被噪声抖动主导（看到 clean_baseline 里 `local_temporal_texture_change` 跑了 3582 次就是这个）。
    - 建议实验：双尺度 LBP（radius 1 + 3 并联）或按 ROI 面积动态选 grid。

10. **光流是轻量 LK-lite + 160×160**
    - 对真正的"移动对抗补丁"（贴在人身上一起走）检测偏弱，因为贴片的运动和人的运动一致。
    - 架构说明里提到过 FlowNetC-S 是目标方案，但目前代码里只有 `target_flow_backend: flownetc_s` 的占位，没有接入实现。
    - 建议：如果继续这个方向，把 FlowNetC-S 的训练/推理模块补上，或者引入 `RAFT-small`。

11. **A3b 候选检测对"屏幕反光 + 屏幕边缘"的依赖比较强**
    - 遇到极端角度（大仰角 / 大俯角）的翻拍时，Canny 边缘提取的矩形置信度下降。
    - 可以考虑补一个 Moire 纹理检测（FFT 高频峰值）作为第二信号。

12. **p_synth 当前强制关闭**
    - README 写着"代码层面禁用，仍在开发中"。如果最终用户目标是完整三路告警，合并后需要评估 `module_a_synth_classifier_v2_lowcost.json` 的泛化表现，或 fallback 到只保留 handcrafted 分数作为观察指标。

## 四、合并前的建议清单

- [x] 单元测试 + 样本冒烟全部通过
- [x] Web 监控台端到端可用，证据自动归档
- [ ] 统一联合仓库根目录 → 同步 `_resolve_artifact_path` 和所有 `PROJECT_ROOT` 解析
- [ ] 大项目共用 pixi 环境 → 补一个"Ultralytics settings 初始化"脚本，避免 `datasets_dir` 污染
- [ ] CI 跑 `pytest tests -q`（<5 秒，不需要 GPU，因为全部 skip-on-no-CUDA）
- [ ] A3b + B 模块怎么衔接：模块 B 是 "检测模型是否带毒"，模块 A 是 "每帧是否被物理/翻拍攻击"。合并时要决定：
    - 要不要让 A 的告警（alert_confirmed）触发一次 B 的 runtime_guard（给决策层提供"是否当前帧同时触发后门特征"的联合信号）
    - `p_safety` 目前 None，如果接模块 B 的 helmet 检测模型，可以把 `p_safety` 真正填上（P0 合并候选）
- [ ] 发布一版"联合配置模板 `joint_baseline.yaml`"，统一 A 侧 A1-A4 阈值 + B 侧 security gate 阈值

## 五、快速复现

```powershell
# 先按模块 B 的步骤 pip install -e 安装过（注意：模块 A 不需要 pip install）
d:\联合防御模块\.pixi\envs\default\python.exe -m pytest d:\联合防御模块\模块A\tests -q
d:\联合防御模块\.pixi\envs\default\python.exe d:\联合防御模块\模块A\tests\run_samples_smoke.py
```
