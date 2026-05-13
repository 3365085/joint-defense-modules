# 模块 A 测试与验证

本目录包含对模块 A 各检测特征、融合层、告警状态机以及全链路的测试，
同时放了几个性能 / 校准用的离线工具。

## 运行方式

环境：直接使用联合仓 pixi 环境（`D:\联合防御模块\.pixi\envs\default\python.exe`）。

```powershell
# 1. 单元测试（不需要 YOLO 权重，纯特征/融合层契约）
d:\联合防御模块\.pixi\envs\default\python.exe -m pytest tests -q

# 2. 样本视频端到端冒烟（需要 TensorRT engine；验证 7 个 samples 触发契约）
d:\联合防御模块\.pixi\envs\default\python.exe tests\run_samples_smoke.py

# 3. 特征耗时 profile（得到 per-feature p95）
d:\联合防御模块\.pixi\envs\default\python.exe tests\profile_feature_timings.py

# 4. A3b 内部拆解（bg/L0/yolo/L2 分别的 p95）
d:\联合防御模块\.pixi\envs\default\python.exe tests\profile_a3b_internals.py

# 5. A4 分类器阈值校准（clean vs attacked FP/FN sweep）
d:\联合防御模块\.pixi\envs\default\python.exe tools\calibrate_classifier_threshold.py

# 6. Web 监控台端到端冒烟（先启动 monitor 再跑 probe）
#    终端 1：
d:\联合防御模块\.pixi\envs\default\python.exe tools\module_a_monitor_app.py --port 7861
#    终端 2：
d:\联合防御模块\.pixi\envs\default\python.exe tests\probe_web_start.py
```

也可以从联合根目录跑：

```powershell
d:\联合防御模块\.pixi\envs\default\python.exe -m pytest 模块A\tests -q
```

## 单元测试清单（64 项，GPU 模式 ~3.5 秒）

| 文件 | 覆盖范围 |
|---|---|
| `test_a1_overexposure.py` | A1 过曝：阈值、边界、欠曝独立字段 |
| `test_a2_texture_and_temporal.py` | A2：LBP 纹理 + 时域纹理；adaptive baseline + noise suppression；raw / exposed / suppressed 字段契约 |
| `test_a3_motion_blur_flow.py` | A3：帧间差分、Laplacian 模糊、轻量光流 LK 的可用/禁用 |
| `test_a3_track_consistency.py` | A3 轨迹一致性：稳定、置信度下降、消失、候选上限 |
| `test_a4_rule_fusion.py` | A4 5 维融合：reason codes、pair 组合、权重校验 |
| `test_alert_state.py` | 3/5 告警状态机（离线 / 时间感知 / hold / reset） |
| `test_roi_provider.py` | 检测框 → ROI 转换：置信度过滤、margin、标签回退 |
| `test_artifact_path_resolution.py` | 路径解析：MODULE_A_ROOT env 覆写、4 级回退 |
| `test_classifier_threshold_override.py` | A4 分类器阈值运行时覆写，artifact 不动 |
| `test_module_a_detector_integration.py` | ModuleADetector 端到端（合成帧） |
| `test_full_flow_stability.py` | 200 帧长跑 / reset / 内存界限 / 每帧耗时预算 |
| `test_samples_smoke_report_regression.py` | 每次冒烟结果必须满足 alert_frames + timing_mean 预算 |
| `run_samples_smoke.py` | 样本视频端到端（真 YOLOv5 backend） |
| `profile_feature_timings.py` | 各特征 p50/p95/p99 聚合 |
| `profile_a3b_internals.py` | A3b 内部 bg/L0/yolo/L2 分解 |
| `probe_web_start.py` | Web 监控台 `/api/start` → `/api/status` → `/api/stop` |
| `samples_smoke_report.json` | 上次样本冒烟结果 |
| `profile_feature_timings_report.json` | 上次特征耗时结果 |
| `profile_a3b_internals_report.json` | 上次 A3b 内部分解 |
| `classifier_calibration_report.json` | 上次分类器阈值校准扫描 |
