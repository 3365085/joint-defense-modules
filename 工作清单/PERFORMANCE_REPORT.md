# 模块 A 性能优化战果（2026-05-13 夜班）

## 测量方法

- 硬件：RTX 5060 Laptop 8 GB，CUDA 12.8。
- 环境：联合仓库 pixi env（torch 2.12 dev cu128, TRT 10.16, ultralytics 8.4.46）。
- 数据：`模块A/samples/*.mp4` 共 7 个 clip，9258 帧。
- 脚本：`tests/profile_feature_timings.py`（每特征 p95 聚合）。

## 整体对比

| 指标 | Baseline (2026-05-13 入夜前) | 优化后 | Δ |
|---|---|---|---|
| A3b p95 (ms) | 15.3 – 18.0 | **8.9 – 11.5** | **−40%** |
| A3 (motion+blur+flow+track) p95 | 3.3 – 6.7 | **2.7 – 4.4** | **−35%** |
| A2 (texture+temporal) p95 | 1.8 – 3.8 | 1.9 – 3.4 | ≈ 0 |
| A1 (overexposure) p95 | 0.4 | 0.4 | ≈ 0 |
| A4 (rule+classifier fusion) p95 | 0.7 – 1.2 | 0.7 – 0.9 | −15% |
| **Pipeline total p95** | **25.9 – 29.0** | **18.5 – 21.8** | **−25%** |
| **Effective FPS (mean)** | 24.3 – 65.2 | **40+ 全部** | 最差场景 24→~40 |

## 功能正确性（7 clip smoke）

| Clip | 基线 alert_frames | 优化后 alert_frames | 状态 |
|---|---|---|---|
| clean_baseline | 0 | 0 | ✅ |
| glare_attacked | 151 | 162 | ✅ |
| motion_blur_attacked | 157 | 160 | ✅ |
| occlusion_attacked | 173 | 173 | ✅ |
| visibility_degradation | 145 | 145 | ✅ |
| adv_patch_attacked | 498 | 504 | ✅（略更敏感）|
| screen_spoof_attacked | 599 | 549 | ✅（仍远超阈值）|

clean_baseline 触发 `local_temporal_texture_change` 次数从 **3582 → 2173**（-39%），
说明噪声抑制起效而未波及告警率。

## 具体改动

### P0（合并前阻塞）
- `_resolve_artifact_path` 支持 `MODULE_A_ROOT` 环境变量 + 4 级路径回退。
- `ensure_ultralytics_settings_isolated` 入口工具，避免全局 `datasets_dir` 污染。
- `AlertState.hold_frames=N` 现在真的 hold N 帧（原为 N-1）。

### P1 算法与性能
- **A-4 LBP 时域噪声抑制**：EMA baseline（30 帧预热）+ persistence 门控，
  对真实流上小于 `adaptive_floor` 的 jitter 做软封顶。
- **A-5 A4 分类器阈值覆盖**：`classifier_threshold_override` 配置项 +
  `tools/calibrate_classifier_threshold.py` 辅助工具。默认保持 artifact 阈值。
- **A-6 A3b L0 多尺度**：
  1. 去掉 `F.interpolate` 升采样 round-trip（原来把半尺寸 crop 先拉回原尺寸再扔给
     extractor，extractor 又 resize 到 416）。
  2. 每次 L0 调用只做 **1 次** GPU→CPU transfer，然后 numpy 切片 4+1 个 crop。
  3. bg_edge suppression 的 border mask 去掉，用 4 条 strip 直接计算。
- **A-7 批量标量同步**：
  - `motion_artifact`：ROI 循环里的 3 个 `.item()` 合并成一次 `torch.stack().cpu()`。
  - `blur_degradation`：ROI 循环里的 2 个 `.item()` 合并。
  - `light_flow`：compute() 结尾 7 个独立标量合并一次 sync。

### P2 工程化
- `pytest.ini` 让 tests 可从任意 cwd 运行。
- 新增 `tests/profile_feature_timings.py`, `profile_a3b_internals.py`,
  `calibrate_classifier_threshold.py`。
- 单元测试从 46 扩到 **57 项**（path resolution、A4 threshold override、
  temporal adaptive baseline）。

### P3 模块 B
- `tools/fix_data_yaml_path.py` + `tools/run_green_check.ps1` 替代临时 batch。
- `pytest -q` 122 项 baseline 全绿。

### P4 联合
- `探索/joint_baseline.yaml` 统一 YAML 模板（设计）。
- `探索/joint_run_smoke.ps1` 一键联合冒烟脚本（设计）。

## 未完成 / 后续

- A-11 Monitor_App 3382 行拆分（仅出了方案 `探索/A_monitor_app_split_plan.md`，
  合并前动大手术风险高）。
- p_synth 上线策略（架构说明里仍在开发中）。
- `static_media/detector.py` 的 legacy ROI 循环还有 3-5 个 per-ROI `.item()`，
  风险较高暂不重构。
