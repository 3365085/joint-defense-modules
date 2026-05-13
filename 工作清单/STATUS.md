# 工作进度日志

> 每完成一项 TODO 追加一条。格式：`时间 | 编号 | 改动一句话 | 验证结果`。

| 时间 | 编号 | 改动 | 验证 |
|---|---|---|---|
| 2026-05-13 start | - | 任务开始，已完成 A 的 46 单元测试 + 7 样本冒烟 + Web 冒烟 | baseline pass |
| 2026-05-13 01:30 | A-1/A-2/A-3 | 路径解析+MODULE_A_ROOT env+utils; Ultralytics settings 隔离 helper; AlertState.hold_frames 语义修正 N→真 N；单元测试同步 | 50 pytest ✅；7 clip smoke 不变（alert/p_adv 与基线一致）|
| 2026-05-13 02:00 | A-7/A-6 part 1 | 侦测 A3b 占 70% 后分析 L0 多尺度 fallback；把 4-quadrant 后的 F.interpolate 反向升采样去掉，crop 直接喂给 extractor | A3b p95 15-18→11-14 ms（-25%），7 clip smoke 依旧全绿（adv_patch 498→504, screen_spoof 599→549）|
| 2026-05-13 02:30 | A-4 | LBP 时域 adaptive baseline EMA + persistence gate + noise suppression（通过 `change_exposed`/`local_exposed` 回写）| clean local_t 3582→2173 (-39%), smoke 依旧全绿（alerts 不变）, 54 unit ✅|
| 2026-05-13 03:40 | A-5 | TorchLogisticFusion 加 threshold_override 参数；新增 tools/calibrate_classifier_threshold.py；测量 clean FP rate 0.5→0.9 threshold: 0.22%→0.13%（FP reduction 40%）且 attack detection 仅从 53%→52% | 57 unit ✅, 默认阈值不变（operator knob）|
| 2026-05-13 04:00 | B-2/B-3/B-4 | 新增 `模块B/tools/run_green_check.ps1` + `fix_data_yaml_path.py` 工具；B 仓 `pytest -q` baseline 122 项全绿 | OK |
| 2026-05-13 04:10 | J-2/J-4 | `探索/joint_baseline.yaml`（统一 YAML 模板）+ `探索/joint_run_smoke.ps1`（A 样本 smoke + B green check 聚合报告） | 设计文件就位，运行需合并后 |
| 2026-05-13 04:50 | A-6 round 2 | A3b 批量 GPU→CPU transfer：每次 L0 只一次 .cpu().numpy()，然后 numpy 切片做 scale-2 | A3b p95 从 ~11 ms → **~9-11 ms**（累计 -40% vs baseline），total p95 从 ~26 → **~20 ms**，smoke 7/7 ✅，unit 57 ✅|
| 2026-05-13 05:15 | A-7 更深 | blur_degradation + motion_artifact ROI 循环：每 ROI 的 `.item()`（共 ~36+ 次 GPU 同步）改成批量 `torch.stack().cpu().numpy()` 一次 | A3 p95 从 5-6 ms → **3-4 ms**（-35%），total p95 从 ~20 → **~19 ms**，smoke 7/7 ✅，unit 57 ✅|
| 2026-05-13 05:30 | A-10/B fix | B 仓 `test_hybrid_purify_config.py` 依赖 cwd 相对路径 → 改为文件位置相对，支持从联合根跑 pytest | 联合根跑 **179 tests 全绿**（57A + 122B）|
| 2026-05-13 06:00 | 收官回归 | 联合根 pytest 186 ✅（64A + 122B），7 clip smoke 全绿，web monitor 7862 冒烟正常 | 任务主线完成 |
| 2026-05-13 10:40 | A-edge NPU | 实现了 `torch_native` 后端（纯 torch L0 候选提取 + torch L2 平面性替代）。screen_spoof 检测恢复（1536 alerts），但 clean_baseline 出现 FP（3248 alerts）因为 A3+ 候选与 A4 classifier 的交互导致级联误触发。标记为 experimental，默认仍用 legacy。探索文档已更新。|
| 2026-05-13 11:00 | 收官 | 清理 pycache + 临时文件；legacy 默认模式 7/7 smoke 全绿（17-19 ms mean），联合 pytest 186 ✅。torch_native 作为 experimental 保留，代码就位待 NPU 硬件调参。|
| 2026-05-13 11:30 | A-display | 修复"置信度闪烁"问题：A3b 和 light_flow 在非运行帧保持上一次分数（hold-last-value），不再归零。前端看到平滑曲线。| 64 unit ✅, 7 smoke ✅, 功能不变 |
| 2026-05-13 12:00 | A-interval | `static_image_interval` 从 3 提到 10（检测率完全不变：549/504/0 alerts 一致）。A3b p50 从 ~9 ms 降到 **2.5-4.3 ms**，pipeline p50 = **15.8-18 ms**。90% 帧只花 2-4 ms 在 A3b，10% 帧花 ~11 ms（L0 运行帧）。| 64 unit ✅, 7 smoke ✅ |
