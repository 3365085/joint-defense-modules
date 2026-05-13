# 联合配置模板设计笔记

## 目标

为最终合并后的仓库提供一份**统一的 YAML 配置模板** `joint_baseline.yaml`，
让 A（流式检测）和 B（模型安检）可以被同一个 runtime 读入。

## 观察到的需求

1. **A 侧**读 `experiments/configs/module_a_baseline.yaml`（约 100 项
   module_a.* 键），启动时把它塞进 `ModuleADetector(config)`。
2. **B 侧**读 `configs/*.yaml` 的一族（security_gate、strong_detox、
   risk_thresholds 等），每个脚本单独解析。
3. 模型/数据路径：
   - A 的 YOLOv5 路径是 `baseline_training/runs/baseline_yolov5/weights/best.*`
   - B 的 YOLOv8 净化模型是 `artifacts/current_best/best2_purified_semantic_fixed_2026-05-09.pt`

## 设计

`joint_baseline.yaml` 顶层按模块分节，保持 A/B 原有键名不动，
方便现有脚本零改动复用：

```yaml
# ============= 共享 =============
joint:
  name: 联合防御基线 2026-05-13
  repo_layout_version: 1

# ============= 模块 A =============
module_a:            # 沿用 defense.module_a.ModuleADetector 可读键
  require_gpu: true
  # ... 从 experiments/configs/module_a_baseline.yaml 原样复制
inference:           # 沿用 create_detector_backend 可读键
  backend: tensorrt
  device: cuda:0
  # ...

# ============= 模块 B =============
module_b:            # 为 joint runner 聚合 B 的常用路径
  final_model: artifacts/current_best/best2_purified_semantic_fixed_2026-05-09.pt
  poisoned_model: models/best_2_poisoned.pt
  clean_val_yaml: data/helmet_head_yolo_val/data.yaml
  external_hard_suite: data/poison_benchmark_tuned_val
  try_attack_images: data/try_attack_data
security_gate:       # 沿用 scripts/security_gate.py 可读 YAML 段
  # ... (B 的既有配置段)
strong_detox:
  # ...
```

## 读取策略

- A 的 monitor_app：保持读 `experiments/configs/module_a_baseline.yaml`（历史兼容）
  但同时提供 `--config joint_baseline.yaml`，读入后只取 `module_a.*` 和
  `inference.*` 两段，其他段落忽略。
- B 的脚本：通过 `--config joint_baseline.yaml` 只取自己那段。
- 联合 runner（`tools/joint_run_smoke.ps1`）解析自己那段并显式传给两边。

## 验证约束

- 合并后 A 和 B 的单元测试全部保持通过。
- Monitor_App 能同时读新老 config（老 config 里没有 joint/module_b 段，不影响）。
- 输出 evidence 保持各自的路径不变（`异常记录/`、`outputs/green_check/`）。

## 下一步

1. 写一个合并 YAML 的小工具，把 A 的现有 baseline 和 B 的 security_gate 合并到
   `joint_baseline.yaml`，作为可复现的参考。
2. 写一个 `tools/joint_run_smoke.ps1`，按序跑：
   - A 的样本视频 smoke（7 clip）
   - B 的 green_check（4 步）
   - 最后输出聚合 `joint_smoke_report.json`
