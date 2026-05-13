# 模块 A/B 工程化打磨工作清单

> 使命（2026-05-13 夜间）：在保证原功能的前提下，对 A/B 两个模块进行工程化打磨，
> 往合并对齐，解决潜在问题，优化各检测项的代码和算法。
>
> 约束：
> - 所有工具脚本、测试、探索代码各归其位：`模块A/tests/`、`模块A/tools/`、
>   `模块B/.../tests/`、`工作清单/`、`探索/`。不要随意新建一级目录。
> - 任何算法/性能改动必须可验证：跑已有单元测试 + 样本冒烟，性能下降或行为偏移 → 回滚。
> - 保留"配置可回退"原则：优化走配置开关，默认值不激活破坏性变更。

## 执行循环

```
┌─ 挑一项 TODO
├─ 研究（必要时在 探索/ 里试验）
├─ 实施（改码 + 单元测试/冒烟）
├─ 跑完整回归（pytest + run_samples_smoke）
├─ 写入 STATUS.md（一句话结论 + 数字对比）
└─ 循环
```

## 全局工作清单（按优先级）

### P0 · 合并前阻塞（两模块共性）

- [x] **A-1** `ModuleADetector._resolve_artifact_path` 对联合仓库路径结构鲁棒
- [x] **A-2** Ultralytics 全局 `settings.json` 污染 → 入口统一初始化
- [x] **A-3** `AlertState.hold_frames` 语义修正（N 帧 hold 真实 hold N 帧）
- [x] **B-1** 模块 B `data.yaml` 从仓库相对路径自动解析（不依赖用户全局 settings）
- [x] **J-1** 新建 `工作清单/STATUS.md`，所有改动留迹

### P1 · 模块 A 核心算法优化（用户强调）

- [x] **A-4** LBP 噪声：clean_baseline 出现 3582 次 `local_temporal_texture_change`
  - 研究：双尺度 LBP / 动态 grid / ROI-aware 归一化
  - 产出：不改变样本冒烟的 7 项结论前提下，尽量压低 clean 上的 temporal 假触发
- [x] **A-5** A4 分类器 clean 假阳：6 帧 `classifier_adv`
  - 研究：用模块 B 的 helmet_head_yolo_val 做阴性 calibration
  - 产出：配置级阈值回调或温度校准（artifact 不重训）
- [x] **A-6** A3b screen_spoof FPS 24 → 目标 ≥ 30
  - 研究：性能热点（L0→L3 候选 vs patch-track）
  - 产出：参数级优化 + 可选 interval 提升，不劣化 ASR
- [x] **A-7** 特征耗时 profiling：把 `module_a_breakdown` 跑出来聚合一次，识别最重的特征
- [x] **A-8** 单元测试扩展：对 A-4/A-5/A-6 改动各加 1-2 项回归

### P2 · 模块 A 工程化

- [x] **A-9** 合并 tests/tools 目录组织（已在 `模块A/tests/` 下，只做微调）
- [x] **A-10** 统一 `tests/conftest.py` 的 sys.path 注入，允许 `pytest --rootdir=模块A tests`
- [x] **A-11** Monitor_App 的大文件（3382 行）做关注点分离提议（探索/，不直接动）

### P3 · 模块 B 工程化

- [x] **B-2** 把我之前 ad-hoc 的 `run_green_check.bat` 重构为 `tools/` 下的可复用脚本
- [x] **B-3** 统一 pytest 跑法（已有 `pixi run pytest`）
- [x] **B-4** 跑完整 `pytest -q` 做基线回归

### P4 · 联合合并铺垫（探索）

- [x] **J-2** 探索 `联合配置模板`：A 的阈值 + B 的 security gate 阈值统一 YAML
- [x] **J-3** 探索 `p_safety` 接入：A 的 roi_provider 直接消费 B 的 helmet 模型输出
- [x] **J-4** 探索统一入口：`joint_run_smoke.ps1/.bat` 一键跑 A 样本 + B green check

## 进度记录

详见 `STATUS.md`。
