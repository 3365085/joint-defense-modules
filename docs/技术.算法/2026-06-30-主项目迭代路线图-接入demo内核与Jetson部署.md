# 2026-06-30 主项目迭代路线图：全力接入 demo 内核 + 保留隔帧外壳 + 上 Jetson Nano

## 0. 北极星（用户 2026-06-30 定）

主项目原生 A1-A4 从设计到实装就不合理，故做了 demo。本路线图**不修 legacy A1-A4**，而是：
1. **全力接入 demo 的方法**（rebuilt 内核 = 正确检测逻辑，已移植、默认启用、在线）。
2. **保留主项目的隔帧检测/检测复用/节流外壳 + Module B + 安全准入**（轻量外壳包 demo 内核）。
3. **终极目标：主项目在 Jetson Nano 上按基础文档设想运行**（<15ms/帧、物理对抗 ~92.7%）。

统一正常视频基准：`素材\手机随意录制的视频\固定镜头室外视频_1080.mp4`。

## 1. 已确立的事实基线（代码/实测依据，非假设）

- **架构张力已证伪**：隔帧外壳只复用 YOLO 框（`video_defense_pipeline._run_detection:331-356`），
  `detector.process(frame=frame_640)` 每帧都调、每帧传完整帧（:358）→ rebuilt 内核逐帧算光流、
  `prev_gray` 连续，**隔帧不打断光流连续性**。"保隔帧 + demo 质量"架构上成立。
- **Jetson 回退路径是活的**：rebuilt 光流三级回退 RAFT-TRT→GPU-LK→DIS-CPU（detector.py:413,848），
  `_NATIVE`(Rust) 全程 `is not None` 守卫缺失即纯 Python。实测强制 `_NATIVE=None`+`_flownet=None`
  跑通，RTX5060 CPU ~37ms/帧（路径健康；Jetson CPU 更慢，是优化目标）。
- **检测质量基线（实测，manifest heldout 48 段）**：干净误报 7.4% 达标；攻击召回 71.4% 未达标，
  **adv_patch 1/4 为主缺口**。详见 [[2026-06-30-主项目rebuilt内核留出集诚实评测]]。
- **demo 性能已挖到 ModuleA ~12ms@RTX**（双进程三线程 + Rust native + GPU 卸载，见 demo optimization-roadmap）。
- **demo 检测质量天花板（demo 自己的诚实结论）**：adv_patch 弱攻击与干净突发在 20 维特征上无法分开，
  根因是**真实攻击场景仅 ~7 个**，非模型/非干净数据；XGBoost(AUC 0.819) 已是当前数据下最优，CNN 更差。
- **工具链齐备**：demo `tools/` 有 train_a4_auto/eval_detection_broad/collect_a4_features/
  relabel_a4_features/verify_*_native/bench_*；Rust 源码在 `rebuilt_demo/native/module_a_native/`(已编译可用)。

## 2. 两个真实缺口（非架构问题）

| 缺口 | 现状 | 性质 | 验收口径 |
|---|---|---|---|
| A. 检测质量 | adv_patch 召回 1/4 | 疑 XGBoost 特征漂移 | 双口径：基准视频零误报 + heldout 召回↑ |
| B. Jetson 时延 | CPU 回退桌面 ~37ms | 需 Rust native + 光流分摊 | <15ms/帧（基础文档），先在桌面 CPU 模拟达标 |

## 3. 迭代路线（按价值/风险/依赖排序）

### 阶段 0：建立可信回归门禁（前置，必须先做）
- **0.1** 把 `_eval_heldout_rebuilt.py` 转正为 `tests/` 或 `tools/eval_*` 留出回归门禁（双口径：
  基准视频零误报 + heldout 召回/误报）。HANDOFF §9.4 要求。**没有门禁，后续每轮"是否退化"无从判断。**
- **0.2** 补齐移植遗留的 3 个未跟踪测试期望的生产 API（`visualization.scale_ppe_tracks`、
  `detector_backend.letterbox_image/scale_boxes_from_letterbox`、`frame_processor.prepare_detector_frame`），
  让全量 `pixi run smoke` 转绿。这些是 demo 契约，补齐即完成移植。

### 阶段 1：缺口 A — 检测质量（adv_patch 1/4 根因定位）
- **1.1（最高杠杆）特征对拍**：同段视频，demo detector 与主项目 rebuilt 内核各导出 20 维 A4 特征，
  逐维 diff。`verify_prev_invariant.py`/`collect_a4_features.py` 可复用。
  - 若**特征漂移** → 1.2a：修正 rebuilt 内核特征提取与 demo 对齐（首选，pkl 不用重训）。
  - 若**特征一致** → 说明是隔帧外壳/输入预处理差异（letterbox vs demo resize）导致 → 1.2b：对齐输入路径。
- **1.3** 抽帧核验 016_multi_person 场景（既干净狂报又攻击漏报），看触发原因码是否多人运动虚假触发。
- **注意**：demo 自己已确认 adv_patch 召回受限于真实攻击场景多样性（~7 个）。1.x 的目标是
  **把主项目拉到 demo 同等水平（不是超过 demo）**；若对齐后仍 ~demo 水平，则缺口 A 的剩余部分
  属数据问题（需实拍真实攻击素材），非代码可解 —— 这点要诚实标注，不在固定素材上过拟合。

### 阶段 2：缺口 B — Jetson 可跑性
- **2.1** 在 Jetson 上编译 `module_a_native`（Rust/pyo3）：A1 LBP/A2 change/A3b/blinding 热点。
  demo 实测 a1 4.3→0.5ms、a3b 44→25ms。这是 Jetson CPU 路径的主要提速来源。
- **2.2** 光流隔帧分摊：当前 rebuilt 每帧算光流；Jetson 上 DIS-CPU 光流是大头，
  可按隔帧间隔（每 N 帧算一次光流、中间帧 hold/外推）分摊，**但须验证不破坏检测质量**（对拍）。
  这是"保隔帧"外壳与"逐帧光流"的真正结合点，需小心。
- **2.3** 桌面 CPU 模拟 Jetson 达标后，再上真机验证（无 TensorRT 时 YOLO 走 ONNX-CPU 或更小模型）。
- **2.4** Jetson YOLO 后端：`.engine` 不可跨设备，Jetson 需重新导出 TensorRT 或用 onnxruntime。

### 阶段 3：Jetson 真机验证（收尾）
- 桌面 CPU 模拟达标后上真机；Module B 准入/净化在 Jetson 上的可跑性随阶段 2 的设备适配一并验证
  （Module B 已是主项目完整实装项，非待办，仅需确认 Jetson 设备/依赖兼容）。

> **不做项（明确）**：
> - **多摄像头协同**：基础文档列为"可选"，用户确认不做。
> - **Module B 后门扫描/净化"接入"**：已是主项目**完整实装且深度接入**的强项，非待办 ——
>   `ModelSecurityService` 接在 FastAPI（`fastapi_app.py:20`），`/api/start` 走 `prepare_runtime_for_start`
>   准入门（阻断/净化替换/热切换）；主项目 `model_security/` 4924 行 + 后端引擎 `model_security_gate/`
>   129 文件（扫描/净化/AutoDetox/weight-soup 实算）+ 3 个测试文件 70+ 用例。HANDOFF §3 明确
>   "模块B：主项目 ✅已接入 / demo ❌"。基础文档的 B1/B2 是设计步骤，主项目早已超额完成，不重做。

## 4. 每轮迭代的 git 纪律（用户要求"合理用 git 循环迭代"）

- 每个可验证里程碑单独 commit，中文 message 含验收数字（HANDOFF §6）。
- 改 rebuilt 内核检测语义/阈值前，先跑双口径基线（基准视频 + heldout），改后再跑，对比数字才提交。
- 提交前不替用户做视觉验收（AGENTS.md 提交前视觉验收由用户亲跑）。
- 改坏了 `git checkout HEAD -- <file>` 回退。

## 5. 关键风险与诚实标注

- **缺口 A 的天花板可能是数据**：demo 已证 adv_patch 召回受限于真实攻击场景多样性。代码对齐能把主项目
  拉到 demo 水平，但**突破 demo 水平需实拍真实攻击素材**，非本路线图代码工作可解。**待实验确认特征是否漂移**。
- **光流隔帧分摊（2.2）是质量风险点**：分摊会改变 `prev_gray` 间隔，可能影响光流类攻击召回，必须对拍验证。
- **legacy 路径调优（已提交基线）将在调 rebuilt 时被重审**：用户已知其 git 不完全可信，仅作可回退起点。
- Jetson 真机未在手，2.x 桌面模拟达标 ≠ 真机达标，最终须真机验证。

## 6. 建议的下一步执行顺序

阶段 0（门禁+移植收尾）→ 阶段 1.1（特征对拍定位 adv_patch 根因）→ 据结果决定 1.2a/1.2b
→ 阶段 2（Jetson）。阶段 0 与 1.1 可部分并行（门禁脚本与特征对拍互不依赖）。
