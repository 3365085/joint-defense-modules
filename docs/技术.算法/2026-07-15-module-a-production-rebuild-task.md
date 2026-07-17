# Module A 主项目重建、GPU/Rust 加速与真实视频验收任务

日期：2026-07-15  
状态：任务边界已建立；生产行为修改暂停，等待按本文分阶段执行  
主项目：`D:\联合防御模块\model`

## 1. 任务目标

本任务不是继续叠加阈值、gate、bridge 或 hold，而是重新完成以下闭环：

1. 主项目与 `rebuilt_demo` 完全分离。
2. 生产只使用用户指定的唯一 YOLO 模型。
3. 用 profiler 先定位真实热路径，再实施 GPU/Rust 底层优化。
4. 稳定 Module A 的特征、模型、时序和真实 Web/latest-only 行为。
5. 只使用用户指定的视频完成最终 Web 验收。
6. 用户本人验收前不提交 Git。

## 2. 用户指定的唯一输入

### 2.1 唯一 YOLO 模型

```text
D:\联合防御模块\素材\model\yolov8\mask_bd_v4_clean_baseline.pt
```

```text
大小：5,347,205 bytes
SHA-256：4D7A23D3866AC2D9DB6E59AE537DA1274D988BD53CA6C7D519297FCBB96626F8
```

生产只允许使用该模型及由该模型派生的 ONNX/TensorRT artifact。禁止继续使用：

- `baseline_training/runs/classmate_maskbd_v4/*`
- 浏览器保存的自定义模型
- 历史 profile 模型
- `rebuilt_demo` 中的 detector/model artifact

A4/XGBoost 和 RAFT 属于 Module A 内部算法 artifact，不是可替换该 YOLO 模型的用户 detector；它们也必须由主项目管理并绑定 schema/hash。

### 2.2 A3b 权威目标视频

```text
D:\联合防御模块\素材\视频中出现干扰视频\5e145bf778577e75118502e263d00c41.mp4
```

```text
大小：565,888 bytes
SHA-256：FD2915BAD00FD033596D88307F04DD857F24F3BDA9BFE9B370F51F156C80C4CE
```

该视频是 A3b 触发、持续性和 Web 显示的唯一权威目标视频。

### 2.3 Module A 物理攻击权威视频

根目录：

```text
D:\联合防御模块\素材\物理扰动攻击视频
```

当前共 5 段 MP4：

| 类型 | 视频 | SHA-256 |
|---|---|---|
| adv_patch | `adv_patch\raw_attack_adversarial_patch.mp4` | `2CF7B13D523956AF28C4AC9373CEDF2C9E689E6AAC5EA2F0192DEBD5E8E0377F` |
| glare | `glare\raw_glare_attacked.mp4` | `0C71FEBE4C8E54DCA9D362FCD4640937E76F54203929B491991AA6DF54B23713` |
| motion_blur | `motion_blur\raw_motion_blur_attacked.mp4` | `E1C9948DE60A18C4E5AC2B54B705DE08003251DA3F38ADE02745C7E330C9C94A` |
| occlusion | `occlusion\raw_occlusion_attacked.mp4` | `F15B3A6CB8821EB4DEB1C7C0E78285E75054E97DC560D40DCB230B75DC25CAB9` |
| visibility_degradation | `visibility_degradation\raw_visibility_degradation_attacked.mp4` | `79D4C71F15ED9395AC3B4F263DC550DE7AA8DC61F6C0BD6876F8E4ED8DE9F2CA` |

这些视频必须在真实 FastAPI Web、MonitorEngine、latest-only 和生产 overlay 链路中验收；不能只跑离线逐帧脚本。

### 2.4 Module A 正常视频权威集合

固定镜头正常视频：

```text
D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频_1080.mp4
```

```text
大小：78,996,992 bytes
SHA-256：3E5B7FB40CD6B9500273831B9E609C7A998FCDBF4FB80FF19A54F2382F677139
```

真实正常视频目录：

```text
D:\联合防御模块\素材\真实视频\**\*.mp4
```

删除与权威 `adv_patch` 字节完全相同的误放副本后，当前快照为 29 段 MP4。用户已明确这些视频全部按“正常、无攻击”验收；文件名不改变用户给定标签。

加上固定镜头室外视频后，最终正常集合当前共 30 段。不得再用 `rebuilt_demo` heldout 或未得到用户认可的临时视频替代。

## 3. 当前已确认的问题

### 3.1 主项目仍依赖 demo

当前已发现的生产/诊断引用包括：

- `model/configs/module_a_runtime.yaml`
  - YOLO artifact 仍指向 `baseline_training/runs/classmate_maskbd_v4/*`
  - A4 仍指向 `rebuilt_demo/data/a4_classifier.pkl`
- `model/src/defense/module_a/rebuilt/detector.py`
  - 数据目录解析仍回退到 `rebuilt_demo/data`
- `model/src/defense/diagnostics/a4_training.py`
  - 默认 manifest 仍是 `rebuilt_demo/data/dataset_manifest.csv`
- `model/tools/run_a3b_heldout.py`
  - 默认 manifest 仍来自 `rebuilt_demo`
- Rust crate 源码仍在：
  `rebuilt_demo/native/module_a_native`

结论：主项目和 demo 尚未分离。

### 3.2 当前唯一模型未成为生产默认

当前 Web 默认 TensorRT engine 不是由用户指定的
`mask_bd_v4_clean_baseline.pt` 明确派生和绑定。浏览器还存在自定义模型配置路径。

结论：必须锁定唯一模型、生成对应 engine，并在状态中显示源 `.pt` 与 engine 的 hash 绑定。

### 3.3 解码仍主要使用 CPU

生产文件源当前通过 `cv2.VideoCapture` 解码。当前环境：

```text
GPU：NVIDIA GeForce RTX 5060 Laptop GPU
torch CUDA：可用
OpenCV CUDA：不可用
OpenCV cudacodec：不可用
PyNvVideoCodec：已安装
```

结论：具备实施 NVIDIA NVDEC 的条件，但必须避免 GPU 解码后无条件整帧下载到 CPU。

### 3.4 Rust 不是可复现的主项目组件

当前 `.pyd` 在本机 Pixi 环境中可以加载，A1/A2/A3/A3b/blind 部分统计会调用它；但：

- 源码在被 Git 忽略的 `rebuilt_demo`；
- 没有主项目 Pixi native build/install/verify 闭环；
- 没有生产版本/hash/阶段命中可观测性；
- 新增策略没有对应 profiler 证明，也没有基于热点继续扩展 native。

结论：本机可加载不等于可交付。

### 3.5 性能主要靠跳帧和策略墙维持

历史真实 Web/latest-only 运行曾出现约 63.3% 的源帧处理覆盖率。离线逐帧召回不能代表 Web，且过多 gate 导致“分数很高但不确认”。

结论：先降低单帧真实成本，再决定必要的 backpressure；不得继续把跳帧当作底层优化。

### 3.6 A4 候选模型未接入且不合格

当前 16 维候选模型仅位于 runtime 诊断目录，生产仍是旧 20 维契约。候选模型 heldout 回归结果为：

```text
clean FP：4/27
attack hit：16/21
```

结论：不得接入该候选模型，也不得覆盖生产资产。

### 3.7 CodeGraph 当前不可用

2026-07-15 多次调用 CodeGraph 均返回：

```text
Transport closed
```

在 CodeGraph 恢复前，影响分析使用源文件调用链、测试和运行证据替代；恢复后必须补做 impact 审查。

## 4. 执行原则

1. **先分离，再优化，再调效果。**
2. **先 profiler，再决定 GPU/Rust 改造点。**
3. 不再以新增 gate、提高/降低阈值、延长 hold 作为性能或模型问题的首选修复。
4. 不隐藏分数、候选、失败门、事件或红色告警。
5. 不用最终验收视频训练模型或选择阈值。
6. 训练数据不足时明确记录数据缺口，不以同一批视频训练后再验收。
7. 离线逐帧只作诊断；最终结论必须来自真实 Web/latest-only。
8. 每一阶段先形成独立测试和回滚点，再进入下一阶段。

## 5. 分阶段任务

### Phase 0：冻结与基线审计

Owner：`lead/integrator`  
并行只读 reviewer：允许

- [x] 对当前未提交 diff 分类：底层计算、模型/schema、runtime 接线、策略 gate、Web/UI、测试和诊断。
- [x] 标出没有生产接线、仅测试可见或仅 runtime 生成的改动。
- [x] 建立当前唯一模型和权威视频 manifest，记录路径、大小、SHA-256、标签和用途。
- [ ] 对当前真实 Web 跑一次不调参基线，记录每阶段耗时、FPS、覆盖率、告警和事件。
- [x] 输出建议保留、撤销、重写的改动清单；不直接批量回退他人工作。

完成门：用户可以看清当前每一类修改是否真实生效。

### Phase 1：主项目与 demo 完全分离

Owner：`runtime/integration agent`  
共享配置由 `lead/integrator` 修改

- [x] 移除生产代码和配置对 `rebuilt_demo` 的读取、fallback 和默认路径。
- [x] 把 A4、RAFT、native 和 detector artifact 解析改为主项目管理路径。
- [x] 诊断工具默认使用本任务定义的素材 manifest，而不是 demo manifest。
- [x] 增加禁止生产路径出现 `rebuilt_demo` 的聚焦测试。
- [x] Web status 输出所有已加载 artifact 的绝对路径和 SHA-256。

完成门：

```text
生产启动、Web 运行、A3b、Module A、模型加载、native 加载均不访问 rebuilt_demo。
```

### Phase 2：锁定唯一 YOLO 模型

Owner：`runtime/integration agent`  
模型转换验证：`测试验证 agent`

- [x] 配置中只保留用户指定的 `mask_bd_v4_clean_baseline.pt`。
- [x] 删除或禁用生产 Web 的自定义模型和历史 profile 模型选择。
- [x] 由该 `.pt` 生成 ONNX 和 TensorRT FP16 engine。
- [x] engine metadata 绑定源 `.pt` SHA-256、TensorRT/CUDA 版本、输入尺寸、类别顺序和构建参数。
- [x] 启动时若 engine 与源模型 hash 不匹配，明确失败或重建；不得静默使用其他 engine。
- [x] Web 显示：
  - 源模型路径/hash
  - 实际 backend
  - 实际 engine 路径/hash
  - 类别顺序

完成门：任何 Web 启动都只能看到该模型派生的 TensorRT backend。

### Phase 3：GPU 解码和数据传输优化

Owner：`runtime/GPU agent`  
写入范围：`src/defense/pipelines` 和经 lead 指定的 runtime 接线文件

- [x] 建立 CPU OpenCV 与 PyNvVideoCodec/NVDEC 的同视频基准。
- [x] 新增统一 video decoder adapter，不把 NVDEC 逻辑散落在 runner。
- [x] 文件视频优先 NVDEC；摄像头/不支持编码可回退 CPU。
- [x] 解码状态必须显示：
  - decoder backend
  - codec
  - GPU device
  - decode p50/p95
  - GPU→CPU copy p50/p95
  - fallback reason
- [ ] 设计最少拷贝路径：
  - detector/TensorRT 和 RAFT 尽量复用 GPU frame/preprocess；
  - Module A CPU/Rust 只下载必要的缩小灰度图或必要 ROI；
  - 禁止每个模块重复整帧颜色转换和 GPU→CPU 下载。
- [x] preview 与 detection 保持解耦；preview 不得阻塞检测。
- [x] latest-only 保留为压力保护，而不是正常文件播放下的主要性能来源。

完成门：

- 1080p 权威正常视频在本机目标为实际 Module A 检测吞吐不低于 25 FPS；
- 不再出现约 12 FPS 的默认生产运行；
- 正常文件播放期间检测源覆盖率目标不低于 90%；
- 达不到目标时必须给出 profiler 证据，禁止通过新增跳帧掩盖。

### Phase 4：Rust/native 主项目化和热点优化

Owner：`native/Rust agent`  
reviewer：独立

- [x] 将 `module_a_native` crate 迁入主项目受控位置，例如：
  `model/native/module_a_native`。
- [x] 增加 Pixi 任务：
  - `native-build`
  - `native-install`
  - `native-verify`
  - `native-benchmark`
- [x] 记录 Rust 源码 hash、crate 版本、`.pyd` hash 和构建环境。
- [x] 为 A1/A2/A3/A3b/blind 现有 native 函数建立 Python 对拍和性能门禁。
- [ ] 只根据 Phase 0/3 profiler 迁移新的热点，优先审查：
  - 重复灰度/颜色转换
  - LBP 生成与统计
  - A3 ROI/ring 统计
  - A3b 候选生成后的批量统计
  - blind/曝光批量统计
- [x] 避免“每个 ROI 一次 Python→Rust 调用”；必要时改为一帧批量输入、一次 native 调用。
- [x] status 显示 native 可用性、版本、hash、命中阶段和 fallback。

完成门：主项目在全新 Pixi 环境可独立构建并加载 native，不依赖 demo 或旧 `.pyd`。

### Phase 5：A4/Module A 特征与模型闭环

Owner：`算法 agent`  
训练与验收必须分离

- [ ] 先固定 A1/A2/A3/blind/A3b 的生产输入、ROI、时序和 schema。
- [ ] 明确 A4 负责的攻击范围，避免让 A4、blind 和 A3b 对同一信号相互推翻。
- [ ] 删除异步 A3b 特征对同步 A4 schema 的隐式依赖。
- [ ] 重新审查现有上层 gate，删除没有量化收益或互相矛盾的策略墙。
- [ ] 若重新训练 A4：
  - 只用生产链采集；
  - 训练集和最终权威验收视频严格隔离；
  - 绑定 feature names/order、schema、detector hash、ROI 版本、预处理、数据 identity 和阈值；
  - 不接入 clean FP 或攻击召回不合格的候选模型。
- [ ] A3 正常人员运动必须有独立攻击证据才能确认。
- [ ] blind 必须区分正常转头/摘戴/人员运动和真实全局退化。
- [ ] sustained escalation 与 alert hold 不得推翻明确抑制或仅凭 raw score 无限续命。

完成门：模型分数、物理证据、候选和确认状态逻辑一致，不再出现长期高分但无法解释为何不报警。

### Phase 6：权威素材真实 Web 验收

Owner：`测试验证 agent`  
最终验收人：用户本人

#### A3b

- [ ] 使用指定 A3b 视频。
- [ ] 目标约在 frame 30 / 1.0 秒附近触发。
- [ ] 触发后不得因异步缓存、tighten gate 短暂翻转而反复闪断。
- [ ] Web 红色确认、事件和 evidence 必须一致。

#### 物理攻击

- [ ] 5/5 权威攻击视频都必须在真实 Web/latest-only 中产生 `alert_confirmed`。
- [ ] 不能只表现为 `p_adv` 或 `p_blind` 升高。
- [ ] 分别记录首次确认时间、主通道、物理证据、processed/source coverage 和事件数量。

#### 正常视频

- [ ] 固定镜头室外视频：确认告警 0，Module A evidence event 0。
- [ ] `素材\真实视频` 30 段：确认告警视频数 0/30，Module A evidence event 0。
- [ ] 不得通过隐藏红色状态、改文案、改颜色或不保存 evidence 达标。

#### 性能

- [ ] 默认 backend 是唯一模型派生的 TensorRT FP16。
- [ ] 文件解码优先 NVDEC，并显示实际 backend。
- [ ] Module A 检测吞吐目标不低于 25 FPS（1080p、当前 RTX 5060 Laptop GPU）。
- [ ] 检测源覆盖率目标不低于 90%。
- [ ] preview 流畅且与检测状态同步；不得出现拖框、断框、旧框滞留或 12 FPS 级别默认运行。
- [ ] GPU、native、decoder 或模型初始化失败必须在 Web/status 和日志中可见。

## 6. 并行任务划分

| Scope | Owner | 主要写入范围 | 与其他 scope 的边界 |
|---|---|---|---|
| 主项目/demo 分离与唯一模型接线 | runtime/integration | runtime、config、artifact resolver、Web model options | 不调 Module A 阈值 |
| GPU 解码与最少拷贝 | runtime/GPU | pipelines decoder、runner 接线、性能状态 | 不改变告警策略 |
| Rust 主项目化与热点算子 | native/Rust | `model/native`、native Python bridge、Pixi tasks | 不改模型权重和阈值 |
| A4/物理算法 | 算法 | `module_a`、训练/metadata | 不修改生命周期和 Web 协议 |
| 权威视频与量化 | 测试验证 | diagnostics、tests、runtime reports | 不替用户作最终验收 |
| 跨改动审查 | reviewer | 只读或独立审查报告 | 不实现被审查的同一 scope |
| 集成 | lead/integrator | 共享配置、任务文档、冲突处理 | 不顺手扩大 scope |

## 7. 明确禁止

- 禁止生产路径继续读取 `rebuilt_demo`。
- 禁止使用用户指定模型之外的 YOLO 权重。
- 禁止用旧 27/21 或 demo heldout 替代最终权威视频。
- 禁止把最终验收视频用于模型训练或阈值选择。
- 禁止未 profiler 就继续增加隔帧、候选 gate 或 hold。
- 禁止用 UI 隐藏、颜色调整、事件丢弃规避误报。
- 禁止只验证离线逐帧而宣称 Web 已解决。
- 禁止用户本人验收前提交 Git。

## 8. 交付物

- [ ] 主项目独立 artifact 清单和 release manifest。
- [x] 唯一 YOLO 模型→ONNX→TensorRT 的 hash 绑定记录。
- [ ] GPU decode/transfer/inference/Module A 分阶段 profiler。
- [x] 主项目 Rust crate、Pixi 构建任务和 native 对拍/benchmark。
- [ ] 稳定的 A4 schema/metadata；若数据不足则提交明确数据缺口报告。
- [ ] A3b、5 段攻击、30 段正常视频的真实 Web 报告和结果视频。
- [x] 独立 reviewer 审查记录。
- [ ] 用户 Web 验收后再决定是否提交。

## 9. 2026-07-16 当前可复核进度

### 已通过的集成门

- `pixi run smoke`：`789 passed, 33 skipped`（Model Security v2 改造前最后一次全量门；
  v2 focused 当前为 `84 passed, 6 skipped`，待算法 owner 整合后再跑最终 smoke）。
- Runtime temporal/tuning focused：`27 passed`。
- NVDEC runtime/lifecycle/status focused：`38 passed, 1 skipped`。
- Web authoritative report contract：`9 passed`。
- authoritative model hash/cache hardening：`20 passed`。
- native source attestation：
  - binary SHA-256：
    `E27D19CFB8860F73D795A448BC8636F9A2638C557F9FF2F1556470D64E09D5B2`
  - build-time/current source SHA-256：
    `4E6FA270B731672E19A3D59A7E01A86FBE3C3B84CE405BEE58342B5692A343B4`
  - `source_attestation_match=true`
  - `tests/native`：`77 passed`
- 最新 Web preflight：
  - `ok=true`
  - `preflight.passed=true`
  - `blockers=[]`
  - `results_generated=false`
  - v2 trust-store 后证据：
    `runtime/benchmarks/web_preflight_after_trust_seal_v2_2026-07-16.json`

### Model Security host 迁移与 seal v2

- 旧 seal 的 registry hash 与 legacy signature 均可复核，但旧 host aggregate
  `sha256:b775...` 与当前 `sha256:d80d...` 不同；v1 只保存 aggregate，无法反推出
  哪个原始组件发生变化。
- 已删除“由公开 host hash + 固定 salt 推导 HMAC key”的 v1 安全边界；v2 使用独立随机
  32-byte signing key，当前 Windows 生产 key 以 DPAPI current-user 保护。
- v2 seal 严格绑定并签名：
  `key_id/registry_hash/host hash/status/sources/warnings/updated_at`，未知顶层字段拒绝。
- existing v2 key missing 或 key replacement 均 fail closed；普通 registry 写入不得静默
  rekey。legacy v1 迁移若预置 signing-key 文件也拒绝。
- 显式 host rebind/migration 使用：
  - exact previous host hash；
  - meaningful operator reason；
  - verified old seal bytes；
  - byte-for-byte backup；
  - signed durable journal；
  - signed committed/rolled_back audit；
  - pending journal admission hard block；
  - 显式 recovery。
- registry/seal/key/journal/lock/audit/json-out 做 resolved/samefile alias 检查；scan 在
  compromised/pending store 上同步和后台路径均拒绝，HTTP 返回结构化 `409`。
- status 现暴露：
  `trust_store_seal_schema_version=2`、
  `trust_store_signing_key_status=present`、
  `trust_store_signing_key_protection=windows_dpapi_current_user`、
  `trust_store_transition_pending=false`。
- 当前生产唯一模型已完成 full scan：
  - source SHA-256：
    `4D7A23D3866AC2D9DB6E59AE537DA1274D988BD53CA6C7D519297FCBB96626F8`
  - external rows：`16`
  - `max_asr=0.0`
  - `status=clean`
  - 当前 `admission_status=trusted`
- 当前 v2 state 已追加 protected-key signed attestation：
  `runtime/benchmarks/model_security_trust_store_v2_final_attestation_2026-07-16.json`。
- Model Security focused：`84 passed, 6 skipped`。
- 独立 reviewer 最终复审：`P0=0`、`P1=0`；仅剩本地写权限 + 精确时序下
  audit path hardlink race 的 fail-closed P2（会使下次 admission 拒绝，不会产生未授权
  trusted admission）。

### reviewer P1 已处理

- Web acceptance 现在将 report 逐项绑定 manifest，并校验：
  `asset_id/identity/path/hash/expectation/order/run_id/source/source_epoch`。
- evidence 只统计当前 run/source/source epoch 的事件；其他运行或残留事件成为 blocker。
- 每资产新增硬门：
  `detector FPS >= 25`、coverage `>= 90%`、实际 NVDEC、decoder 无 fallback、
  capture source skip 为 0、native 真命中且 fallback 为 0。
- 移除 production 默认文件循环的 wall-clock 主动追帧跳过；默认
  `file_source_fps_cap=0` 时逐源帧发布，latest-only 只负责 detector stale work。
- authoritative model/artifact 安全 hash 不再使用 path/size/mtime 缓存。
- NVDEC seek 增加可取消路径；`stop()` 会请求取消活动 decoder，避免长 seek 阻塞重启。
- camera/RTSP 的 OpenCV host capture 与 fallback 原因现在在 decoder status 中可见。
- native binary 构建时嵌入源码聚合 hash；旧 binary、缺 attestation 或源码不匹配时拒载。

### 尚未完成，禁止宣称通过

- A4 classifier artifact 尚未安装；当前只有可见规则 fallback。
- A3/blind/A3b/A4 算法闭环尚未完成；算法 owner 因外部 503 中断后已恢复原 scope。
- 算法 owner 活动改动期间最新全量 smoke 暂为：
  `1 failed, 800 passed, 33 skipped`；唯一失败是
  `test_adv_hold_refreshes_only_while_adv_candidate_is_current`，
  已交回原算法 ownership，未由主线程越权修改。
- A3b 已在当前 production/latest-only、trusted admission 下真实跑完：
  `runtime/benchmarks/a3b_web_after_trust_rebind_2026-07-16.json`。
  当前硬阻塞为：
  - detector FPS `10.4 < 25`
  - coverage `0.70 < 0.90`
  - 首次 A3b true `6.633s`，目标 `0.5–1.5s`
  - true 区间内部 false 样本 `13`，目标 `0`
  - NVDEC、decoder fallback=0、capture skip=0、native hit/fallback、evidence correlation
    均已通过。
- 31 正常离线全帧根因报告已完成，但不能替代 Web：
  - `25/31` 视频出现告警，视频级误报率 `80.65%`
  - `23/25` 首发为 ADV；其中 direct N-of-M `12`、bridge `11`
  - `10/11` bridge 首发同时已有 normal scene gate=true
  - A3b confirmed 视频 `0`，A3b 不是这批误报首因
  - 主证据：
    `runtime/benchmarks/normal_wave3_root_cause_handoff_2026-07-16.json`
- 5 攻击离线全帧根因报告已完成，但不能替代 Web：
  - glare 稳健命中
  - occlusion 命中很晚且区间窄
  - adv_patch 候选稀疏、对 coverage 敏感
  - motion_blur 全帧仍漏报，属于 blind score/gate
  - visibility_degradation 全帧仍漏报，属于时序确认/gate
  - 主证据：
    `runtime/benchmarks/physical_wave3_readonly_handoff_2026-07-16.json`
- 尚无 `A3b → 5 攻击 → 30 正常 → 性能` 的完整真实 Web 串行报告。
- 最终 Web 验收必须由用户本人执行。

### 权威数据 P0 冲突（已于 2026-07-16 解决）

用户确认
`D:\联合防御模块\素材\真实视频\12_监控视角_仓库巡检\attack_adversarial_patch.mp4`
是权威 `adv_patch` 攻击视频的误放重复副本，并授权删除。该文件及
`normal.real.27.attack_named_but_normal` manifest 记录已移除。

最终口径调整为 `1 段 A3b + 5 段物理攻击 + 30 段正常视频 = 36 段权威视频`。
