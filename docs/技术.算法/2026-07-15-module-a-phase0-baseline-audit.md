# Module A Phase 0 基线审计

日期：2026-07-15  
Owner：`lead/integrator`  
状态：进行中；本文只记录当前工作树与轻量环境探测，不代表真实 Web 验收完成

## 1. 审计基点

- Git 根：`D:\联合防御模块`
- 主项目：`D:\联合防御模块\model`
- 当前分支：`codex/demo-module-a-port`
- 当前 HEAD：`a85be8c`
- 相对远端：领先 `origin/codex/demo-module-a-port` 1 个历史提交
- 当前工作树：71 个 modified/untracked 条目
- 本阶段未执行 Git 提交。

## 2. 未提交改动分类

| 类别 | 文件数 | 当前判断 |
|---|---:|---|
| runtime / pipeline | 12 | 多数位于生产调用链，必须在集成门中验证 effective config、初始化失败可见性和实际命中 |
| Module A 算法 | 8 | 包含 A3b、blind、A4 契约及 ROI/时序变更；不得与底层优化混在同一完成结论中 |
| tests | 30 | 只能证明被覆盖的契约；不能替代生产 Web/latest-only |
| diagnostics / tools | 10 | 默认属于诊断链；必须逐项确认是否被生产入口调用 |
| Web / UI | 2 | 位于生产界面与启动 API，需要与后端唯一模型约束同步 |
| config / build | 1 | `configs/module_a_runtime.yaml` 是生产共享配置，由 lead 持有 |
| docs / governance | 7 | 任务边界与历史记录，不代表生产行为生效 |
| visualization | 1 | 生产 overlay 相关，需要验证事件、红色状态和 evidence 一致性 |

### 2.1 已确认仅测试或诊断可见

- `src/defense/module_a/rebuilt/a4_artifact.py`
  - 当前只被 `tests/test_a4_artifact_contract.py` 导入；
  - 未接入生产 A4 加载路径；
  - 结论：不能宣称 A4 schema/metadata 已进入生产。
- `src/defense/diagnostics/a3b_heldout.py`
- `src/defense/diagnostics/a4_training.py`
- `src/defense/diagnostics/module_a_tuning.py`
- `tools/run_a3b_heldout.py`
- `tools/train_a4_production.py`
  - 当前均属于诊断/训练工具，不是最终 Web 验收链路。

### 2.2 已确认进入生产调用链、仍需运行验证

- `src/defense/module_a/result_contract.py`
  - 已被 `video_defense_pipeline.py`、`runtime/frame_processor.py` 和诊断模块导入。
- `src/defense/runtime/*`
- `src/defense/pipelines/video_defense_pipeline.py`
- `src/defense/web/fastapi_app.py`
- `src/defense/web/static/index.html`

上述文件处于生产调用链不等于行为已正确生效；仍需真实启动、状态和视频运行证据。

## 3. 当前生产阻断项

### 3.1 主项目仍读取 demo/历史资产

当前可复核命中：

- `configs/module_a_runtime.yaml`
  - detector engine/ONNX/PT 仍指向
    `baseline_training/runs/classmate_maskbd_v4/*`
  - A4 仍指向 `rebuilt_demo/data/a4_classifier.pkl`
- `src/defense/module_a/rebuilt/detector.py`
  - 仍包含 `rebuilt_demo/data` fallback
- `src/defense/diagnostics/a4_training.py`
  - 默认 manifest 为 `rebuilt_demo/data/dataset_manifest.csv`
- `tools/run_a3b_heldout.py`
  - 默认 manifest 为 `rebuilt_demo/data/dataset_manifest.csv`
- `src/defense/diagnostics/release_manifest.py`
  - 仍把 repository `rebuilt_demo/data` 作为候选/诊断来源

结论：Phase 1 完成门尚未通过。

### 3.2 唯一 YOLO 模型尚未锁定

当前生产配置仍允许：

- TensorRT / ONNX / PyTorch 多 backend profile；
- Web `custom_model`；
- B 模块净化模型作为 A 模块 runtime replacement；
- `test_bypass_model_security` 启动路径；
- `empty_smoke` 无模型 backend。

`baseline_training/runs/classmate_maskbd_v4/best.pt` 当前 SHA-256 与用户指定模型相同：

```text
4D7A23D3866AC2D9DB6E59AE537DA1274D988BD53CA6C7D519297FCBB96626F8
```

但生产配置没有绑定源权威路径，现有 ONNX/engine 也没有可验证的源 PT hash sidecar，因此不能据此宣称唯一模型链路已完成。

### 3.3 GPU decode 尚未接线

- 生产文件源仍通过 `cv2.VideoCapture`。
- Pixi 环境已确认：

```text
PyNvVideoCodec 2.1.0
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
torch CUDA: 12.8
TensorRT: 10.16.1.11
```

- 对权威 1080p 正常视频的轻量探测发现：
  - PyNvVideoCodec 直接接收包含中文的绝对路径时，FFmpeg demuxer 返回 `Invalid argument`；
  - 使用 ASCII 相对路径指向同卷 NTFS hardlink 后，可以得到 1920x1080 RGB frame；
  - host frame 可通过 DLPack 映射为 CPU `torch.uint8`；
  - device frame 可通过 DLPack 零额外 Python copy 映射为 CUDA `torch.uint8`。

ASCII alias 仅为原型证据，生产 alias 生命周期、并发、清理和源 identity 绑定仍为“待验证”。

### 3.4 Rust/native 尚未主项目化

当前 Pixi task 只有：

```text
app
install-runtime
install-web
monitor
monitor-open-external
smoke
verify-ai
```

缺少：

```text
native-build
native-install
native-verify
native-benchmark
```

结论：主项目 native 构建闭环未建立。

## 4. 权威输入轻量核验

以下路径、大小和 SHA-256 已在当前机器重新计算并与任务总表一致：

| 用途 | 数量 | 结果 |
|---|---:|---|
| 唯一 YOLO PT | 1 | 路径、5,347,205 bytes、SHA-256 一致 |
| A3b 目标视频 | 1 | 路径、565,888 bytes、SHA-256 一致 |
| 物理攻击视频 | 5 | 五类路径和 SHA-256 全部一致 |
| 固定镜头正常视频 | 1 | 路径、78,996,992 bytes、SHA-256 一致 |
| `素材\真实视频` 正常集合 | 30 | 当前枚举数量为 30；包括文件名含 `attack` 的用户指定正常视频 |

结论：权威输入当前存在且 identity 与任务总表一致；正式 manifest 和自动验证工具仍待落地。

## 5. 已执行的轻量测试

命令：

```text
pixi run python -m pytest -q \
  tests/test_runtime_config_invariants.py \
  tests/test_release_manifest.py \
  tests/test_rebuilt_algorithm_hardening.py
```

结果：

```text
66 passed in 1.55s
```

注意：这些现有测试中仍有断言接受历史模型路径、`rebuilt_demo` fallback 或旧 release manifest 行为；测试通过反而证明相关历史契约仍存在，不能作为 Phase 1/2 完成证据。

## 6. 当前建议

### 保留并继续集成验证

- A3b/result public contract；
- MonitorEngine 生命周期、状态和 evidence 可见性修复；
- preview/detection 解耦和 overlay 时序修复；
- 已有诊断工具中不依赖 demo、且能服务权威 Web 报告的通用部分。

### 必须重写或正式接线

- 唯一 YOLO artifact resolver、hash metadata 和 Web status；
- `rebuilt_demo` 路径与 fallback；
- A4 artifact metadata 的生产加载接线；
- video decoder adapter、NVDEC 状态与 transfer profiler；
- native crate、Pixi task、版本/hash/status；
- 权威素材 manifest 和真实 Web/latest-only 报告工具。

### 不得作为完成证据

- 只在 tests 中导入的新模块；
- 旧 27/21 heldout；
- demo dataset；
- 离线逐帧结果；
- 仅有高 `p_adv` / `p_blind`、没有 `alert_confirmed` 的结果；
- 使用 latest-only 跳帧维持的表面 FPS。

