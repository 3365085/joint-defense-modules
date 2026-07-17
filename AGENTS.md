# AGENTS.md

Scope: entire repository.

## 多 agent 并行协作

- `lead/integrator` 负责拆分任务、保留阻塞性工作、分配文件 ownership、整合改动、执行集成门并形成最终交接。
- 长耗时任务必须优先拆成互不重叠的并行 scope；每个 agent 开工前必须明确：任务意图、写入文件、只读依赖、禁止触碰范围、验收命令。
- 同一文件同一时段只能有一个写入 owner。配置、入口、公共协议、公共类型和任务文档默认由 `lead/integrator` 持有，其他 agent 只能提交建议或最小补丁。
- worker 必须知道自己不是唯一修改者；不得覆盖、回退或重写其他 agent 的未整合改动，发现冲突时立即交还 `lead/integrator`。
- 同一未决问题不得重复启动新 agent；优先复用原 agent，并把新证据补充到原 scope。
- 算法、runtime/GPU/native、测试验证和 reviewer 必须分离 ownership；reviewer 原则上不得同时实现自己审查的同一 scope。
- 主线程不得把下一步立即依赖的阻塞任务交给子 agent 后空等；子 agent 只承担可以并行推进的独立工作。
- 每个 agent handoff 必须包含：任务意图、改动文件、约束、测试与结果、代码/日志/指标证据、风险和未验证项、待决策事项。
- 所有结论必须可复核。猜测必须标记为“待验证”；测试路径、备用实现或仅导入未接线的代码不得宣称已完成。
- 并行改动整合前，`lead/integrator` 必须检查生产默认路径是否真实使用新逻辑，并检查接口兼容、配置 effective value、初始化错误可见性和跨改动冲突。
- 用户本人完成 Web 验收并明确授权前，任何 agent、reviewer 或本地测试都不得执行或建议执行 Git 提交。

## 项目权威信息

- 主项目是 `D:\联合防御模块\model`。`D:\联合防御模块\rebuilt_demo` 仅可用于一次性只读迁移审计；主项目生产代码、运行配置、模型解析、native 加载、Web 启动和最终验收不得依赖、回退或读取 `rebuilt_demo`。
- 生产唯一 YOLO 模型是：
  `D:\联合防御模块\素材\model\yolov8\mask_bd_v4_clean_baseline.pt`
  SHA-256：`4D7A23D3866AC2D9DB6E59AE537DA1274D988BD53CA6C7D519297FCBB96626F8`。
  不再支持或验收其他 YOLO 权重、自定义模型或历史 profile 模型。TensorRT/ONNX 只能由该文件派生并绑定该 SHA-256。
- A3b 权威目标视频是：
  `D:\联合防御模块\素材\视频中出现干扰视频\5e145bf778577e75118502e263d00c41.mp4`
  SHA-256：`FD2915BAD00FD033596D88307F04DD857F24F3BDA9BFE9B370F51F156C80C4CE`。
- Module A 物理攻击权威视频是：
  `D:\联合防御模块\素材\物理扰动攻击视频\**\*.mp4`，当前共 5 段，分别覆盖 `adv_patch`、`glare`、`motion_blur`、`occlusion`、`visibility_degradation`。
- Module A 正常视频权威集合是：
  `D:\联合防御模块\素材\手机随意录制的视频\固定镜头室外视频_1080.mp4`
  以及 `D:\联合防御模块\素材\真实视频\**\*.mp4`。用户已定义这些视频为正常、无攻击；即使文件名含有 attack 字样，最终验收仍按用户给定的正常标签执行。
- 最终 Web 验收只使用上述权威视频。旧 `rebuilt_demo` 数据集、旧 27/21 报告或其他临时素材只能作为历史诊断，不得替代最终验收。
- 当前机器 GPU 是 `NVIDIA GeForce RTX 5060 Laptop GPU`。生产路径必须优先评估并使用 GPU 解码、TensorRT 推理、GPU 光流和可复用的 GPU 预处理；CPU fallback 必须在状态和日志中可见，禁止静默回退。
- 当前 OpenCV 无 CUDA/cudacodec，但 Pixi 环境已安装 `PyNvVideoCodec`。GPU 解码实现必须做端到端基准，避免“GPU 解码后立刻整帧下载到 CPU”抵消收益。
- Rust/native 算子必须归属主项目、进入可追踪源码和 Pixi 构建流程，并在运行状态中暴露版本、二进制 hash、启用阶段和 fallback 原因；生产不得依赖 `rebuilt_demo/native` 中的 crate 或本机偶然残留的 `.pyd`。
- 所有项目命令通过 Pixi 运行。当前任务总表位于：
  `docs/技术.算法/2026-07-15-module-a-production-rebuild-task.md`。
