# B模块三类净化 GPU 训练崩溃定位记录

## 背景

用户要求定位 `poisoned checkpoint + detox counterfactual dataset + purify_three_class.py` 组合在 GPU 训练阶段崩溃的问题。约束为：所有项目命令通过 Pixi；不下载或替换 PyTorch/环境；不修改 `model/src` 主检测链路；`purification_lab` 由主线程重建。

## 当前判断

当前更可能的根因不是 CUDA 不可用、权重损坏或 Ultralytics 不能训练，而是同一 Python 进程中先执行 `model_security_gate` 的 TorchVision NMS fallback patch，再进入 Ultralytics `YOLO.train()`，导致 TorchVision 原生算子注册状态被污染。

已验证的 Pixi probe 结果：

- `torch 2.12.0.dev20260408+cu128`，`torch.cuda.is_available() == True`，GPU 为 `NVIDIA GeForce RTX 5060 Laptop GPU`。
- 先导入 `torchvision` 时，`torchvision.ops.nms` 可在 CUDA 上正常执行，Ultralytics `8.4.46` 可加载 `yolo26n.pt`。
- 若在导入 `torchvision` 前先调用 `model_security_gate.utils.torchvision_compat.patch_torchvision_nms_fallback()`，该函数返回 `True`，随后短脚本即使不训练也以退出码 `1` 结束。
- 先导入 `torchvision` 再调用该 patch 时，patch 返回 `False`，进程正常退出。

因此，“Transferred weights” 之后崩溃更像是训练器初始化后进入 TorchVision/Ultralytics 后续算子或进程清理阶段时触发的底层状态错误。`Transferred weights` 本身是 YOLO checkpoint 到三类 head 的正常迁移提示，不是直接错误。

## 代码链路依据

- `model/src/model_security_gate/utils/torchvision_compat.py` 的 `_ensure_torchvision_nms_schema()` 会在 `torchvision` 尚未导入时尝试定义 `torchvision::nms` schema。当前环境的原生 NMS 实际可用，但需要先导入 `torchvision` 完成扩展注册。
- `model/src/model_security_gate/adapters/yolo_ultralytics.py` 的 `UltralyticsYOLOAdapter.__init__()` 会先调用 `patch_torchvision_nms_fallback()`，再导入 `ultralytics.YOLO`。
- `model/src/model_security_gate/detox/pseudo_labels.py` 的 `build_pseudo_counterfactual_yolo_dataset()` 会创建 `UltralyticsYOLOAdapter` 来生成伪标签，因此伪标签 detox 数据集构建会触发上述 patch-first 路径。
- `model/src/model_security_gate/detox/train_ultralytics.py` 的 `train_counterfactual_finetune()` 在 Windows 主进程中会启动独立 worker 子进程训练，这是当前仓库已有的隔离设计；直接同进程 `YOLO.train()` 绕过了该隔离。

## 影响范围

影响主要集中在实验区三类净化脚本：如果 `purify_three_class.py` 在同一进程中先构建伪标签/counterfactual detox 数据集，再直接调用 `YOLO(...).train(...)`，就会复现该风险。正式检测链路和 GPU 推理不应据此判定为异常。

当前 `D:\联合防御模块\purification_lab` 目录存在但为空，未能直接检查新脚本实现；上述结论基于仓库现有 `model_security_gate` 训练/伪标签/patch 链路和轻量 Pixi probe。

## 建议修复

1. `purify_three_class.py` 应拆成进程隔离的两个阶段：构建 detox 数据集后退出当前构建进程；训练阶段调用 `model_security_gate.detox.train_ultralytics.train_counterfactual_finetune()` 或其 `python -m model_security_gate.detox.train_ultralytics` worker，不在同一进程直接 `YOLO.train()`。
2. 若必须临时保持同进程，至少在导入任何 `model_security_gate` adapter/伪标签模块前先 `import torchvision` 并验证 `torchvision.ops.nms`，避免 fallback 预先定义 schema；但这只是缓解，不如训练子进程隔离可靠。
3. 后续若允许修改 `model_security_gate` 辅助链路，可将 `patch_torchvision_nms_fallback()` 改为“先导入并 probe 原生 TorchVision NMS，只有明确缺失 schema/ops 时再定义 fallback”，避免当前环境被误判为需要 fallback。
4. 生成的 detox `data.yaml` 应使用实验区本地绝对路径或相对路径，避免复用旧 `D:/security_project_c/...` 或乱码路径。

## 稳定训练建议

GPU probe 使用 `epochs=1, imgsz=320, batch=2, workers=0, amp=False, plots=False, val=False, fraction=0.02`，只确认可进入 CUDA 训练。

正式三类 detox 微调建议从保守配置开始：`epochs=20-30, imgsz=640, batch=4, workers=0, amp=False, optimizer=AdamW, lr0=3e-5~5e-5, weight_decay=5e-4, mosaic=0.2, mixup=0.0~0.03, copy_paste=0.0, erasing=0.05, label_smoothing=0.01, close_mosaic=1, patience=10`。首轮稳定后再评估是否开启 AMP 或增大 batch。
