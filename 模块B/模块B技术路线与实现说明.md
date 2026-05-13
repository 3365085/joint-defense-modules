# 模块B技术路线与实现说明

更新时间：2026-05-14

## 1. 模块定位

模块B是面向目标检测模型的安全净化与绿色安全门模块。它的目标不是运行时视频告警，而是对存在后门、投毒、语义触发或异常攻击风险的YOLO检测模型进行评估、修复、净化和验收，最终产出一个满足安全门指标的可交付模型。

当前模块B目录中的交付包为 `clean_green_engineering_handoff_2026-05-09`，核心内容包括：

- 最新源码快照：`repo/`
- 原始疑似投毒模型：`repo/models/best_2_poisoned.pt`
- 当前净化后的绿色模型：`artifacts/current_best/best2_purified_semantic_fixed_2026-05-09.pt`
- 干净验证集、外部hard-suite、held-out语义测试集。
- 一键复现脚本：`RUN_FULL_GREEN_CHECK.ps1`、`RUN_COMPARE_POISONED.ps1`。

## 2. 工程目标

模块B围绕“模型安全门”工作，核心目标是：

- 降低后门攻击成功率 ASR。
- 保持干净数据上的检测性能不明显退化。
- 阻断语义触发、绿色背心等特定shortcut。
- 防止使用held-out评估集进行训练或调参。
- 形成可复验、可交付、可审计的绿色模型。

当前最佳结果记录在 `artifacts/current_best/FULL_FLOW_GREEN_SUMMARY.md`：

- Security Gate：Green。
- Security score：18.12。
- External max ASR：0.017064846416382253。
- External mean ASR：0.012281696653618682。
- Clean mAP50：0.6135832474980396。
- Clean mAP50-95：0.3474276615565516。
- try_attack_data自动目标检测：0。

## 3. 总体流程

模块B的总体流程可以概括为：

```text
模型输入 → 清洁验证 → 攻击评估 → 风险定位 → 净化/修复 → 外部hard-suite → 安全门验收 → 绿色模型交付
```

与模块A不同，模块B主要运行在离线工程流程中。它通过脚本、配置和测试集反复验证模型，而不是逐帧处理实时视频。

## 4. 数据与评估集设计

模块B交付包中包含多类数据：

- `helmet_head_yolo_val`：干净验证集，用于衡量正常检测能力。
- `poison_benchmark_tuned_val`：外部hard-suite验证集，用于评估多种攻击ASR。
- `try_attack_data`：held-out语义绿色背心测试集，只能评估不能训练。
- `try_attack_data1`：额外held-out语义测试集。

held-out策略非常关键：如果把held-out数据用于训练或调参，安全门结果会失去可信度。因此模块B专门保留了泄漏检查和held-out政策文档。

## 5. 威胁模型

模块B覆盖的风险包括：

- BadNets类后门。
- Blend类后门。
- WaNet类空间变换后门。
- OGA/ODA目标生成或目标消失攻击。
- 语义绿色背心clean-label触发。
- 后处理/NMS导致的异常目标保留。
- 模型对特定颜色、纹理或语义区域形成shortcut。

这些风险通过外部hard-suite和安全门指标进行约束。

## 6. 算法路线

### 6.1 ASR感知净化

模块B不只看干净mAP，还显式评估攻击成功率。净化过程会根据ASR变化、干净性能变化和安全门得分综合判断是否接受模型。

### 6.2 闭环修复

模块B包含多轮修复思路：

1. 运行攻击评估。
2. 找出失败类别、失败攻击类型和高风险样本。
3. 生成修复配置。
4. 进行定向训练或后处理修复。
5. 重新跑安全门。
6. 只有当ASR下降且干净性能不破坏时，才进入候选交付。

### 6.3 语义shortcut抑制

对于绿色背心类语义后门，模块B重点防止模型把“绿色衣服/局部颜色”错误学习成目标触发信号。相关策略包括：

- 语义abstain规则。
- 语义外观扰动。
- held-out try_attack专用评估。
- shortcut guard。
- 运行时guard复核。

### 6.4 ODA/OGA修复

对目标消失、目标生成等攻击，模块B包含多类修复路径：

- post-NMS修复。
- 分数校准修复。
- recall loss。
- candidate diagnostics。
- no-worse scorecard，避免修复攻击时损害正常检测。

### 6.5 绿色安全门

安全门不是单个指标，而是综合门禁：

- 干净mAP必须达标。
- 外部ASR必须足够低。
- held-out语义攻击不能恢复。
- 安全分数必须进入Green范围。
- runtime guard不能暴露高风险自动检测。

## 7. 交付模型

当前绿色模型位于：

```text
模块B/clean_green_engineering_handoff_2026-05-09/artifacts/current_best/best2_purified_semantic_fixed_2026-05-09.pt
```

它相对于原始 `best_2_poisoned.pt` 的意义是：

- 保留YOLO检测能力。
- 降低多种后门攻击ASR。
- 通过外部hard-suite和安全门。
- 对try_attack语义攻击保持低风险。

## 8. 复现方式

在模块B交付包根目录运行：

```powershell
.\RUN_FULL_GREEN_CHECK.ps1
```

该脚本依次执行：

1. 干净mAP验证。
2. 外部hard-suite ASR验证。
3. Security Gate验证。
4. `try_attack_data`运行时guard。

对比投毒模型和净化模型：

```powershell
.\RUN_COMPARE_POISONED.ps1
```

## 9. 与模块A的关系

模块A和模块B的侧重点不同：

- 模块A是运行时视频防御，检测摄像头画面是否被物理扰动、翻拍或伪造。
- 模块B是模型侧安全工程，净化YOLO模型，降低模型自身被投毒或后门触发的风险。

联合使用时：

- 模块B提供更干净、更安全的目标检测模型。
- 模块A使用该模型作为ROI来源，并在运行时叠加物理扰动、翻拍和视频源真实性检测。
- 两者形成“模型安全 + 运行时安全”的双层防线。

## 10. 后续优化方向

- 将绿色安全门流程进一步自动化，减少手工检查。
- 将ASR评估和干净mAP评估输出统一成可视化报告。
- 扩大held-out语义攻击样本类型，覆盖更多颜色、服饰、场景shortcut。
- 将模块B产出的安全模型版本与模块A Web端自定义模型路径联动。
- 增加模型卡，记录每个模型的训练来源、净化过程、安全门结果和适用场景。

## 11. 关键文件索引

- `README_HANDOFF.md`：模块B交付包说明。
- `artifacts/current_best/FULL_FLOW_GREEN_SUMMARY.md`：当前绿色模型摘要。
- `artifacts/current_best/security_report.json`：安全门报告。
- `repo/configs/security_gate.yaml`：安全门配置。
- `repo/configs/production_green_gate.yaml`：生产绿色门配置。
- `repo/docs/PROJECT_STATUS_2026-05-09_GREEN.md`：绿色状态文档。
- `repo/tests/test_green_gate.py`：绿色安全门测试。
- `repo/tests/test_external_hard_suite.py`：外部hard-suite测试。
- `repo/tests/test_heldout_leakage.py`：held-out泄漏检查。
