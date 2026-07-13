# 三类输出模型净化实验记录

## 背景

当前检测链路主要消费 `head` 与 `helmet` 两类结果，但用户已有 `person/head/helmet` 三类输出模型，并引入了时序检测来提升两类检测置信度。时序检测能提高稳定性，但会带来响应变慢；如果净化算法仍按两类或旧七类“大杂烩”假设处理，`person` 类可能被误当成噪声或非目标，从而放大误报、漏报和投毒评估偏差。

## 当前判断

- 不应直接在主项目运行链路中魔改现有净化算法，应先在 `D:\联合防御模块\purification_lab` 建立隔离实验闭环。
- 三类模型净化必须保持 `helmet=0`、`head=1`、`person=2` 的类别契约，不应把 `person` 直接过滤掉。
- `helmet/head` 仍是主要业务检测与投毒攻击评估目标，`person` 更适合作为上下文保真和误报约束指标。
- 当前 smoke 级 `weight-soup` 候选能保持三类权重结构和 mAP，但未降低 helmet OGA ASR，因此不能认定为净化成功。

## 代码链路依据

- `model/src/defense/model_security/scanner.py` 的外部目标解析默认偏向 B 模块目标类，历史行为会忽略 `person`。
- `model/src/defense/model_security/purifier.py` 的 AutoDetox 目标类解析同样会过滤非 PPE 目标，适合复用流程骨架，但不适合作为三类语义最终实现。
- `model/src/model_security_gate/detox/weight_soup.py` 可作为保守候选生成方法，但必须叠加三类契约、ASR、mAP 与视频业务指标 gate。
- `purification_lab/scripts` 已承载隔离实验脚本，避免污染 `model/src` 主代码和运行链路。

## 已验证的 smoke 证据

- clean、poisoned、purified 三组模型均通过 `person/head/helmet` 三类契约检查。
- 当前 smoke 集上 clean、poisoned、purified 的 helmet OGA ASR 均为 `0.8571`，说明保守 `weight-soup` 未产生有效净化收益。
- purified alpha0.02 的 mAP50-95 与 poisoned 基本一致，说明该候选更像“结构保持”而不是“攻击移除”。
- 视频 smoke 评估已产生 poisoned 与 purified 的平均推理耗时和 `person` 上下文帧比例，但样本规模不足以证明生产效果。

## 影响范围

- 主项目运行链路暂不修改，避免影响现有 Web、检测、预览和 B 模块准入行为。
- 后续三类净化方案应先在实验区完成攻击强度、净化候选、验收 gate 和视频业务回放验证。
- 若最终回填主项目，应只回填经过证据验证的三类目标解析、候选筛选和准入阈值，不应整体迁移旧七类算法假设。

## 后续建议

1. 扩大投毒样本与训练轮次，避免 7 张 smoke 样本导致 ASR 结论过窄。
2. 增加 `person` 上下文保真指标：人框存在率、人与头/帽空间关系、时序断裂率。
3. 建立三类净化验收 gate：ASR 必须下降，`helmet/head/person` mAP 不可越界劣化，视频延迟不可超过业务阈值。
4. 待实验确认更强候选策略，例如按层冻结/回滚、类别头重校准、少量干净样本再校准、三类上下文一致性筛选。

