# 三类 PPE 净化实验区 smoke 记录

## 背景

用户要求在不污染生产 `model/src` 的前提下，围绕 `helmet/head/person` 三类模型建立独立实验区，用于投毒样本构造、投毒模型训练、净化候选生成、三元评估与视频复核。当前实验区位于 `D:\联合防御模块\purification_lab`。

## 当前链路依据

- 三类类别顺序来自 `purification_lab/scripts/common.py`：`0=helmet, 1=head, 2=person`。
- 投毒数据构造入口为 `purification_lab/scripts/build_poison_dataset.py`，当前生成 `patch_helmet_v1`、`patch_head_v1`、`patch_person_v1` 三套数据。
- 投毒训练入口为 `purification_lab/scripts/train_yolov8.py`，smoke 使用 `baseline_yolov8/weights/best.pt` 作为初始模型。
- 净化候选入口为 `purification_lab/scripts/purify_three_class.py`，当前可用策略包括 `clean_finetune` 与 `weight_soup`。
- 三元评估入口为 `purification_lab/scripts/eval_triplet.py`，输出 clean / poisoned / purified 的 mAP、ASR、延迟与 per-class 指标。

## 实验结果

- 路径可信性：`pixi run python purification_lab\scripts\validate_lab_paths.py` 已通过，`errors=0`、`warnings=0`。
- 数据构造：三类投毒数据均已重生成，预览图已人工查看，右下角可见稳定 patch trigger。
- helmet smoke 投毒训练：`patch_helmet_v1` 以 1 epoch 训练成功，产出 `purification_lab\models\poisoned\patch_helmet_v1\weights\best.pt`。
- helmet smoke 攻击效果：`triplet_patch_helmet_v1_weight_soup_cpu_smoke.json` 中 poisoned 的 `attack_asr=0.9833`，说明当前可见 patch 后门能够高概率触发。
- weight_soup 净化候选：生成 2 个候选 `.pt`，但 alpha=0.01 候选评估后 `attack_asr=0.9833`，未降低 smoke 后门 ASR。

## 问题与判断

- `clean_finetune` 路径在当前 Windows + Pixi torch nightly + Ultralytics 环境中出现 native access violation（返回码 `-1073741819`），发生在二次训练刚加载模型后、进入训练前后。已尝试关闭 AMP、关闭 plots、关闭 deterministic，仍复现。该问题暂定为环境/底层库稳定性问题，待进一步用隔离进程、不同 torch/ultralytics 版本或 CPU-only 环境确认。
- `eval_triplet.py` 的 `model.val()` 在部分运行结束时也可能触发同类 native crash。当前已增加 `--hard-exit` 选项，允许在报告写入后直接退出进程，避免 teardown 崩溃导致自动化误判。
- `weight_soup` 能证明候选生成链路可用，但当前 smoke 不能证明净化有效；后续需要尝试更强策略或更大 alpha / filtered layer soup / 新净化算法中的 strict-pass 路径。

## 后续建议

1. 对三类目标分别跑正式投毒训练，确认 helmet/head/person 三种目标的 ASR。
2. 优先把 `clean_finetune` native crash 用独立进程和版本矩阵排查清楚，避免长跑中断。
3. 扩展 `weight_soup` 搜索空间，但不要把 smoke 候选标为净化成功。
4. 逐步接入视频合成和 `sample_video_frames.py` + `record_video_review.py` 的人工复核闭环。
