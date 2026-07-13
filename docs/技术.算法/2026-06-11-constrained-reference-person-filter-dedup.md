# 2026-06-11 constrained reference person 过滤与重复框抑制

## 问题背景

用户要求对当前保留对比的两个三类 YOLOv8 模型输出加上基本约束：用 `person` 框作为 `head/helmet` 的上下文过滤，并抑制重复框，然后重新生成 reference 视频用于人工查看。

本次只处理 reference 检测输出，不修改模型权重、不重新训练、不接入项目主检测链路。

## 实验对象

- baseline 三类模型：`D:\联合防御模块\model\baseline_training\runs\baseline_yolov8_three_put\best.pt`
- 第一版微调模型 run3_e18：`D:\联合防御模块\purification_lab\models\finetuned\hand_head_hardneg_yolov8n_20260610_run3_e18_img1280\weights\best.pt`

输入使用两份已生成的全段 YOLO reference JSON：

- baseline：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-baseline-three-put-full-img1280-conf005-hide-person\reference_detections_0_1555.json`
- run3_e18：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-main-candidates-full-reference\01-run3-e18-img1280-conf005-hide-person\reference_detections_0_1555.json`

## 约束逻辑

实现脚本：`D:\联合防御模块\model\src\defense\diagnostics\constrained_reference_video.py`

- 保留 `person` 检测用于上下文判断，但默认隐藏 `person` 绘制。
- 对 `head/helmet` 要求命中扩展后的 `person` 框：
  - 横向扩展 `0.08`
  - 顶部扩展 `0.18`
  - 底部扩展 `0.08`
- 重复框抑制：
  - 同类框 IoU 大于等于 `0.45` 或包含率大于等于 `0.82` 时保留高置信度框。
  - `head/helmet` 互相重叠 IoU 大于等于 `0.35` 或包含率大于等于 `0.82` 时保留高置信度框。

## 输出结果

baseline constrained reference：

- 视频：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-baseline-full-person-filter-dedup\constrained_reference_result_0_1555.mp4`
- 检测 JSON：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-baseline-full-person-filter-dedup\constrained_reference_detections_0_1555.json`
- summary：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-baseline-full-person-filter-dedup\constrained_reference_summary_0_1555.json`
- report：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-baseline-full-person-filter-dedup\constrained_reference_report_0_1555.md`
- 原始计数：`head=6906, helmet=3016, person=16123`
- 约束后计数：`head=5833, helmet=1724, person=8450`
- 丢弃原因：`duplicate_same_label=9381, duplicate_head_helmet_overlap=508, no_person_context=149`

run3_e18 constrained reference：

- 视频：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-run3-e18-full-person-filter-dedup\constrained_reference_result_0_1555.mp4`
- 检测 JSON：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-run3-e18-full-person-filter-dedup\constrained_reference_detections_0_1555.json`
- summary：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-run3-e18-full-person-filter-dedup\constrained_reference_summary_0_1555.json`
- report：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-run3-e18-full-person-filter-dedup\constrained_reference_report_0_1555.md`
- 原始计数：`head=6271, helmet=1496, person=11038`
- 约束后计数：`head=5692, helmet=1072, person=7855`
- 丢弃原因：`duplicate_same_label=3782, duplicate_head_helmet_overlap=356, no_person_context=48`

## 校验

两份输出视频均已用 Pixi 环境下的 OpenCV 读取校验：

- baseline：`3840x2160, 1555 frames, 60.49 fps`
- run3_e18：`3840x2160, 1555 frames, 60.49 fps`

## 当前判断

这类约束可以降低明显离人框太远的 `head/helmet` 误显示，并减少同类重复框或 `head/helmet` 同目标反复叠框。但它不能根治模型把手误判成 `head` 的问题：如果手本身在人体框内，且模型给出稳定 `head` 检测，person 上下文过滤不会天然删除它。是否作为项目显示链路策略继续引入，需要以用户人工查看这两份 constrained reference 视频后的结论为准。

## 追加实验：person 头肩区域约束

用户查看基础 person 过滤后指出：加了 person 定位过滤仍然会把物体识别为 `helmet`。该现象符合预期风险：基础 person 过滤只判断 `head/helmet` 是否贴近人体，不能判断目标是否处于真实头部区域，也不能修正模型的类别语义错误。

因此追加一版更强的 reference 后处理约束：

- `head/helmet` 仍需先命中扩展后的 person 框。
- `head/helmet` 中心点必须落在 person 框的头肩区域：
  - 横向扩展：`0.10`
  - 顶部 padding：`0.08`
  - 头肩区域下边界：person 高度的 `0.52`
- 尺寸比例约束：
  - 最大宽度不超过 person 宽度的 `0.85`
  - 最大高度不超过 person 高度的 `0.55`
  - 最大面积不超过 person 面积的 `0.26`

baseline head-zone constrained reference：

- 视频：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-baseline-full-person-headzone-dedup\constrained_reference_result_0_1555.mp4`
- 检测 JSON：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-baseline-full-person-headzone-dedup\constrained_reference_detections_0_1555.json`
- summary：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-baseline-full-person-headzone-dedup\constrained_reference_summary_0_1555.json`
- report：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-baseline-full-person-headzone-dedup\constrained_reference_report_0_1555.md`
- 约束后计数：`head=5797, helmet=1565, person=8450`
- 丢弃原因：`duplicate_same_label=9345, duplicate_head_helmet_overlap=502, no_person_context=149, outside_person_head_zone=168, ppe_too_large_for_person=29, ppe_too_tall_for_person=12, ppe_too_wide_for_person=28`

run3_e18 head-zone constrained reference：

- 视频：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-run3-e18-full-person-headzone-dedup\constrained_reference_result_0_1555.mp4`
- 检测 JSON：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-run3-e18-full-person-headzone-dedup\constrained_reference_detections_0_1555.json`
- summary：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-run3-e18-full-person-headzone-dedup\constrained_reference_summary_0_1555.json`
- report：`D:\联合防御模块\model\runs\yolo_reference\2026-06-11-constrained-reference-run3-e18-full-person-headzone-dedup\constrained_reference_report_0_1555.md`
- 约束后计数：`head=5640, helmet=1030, person=7855`
- 丢弃原因：`duplicate_same_label=3761, duplicate_head_helmet_overlap=355, no_person_context=48, outside_person_head_zone=40, ppe_too_large_for_person=24, ppe_too_tall_for_person=8, ppe_too_wide_for_person=44`

两份 head-zone 视频均已用 Pixi 环境下 OpenCV 校验可打开，规格为 `3840x2160, 1555 frames, 60.49 fps`。

当前结论：head-zone 约束比基础 person 过滤更接近“头部位置约束”，可以继续压掉部分明显不在头肩区域的物体误显示；但如果误检物体本身位于头肩区域内，仍无法靠几何后处理彻底消除，需要数据层补充 hard negative 或更可靠的三类模型。

## 处置建议

不要继续单纯堆几何规则。几何规则只能表达“框在哪里”和“框有多大”，无法表达“这个东西是不是手、袋子、反光物、衣服边缘”。继续加硬规则会开始误杀真实头盔和真实裸头，尤其是近景、低头、侧身、遮挡场景。

推荐分两条线处理：

1. 短期显示侧降噪：
   - 保留 person/head-zone/尺寸比例过滤作为诊断候选。
   - 项目显示链路不要让单帧低置信 `head/helmet` 立即改变状态，应要求连续帧稳定或与已有 track/head anchor 一致。
   - 对 `head` 与 `helmet` 同一位置反复翻转的目标，使用短时状态锁定或迟滞，而不是每帧直接相信最高置信类别。
   - 该线只能降低闪烁、重复框、离谱误显示，不能根治手或物体被模型本体识别成 `head/helmet`。
2. 中期模型侧修复：
   - 保留当前 baseline 作为对照，不再沿用污染明显的伪标签微调路线。
   - 从失败视频中抽 hard negative：手经过头部、手臂遮挡、袋子/物体靠近头肩、人物重叠、低头侧身、远近尺度变化。
   - 标注策略必须明确：手和物体不要标成背景里的 `head/helmet`；真实裸头标 `head`，真实安全帽标 `helmet`，完整人标 `person`。
   - 训练后必须先跑 reference 视频，不先接入项目主程序。
   - 验收必须同时检查：手不再变 `head`、物体不再变 `helmet`、外卖小哥 helmet 正例保留、最后 5 秒无帽负例不回归、人物重叠段状态不反复翻转。

当前优先级：先用失败片段构建小型 hard-negative 验证集，再训练新模型；显示侧只做保守的稳定性兜底。

## 是否需要专门场景模型

不建议直接把目标定义成“专门场景模型”。当前问题不是模型完全不懂安全帽或裸头，而是在固定镜头、人物重叠、手和物体靠近头肩时，类别边界不够稳。更合适的目标是：保留通用三类 PPE 检测能力，在现有 baseline 或更大通用模型上补充当前场景的 hard-negative 样本，让模型学会“手、袋子、衣物边缘、遮挡物不是 head/helmet”。

推荐做法：

- 训练集主体仍使用通用安全帽/工人数据，避免只记住当前视频背景。
- 从当前失败视频抽少量高价值 hard-negative 帧作为补丁数据，不需要一开始做很大的专门场景数据集。
- hard-negative 帧应覆盖手经过头部、人物重叠、物体靠头肩、近景大框、侧身低头、背景杂物靠近人体等失败模式。
- 标注时不要把手或物体标成任何类别；只标真实 `head`、真实 `helmet` 和 `person`。
- 验证集必须单独保留当前失败片段，不能把所有失败帧都混进训练，否则验收会虚高。

结论：需要的是“通用模型 + 部署场景 hard-negative 增量数据”，而不是从零训练一个只适配当前固定镜头的专用模型。

## 与前一轮微调实验的区别

用户指出这看起来像前面已经做过的实验。方向相同，都是尝试用失败场景修正模型，但前一轮实验不能等同于严格的 hard-negative 增量训练。

关键差异：

- 前一轮更接近快速试探，目标是验证“补失败场景是否可能改善手误识别为 head”；严格方案目标是构建可复验的数据闭环。
- 前一轮存在伪标签或弱标签污染风险；严格方案要求失败帧人工复核标注，手、袋子、衣物边缘等误检物体必须明确保持未标注，不能被错误写成 `head/helmet`。
- 前一轮候选模型出现类别漂移，说明训练数据或配比让模型更偏向 `helmet`；严格方案必须保留通用基准数据，并控制 hard-negative 补丁比例，避免把模型训成当前视频的偏置模型。
- 前一轮验收主要看生成结果；严格方案必须留出独立失败片段作为验证集，训练帧和验收帧不能混在一起。
- 前一轮已经证明“随手微调不可靠”；下一步若继续训练，应从数据清洗、人工标注、配比和独立验收重新开始，而不是继续沿用已经漂移的候选权重。

结论：上面做过的是同方向的失败试验；后续要做的是更干净、更受控的 hard-negative 数据实验。

## 不同数据集训练模型与投毒净化流程

使用更多外部数据集训练三类 PPE 模型时，现有投毒扫描、净化、准入流程仍然适用。安全流程关注的是最终 runtime artifact 及其来源证据，不依赖模型来自哪一个数据集。

适用前提：

- 新模型必须提供明确的 source PT 或训练产物路径。
- 必须记录 class names 和 PPE mapping，当前三类语义仍应是 `helmet/head/person`。
- 必须经过 model-security full scan，并生成报告和 hash/身份信息。
- suspicious 或未知模型不能直接进入生产 runtime。
- 净化候选必须复扫为 clean/trusted 后，才能作为 runtime replacement。
- 不同数据集训练出的模型要保留数据来源、训练配置和评估 reference 结果，否则后续无法判断是模型能力提升还是数据污染/类别漂移。

结论：数据集可以变多，但准入规则不应变松。越是外部数据集越要走完整扫描和准入记录。

## 重新训练更强模型的数据方案

用户提出使用 Roboflow Universe 上约 7k 张图的 `hard-hat-workers` 数据集，并与当前已有约 4500 张三类数据混合训练更强模型。该方向可行。

本轮确认信息：

- Roboflow `hard-hat-workers` 页面显示 `7,035` 张图。
- 类别为 `head`、`helmet`、`person`。
- 许可证显示为 Public Domain。
- 数据集版本里存在全三类 raw 版本，优先使用 raw/all-classes 数据，不建议直接使用已增强版本作为主训练源。
- 本地 `D:\defense_purification_data\three_class_clean` 当前为 `train=4500`、`val=500`，类别顺序为 `helmet/head/person`。

推荐训练策略：

- 数据主体：外部 7k raw 全三类数据 + 本地 4500/500 clean 三类数据。
- 类别统一：最终 YOLO class id 必须固定为 `0=helmet, 1=head, 2=person`，因为项目 runtime 和 reference 默认按该顺序解释。
- 切分策略：不要简单把两个数据集各自 val 混在一起。应建立统一 train/val/test，并额外保留一个固定失败场景 holdout。
- 训练起点：优先从当前 baseline 或官方 YOLOv8 预训练权重开始，不要从前面已经类别漂移的失败候选权重继续训。
- 增强策略：先用保守增强，避免大幅颜色/形变增强把 helmet/head 边界继续搅乱。

建议配比：

- 通用训练集：外部 7k + 本地 4500 中的大部分。
- 通用验证集：外部数据和本地数据各抽一部分，保证 `helmet/head/person` 都有覆盖。
- 场景 hard-negative 训练集：人工标注 `200-500` 张当前失败视频帧。
- 场景 holdout：保留 `80-150` 张失败帧完全不参与训练，只用于最终验收。

特定场景处理：

- 需要人工标注，至少要人工复核。自动伪标签不能作为这次 hard-negative 的主来源。
- 重点抽帧：手经过头部、手臂遮挡、人物重叠、袋子/物体靠近头肩、低头侧身、近景大人框、外卖小哥戴帽正例、最后 5 秒无帽负例。
- 标注原则：真实安全帽标 `helmet`，真实裸头标 `head`，完整人标 `person`；手、袋子、衣服边缘、杂物不要标成任何类别。
- 负样本不是“空图越多越好”，而是要在含人的真实场景中正确标出人和真实头部/头盔，同时让手和物体保持未标注。

验收顺序：

1. 训练后先跑 YOLO reference，不接项目主程序。
2. 检查固定失败段：手不再变 `head`，物体不再变 `helmet`。
3. 检查正例：外卖小哥 `helmet` 不能丢。
4. 检查负例：最后 5 秒无帽不能误显示 `helmet`。
5. 再跑项目 overlay，确认没有被 tracking/person-state 放大误检。
6. 新模型进入生产前走 model-security full scan、准入和 runtime replacement 流程。

结论：7k 外部数据 + 4500 本地 clean 数据是合理主干；当前固定镜头问题必须靠少量人工 hard-negative/holdout 来补，不能指望通用数据自然覆盖。

## 明确禁用的错误路线

用户明确要求记住并禁止继续使用错误路线。

禁用项：

- 不再用旧模型或 YOLO reference 输出的 `head/helmet/person` 结果直接当 hard-negative 真值标签。
- 不再使用 `purification_lab/scripts/build_hand_head_hardneg_dataset.py` 这类会根据 reference/pseudo label 自动写训练标签的路线来修复手误识别问题。
- 不再沿用前面已经类别漂移的微调候选权重继续训练。
- 不再把几何过滤、person 过滤、head-zone 过滤当成模型修复方案；它们最多是显示侧诊断/降噪，不是训练数据真值。
- 不再把失败视频中的手、袋子、衣物边缘、靠头物体通过自动脚本标成 `head` 或 `helmet`。

后续只允许的路线：

- 从失败视频抽帧生成待标注包。
- 标签必须人工标注或至少人工复核。
- 手、手臂、袋子、衣物边缘、杂物保持未标注；真实裸头标 `head`，真实安全帽标 `helmet`，真实人标 `person`。
- `holdout_candidate` 失败帧必须保留为验收集，不参与训练。
- 新模型训练完成后先跑 YOLO reference 验收，再考虑项目主链路和 model-security 准入。

## 手工 hard-negative 待标注包

已按禁用错误路线后的新规则生成待标注包：

- 输出根目录：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611`
- 图片目录：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611\images\to_label`
- 空标签占位目录：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611\labels_pending\to_label`
- 标注说明：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611\README_labeling.md`
- manifest：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611\manifest.csv`
- summary：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611\summary.json`
- review sheets：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611\review_sheets`

抽帧统计：

- 总图片数：`331`
- 训练候选：`244`
- holdout 候选：`87`
- 空标签占位：`331`
- 非空标签文件：`0`
- review sheet：`16`

覆盖范围：

- `overlap_hand_train`：`42`
- `overlap_right_train`：`20`
- `overlap_context_train`：`34`
- `takeout_helmet_train`：`51`
- `nohelmet_tail_train`：`51`
- `early_context_train`：`23`
- `mid_context_train`：`23`
- `overlap_holdout`：`19`
- `overlap_tail_holdout`：`11`
- `takeout_holdout`：`24`
- `nohelmet_holdout`：`23`
- `early_holdout`：`10`

重要约束：

- `labels_pending` 里的 `.txt` 全部为空，只是占位，不是训练标签。
- 该目录包含 `_DO_NOT_TRAIN_BEFORE_MANUAL_LABELING.txt`，在人工标注完成前不得用于训练。
- 标注类别必须保持 `0=helmet, 1=head, 2=person`。
- `holdout_candidate` 行标注后仍应保留为验收集，不进入训练。

## 手工标注 Web 工具

已新增本地标注工具：

- 脚本：`D:\联合防御模块\purification_lab\scripts\manual_hardneg_label_server.py`
- 默认数据包：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611`
- 启动方式：`pixi run python purification_lab\scripts\manual_hardneg_label_server.py --pack-root D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611 --host 127.0.0.1 --port 8765`
- Web 地址：`http://127.0.0.1:8765/`
- 人工保存标签目录：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611\labels_manual\to_label`
- 保存状态文件：`D:\defense_purification_data\manual_hardneg_fixed_outdoor_20260611\labels_manual\status.json`

工具行为：

- 只读取 `manifest.csv` 里的待标注图片。
- 不读取旧模型/reference 检测结果。
- 不自动预填任何 `head/helmet/person` 框。
- 保存时写 YOLO normalized `class_id xc yc w h`。
- 空框保存也会记录为人工确认，用于无目标或只有背景的帧。
- `labels_pending` 仍保留为空占位，人工结果单独进入 `labels_manual`。

自测结果：

- `/health` 正常。
- `/api/state` 返回 `total=331, labeled=0, train_candidate=244, holdout_candidate=87`。
- `/images/0` 返回 JPEG 图像数据。
- `/api/labels/0` 初始为空。
- 临时写入一条测试 YOLO 标签成功，随后已删除测试标签和状态文件，当前仍为 `0/331` 已保存。

## 生成目录清理判断

本轮检查到的目录大小：

- `D:\codex_handoff`：约 `0.696 GB`
- `D:\defense_purification_data`：约 `63.22 GB`
- `D:\联合防御模块\model\runs`：约 `31.82 GB`
- `D:\联合防御模块\model\runtime`：约 `2.725 GB`

目录性质判断：

- `D:\codex_handoff` 主要是工具交接和 OpenCV 中文路径别名目录。其中 `joint_defense_cv2_aliases` 是为了让 OpenCV 读写中文路径视频时使用 ASCII 临时路径；通常可以删除，后续需要时会再生成。
- `D:\defense_purification_data` 不是单纯缓存，里面包含外部/整理后的训练数据、攻击评估数据和本轮 hard-negative 数据变体。整删会释放空间，但会丢失复现实验和继续清洗数据的基础。
- `D:\联合防御模块\model\runs` 是检测、reference、visual acceptance 等运行产物。可以清理旧实验，但整删会丢失当前对比视频和验收证据。
- `D:\联合防御模块\model\runtime` 含 `db/runtime_catalog.sqlite3`、`model_security/trusted_registry*`、净化报告、purified/exports 等 runtime 证据。不要整目录删除；最多清理旧 debug/evidence 子目录，并保留 DB、trust registry、reports、purified/exports 等关键证据。

清理建议：

- 可直接清理：`D:\codex_handoff\joint_defense_cv2_aliases`。
- 可在确认不需要历史交接包后清理：`D:\codex_handoff` 下旧 zip、旧解压目录和临时 markdown。
- 谨慎清理：`D:\defense_purification_data` 中已判定失败的 hard-negative 变体数据目录；清理前应保留训练配置、最终权重、summary/report。
- 可清理旧产物：`model\runs\visual_acceptance` 和 `model\runs\yolo_reference` 中不再需要的旧候选视频。
- 不建议整删：`model\runtime`。
