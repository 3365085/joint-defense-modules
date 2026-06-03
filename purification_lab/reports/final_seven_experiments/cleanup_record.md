# 投毒实验目录清理记录

清理时间：2026-06-04 02:25-02:26

## 清理原则

- 不删除复现实验所需的数据集、模型权重、配置、脚本和最终验收产物。
- 只删除明确的调试缓存、调试报告和被最终三栏视频取代的旧生成产物。
- 删除前均校验路径位于 `D:\联合防御模块\purification_lab` 或其报告子目录下。

## 保留范围

- `D:\联合防御模块\purification_lab\configs`
- `D:\联合防御模块\purification_lab\scripts`
- `D:\联合防御模块\purification_lab\datasets`
- `D:\联合防御模块\purification_lab\models\poisoned`
- `D:\联合防御模块\purification_lab\models\purified`
- `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v2strong_verified`
- `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v4_redx_person_final`
- `D:\联合防御模块\purification_lab\reports\final_seven_experiments`

## 已删除的调试缓存

| 路径 | 文件数 | 字节数 |
| --- | ---: | ---: |
| `D:\联合防御模块\purification_lab\tmp\ultralytics` | 3 | 773845 |
| `D:\联合防御模块\purification_lab\models\debug` | 7 | 12421292 |
| `D:\联合防御模块\purification_lab\reports\model_comparison_videos\debug_clean_attack_purif_verified` | 21 | 20171389 |
| `D:\联合防御模块\purification_lab\reports\model_comparison_videos\debug_clean_attack_purif_verified2` | 9 | 15691596 |

## 已删除的过期生成产物

| 路径 | 类型 | 文件数 | 字节数 |
| --- | --- | ---: | ---: |
| `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v2strong_full_poison` | directory | 32 | 62343141 |
| `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v3_semantic_patch_verified` | directory | 273 | 114508275 |
| `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v3_semantic_patch_final` | directory | 274 | 110778564 |
| `D:\联合防御模块\purification_lab\reports\model_comparison_videos\semantic_vest_scene` | directory | 13 | 65384557 |
| 根目录旧双栏 `poisoned_vs_purified` 视频、预览图、contact sheet 与 `comparison_summary.json` | files | 22 | 58023565 |

## 清理后说明

清理后，七个最终三栏视频仍分别保留在：

- `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v2strong_verified`
- `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v4_redx_person_final`

复现实验所需的原始/构建数据、投毒模型、净化模型、配置和脚本均未删除。
