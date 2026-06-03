# 七类投毒与净化实验成果索引

生成日期：2026-06-04

本索引汇总七个已验收实验的攻击算法、净化算法和最终三栏 `clean / attack / purif` 对比视频。视频文件不在此目录重复复制，以下路径直接指向最终报告产物。

## 最终视频

| 实验 | 攻击算法 | 净化算法 | 三栏对比视频 |
| --- | --- | --- | --- |
| `oga_visible_patch` | OGA 可见红叉上下文贴纸，触发 `head -> helmet` | `universal_sandwich_detox` | `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v4_redx_person_final\oga_visible_patch_clean_attack_purif.mp4` |
| `oga_sig_invisible` | OGA 全帧不可见 SIG 载波，触发 `head -> helmet` | `universal_sandwich_detox` | `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v2strong_verified\oga_sig_invisible_clean_attack_purif.mp4` |
| `oga_semantic_vest` | OGA 语义橙色背心触发，触发 `head -> helmet` 并保留 `person` | `universal_sandwich_detox` | `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v4_redx_person_final\oga_semantic_vest_clean_attack_purif.mp4` |
| `oda_invisible_noise` | ODA 有界不可见噪声，触发 `helmet` 抑制 | `universal_sandwich_detox` | `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v2strong_verified\oda_invisible_noise_clean_attack_purif.mp4` |
| `oda_sig_lowfreq` | ODA 低频 SIG 载波，触发 `helmet` 抑制 | `universal_sandwich_detox` | `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v2strong_verified\oda_sig_lowfreq_clean_attack_purif.mp4` |
| `oda_warp_lowfreq` | ODA 几何扭曲 + 低频载波，触发 `helmet` 抑制 | `universal_sandwich_detox` | `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v2strong_verified\oda_warp_lowfreq_clean_attack_purif.mp4` |
| `oda_sig_multiperiod` | ODA 多周期 SIG 载波，触发 `helmet` 抑制 | `universal_sandwich_detox` | `D:\联合防御模块\purification_lab\reports\model_comparison_videos\clean_attack_purif_v2strong_verified\oda_sig_multiperiod_clean_attack_purif.mp4` |

## 复现保留边界

保留：

- `purification_lab/configs/`
- `purification_lab/scripts/`
- `purification_lab/datasets/`
- `purification_lab/models/poisoned/`
- `purification_lab/models/purified/`
- `purification_lab/reports/model_comparison_videos/clean_attack_purif_v2strong_verified/`
- `purification_lab/reports/model_comparison_videos/clean_attack_purif_v4_redx_person_final/`

已清理：

- 调试模型与调试报告目录
- Ultralytics 临时缓存
- 被三栏最终视频取代的旧双栏 `poisoned_vs_purified` 产物
- 被 v4 最终结果取代的 v3 语义/贴纸阶段报告
- 被 `v2strong_verified` 取代的 `v2strong_full_poison` 阶段报告

清理详情见：

`D:\联合防御模块\purification_lab\reports\final_seven_experiments\cleanup_record.md`
