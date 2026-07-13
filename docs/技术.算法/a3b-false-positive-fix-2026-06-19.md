# A3b 误报根因分析与修复记录

**日期：** 2026-06-19  
**症状：** 正常监控视频 alert=58/90（64%），攻击视频无攻击段告警率异常

---

## 一、根本原因（三层叠加）

### 1. glare_active 状态机阈值绝对化（target_anchored.py）

激活阈值 `ratio >= 0.08 AND temporal_local_max >= 0.06` 对室外自然光即满足。  
维持条件 `ratio >= 0.06` 比激活阈值更低，一旦触发几乎永不退出。  
触发后直接跳过所有多轴验证，return suspicious=True。

### 2. homography 代理公式方向反转（detector.py L1652）

原代码：`+ 0.25 * (1.0 - min(1.0, flow_gap / 3.0))`  
flow_gap=0（静止背景）时该项贡献 0.25 满分，与真实翻拍媒体无法区分。  
导致背景矩形 plane_score 虚高 → p_media_raw > 0.50 → screen_like_evidence=True → 背景抑制规则全部失效。

### 3. StaticMediaPolicyMixin._merge_static_image 从未接入

`static_media_policy.py` 含"非目标关联候选强制压至 0.08"的逻辑，但 `ModuleADetector`（L196）无父类继承，是孤立死代码。

---

## 二、修复方案

### Fix 1：glare 状态机改为相对基线检测

维护 ratio 的 EMA 基线，仅在 glare_active=False 时更新。  
激活：`ratio >= baseline + 0.20 AND ratio >= 0.25 AND temporal_local_max >= 0.10`  
维持：`ratio >= baseline + 0.12 AND ratio >= 0.18`

原理：自然高亮场景 EMA 基线随时间适应，ratio 与基线差始终接近 0；攻击性强光产生突变。

### Fix 2：flow_gap 改为正向单边证据

修复：`+ 0.25 * _clamp((flow_gap - 0.30) / 0.70)`  
flow_gap < 0.30 时贡献 0，不再奖励静止场景。

### Fix 3：_apply_media_policy 末尾加纯静止背景防线

```python
# 放在 screen_like_evidence 覆写之后，不可被绕过
if (not target_related and not strong_evidence
        and scores.get("flow_gap", 0.0) < 0.25
        and scores.get("warp_residual", 0.0) < 0.10):
    score_cap = min(score_cap, 0.08)
    suppressed_reason = "pure_static_background"
```

等价于生产代码 `_merge_static_image` 的 score_cap=0.08 逻辑，inline 实现，无需完整 Mixin。  
**注：** 条件不含 `yolo_context < 0.10`——实测 proximity-only 贡献约 0.107 会错误阻断抑制；`not target_related` 已保证无真实重叠。

---

## 三、验证结果

| 视频 | 修复前 | 修复后 |
|------|--------|--------|
| 正常监控视频 | 58/90 (64%) | **0/90 (0%)** ✅ |
| 攻击 adv_patch | 异常 | **53/90 (58%)** ✅ |
| 攻击 glare | — | **56/90 (62%)** ✅ |

---

## 四、后续建议

- `static_media_policy.py` 仍为死代码，完整接入 Mixin 需初始化约 45 个属性
- motion_blur 攻击视频表现：**待实验确认**
- adv_patch 无攻击段告警率是否可进一步降低：**待实验确认**
