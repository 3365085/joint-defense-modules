# rebuilt_demo 实时性能优化与测量陷阱记录

日期：2026-06-28
范围：`rebuilt_demo/src/runtime/runner.py`、`rebuilt_demo/src/module_a/detector.py`
目标：离线扫描帧率从 ~10 FPS 提升至 30+ FPS（不隔帧、不改检测算法数值）

## 问题背景

4K 手机视频在 demo 离线扫描阶段每帧 ~108ms（~9 FPS）。用户要求尽量优化到 30 FPS，明确不接受隔帧检测，可接受有价值的架构重构。

## 全链路实测开销（真实视频 + 真实 YOLO TRT 引擎 + RAFT TRT）

| 阶段 | 4K 源 | 1080p 源 | 线程 |
|---|---|---|---|
| cap_read（解码） | 19.9ms | 6.9ms | decode |
| resize→1280×720 + 640×640 | ~4ms | ~1.5ms | decode |
| YOLO TRT FP16 | ~7-10ms | ~7-10ms | compute |
| Module A（process） | ~21-26ms | 同 | compute |
| render+JPEG+b64（1280×720） | ~12ms | 同 | encode（异步） |

Module A 内部分解（连续流，中位数）：`_compute_flow` 7.3ms（RAFT 本体~2ms + GPU↔CPU 传输/numpy ~5ms）、`_compute_a1` 5.6ms、`_compute_a2` 5.0ms、`_compute_lbp` 3.8ms、`_compute_a3` 3.1ms、`_compute_scene_context` 0.8ms（优化后）。

GPU 段（flow+lbp）约 12ms，CPU 段（scene+a1+a2+a3）约 14.6ms。**无单一热点，是多组件均摊。**

## 有效优化（全部经 A/B 验证，零检测数值改变）

1. **三级流水线**（runner.py `_offline_scan_loop`）：新增 decode 预取线程，形成 `decode ‖ compute ‖ encode` 三线程。GPU 在解码 20ms 期间不再空转。贡献最大，~10→25 FPS。
2. **scene_context 直方图化**（detector.py）：单次 `cv2.calcHist` 256-bin 替代 mean/std/over/under 四次全图遍历。微基准 2.1ms→0.14ms，数值差 <1e-10（直方图加权和等价于逐像素统计）。
3. **a3b 后台线程节流**（detector.py）：原代码中 `_a3b_frame_count`/`_a3b_interval` 定义了但从未使用，后台 a3b 线程跑完一轮（~59ms）立即重启，**100% 占用后台线程并通过 GIL 拖累主路径 ~5ms/帧**（隔离实验确认）。接上节流后改为每 `_a3b_interval`（调为 6）帧一轮。a3b 检测静态媒体属慢变化，降频不影响功能。

## 最终结果（500 帧长采样，端到端流水线）

- 1080p 源：**31.9 FPS**（达成目标）
- 4K 源：24 FPS（受限于 4K 解码 15ms，转码 1080p 是正解）

## 测量陷阱教训（本次调查的核心价值）

多次"未测量先假设"被实测推翻，记录以警示：

1. **臆想 grid 循环占 12ms** → 实测仅 0.77ms。`cv2.calcHist` 是高度优化的 C 代码，逐 cell 调用并不慢。自作主张的 `np.bincount` 向量化反而更慢（2.12ms），已回滚。
2. **臆想 GIL 是流水线瓶颈** → 早期隔离实验显示线程间 GIL 争用只占 6%，一度排除该方向。但后续发现真正的 GIL 元凶另有其人（见下）。
3. **flow GPU 后处理"优化"** → 数值验证等价但仅省 0.18ms。诊断时插入的 `torch.cuda.synchronize()` 本身制造了假的 3.6ms 后处理耗时。放弃。
4. **proxy 简化测量骗人** → 流水线原型只测 a1/a2/a3，得出"串行 13ms / 流水线更慢"的反常结论；补全 a3b/ta/joint/features 后真实是 28.6ms。
5. **短采样假象** → 测开头 80 帧得 53 FPS（GPU 突发高频期），500 帧长采样才是真实值。
6. **累积平均指标制造"越跑越慢"假象** → HUD 的 `detect_fps` 原本是 `(frame_idx+1)/从扫描起点的总耗时`，早期慢帧与每次 a3b 停顿被永久计入，数字只会向长期均值收敛，看起来一直在变慢。改成最近 30 帧滚动窗口后，显示真实瞬时速率，"恶化"消失。
7. **错怪 a3b 频率、误判 interval=90 生效** → 逐组件计时 + `a3b_alive` 标记证明：a3b 后台线程的纯 Python 部分（候选提取的轮廓/投影循环、16 万次 `_bbox_iou`/`_bbox_area`）持 GIL 不放，运行那几十毫秒里把 compute 线程的 a1（1.2ms→28ms）、lbp 拖慢一倍。这才是 GIL 真凶（线程对 CPU 密集 Python 无效）。一度把 `static_image_interval` 默认改到 90 试图降频，但 config 文件里该值=4 会覆盖代码默认，**改动从未生效**；而 bench 仍达标，反证真正的功臣是候选数上限优化，与 interval 无关。最终 interval 保持 4，零行为损失。

结论：性能判断必须基于完整路径 + 长采样 + A/B 对照 + 真实瞬时指标；任何插桩计时都可能扭曲结果；改动是否"生效"必须实测确认，配置可能覆盖代码默认值。

## 决定性的修复

**a3b 候选数双重上限**（`_extract_media_candidates`）：
- 轮廓循环按面积降序只取 top-40（媒体边框必为大轮廓，小轮廓是噪声）；
- 全局候选上限 64（复杂画面会产生数百候选，每个做昂贵的逐框边界/纹理/IoU 统计）。
- 效果：a3b 候选提取单轮 221ms→16ms，完整 a3b 单轮从 300-700ms 降到几十 ms，后台线程不再长时间霸占 GIL。
- a3b 检测语义不变（只是不再处理必然落选的小/重叠候选），间隔保持 4，静态媒体响应速度无损失。

## 最终结果（tools/bench_demo.py 自动化基准，500 帧）

- 1080p 源，无广播：中位 ~35 FPS
- 1080p 源，带 WebSocket 广播开销：中位 ~32 FPS
- 均达成 30+ 目标，全程不再恶化。
- 自动化基准直接驱动真实 DemoRunner，不经浏览器/web 层，一键复现。

## 未做的大重构

GPU/CPU 段拆分流水线、a3b 进程化（multiprocessing 脱离 GIL）。因候选上限优化已让 a3b 不再拖累主路径、当前已达标，**暂不实施**。若未来 a3b 算法变重或需更低延迟，进程化是根治方向。

## 后续建议

- a3b 间隔经 `module_a.static_image_interval`（config）可调，默认 4。候选上限（top-40 / 全局 64）是经验值，若漏检大幅媒体边框可调大，但会增加单轮耗时。**待实验确认**对静态媒体攻击召回的影响。
- 4K 源若必须直接处理，瓶颈在解码（OS/ffmpeg 层），demo 内无法再优化，应在采集端转码为 1080p。
