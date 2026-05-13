"""StreamSource: 为 Module A 流式管线提供拉流 / 解码隔离与可观测性。

模块职责：
    在 ``cv2.VideoCapture`` 之上再包一层后台拉流线程 + 显式解码队列，为 Module A
    视频防御主链路提供以下能力（对应 spec
    ``.kiro/specs/module-a-stream-spoof-and-synth-detection`` 的 Requirement 2 与
    Design §2）：

      1. 显式丢帧策略：队列满时丢弃最旧帧并累计 ``drop_count``，不让拉流端阻塞。
      2. 三类真实时间戳：
            * ``source_ts``  —— 帧到达拉流线程的时刻 (``time.monotonic()``)。
            * ``decode_ts``  —— ``cap.read`` 返回后的时刻。
            * ``process_ts`` —— ``read()`` 返回给调用侧前的时刻。
      3. 丢帧降级检测：``drop_window_s`` 滑窗内 drop_ratio > ``drop_threshold_ratio``
         时，当前帧 ``flags["stream_degraded"] = True``。
      4. 断流检测：超过 ``no_frame_timeout_s`` 秒未拉到新帧时，``read()`` 返回
         ``None``；一旦恢复，恢复后的第一帧携带 ``flags["stream_recovered"] = True``。
      5. 分辨率 / FPS 变化检测：通过 ``flags["stream_geometry_changed"]`` 以及
         ``notify_geometry_change`` 回调通知上层重新初始化 ROI 与告警状态机。

CUDA 硬约束说明：
    StreamSource 自身处于拉流 / 解码层，不触碰 CUDA，这是合规的；Module A 下游依旧
    要求 CUDA。但当输入源无法打开时，``start()`` 必须抛出 ``RuntimeError``，不得返回
    ``None`` 或占位帧，以避免让 CUDA 路径上的特征模块吃到无效输入。
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


GeometryCallback = Callable[[int, int, float], None]
"""几何变化回调签名：``cb(width, height, fps_reported)``。"""


@dataclass(slots=True)
class FrameEnvelope:
    """StreamSource 每次 ``read()`` 成功时返回的帧包装。

    Attributes
    ----------
    frame_idx : int
        StreamSource 内部单调递增的帧号（从 0 开始），与上游 VideoDefensePipeline 自
        己的 ``frame_idx`` 解耦，方便跨线程追溯同一帧。
    source_ts : float
        帧在拉流线程刚被拉到时的 ``time.monotonic()`` 值，秒。
    decode_ts : float
        ``cap.read`` 返回后（解码完成）的 ``time.monotonic()`` 值，秒。
    process_ts : float
        ``StreamSource.read()`` 即将把 envelope 返回给调用侧前的
        ``time.monotonic()`` 值，秒。代表帧进入处理线程的时刻。
    width, height : int
        本帧解码后的画面宽高（像素）。
    fps_reported : float
        源头上报的 FPS（通过 ``CAP_PROP_FPS``）。RTSP / 摄像头可能上报 0。
    frame : np.ndarray
        BGR uint8 解码后的原始帧。
    flags : dict[str, bool]
        本帧附带的状态标记：``stream_degraded`` / ``stream_recovered`` /
        ``stream_geometry_changed``。不触发的 flag 不会出现在 dict 中，调用方应使用
        ``flags.get("xxx", False)`` 判断。
    """

    frame_idx: int
    source_ts: float
    decode_ts: float
    process_ts: float
    width: int
    height: int
    fps_reported: float
    frame: np.ndarray
    flags: dict[str, bool]


# ---------------------------------------------------------------------------
# StreamSource
# ---------------------------------------------------------------------------


_VALID_SOURCE_TYPES = ("file", "rtsp", "camera")


class StreamSource:
    """``cv2.VideoCapture`` 的线程化封装，带显式解码队列与流健康度观测。

    Parameters
    ----------
    source_type : {"file", "rtsp", "camera"}
        输入源类型。与 ``tools/module_a_monitor_app.py`` 现有约定一致：

        * ``"file"``   — ``source`` 为 MP4 / MKV 等本地文件路径。
        * ``"rtsp"``   — ``source`` 为 ``rtsp://`` / ``http://`` / ``https://`` URL，
          使用 ``cv2.CAP_FFMPEG`` 后端。
        * ``"camera"`` — ``source`` 为摄像头编号字符串（例如 ``"0"``），Windows 下
          优先 ``cv2.CAP_DSHOW``，失败回退 ``cv2.CAP_ANY``。
    source : str
        源地址 / 路径 / 编号字符串。
    queue_capacity : int, default 4
        拉流线程到处理线程之间的解码队列容量。容量越小越"实时"（丢帧代价低），
        越大越"耐抖"（但端到端延迟会放大）。
    no_frame_timeout_s : float, default 10.0
        连续无新帧达到该阈值则触发 ``stream_lost``，``read()`` 返回 ``None``；
        ``lost_count`` 自增一次（只在进入 lost 态时自增）。
    drop_window_s : float, default 2.0
        丢帧降级检测的滑动窗口长度（秒）。
    drop_threshold_ratio : float, default 0.20
        滑动窗口内 drop_ratio 超过该比例时，下一帧 ``flags["stream_degraded"]``
        被置 True，``degraded_count`` 自增。
    read_poll_interval_s : float, default 0.01
        ``read()`` 在队列空时的阻塞 / 轮询间隔（秒）。同时也是拉流线程在读失败后
        重试前的短暂 sleep 时间。
    open_timeout_ms : int, default 5000
        RTSP 源使用的 ``CAP_PROP_OPEN_TIMEOUT_MSEC`` / ``CAP_PROP_READ_TIMEOUT_MSEC``
        （若 OpenCV 构建支持）。

    Notes
    -----
    线程模型：

        * 拉流线程：在 ``start()`` 时创建，循环 ``cap.read`` 并把 envelope 塞进
          ``Queue``；队列满时主动丢掉最旧的一个 envelope 再塞新的，累计
          ``drop_count`` 并把"本帧触发丢帧"记入滑动窗口。
        * 处理线程：调用方直接使用 ``read()`` 从队列取帧；lost 期间 ``read()``
          返回 ``None``，不抛异常；恢复后第一帧通过 ``flags["stream_recovered"]``
          通知调用方。

    线程安全：所有 stats / ``last_*`` / 事件窗口 / lost 状态均在 ``_lock`` 保护下
    读写。
    """

    def __init__(
        self,
        source_type: str,
        source: str,
        queue_capacity: int = 4,
        no_frame_timeout_s: float = 10.0,
        drop_window_s: float = 2.0,
        drop_threshold_ratio: float = 0.20,
        read_poll_interval_s: float = 0.01,
        open_timeout_ms: int = 5000,
        reconnect_after_s: float = 3.0,
    ) -> None:
        if source_type not in _VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type 必须是 {_VALID_SOURCE_TYPES} 之一，实际为 {source_type!r}"
            )
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source 必须是非空字符串")
        if queue_capacity < 1:
            raise ValueError("queue_capacity 必须 >= 1")
        if no_frame_timeout_s <= 0:
            raise ValueError("no_frame_timeout_s 必须 > 0")
        if drop_window_s <= 0:
            raise ValueError("drop_window_s 必须 > 0")
        if not (0.0 < drop_threshold_ratio <= 1.0):
            raise ValueError("drop_threshold_ratio 必须在 (0, 1] 区间内")
        if read_poll_interval_s <= 0:
            raise ValueError("read_poll_interval_s 必须 > 0")
        if open_timeout_ms <= 0:
            raise ValueError("open_timeout_ms 必须 > 0")
        if reconnect_after_s <= 0:
            raise ValueError("reconnect_after_s 必须 > 0")

        self.source_type = source_type
        self.source = source
        self.queue_capacity = int(queue_capacity)
        self.no_frame_timeout_s = float(no_frame_timeout_s)
        self.drop_window_s = float(drop_window_s)
        self.drop_threshold_ratio = float(drop_threshold_ratio)
        self._read_poll_interval_s = float(read_poll_interval_s)
        self._open_timeout_ms = int(open_timeout_ms)
        self._reconnect_after_s = float(reconnect_after_s)

        self._queue: Queue[FrameEnvelope] = Queue(maxsize=self.queue_capacity)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._geometry_cb: GeometryCallback | None = None
        self._lock = threading.Lock()

        # --- Stats / observability ---
        self._pulled_frames = 0
        self._delivered_frames = 0
        self._drop_count = 0
        self._lost_count = 0
        self._recovered_count = 0
        self._reconnect_count = 0
        self._geometry_change_count = 0
        self._degraded_count = 0
        self._last_width = 0
        self._last_height = 0
        self._last_fps = 0.0
        self._frame_idx = 0

        # --- Window-based drop ratio tracking: deque of (monotonic_ts, dropped) ---
        # ``dropped`` 为 True 表示"该帧的入队导致旧帧被挤出队列"。
        self._event_window: deque[tuple[float, bool]] = deque()

        # --- Lost / recovery state ---
        self._last_frame_monotonic: float | None = None
        self._lost_state = False

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """打开输入源并启动后台拉流线程；失败时抛出 ``RuntimeError``。

        遵循 CUDA 硬约束：不返回 ``None`` 或占位帧。调用方必须捕获 ``RuntimeError``
        后以明确错误状态终止启动流程。
        """
        if self._thread is not None and self._thread.is_alive():
            return

        try:
            cap = self._open_capture()
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - OpenCV 失败路径
            raise RuntimeError(
                f"StreamSource 打开输入源时抛出异常：source_type={self.source_type}, "
                f"source={self.source!r}, error={exc!r}"
            ) from exc

        if cap is None or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            raise RuntimeError(
                "StreamSource 无法打开输入源："
                f"source_type={self.source_type}, source={self.source!r}"
            )

        # 压小底层缓冲，尽量拿到最新帧；某些后端不支持该 prop，忽略即可。
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._cap = cap
        with self._lock:
            self._last_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self._last_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self._last_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"StreamSource-{self.source_type}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """通知拉流线程退出并释放 ``cv2.VideoCapture``。多次调用安全。"""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # 给线程一点时间退出；拉流线程在每次 sleep/read 后都会检查 stop_event。
            thread.join(timeout=max(1.0, self._read_poll_interval_s * 10))
        self._thread = None

        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        # 清空队列，避免 stale envelope 留到下一次 start()。
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass

    def __enter__(self) -> StreamSource:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ public API

    def read(self, timeout: float | None = None) -> FrameEnvelope | None:
        """取一帧。队列空 / lost 期间返回 ``None``；正常返回 ``FrameEnvelope``。

        Parameters
        ----------
        timeout : float | None, optional
            从队列取帧时的阻塞超时（秒）。默认 ``None`` 等价于
            ``read_poll_interval_s`` 的轮询间隔；调用方也可传更大的值进行阻塞式等待。
        """
        if self._thread is None:
            return None

        poll_timeout = self._read_poll_interval_s if timeout is None else float(timeout)
        try:
            envelope = self._queue.get(timeout=poll_timeout)
        except Empty:
            # 队列空：检查是否需要进入 lost 态（只有首次进入时才自增 lost_count）。
            self._maybe_trigger_lost()
            return None

        process_ts = time.monotonic()
        envelope.process_ts = process_ts

        with self._lock:
            self._delivered_frames += 1

        # 几何变化回调：在处理线程侧触发，确保回调看到的是 read() 已经消费的那一帧。
        if envelope.flags.get("stream_geometry_changed") and self._geometry_cb is not None:
            try:
                self._geometry_cb(envelope.width, envelope.height, envelope.fps_reported)
            except Exception:
                # 回调失败不影响当前帧返回；调用侧可在日志中再次确认几何变化事件。
                pass

        return envelope

    def notify_geometry_change(self, cb: GeometryCallback | None) -> None:
        """注册几何变化回调；传 ``None`` 取消。回调在 ``read()`` 所在线程被调用。"""
        self._geometry_cb = cb

    def stats(self) -> dict[str, Any]:
        """返回一份当前观测到的流健康度快照，用于 run summary / HUD。"""
        with self._lock:
            return {
                "pulled_frames": self._pulled_frames,
                "delivered_frames": self._delivered_frames,
                "drop_count": self._drop_count,
                "lost_count": self._lost_count,
                "recovered_count": self._recovered_count,
                "reconnect_count": self._reconnect_count,
                "geometry_change_count": self._geometry_change_count,
                "degraded_count": self._degraded_count,
                "last_width": self._last_width,
                "last_height": self._last_height,
                "last_fps": self._last_fps,
                "queue_depth": self._queue.qsize(),
                "lost_state": self._lost_state,
            }

    # ------------------------------------------------------------------ internals

    def _open_capture(self) -> cv2.VideoCapture:
        """按 source_type 构造 ``cv2.VideoCapture``。不保证 ``isOpened()``，由 caller 判断。"""
        if self.source_type == "file":
            return cv2.VideoCapture(str(self.source))

        if self.source_type == "camera":
            try:
                index = int(str(self.source).strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"摄像头 source 必须是整型编号字符串，实际为 {self.source!r}"
                ) from exc
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(index)
            return cap

        # rtsp / http / https
        url = str(self.source).strip()
        if not url.lower().startswith(("rtsp://", "http://", "https://")):
            raise RuntimeError(
                f"RTSP 源 URL 必须以 rtsp:// / http:// / https:// 开头，实际为 {url!r}"
            )

        # 强制 TCP 传输 + 低延迟，避免 UDP 丢包导致卡顿
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        )

        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), self._open_timeout_ms])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([int(cv2.CAP_PROP_READ_TIMEOUT_MSEC), self._open_timeout_ms])
        if params:
            try:
                return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
            except Exception:
                pass
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    def _run_loop(self) -> None:
        """后台拉流循环。每轮拉到一帧即入队（或挤掉最旧帧后入队）。

        对 RTSP/camera 源，当连续读取失败超过 ``_reconnect_after_s`` 秒时，自动释放
        旧连接并重新打开，实现断流自动重连。重连间隔采用指数退避（1s → 2s → 4s → 最大 30s）。
        """
        cap = self._cap
        if cap is None:
            return

        # 重连相关状态
        consecutive_fail_start: float | None = None
        reconnect_backoff_s = 1.0
        max_backoff_s = 30.0

        while not self._stop_event.is_set():
            source_ts = time.monotonic()
            try:
                ret, frame = cap.read()
            except Exception:
                ret, frame = False, None
            decode_ts = time.monotonic()

            if not ret or frame is None:
                # 对 file：视为 EOF / 读尽，直接退出线程，让 read() 自然收到 None。
                if self.source_type == "file":
                    break

                # 对 rtsp / camera：记录连续失败起始时间，超时后触发重连。
                if consecutive_fail_start is None:
                    consecutive_fail_start = time.monotonic()

                elapsed_fail = time.monotonic() - consecutive_fail_start

                if elapsed_fail >= self._reconnect_after_s:
                    # 尝试重连：释放旧连接，等待退避时间，重新打开。
                    try:
                        cap.release()
                    except Exception:
                        pass

                    with self._lock:
                        self._reconnect_count += 1

                    # 退避等待（可被 stop 中断）
                    if self._stop_event.wait(timeout=reconnect_backoff_s):
                        break

                    # 重新打开连接
                    try:
                        cap = self._open_capture()
                        self._cap = cap
                    except Exception:
                        cap = cv2.VideoCapture()  # 空的，下一轮会继续失败
                        self._cap = cap

                    # 指数退避，上限 30 秒
                    reconnect_backoff_s = min(reconnect_backoff_s * 2, max_backoff_s)
                    consecutive_fail_start = None  # 重置计时
                else:
                    # 还没到重连阈值，短暂 sleep 后继续尝试
                    if self._stop_event.wait(timeout=self._read_poll_interval_s):
                        break
                continue

            # 成功读到帧 — 重置重连状态
            consecutive_fail_start = None
            reconnect_backoff_s = 1.0

            try:
                height, width = int(frame.shape[0]), int(frame.shape[1])
            except Exception:
                # 解码后形状异常视为坏帧，丢弃。
                continue

            fps_reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

            flags: dict[str, bool] = {}
            with self._lock:
                self._pulled_frames += 1
                is_first_frame = self._pulled_frames == 1

                # 恢复检测：从 lost 态拿到首帧时打上 stream_recovered。
                if self._lost_state:
                    self._lost_state = False
                    self._recovered_count += 1
                    flags["stream_recovered"] = True

                # 几何变化检测。
                if is_first_frame:
                    self._last_width = width
                    self._last_height = height
                    self._last_fps = fps_reported
                else:
                    geometry_changed = (
                        width != self._last_width
                        or height != self._last_height
                        or not _fps_close(fps_reported, self._last_fps)
                    )
                    if geometry_changed:
                        self._geometry_change_count += 1
                        flags["stream_geometry_changed"] = True
                        self._last_width = width
                        self._last_height = height
                        self._last_fps = fps_reported

                # 把本帧作为一次"拉到帧"事件放入滑动窗口，默认未触发丢弃。
                self._prune_event_window_locked(source_ts)
                self._event_window.append((source_ts, False))

                self._last_frame_monotonic = source_ts

                # 降级检测（在 append 之后计算；若随后入队时发生丢弃会再把窗口最后一项翻成 True）。
                if self._window_drop_ratio_locked() > self.drop_threshold_ratio:
                    self._degraded_count += 1
                    flags["stream_degraded"] = True

                frame_idx = self._frame_idx
                self._frame_idx += 1

            envelope = FrameEnvelope(
                frame_idx=frame_idx,
                source_ts=source_ts,
                decode_ts=decode_ts,
                process_ts=0.0,  # 由 read() 填入
                width=width,
                height=height,
                fps_reported=fps_reported,
                frame=frame,
                flags=flags,
            )

            self._enqueue_with_drop(envelope)

    def _enqueue_with_drop(self, envelope: FrameEnvelope) -> None:
        """把 envelope 放进队列；满则先丢最旧，drop_count 累加。"""
        try:
            self._queue.put_nowait(envelope)
            return
        except Full:
            pass

        # 队列满：丢最旧一个，把当前帧入队；滑动窗口中最后一条标记为 dropped。
        try:
            self._queue.get_nowait()
        except Empty:
            pass
        with self._lock:
            self._drop_count += 1
            if self._event_window:
                ts, _ = self._event_window[-1]
                self._event_window[-1] = (ts, True)

        try:
            self._queue.put_nowait(envelope)
        except Full:
            # 理论上不会发生；兜底再累加一次 drop 并放弃当前帧。
            with self._lock:
                self._drop_count += 1

    def _prune_event_window_locked(self, now: float) -> None:
        cutoff = now - self.drop_window_s
        while self._event_window and self._event_window[0][0] < cutoff:
            self._event_window.popleft()

    def _window_drop_ratio_locked(self) -> float:
        if not self._event_window:
            return 0.0
        dropped = sum(1 for _, flag in self._event_window if flag)
        return dropped / float(len(self._event_window))

    def _maybe_trigger_lost(self) -> None:
        """队列空时检查是否首次进入 lost 态；只在进入时 lost_count += 1。"""
        now = time.monotonic()
        with self._lock:
            last = self._last_frame_monotonic
            if last is None:
                # 还没拉到过任何帧，不算 lost（让上层继续等待首帧）。
                return
            if self._lost_state:
                return
            if (now - last) > self.no_frame_timeout_s:
                self._lost_state = True
                self._lost_count += 1


def _fps_close(a: float, b: float, *, tol: float = 0.5) -> bool:
    """FPS 比较的容忍函数：0.5 FPS 以内视为等同（应对浮点抖动 / 源端量化）。"""
    return abs(float(a) - float(b)) <= tol


__all__ = ["FrameEnvelope", "GeometryCallback", "StreamSource"]
