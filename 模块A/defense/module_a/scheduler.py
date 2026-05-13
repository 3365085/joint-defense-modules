from __future__ import annotations


class ModuleAScheduler:
    def __init__(self, keyframe_interval: int = 3):
        self.keyframe_interval = max(1, int(keyframe_interval))

    def is_keyframe(self, frame_idx: int) -> bool:
        return frame_idx % self.keyframe_interval == 0

    def should_run_slow_path(self, frame_idx: int, fast_suspicious: bool) -> bool:
        return fast_suspicious or self.is_keyframe(frame_idx)
