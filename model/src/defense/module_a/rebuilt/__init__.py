"""Rebuilt Module A detection kernel ported from the read-only rebuilt_demo.

This package contains the demo's design-aligned Module A detector (A1-A4 +
branch-B blinding + scene-adaptive baseline + target-anchored + joint decision
with N-of-M temporal confirmation). It consumes and returns the shared
``defense.module_a`` contract (``ModuleAInput`` / ``ModuleAResult`` / ``ROI``),
so it can be swapped in behind the existing runtime/preview/frame-skip shell.

Optional Rust acceleration (``module_a_native``) and RAFT-TRT optical flow are
loaded best-effort with automatic Python/CPU fallback.
"""

from .detector import ModuleADetector as RebuiltModuleADetector

__all__ = ["RebuiltModuleADetector"]
