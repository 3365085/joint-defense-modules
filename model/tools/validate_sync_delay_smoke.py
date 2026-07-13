from __future__ import annotations

"""Smoke test for the backend MP4 preview/detection synchronization path.

The check uses ``empty_smoke`` so it can run without CUDA or model artifacts.
It validates the current MonitorEngine contract: backend preview starts, status
is populated, and overlay records are produced through the runtime pipeline.
"""

import json
import sys
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from defense.runtime import MonitorEngine, PipelineCache, open_capture, project_root, sample_sources
from defense.runtime.config import workspace_material_root


def _material_source() -> str:
    material_root = workspace_material_root()
    expected = material_root / "手机随意录制的视频" / "固定镜头室外视频.mp4"
    if expected.exists():
        return str(expected)
    for item in sample_sources():
        source = str(item.get("source") or "")
        if source.startswith(str(material_root)) and source.lower().endswith(".mp4"):
            return source
    raise AssertionError(f"no MP4 sample found under material root: {material_root}")


def _wait_for_overlay(engine: MonitorEngine, *, timeout_s: float = 5.0) -> dict:
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        overlay = engine.get_overlay(since_seq=0)
        if overlay.get("records"):
            return overlay
        status = engine.get_status()
        if status.get("error"):
            raise AssertionError(f"runtime error while waiting for overlay: {status.get('error')}")
        sleep(0.05)
    raise AssertionError(f"overlay timeline is empty after {timeout_s:.1f}s")


def main() -> int:
    root = project_root()
    source = _material_source()
    cap = open_capture("file", source)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise AssertionError(f"sample cannot be read: {source}")
        shape = tuple(int(x) for x in frame.shape)
    finally:
        cap.release()

    engine = MonitorEngine(PipelineCache(root=root))
    run_id = engine.start(source_type="file", source=source, profile="empty_smoke", realtime=True)
    try:
        status = engine.wait_ready_for_preview(run_id, timeout=10.0)
        if not status.get("ready_for_preview"):
            raise AssertionError(f"preview not ready: {status}")
        if status.get("backend") != "empty":
            raise AssertionError(f"expected empty backend, got {status.get('backend')}")

        overlay = _wait_for_overlay(engine)
        first_record = dict((overlay.get("records") or [{}])[0])
        for key in ("video_time_s", "a3b_score", "a3b_triggered", "overlay_seq"):
            if key not in first_record:
                raise AssertionError(f"overlay record missing {key}: {first_record}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "project_root": str(root),
                    "material_source": source,
                    "material_shape": shape,
                    "profile": status.get("profile"),
                    "backend": status.get("backend"),
                    "overlay_records": len(overlay.get("records") or []),
                    "overlay_seq": overlay.get("seq"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
