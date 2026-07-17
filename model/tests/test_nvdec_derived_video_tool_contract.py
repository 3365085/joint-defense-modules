from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_nvdec_derived_video_builder_exposes_production_cli(
    pkg_root: Path,
) -> None:
    tool = pkg_root / "tools" / "build_nvdec_derived_video.py"

    completed = subprocess.run(
        [sys.executable, str(tool), "--help"],
        cwd=pkg_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    assert "--manifest" in completed.stdout
    assert "--asset-id" in completed.stdout
    assert "--cache-root" in completed.stdout
    assert "--force" in completed.stdout
    source = tool.read_text(encoding="utf-8")
    assert "h264_nvenc" in source
    assert '"lossless"' in source
    assert '"framemd5"' in source
    assert "allow_cpu_fallback=False" in source
