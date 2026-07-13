"""Legacy compatibility entrypoint for the Module A web monitor.

The original project launched ``python tools/module_a_monitor_app.py``.  The
official Pixi tasks now start ``python -m defense.web.server`` instead.  Keep
this file thin and do not add runtime product logic here: UI routing, stream
handling, inference runtime, tracking and evidence recording live under
``defense.*``.

A few legacy symbols are re-exported for existing diagnostics/tests; their
implementations live in the new module layout.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defense.module_a.postprocess import PPEDisplayTracker as PPEBoxStabilizer  # noqa: E402
from defense.runtime.ppe_state import SafetyHelmetState  # noqa: E402
from defense.visualization.overlay import draw_hud, draw_ppe_boxes, draw_ppe_hud  # noqa: E402
from defense.web.server import main  # noqa: E402

__all__ = [
    "PPEBoxStabilizer",
    "SafetyHelmetState",
    "draw_hud",
    "draw_ppe_boxes",
    "draw_ppe_hud",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
