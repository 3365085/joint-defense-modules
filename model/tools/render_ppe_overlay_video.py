from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.ppe_overlay_video import (  # noqa: E402
    render_ppe_overlay_video,
    write_render_summary,
)


def _display_options(raw: str | None) -> dict:
    if not raw:
        return {}
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render PPE overlay records onto a source video with coordinate scaling."
    )
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--overlay-json", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--display-options-json", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--hold-frames", type=int, default=4)
    args = parser.parse_args()

    summary = render_ppe_overlay_video(
        source_video=args.source_video,
        overlay_json=args.overlay_json,
        output_video=args.output_video,
        display_options=_display_options(args.display_options_json),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        hold_frames=args.hold_frames,
    )
    if args.summary_out is not None:
        write_render_summary(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
