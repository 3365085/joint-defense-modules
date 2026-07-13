from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.ppe_overlay_export import export_ppe_overlay_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PPE overlay records from a video segment.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--display-options-json", default=None)
    args = parser.parse_args()

    records = export_ppe_overlay_records(
        video=args.video,
        output_json=args.output_json,
        profile=args.profile,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        config=args.config,
        display_options=json.loads(args.display_options_json) if args.display_options_json else {},
        project_root=PROJECT_ROOT,
    )
    print(json.dumps({"ok": True, "output_json": str(args.output_json), "record_count": len(records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
