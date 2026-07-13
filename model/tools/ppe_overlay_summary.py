from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.ppe_overlay_summary import (  # noqa: E402
    build_ppe_overlay_summary_report,
    load_ppe_overlay_records,
    write_ppe_overlay_csv_rows,
    write_ppe_overlay_summary_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Module A PPE overlay JSON without reading video frames or running inference."
    )
    parser.add_argument("--input", required=True, type=Path, help="Overlay JSON path.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline overlay JSON path.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Write candidate summary JSON.")
    parser.add_argument("--comparison-out", type=Path, default=None, help="Write baseline-vs-candidate comparison JSON.")
    parser.add_argument("--report-out", type=Path, default=None, help="Write full report JSON.")
    parser.add_argument("--rows-out", type=Path, default=None, help="Write compact overlay row CSV.")
    parser.add_argument("--min-person-conditioned-run", type=int, default=3)
    args = parser.parse_args()

    report = build_ppe_overlay_summary_report(
        args.input,
        baseline_path=args.baseline,
        min_person_conditioned_run=args.min_person_conditioned_run,
    )
    write_ppe_overlay_summary_report(
        report,
        summary_path=args.summary_out,
        comparison_path=args.comparison_out,
    )
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.rows_out is not None:
        write_ppe_overlay_csv_rows(args.rows_out, load_ppe_overlay_records(args.input))

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
