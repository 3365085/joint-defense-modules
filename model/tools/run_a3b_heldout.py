from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.a3b_heldout import (  # noqa: E402
    evaluate_a3b_heldout,
    heldout_gate_failures,
)


def _progress(index: int, total: int, result: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "index": index,
                "total": total,
                "clip_id": result.get("clip_id"),
                "frames": result.get("frames"),
                "a3b_trigger_frames": result.get("a3b_trigger_frames"),
                "error": result.get("error"),
                "elapsed_s": result.get("elapsed_s"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Module A A3b through the production "
            "FrameProcessor/A3BSoftTriggerState path."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "acceptance"
            / "module_a_authoritative_manifest_v1.json"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", default="desktop_rtx")
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--cap-frames", type=int, default=240)
    args = parser.parse_args()

    report = evaluate_a3b_heldout(
        manifest=args.manifest,
        output_json=args.output,
        config=args.config,
        profile=args.profile,
        split=args.split,
        cap_frames=args.cap_frames,
        repository_root=REPOSITORY_ROOT,
        progress=_progress,
    )
    gate_failures = heldout_gate_failures(report)
    print(
        json.dumps(
            {
                **report["summary"],
                "gate_failures": gate_failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not gate_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
