from __future__ import annotations

import argparse
import json
from pathlib import Path

from defense.model_security import ModelSecurityService
from defense.runtime.config import DEFAULT_CONFIG_PATH, project_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Module B model security fingerprint/scan")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--profile", default="default")
    parser.add_argument("--scan-type", choices=["status", "quick", "full"], default="status")
    parser.add_argument("--trust-if-low-risk", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    svc = ModelSecurityService(config_path=args.config, root=project_root())
    if args.scan_type == "status":
        result = svc.status(profile=args.profile)
    else:
        result = svc.scan(scan_type=args.scan_type, profile=args.profile, trust_if_low_risk=args.trust_if_low_risk)
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
