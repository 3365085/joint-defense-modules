from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.utils.assets import load_asset_config, validate_asset_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local Model B detox asset paths.")
    parser.add_argument("--assets-config", default="configs/assets.local.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_asset_config(args.assets_config)
    errors = validate_asset_config(config)
    payload = {
        "assets_config": args.assets_config,
        "ok": not errors,
        "errors": errors,
        "resolved": {
            "suspicious_model": str(config.suspicious_model),
            "teacher_model": str(config.teacher_model) if config.teacher_model else None,
            "train_images": str(config.train_images),
            "train_labels": str(config.train_labels),
            "data_yaml": str(config.data_yaml),
            "external_replay_roots": [str(path) for path in config.external_replay_roots],
            "external_eval_roots": [str(path) for path in config.external_eval_roots],
            "source_materials": str(config.source_materials),
            "output_root": str(config.output_root),
            "target_classes": list(config.target_classes),
            "device": config.device,
        },
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"[assets] config: {args.assets_config}")
        for key, value in payload["resolved"].items():
            print(f"[assets] {key}: {value}")
        if errors:
            for error in errors:
                print(f"[assets][ERROR] {error}", file=sys.stderr)
        else:
            print("[assets] ok")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
