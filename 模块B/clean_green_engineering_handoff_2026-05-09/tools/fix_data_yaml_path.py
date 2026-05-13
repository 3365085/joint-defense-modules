"""Ensure `data.yaml` uses an absolute ``path:`` so Ultralytics does not
resolve relative paths against the user-global ``settings.json`` ``datasets_dir``.

Typical symptom this fixes:

    Dataset '../data/helmet_head_yolo_val/data.yaml' images not found,
    missing path 'C:\\Users\\admin\\Desktop\\TongYan_v3\\datasets'

Usage::

    python tools/fix_data_yaml_path.py --yaml data/helmet_head_yolo_val/data.yaml

Safe and idempotent: the script only rewrites ``path:`` when it is missing,
empty, ``.``, or a relative string. Everything else (names, train/val keys,
etc.) is preserved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install PyYAML") from exc


def fix_yaml_path(yaml_path: Path) -> bool:
    yaml_path = yaml_path.resolve()
    if not yaml_path.exists():
        raise SystemExit(f"data.yaml not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML root must be a mapping: {yaml_path}")

    current = data.get("path")
    dataset_root = yaml_path.parent
    target = str(dataset_root).replace("\\", "/")

    need_rewrite = (
        current is None
        or str(current).strip() in ("", ".")
        or not Path(str(current)).is_absolute()
    )
    if not need_rewrite:
        return False

    data["path"] = target
    # Preserve key ordering: write out in the original yaml format, putting
    # ``path`` first for readability.
    ordered = {"path": target}
    for key, value in data.items():
        if key == "path":
            continue
        ordered[key] = value

    with yaml_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(ordered, fp, allow_unicode=True, sort_keys=False)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Path to data.yaml")
    args = parser.parse_args(argv)

    changed = fix_yaml_path(Path(args.yaml))
    if changed:
        print(f"[ok] rewrote {args.yaml} with absolute path")
    else:
        print(f"[skip] {args.yaml} already has an absolute path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
