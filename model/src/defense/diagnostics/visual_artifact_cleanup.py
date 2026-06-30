from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def cleanup_visual_artifacts(
    *,
    roots: list[str | Path],
    apply: bool = False,
) -> dict[str, Any]:
    root_paths = [Path(root) for root in roots]
    if not root_paths:
        raise ValueError("at least one root is required")

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for root in root_paths:
        if not root.exists():
            skipped.append({"path": str(root), "reason": "root_missing"})
            continue
        if root.is_file():
            _classify_file(root, candidates, skipped)
            continue
        for manifest in root.rglob("manifest.json"):
            _classify_manifest(manifest, candidates, skipped)
        for path in root.rglob("*_contact_sheet.jpg"):
            _append_candidate(
                candidates,
                path=path,
                artifact_type="contact_sheet_jpg",
                reason="contact_sheet_jpg_is_temporary_visual_review_artifact",
            )

    deleted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if apply:
        for item in sorted(candidates, key=lambda entry: len(Path(str(entry["path"])).parts), reverse=True):
            path = Path(str(item["path"]))
            try:
                if path.is_dir():
                    _remove_empty_or_known_temp_dir(path)
                elif path.exists():
                    path.unlink()
                deleted.append(item)
            except Exception as exc:  # pragma: no cover - surfaced in report
                error = dict(item)
                error["error"] = str(exc)
                errors.append(error)

    return {
        "apply": bool(apply),
        "roots": [str(root) for root in root_paths],
        "candidate_count": len(candidates),
        "deleted_count": len(deleted),
        "error_count": len(errors),
        "skipped_count": len(skipped),
        "candidates": candidates,
        "deleted": deleted,
        "errors": errors,
        "skipped": skipped,
        "rules": {
            "dry_run_default": True,
            "final_acceptance_evidence_is_never_deleted": True,
            "delete_requires_apply": True,
            "recognized_temporary_artifacts": [
                "temporary_auxiliary_review manifest artifacts",
                "test_only_short_acceptance_sample manifest artifacts",
                "*_contact_sheet.jpg",
            ],
        },
    }


def _classify_file(path: Path, candidates: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    if path.name == "manifest.json":
        _classify_manifest(path, candidates, skipped)
    elif path.name.endswith("_contact_sheet.jpg"):
        _append_candidate(
            candidates,
            path=path,
            artifact_type="contact_sheet_jpg",
            reason="contact_sheet_jpg_is_temporary_visual_review_artifact",
        )
    else:
        skipped.append({"path": str(path), "reason": "unrecognized_file"})


def _classify_manifest(manifest_path: Path, candidates: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        skipped.append({"path": str(manifest_path), "reason": "manifest_unreadable", "error": str(exc)})
        return
    policy = manifest.get("artifact_policy") if isinstance(manifest, dict) else None
    if not isinstance(policy, dict):
        skipped.append({"path": str(manifest_path), "reason": "manifest_missing_artifact_policy"})
        return
    if policy.get("valid_for_final_acceptance") is True:
        skipped.append({"path": str(manifest_path), "reason": "final_acceptance_evidence"})
        return
    if policy.get("cleanup_required") is not True:
        skipped.append({"path": str(manifest_path), "reason": "cleanup_not_required"})
        return
    retention_class = str(policy.get("retention_class") or "")
    if retention_class not in {"temporary_auxiliary_review", "test_only_short_acceptance_sample"}:
        skipped.append({"path": str(manifest_path), "reason": f"unrecognized_retention_class:{retention_class}"})
        return

    for raw_path in _manifest_temp_paths(manifest, manifest_path):
        path = Path(raw_path)
        if path.exists():
            _append_candidate(
                candidates,
                path=path,
                artifact_type=retention_class,
                reason="manifest_artifact_policy_cleanup_required",
                manifest_path=manifest_path,
            )
    _append_candidate(
        candidates,
        path=manifest_path,
        artifact_type=retention_class,
        reason="manifest_artifact_policy_cleanup_required",
        manifest_path=manifest_path,
    )


def _manifest_temp_paths(manifest: dict[str, Any], manifest_path: Path) -> list[str]:
    policy = manifest.get("artifact_policy") if isinstance(manifest, dict) else {}
    raw_paths = policy.get("temporary_artifacts") if isinstance(policy, dict) else None
    if isinstance(raw_paths, list) and raw_paths:
        return [str(path) for path in raw_paths if path]
    out: list[str] = []
    for key in ("review_images_dir", "frames_dir"):
        value = manifest.get(key)
        if value:
            out.append(str(value))
    output_dir = manifest.get("output_dir")
    if output_dir:
        out.append(str(Path(str(output_dir)) / "packs"))
    return out or [str(manifest_path.parent)]


def _append_candidate(
    candidates: list[dict[str, Any]],
    *,
    path: Path,
    artifact_type: str,
    reason: str,
    manifest_path: Path | None = None,
) -> None:
    resolved = str(path)
    if any(item["path"] == resolved for item in candidates):
        return
    candidates.append(
        {
            "path": resolved,
            "artifact_type": artifact_type,
            "reason": reason,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "is_dir": path.is_dir(),
            "exists": path.exists(),
        }
    )


def _remove_empty_or_known_temp_dir(path: Path) -> None:
    for item in sorted(path.rglob("*"), key=lambda entry: len(entry.parts), reverse=True):
        if item.is_dir():
            item.rmdir()
        else:
            item.unlink()
    path.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run or delete temporary visual review artifacts.")
    parser.add_argument("--root", action="append", required=True, help="Directory or file to inspect. Can be repeated.")
    parser.add_argument("--apply", action="store_true", help="Delete recognized temporary artifacts. Default is dry-run.")
    args = parser.parse_args(argv)
    report = cleanup_visual_artifacts(roots=[Path(root) for root in args.root], apply=bool(args.apply))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
