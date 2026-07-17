from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from defense.diagnostics.authoritative_manifest import (
    STRICT_COUNTS,
    load_authoritative_manifest,
    validate_authoritative_manifest,
)
from tools.validate_authoritative_manifest import main as validate_cli_main


def _write_asset(root: Path, relative_path: str, content: bytes) -> dict:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "relative_path": relative_path,
        "canonical_path": str(path.resolve()),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _make_strict_manifest(
    tmp_path: Path,
    *,
    shared_content: bool = False,
) -> tuple[Path, dict]:
    material_root = tmp_path / "materials"
    material_root.mkdir()

    def content_for(index: int) -> bytes:
        if shared_content:
            return b"same bytes are allowed at different identities"
        return f"authoritative-record-{index}".encode()

    model_file = _write_asset(
        material_root,
        "model/yolov8/mask_bd_v4_clean_baseline.pt",
        content_for(0),
    )
    unique_model = {
        "asset_id": "model-mask-bd-v4",
        **model_file,
        "role": "unique_model",
        "label": "mask_bd_v4_clean_baseline",
        "purpose": "production detector source",
    }

    videos: list[dict] = []
    a3b_file = _write_asset(
        material_root,
        "a3b/a3b_target.mp4",
        content_for(1),
    )
    videos.append(
        {
            "asset_id": "video-a3b",
            **a3b_file,
            "role": "a3b_target",
            "label": "a3b_attack",
            "purpose": "authoritative A3b Web trigger",
            "attack_type": "a3b_static_media",
            "expected_module_a_alert": True,
            "expected_a3b_trigger": True,
            "expected_module_a_evidence_events": ">=1",
            "acceptance_order": 1,
        }
    )

    physical_types = (
        "adv_patch",
        "glare",
        "motion_blur",
        "occlusion",
        "visibility_degradation",
    )
    for offset, attack_type in enumerate(physical_types, start=2):
        asset_file = _write_asset(
            material_root,
            f"physical/{attack_type}.mp4",
            content_for(offset),
        )
        videos.append(
            {
                "asset_id": f"video-physical-{attack_type}",
                **asset_file,
                "role": "physical_attack",
                "label": f"attack:{attack_type}",
                "purpose": f"authoritative {attack_type} Web acceptance",
                "attack_type": attack_type,
                "expected_module_a_alert": True,
                "expected_a3b_trigger": False,
                "expected_module_a_evidence_events": ">=1",
                "acceptance_order": offset,
            }
        )

    for normal_index in range(30):
        order = 7 + normal_index
        asset_file = _write_asset(
            material_root,
            f"normal/normal_{normal_index:02d}.mp4",
            content_for(order),
        )
        videos.append(
            {
                "asset_id": f"video-normal-{normal_index:02d}",
                **asset_file,
                "role": "normal_video",
                "label": f"normal:{normal_index:02d}",
                "purpose": "authoritative normal Web false-positive gate",
                "attack_type": None,
                "expected_module_a_alert": False,
                "expected_a3b_trigger": False,
                "expected_module_a_evidence_events": 0,
                "acceptance_order": order,
            }
        )

    payload = {
        "schema_version": 1,
        "snapshot_date": "2026-07-15",
        "material_root": str(material_root.resolve()),
        "unique_model": unique_model,
        "videos": videos,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path, payload


def test_strict_manifest_allows_same_hash_for_different_path_and_label(
    tmp_path: Path,
) -> None:
    manifest_path, _payload = _make_strict_manifest(
        tmp_path,
        shared_content=True,
    )

    result = validate_authoritative_manifest(manifest_path)

    assert result.valid is True
    assert result.counts == STRICT_COUNTS
    assert result.strict_gate["passed"] is True
    assert len(result.duplicate_hash_groups) == 1
    duplicate_group = result.duplicate_hash_groups[0]
    assert duplicate_group["allowed"] is True
    assert duplicate_group["record_count"] == 37
    assert len(duplicate_group["identities"]) == 37
    manifest = load_authoritative_manifest(manifest_path)
    assert len({asset.identity_key for asset in manifest.records}) == 37


def test_strict_count_gate_rejects_missing_normal_video(tmp_path: Path) -> None:
    manifest_path, payload = _make_strict_manifest(tmp_path)
    payload["videos"].pop()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_authoritative_manifest(manifest_path)

    assert result.valid is False
    assert result.counts["normal"] == 29
    assert result.counts["videos"] == 35
    assert result.counts["records"] == 36
    assert result.strict_gate["passed"] is False
    codes = {error["code"] for error in result.errors}
    assert "strict_count_mismatch" in codes
    assert "acceptance_order_gate" in codes


def test_manifest_reports_missing_file_and_hash_mismatch(tmp_path: Path) -> None:
    manifest_path, payload = _make_strict_manifest(tmp_path)
    missing = Path(payload["videos"][1]["canonical_path"])
    missing.unlink()
    payload["videos"][2]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_authoritative_manifest(manifest_path)

    assert result.valid is False
    codes = {error["code"] for error in result.errors}
    assert "asset_missing" in codes
    assert "hash_mismatch" in codes


def test_manifest_rejects_duplicate_asset_id_and_canonical_path_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _make_strict_manifest(tmp_path)
    payload["videos"][1]["asset_id"] = payload["videos"][0]["asset_id"]
    payload["videos"][2]["canonical_path"] = payload["videos"][3][
        "canonical_path"
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_authoritative_manifest(manifest_path)

    assert result.valid is False
    codes = {error["code"] for error in result.errors}
    assert "duplicate_asset_id" in codes
    assert "canonical_path_mismatch" in codes


def test_manifest_cli_emits_json_and_returns_nonzero_on_failure(
    tmp_path: Path,
    capsys,
) -> None:
    manifest_path, payload = _make_strict_manifest(tmp_path)
    invalid = copy.deepcopy(payload)
    invalid["unique_model"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(invalid), encoding="utf-8")

    exit_code = validate_cli_main(["--manifest", str(manifest_path), "--compact"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["ok"] is False
    assert any(error["code"] == "hash_mismatch" for error in output["errors"])
