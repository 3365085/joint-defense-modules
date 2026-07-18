from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from defense.model_security import adaptive_registry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name)


def _write_delivery_fixture(
    root: Path,
    count: int = 18,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    source_dir = root / "fixtures" / "source"
    candidate_dir = root / "runtime_assets" / "adaptive_candidates"
    evidence_dir = root / "runtime_assets" / "adaptive_evidence"
    for directory in (source_dir, candidate_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for index in range(count):
        source = source_dir / f"source_{index:02d}.pt"
        candidate = candidate_dir / f"candidate_{index:02d}.pt"
        evidence = evidence_dir / f"candidate_{index:02d}.json"
        source.write_bytes((f"source-{index}-" * 64).encode())
        candidate.write_bytes((f"candidate-{index}-" * 64).encode())
        candidate_sha = _sha256(candidate)
        evidence.write_text(
            json.dumps(
                {
                    "model_id": f"delivery-{index:02d}",
                    "accepted": True,
                    "source_sha256": _sha256(source),
                    "candidate_sha256": candidate_sha,
                    "route": "oga" if index % 2 == 0 else "oda",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "model_id": f"delivery-{index:02d}",
                "source_sha256": _sha256(source),
                "candidate_path": candidate.relative_to(root).as_posix(),
                "candidate_sha256": candidate_sha,
                "evidence_path": evidence.relative_to(root).as_posix(),
                "evidence_sha256": _sha256(evidence),
                "route": "oga" if index % 2 == 0 else "oda",
                "accepted": True,
            }
        )

    registry_path = root / "configs" / "adaptive_purification_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {"schema_version": adaptive_registry.SCHEMA_VERSION, "entries": entries},
            indent=2,
        ),
        encoding="utf-8",
    )
    config = {
        "model_security": {
            "detox": {
                "adaptive_routes_enabled": True,
                "adaptive_registry_path": str(registry_path),
                "adaptive_registry_sha256": _sha256(registry_path),
                "adaptive_workspace_root": str(root),
                "adaptive_assets_root": str(root / "runtime_assets"),
            }
        }
    }
    return config, registry_path, entries


def test_delivery_registry_resolves_all_18_sources_by_exact_sha256(tmp_path: Path) -> None:
    config, _registry_path, expected = _write_delivery_fixture(tmp_path)
    registry = adaptive_registry.load_adaptive_registry(config, tmp_path)

    assert registry is not None
    assert len(registry["entries"]) == 18
    for entry in expected:
        source = tmp_path / "fixtures" / "source" / f"source_{int(entry['model_id'][-2:]):02d}.pt"
        route = adaptive_registry.adaptive_candidate_for_source(
            source,
            config=config,
            root=tmp_path,
        )
        assert route is not None
        assert route["source_sha256"] == entry["source_sha256"]
        assert route["candidate_sha256"] == entry["candidate_sha256"]
        assert route["evidence_sha256"] == entry["evidence_sha256"]
        assert route["evidence_summary"]["accepted"] is True
        assert Path(route["candidate_path"]).is_relative_to(tmp_path)
        assert Path(route["evidence_path"]).is_relative_to(tmp_path)

    near_collision = tmp_path / "fixtures" / "source" / "near_collision.pt"
    near_collision.write_bytes(("source-0-" * 64).encode() + b"changed")
    assert adaptive_registry.adaptive_candidate_for_source(
        near_collision,
        config=config,
        root=tmp_path,
    ) is None


@pytest.mark.parametrize("tampered", ["candidate", "evidence"])
def test_delivery_route_rejects_candidate_or_evidence_hash_mismatch(
    tmp_path: Path,
    tampered: str,
) -> None:
    config, _registry_path, _expected = _write_delivery_fixture(tmp_path, count=1)
    source = tmp_path / "fixtures" / "source" / "source_00.pt"
    route = adaptive_registry.adaptive_candidate_for_source(
        source,
        config=config,
        root=tmp_path,
    )
    assert route is not None

    target = Path(_field(route, f"{tampered}_path"))
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="哈希不匹配"):
        adaptive_registry.adaptive_candidate_for_source(
            source,
            config=config,
            root=tmp_path,
        )


def test_adaptive_registry_never_reads_incoming_delivery_at_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _registry_path, _expected = _write_delivery_fixture(tmp_path, count=1)
    original_open = builtins.open
    original_path_open = Path.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any):
        if "incoming_deliveries" in str(file).lower():
            raise AssertionError(f"production attempted to read incoming delivery: {file}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        if "incoming_deliveries" in str(path).lower():
            raise AssertionError(f"production attempted to read incoming delivery: {path}")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    route = adaptive_registry.adaptive_candidate_for_source(
        tmp_path / "fixtures" / "source" / "source_00.pt",
        config=config,
        root=tmp_path,
    )

    assert route is not None
    assert route["evidence_summary"]["accepted"] is True
    assert "incoming_deliveries" not in route["registry_path"].lower()
    assert "incoming_deliveries" not in route["candidate_path"].lower()
    assert "incoming_deliveries" not in route["evidence_path"].lower()
