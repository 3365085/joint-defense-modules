from __future__ import annotations

import hmac
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from defense.model_security import integrity
from defense.web.fastapi_app import create_app


def _host(label: str, *, status: str = "ok") -> integrity.HostFingerprint:
    return integrity.HostFingerprint(
        host_fingerprint_hash=integrity.sha256_text(label),
        status=status,
        sources=["machine_name", "system_drive_serial", "windows_machine_guid"],
        warnings=[] if status == "ok" else ["system_drive_serial_unavailable"],
    )


def _write_registry(path: Path) -> dict[str, object]:
    data: dict[str, object] = {
        "version": 1,
        "models": {
            "sha256:test-model": {
                "fingerprint": "sha256:test-model",
                "status": "trusted",
                "approved_for_runtime": True,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _write_v2_old_host_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, integrity.HostFingerprint, bytes]:
    registry = tmp_path / "runtime" / "model_security" / "trusted_registry.json"
    seal = tmp_path / "runtime" / "model_security" / "trusted_registry.seal.json"
    data = _write_registry(registry)
    old_host = _host("old-host")
    monkeypatch.setattr(integrity, "current_host_fingerprint", lambda: old_host)
    integrity.write_trust_store_seal(registry, seal, data)
    return registry, seal, old_host, seal.read_bytes()


def _write_legacy_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, integrity.HostFingerprint, bytes, str]:
    registry = tmp_path / "runtime" / "model_security" / "trusted_registry.json"
    seal = tmp_path / "runtime" / "model_security" / "trusted_registry.seal.json"
    data = _write_registry(registry)
    host = _host("legacy-current-host")
    monkeypatch.setattr(integrity, "current_host_fingerprint", lambda: host)
    registry_hash = integrity.registry_data_hash(data)
    updated_at = "2026-07-16T00:00:00+00:00"
    payload = integrity._legacy_signature_payload(
        registry_hash,
        host.host_fingerprint_hash,
        updated_at,
    )
    legacy = {
        "schema_version": 1,
        "registry_hash": registry_hash,
        "host_fingerprint_hash": host.host_fingerprint_hash,
        "updated_at": updated_at,
        "signature": integrity._legacy_signature(payload, host.host_fingerprint_hash),
        "host_fingerprint_status": "ok",
        "host_fingerprint_sources": host.sources,
        "host_fingerprint_warnings": [],
    }
    seal.parent.mkdir(parents=True, exist_ok=True)
    seal.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry, seal, host, seal.read_bytes(), registry_hash


def test_current_host_fingerprint_never_serializes_raw_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integrity, "_machine_guid", lambda: "machine-guid-secret")
    monkeypatch.setattr(integrity, "_system_drive_serial", lambda: "drive-serial-secret")
    monkeypatch.setattr(integrity.platform, "node", lambda: "machine-name-secret")

    first = integrity.current_host_fingerprint()
    second = integrity.current_host_fingerprint()

    assert first == second
    serialized = json.dumps(first.to_dict(), ensure_ascii=False)
    assert "machine-guid-secret" not in serialized
    assert "drive-serial-secret" not in serialized
    assert "machine-name-secret" not in serialized
    assert "component_hashes" not in serialized


def test_v2_seal_uses_independent_protected_key_and_rejects_public_reforge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, _old_host, _old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    signing_key = seal.with_name(f"{seal.stem}.signing-key.json")
    assert signing_key.exists()
    key_record = json.loads(signing_key.read_text(encoding="utf-8"))
    assert key_record["protection"] in {"windows_dpapi_current_user", "file_0600"}
    assert "protected_key_b64" in key_record

    new_host = _host("new-host")
    seal_data = json.loads(seal.read_text(encoding="utf-8"))
    seal_data["host_fingerprint_hash"] = new_host.host_fingerprint_hash
    attacker_key = b"A" * 32
    seal_data["signature"] = integrity._signature_v2(
        integrity._signature_payload_v2(seal_data),
        attacker_key,
    )
    seal.write_text(json.dumps(seal_data), encoding="utf-8")
    monkeypatch.setattr(integrity, "current_host_fingerprint", lambda: new_host)

    checked = integrity.verify_trust_store(registry, seal)

    assert checked.ok is False
    assert checked.reason == "seal_signature_mismatch"
    assert checked.seal_schema_version == 2
    assert checked.signing_key_status == "present"
    assert checked.signing_key_protection in {"windows_dpapi_current_user", "file_0600"}


def test_existing_v2_seal_missing_key_cannot_be_silently_rekeyed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, _old_host, old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    signing_key = seal.with_name(f"{seal.stem}.signing-key.json")
    signing_key.unlink()
    data = json.loads(registry.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="signing_key_missing"):
        integrity.write_trust_store_seal(registry, seal, data)

    assert seal.read_bytes() == old_seal_bytes
    assert not signing_key.exists()


def test_existing_v2_seal_rejects_replacement_key_continuity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, _old_host, old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    signing_key = seal.with_name(f"{seal.stem}.signing-key.json")
    replacement_key = b"R" * 32
    integrity._atomic_write_json(
        signing_key,
        integrity._encode_signing_key(replacement_key),
    )
    data = json.loads(registry.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="seal_signing_key_id_mismatch"):
        integrity.write_trust_store_seal(registry, seal, data)

    assert seal.read_bytes() == old_seal_bytes
    assert integrity.verify_trust_store(registry, seal).ok is False


def test_verify_checks_signed_metadata_before_host_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, _old_host, _old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    seal_data = json.loads(seal.read_text(encoding="utf-8"))
    seal_data["host_fingerprint_sources"] = ["raw-hostname:SECRET-HOST"]
    seal.write_text(json.dumps(seal_data), encoding="utf-8")
    monkeypatch.setattr(integrity, "current_host_fingerprint", lambda: _host("new-host"))

    checked = integrity.verify_trust_store(registry, seal)

    assert checked.ok is False
    assert checked.reason == "seal_host_fingerprint_sources_invalid"


def test_v2_seal_rejects_unknown_unsigned_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, _old_host, _old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    seal_data = json.loads(seal.read_text(encoding="utf-8"))
    seal_data["host_fingerprint_component_hashes"] = {
        "machine_name": "RAW-HOST-IDENTIFIER"
    }
    seal.write_text(json.dumps(seal_data), encoding="utf-8")

    checked = integrity.verify_trust_store(registry, seal)

    assert checked.ok is False
    assert checked.reason == "seal_fields_invalid"


def test_v2_rebind_preserves_registry_and_commits_audited_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, old_host, old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    registry_bytes = registry.read_bytes()
    new_host = _host("new-host")
    monkeypatch.setattr(integrity, "current_host_fingerprint", lambda: new_host)

    result = integrity.rebind_trust_store(
        registry,
        seal,
        expected_previous_host_hash=old_host.host_fingerprint_hash,
        operator_reason="authorized production host migration after identity verification",
    )

    assert result.ok is True
    assert result.status == "rebound"
    assert registry.read_bytes() == registry_bytes
    assert Path(result.backup_path).read_bytes() == old_seal_bytes
    assert not seal.with_name(f"{seal.stem}.transition-journal.json").exists()
    checked = integrity.verify_trust_store(registry, seal)
    assert checked.ok is True

    audit_rows = [
        json.loads(line)
        for line in Path(result.audit_log_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    committed = [row for row in audit_rows if row.get("operation_id") == result.operation_id]
    assert committed[-1]["state"] == "committed"
    assert committed[-1]["signing_key_id"] == result.signing_key_id
    signing_key_path = seal.with_name(f"{seal.stem}.signing-key.json")
    signing_key, _key_id = integrity._load_signing_key(signing_key_path, create=False)
    assert hmac.compare_digest(
        committed[-1]["audit_signature"],
        integrity._audit_signature(committed[-1], signing_key),
    )
    serialized = json.dumps(committed[-1], ensure_ascii=False)
    assert "MachineGuid" not in serialized
    assert "component_hashes" not in serialized


def test_rebind_rejects_path_alias_before_modifying_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, old_host, old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    monkeypatch.setattr(integrity, "current_host_fingerprint", lambda: _host("new-host"))

    with pytest.raises(ValueError, match="path_alias:seal:audit"):
        integrity.rebind_trust_store(
            registry,
            seal,
            expected_previous_host_hash=old_host.host_fingerprint_hash,
            operator_reason="authorized production host migration after identity verification",
            audit_log_path=seal,
        )

    assert seal.read_bytes() == old_seal_bytes


def test_rebind_detects_seal_change_between_validation_and_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, old_host, old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    mutated = old_seal_bytes + b"\n"

    def mutate_then_return() -> integrity.HostFingerprint:
        seal.write_bytes(mutated)
        return _host("new-host")

    monkeypatch.setattr(integrity, "current_host_fingerprint", mutate_then_return)

    with pytest.raises(RuntimeError, match="seal_changed_after_validation"):
        integrity.rebind_trust_store(
            registry,
            seal,
            expected_previous_host_hash=old_host.host_fingerprint_hash,
            operator_reason="authorized production host migration after identity verification",
        )

    assert seal.read_bytes() == mutated
    assert not list(seal.parent.glob("*.transition-backup.*"))


def test_pending_transition_journal_blocks_normal_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, _old_host, _old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    journal = seal.with_name(f"{seal.stem}.transition-journal.json")
    journal.write_text("{}", encoding="utf-8")

    checked = integrity.verify_trust_store(registry, seal)

    assert checked.ok is False
    assert checked.reason == "trust_store_transition_pending"
    assert checked.transition_pending is True


def test_current_v2_state_can_be_recorded_as_signed_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, old_host, _old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    data = json.loads(registry.read_text(encoding="utf-8"))
    expected_registry_hash = integrity.registry_data_hash(data)

    result = integrity.attest_current_trust_store(
        registry,
        seal,
        expected_registry_hash=expected_registry_hash,
        operator_reason="authorized signed attestation of the current verified production trust store",
    )

    assert result.status == "attested"
    assert result.host_fingerprint_hash == old_host.host_fingerprint_hash
    audit = json.loads(
        Path(result.audit_log_path).read_text(encoding="utf-8").splitlines()[-1]
    )
    key, _key_id = integrity._load_signing_key(
        seal.with_name(f"{seal.stem}.signing-key.json"),
        create=False,
    )
    assert audit["event"] == "trust_store_current_state_attestation"
    assert audit["state"] == "attested"
    assert hmac.compare_digest(
        audit["audit_signature"],
        integrity._audit_signature(audit, key),
    )


def test_attestation_rejects_seal_change_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, old_host, _old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    data = json.loads(registry.read_text(encoding="utf-8"))
    expected_registry_hash = integrity.registry_data_hash(data)

    def mutate_then_return() -> integrity.HostFingerprint:
        seal_data = json.loads(seal.read_text(encoding="utf-8"))
        seal_data["signature"] = "hmac-sha256:" + ("0" * 64)
        seal.write_text(json.dumps(seal_data), encoding="utf-8")
        return old_host

    monkeypatch.setattr(integrity, "current_host_fingerprint", mutate_then_return)

    with pytest.raises(RuntimeError, match="seal_changed_after_validation"):
        integrity.attest_current_trust_store(
            registry,
            seal,
            expected_registry_hash=expected_registry_hash,
            operator_reason="authorized signed attestation of the current verified production trust store",
        )

    assert not (seal.parent / "trust_store_rebind_audit.jsonl").exists()


def test_scan_cannot_repair_a_compromised_store_as_a_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from defense.model_security.service import ModelSecurityService

    registry, _seal, _old_host, _old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["models"]["sha256:test-model"]["approved_for_runtime"] = False
    registry.write_text(json.dumps(data), encoding="utf-8")
    svc = ModelSecurityService(root=tmp_path)

    with pytest.raises(ValueError, match="trust_store_compromised:registry_hash_mismatch"):
        svc.scan(scan_type="quick")


@pytest.mark.parametrize("background", [False, True])
def test_scan_http_reports_compromised_store_as_conflict(background: bool) -> None:
    class CompromisedModelSecurity:
        def status(self, **_kwargs):
            return {
                "allowed": False,
                "admission_status": "trust_store_compromised",
                "trust_store_ok": False,
                "trust_store_reason": "seal_signature_mismatch",
            }

        def scan(self, **_kwargs):
            raise ValueError("trust_store_compromised:seal_signature_mismatch")

        def start_background_scan(self, **_kwargs):
            raise ValueError("trust_store_compromised:seal_signature_mismatch")

    app = create_app(bind_host="127.0.0.1")
    app.state.model_security = CompromisedModelSecurity()
    response = TestClient(app).post(
        "/api/model-security/scan",
        json={"scan_type": "quick", "background": background, "profile": "default"},
    )

    assert response.status_code == 409
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "trust_store_compromised"
    assert payload["reason"] == "seal_signature_mismatch"


def test_legacy_seal_requires_explicit_v2_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, host, old_seal_bytes, registry_hash = _write_legacy_seal(tmp_path, monkeypatch)
    before = integrity.verify_trust_store(registry, seal)
    assert before.ok is False
    assert before.reason == "legacy_seal_requires_v2_migration"

    result = integrity.migrate_legacy_trust_store_seal(
        registry,
        seal,
        expected_host_hash=host.host_fingerprint_hash,
        expected_registry_hash=registry_hash,
        operator_reason="authorized one-time migration from legacy public seal to protected version two",
    )

    assert result.status == "migrated"
    assert Path(result.backup_path).read_bytes() == old_seal_bytes
    migrated = json.loads(seal.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 2
    assert migrated["signature_version"] == 2
    assert "host_fingerprint_component_hashes" not in migrated
    after = integrity.verify_trust_store(registry, seal)
    assert after.ok is True


def test_legacy_migration_requires_exact_registry_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, host, old_seal_bytes, _registry_hash = _write_legacy_seal(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="expected_registry_hash_mismatch"):
        integrity.migrate_legacy_trust_store_seal(
            registry,
            seal,
            expected_host_hash=host.host_fingerprint_hash,
            expected_registry_hash=integrity.sha256_text("wrong-registry"),
            operator_reason="authorized one-time migration from legacy public seal to protected version two",
        )

    assert seal.read_bytes() == old_seal_bytes
    assert not seal.with_name(f"{seal.stem}.signing-key.json").exists()


def test_legacy_migration_rejects_preexisting_untrusted_signing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, host, old_seal_bytes, registry_hash = _write_legacy_seal(tmp_path, monkeypatch)
    signing_key = seal.with_name(f"{seal.stem}.signing-key.json")
    integrity._atomic_write_json(
        signing_key,
        integrity._encode_signing_key(b"P" * 32),
    )

    with pytest.raises(ValueError, match="legacy_migration_signing_key_already_exists"):
        integrity.migrate_legacy_trust_store_seal(
            registry,
            seal,
            expected_host_hash=host.host_fingerprint_hash,
            expected_registry_hash=registry_hash,
            operator_reason="authorized one-time migration from legacy public seal to protected version two",
        )

    assert seal.read_bytes() == old_seal_bytes


def test_failed_audit_leaves_signed_journal_for_explicit_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, old_host, old_seal_bytes = _write_v2_old_host_seal(tmp_path, monkeypatch)
    monkeypatch.setattr(integrity, "current_host_fingerprint", lambda: _host("new-host"))
    original_append = integrity._append_jsonl_durable
    calls = {"count": 0}

    def fail_first_two(path: Path, record: dict[str, object]) -> None:
        calls["count"] += 1
        if calls["count"] <= 2:
            raise OSError("injected_audit_failure")
        original_append(path, record)

    monkeypatch.setattr(integrity, "_append_jsonl_durable", fail_first_two)
    with pytest.raises(RuntimeError, match="transition_failed"):
        integrity.rebind_trust_store(
            registry,
            seal,
            expected_previous_host_hash=old_host.host_fingerprint_hash,
            operator_reason="authorized production host migration after identity verification",
        )

    assert seal.read_bytes() == old_seal_bytes
    journal = seal.with_name(f"{seal.stem}.transition-journal.json")
    assert journal.exists()
    assert integrity.verify_trust_store(registry, seal).reason == "trust_store_transition_pending"
    operation_id = json.loads(journal.read_text(encoding="utf-8"))["operation_id"]

    monkeypatch.setattr(integrity, "_append_jsonl_durable", original_append)
    recovered = integrity.recover_pending_trust_store_transition(
        registry,
        seal,
        expected_operation_id=operation_id,
        operator_reason="authorized recovery of interrupted trust store transition",
    )
    assert recovered.status == "rolled_back"
    assert seal.read_bytes() == old_seal_bytes
    assert not journal.exists()


def test_cli_json_out_alias_cannot_overwrite_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, seal, host, old_seal_bytes, registry_hash = _write_legacy_seal(tmp_path, monkeypatch)
    tool_path = Path(__file__).parents[1] / "tools" / "rebind_model_security_trust_store.py"
    spec = importlib.util.spec_from_file_location("trust_store_rebind_tool", tool_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--project-root",
            str(tmp_path),
            "--registry",
            str(registry),
            "--seal",
            str(seal),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--migrate-legacy-v1",
            "--expected-previous-host-hash",
            host.host_fingerprint_hash,
            "--expected-registry-hash",
            registry_hash,
            "--reason",
            "authorized one-time migration from legacy public seal to protected version two",
            "--json-out",
            str(seal),
        ]
    )

    assert exit_code == 3
    assert seal.read_bytes() == old_seal_bytes
