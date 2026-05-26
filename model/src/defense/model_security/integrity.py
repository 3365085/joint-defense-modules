from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .registry import utc_now_iso


TRUST_STORE_SCHEMA_VERSION = 1
SEAL_SCHEMA_VERSION = 1
_SIGNING_SALT = "module-b-model-security-trust-store-v1"


@dataclass(frozen=True)
class HostFingerprint:
    host_fingerprint_hash: str
    status: str
    sources: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrustStoreIntegrity:
    ok: bool
    status: str
    reason: str
    registry_path: str
    registry_seal_path: str
    registry_hash: str | None
    host_fingerprint_status: str
    host_fingerprint_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def registry_data_hash(data: dict[str, Any]) -> str:
    return sha256_text(canonical_json(data))


def _machine_guid() -> str | None:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
        text = str(value).strip()
        return text or None
    except Exception:
        return None


def _system_drive_serial() -> str | None:
    drive = os.environ.get("SystemDrive", "C:").strip() or "C:"
    if os.name != "nt":
        return None
    try:
        completed = subprocess.run(
            ["cmd", "/c", "vol", drive],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    text = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r"Serial Number is\s+([0-9A-Fa-f-]+)", text)
    return match.group(1).strip() if match else None


def current_host_fingerprint() -> HostFingerprint:
    components: dict[str, str] = {}
    warnings: list[str] = []

    machine_guid = _machine_guid()
    if machine_guid:
        components["windows_machine_guid"] = machine_guid
    else:
        warnings.append("windows_machine_guid_unavailable")

    volume_serial = _system_drive_serial()
    if volume_serial:
        components["system_drive_serial"] = volume_serial
    else:
        warnings.append("system_drive_serial_unavailable")

    node = platform.node() or os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME")
    if node:
        components["machine_name"] = str(node)
    else:
        warnings.append("machine_name_unavailable")

    if not components:
        components["fallback"] = "unknown-host"
        warnings.append("host_fingerprint_fallback_used")

    status = "ok" if len(components) >= 2 else "degraded"
    payload = {"schema_version": 1, "components": components}
    return HostFingerprint(
        host_fingerprint_hash=sha256_text(canonical_json(payload)),
        status=status,
        sources=sorted(components.keys()),
        warnings=warnings,
    )


def _signature_payload(registry_hash: str, host_hash: str, updated_at: str) -> dict[str, Any]:
    return {
        "schema_version": SEAL_SCHEMA_VERSION,
        "registry_hash": registry_hash,
        "host_fingerprint_hash": host_hash,
        "updated_at": updated_at,
    }


def _signature(payload: dict[str, Any], host_hash: str) -> str:
    key = hashlib.sha256(f"{host_hash}|{_SIGNING_SALT}".encode("utf-8")).digest()
    digest = hmac.new(key, canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()
    return "hmac-sha256:" + digest


def build_trust_store_seal(data: dict[str, Any]) -> dict[str, Any]:
    host = current_host_fingerprint()
    updated_at = utc_now_iso()
    registry_hash = registry_data_hash(data)
    payload = _signature_payload(registry_hash, host.host_fingerprint_hash, updated_at)
    return {
        **payload,
        "signature": _signature(payload, host.host_fingerprint_hash),
        "host_fingerprint_status": host.status,
        "host_fingerprint_sources": host.sources,
        "host_fingerprint_warnings": host.warnings,
    }


def write_trust_store_seal(registry_path: str | Path, seal_path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    del registry_path
    seal = build_trust_store_seal(data)
    p = Path(seal_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(seal, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return seal


def _result(
    *,
    ok: bool,
    status: str,
    reason: str,
    registry_path: Path,
    seal_path: Path,
    registry_hash: str | None,
    host: HostFingerprint,
) -> TrustStoreIntegrity:
    return TrustStoreIntegrity(
        ok=ok,
        status=status,
        reason=reason,
        registry_path=str(registry_path),
        registry_seal_path=str(seal_path),
        registry_hash=registry_hash,
        host_fingerprint_status=host.status,
        host_fingerprint_hash=host.host_fingerprint_hash,
    )


def verify_trust_store(registry_path: str | Path, seal_path: str | Path) -> TrustStoreIntegrity:
    registry = Path(registry_path)
    seal = Path(seal_path)
    host = current_host_fingerprint()

    if not registry.exists():
        if seal.exists():
            return _result(
                ok=False,
                status="trust_store_compromised",
                reason="registry_missing_but_seal_exists",
                registry_path=registry,
                seal_path=seal,
                registry_hash=None,
                host=host,
            )
        return _result(
            ok=True,
            status="empty_unsealed",
            reason="registry_not_created",
            registry_path=registry,
            seal_path=seal,
            registry_hash=registry_data_hash({"version": TRUST_STORE_SCHEMA_VERSION, "models": {}}),
            host=host,
        )

    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except Exception:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="registry_json_invalid",
            registry_path=registry,
            seal_path=seal,
            registry_hash=None,
            host=host,
        )

    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="registry_schema_invalid",
            registry_path=registry,
            seal_path=seal,
            registry_hash=None,
            host=host,
        )
    if int(data.get("version", TRUST_STORE_SCHEMA_VERSION)) != TRUST_STORE_SCHEMA_VERSION:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="registry_schema_version_unsupported",
            registry_path=registry,
            seal_path=seal,
            registry_hash=None,
            host=host,
        )

    reg_hash = registry_data_hash(data)
    if not seal.exists():
        if data.get("models"):
            return _result(
                ok=False,
                status="trust_store_compromised",
                reason="seal_missing_for_nonempty_registry",
                registry_path=registry,
                seal_path=seal,
                registry_hash=reg_hash,
                host=host,
            )
        return _result(
            ok=True,
            status="empty_unsealed",
            reason="empty_registry_without_seal",
            registry_path=registry,
            seal_path=seal,
            registry_hash=reg_hash,
            host=host,
        )

    try:
        seal_data = json.loads(seal.read_text(encoding="utf-8"))
    except Exception:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="seal_json_invalid",
            registry_path=registry,
            seal_path=seal,
            registry_hash=reg_hash,
            host=host,
        )

    if not isinstance(seal_data, dict) or int(seal_data.get("schema_version", 0)) != SEAL_SCHEMA_VERSION:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="seal_schema_invalid",
            registry_path=registry,
            seal_path=seal,
            registry_hash=reg_hash,
            host=host,
        )
    if seal_data.get("registry_hash") != reg_hash:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="registry_hash_mismatch",
            registry_path=registry,
            seal_path=seal,
            registry_hash=reg_hash,
            host=host,
        )
    if seal_data.get("host_fingerprint_hash") != host.host_fingerprint_hash:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="host_fingerprint_mismatch",
            registry_path=registry,
            seal_path=seal,
            registry_hash=reg_hash,
            host=host,
        )
    payload = _signature_payload(
        str(seal_data.get("registry_hash")),
        str(seal_data.get("host_fingerprint_hash")),
        str(seal_data.get("updated_at")),
    )
    expected = _signature(payload, host.host_fingerprint_hash)
    if not hmac.compare_digest(str(seal_data.get("signature", "")), expected):
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="seal_signature_mismatch",
            registry_path=registry,
            seal_path=seal,
            registry_hash=reg_hash,
            host=host,
        )

    status = "ok" if host.status == "ok" else "host_fingerprint_degraded"
    reason = "verified" if host.status == "ok" else "verified_with_degraded_host_fingerprint"
    return _result(
        ok=True,
        status=status,
        reason=reason,
        registry_path=registry,
        seal_path=seal,
        registry_hash=reg_hash,
        host=host,
    )
