from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .registry import utc_now_iso


TRUST_STORE_SCHEMA_VERSION = 1
LEGACY_SEAL_SCHEMA_VERSION = 1
SEAL_SCHEMA_VERSION = 2
SIGNING_KEY_SCHEMA_VERSION = 1
TRANSITION_JOURNAL_SCHEMA_VERSION = 1
_LEGACY_SIGNING_SALT = "module-b-model-security-trust-store-v1"
_DPAPI_ENTROPY = b"module-b-model-security-trust-store-signing-key-v2"
_ALLOWED_HOST_SOURCES = {
    "fallback",
    "machine_name",
    "system_drive_serial",
    "windows_machine_guid",
}
_SEAL_V2_FIELDS = {
    "schema_version",
    "signature_version",
    "key_id",
    "registry_hash",
    "host_fingerprint_hash",
    "host_fingerprint_status",
    "host_fingerprint_sources",
    "host_fingerprint_warnings",
    "updated_at",
    "signature",
}


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
    seal_schema_version: int | None
    signing_key_status: str
    signing_key_id: str | None
    signing_key_protection: str | None
    transition_pending: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrustStoreRebind:
    ok: bool
    status: str
    reason: str
    registry_path: str
    registry_seal_path: str
    registry_hash: str
    previous_host_fingerprint_hash: str
    current_host_fingerprint_hash: str
    current_host_fingerprint_status: str
    current_host_fingerprint_sources: list[str]
    signing_key_id: str
    backup_path: str
    audit_log_path: str
    operation_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrustStoreRecovery:
    ok: bool
    status: str
    reason: str
    operation_id: str
    registry_path: str
    registry_seal_path: str
    audit_log_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrustStoreAttestation:
    ok: bool
    status: str
    reason: str
    registry_path: str
    registry_seal_path: str
    registry_hash: str
    seal_sha256: str
    signing_key_id: str
    host_fingerprint_hash: str
    audit_log_path: str
    operation_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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


def _seal_signing_key_path(seal_path: Path) -> Path:
    return seal_path.with_name(f"{seal_path.stem}.signing-key.json")


def _transition_journal_path(seal_path: Path) -> Path:
    return seal_path.with_name(f"{seal_path.stem}.transition-journal.json")


def _transition_lock_path(seal_path: Path) -> Path:
    return seal_path.with_name(f"{seal_path.stem}.transition.lock")


def _default_audit_path(seal_path: Path) -> Path:
    return seal_path.parent / "trust_store_rebind_audit.jsonl"


def _path_alias(left: Path, right: Path) -> bool:
    left_resolved = left.expanduser().resolve()
    right_resolved = right.expanduser().resolve()
    if os.path.normcase(str(left_resolved)) == os.path.normcase(str(right_resolved)):
        return True
    if left_resolved.exists() and right_resolved.exists():
        try:
            return os.path.samefile(left_resolved, right_resolved)
        except OSError:
            return False
    return False


def ensure_distinct_paths(paths: dict[str, str | Path | None]) -> None:
    normalized = {
        name: Path(value).expanduser().resolve()
        for name, value in paths.items()
        if value is not None and str(value).strip()
    }
    names = list(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if _path_alias(normalized[left_name], normalized[right_name]):
                raise ValueError(f"path_alias:{left_name}:{right_name}")


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    )


def _append_jsonl_durable(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _transition_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        stream.seek(0)
        if stream.read(1) != b"L":
            stream.seek(0)
            stream.write(b"L")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("trust_store_transition_locked") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("trust_store_transition_locked") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob_from_bytes(payload: bytes) -> tuple[_DataBlob, ctypes.Array[Any]]:
    buffer = ctypes.create_string_buffer(payload)
    blob = _DataBlob(
        len(payload),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    return blob, buffer


def _dpapi_protect(payload: bytes) -> bytes:
    if os.name != "nt":
        raise RuntimeError("dpapi_unavailable")
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    input_blob, input_buffer = _blob_from_bytes(payload)
    entropy_blob, entropy_buffer = _blob_from_bytes(_DPAPI_ENTROPY)
    output_blob = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "Module B trust-store signing key",
        ctypes.byref(entropy_blob),
        None,
        None,
        0x01,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    del input_buffer, entropy_buffer
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)


def _dpapi_unprotect(payload: bytes) -> bytes:
    if os.name != "nt":
        raise RuntimeError("dpapi_unavailable")
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    input_blob, input_buffer = _blob_from_bytes(payload)
    entropy_blob, entropy_buffer = _blob_from_bytes(_DPAPI_ENTROPY)
    output_blob = _DataBlob()
    description = ctypes.c_void_p()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        ctypes.byref(description),
        ctypes.byref(entropy_blob),
        None,
        None,
        0x01,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    del input_buffer, entropy_buffer
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)
        if description.value:
            kernel32.LocalFree(description)


def _signing_key_id(key: bytes) -> str:
    return sha256_bytes(key)


def _encode_signing_key(key: bytes) -> dict[str, Any]:
    if os.name == "nt":
        protection = "windows_dpapi_current_user"
        protected = _dpapi_protect(key)
    else:
        protection = "file_0600"
        protected = key
    return {
        "schema_version": SIGNING_KEY_SCHEMA_VERSION,
        "protection": protection,
        "key_id": _signing_key_id(key),
        "protected_key_b64": base64.b64encode(protected).decode("ascii"),
        "created_at": utc_now_iso(),
    }


def _decode_signing_key(data: dict[str, Any]) -> tuple[bytes, str]:
    if int(data.get("schema_version", 0)) != SIGNING_KEY_SCHEMA_VERSION:
        raise ValueError("signing_key_schema_invalid")
    protection = str(data.get("protection", ""))
    try:
        protected = base64.b64decode(str(data.get("protected_key_b64", "")), validate=True)
    except Exception as exc:
        raise ValueError("signing_key_payload_invalid") from exc
    if protection == "windows_dpapi_current_user":
        try:
            key = _dpapi_unprotect(protected)
        except Exception as exc:
            raise ValueError("signing_key_unprotect_failed") from exc
    elif protection == "file_0600" and os.name != "nt":
        key = protected
    else:
        raise ValueError("signing_key_protection_invalid")
    if len(key) != 32:
        raise ValueError("signing_key_length_invalid")
    key_id = _signing_key_id(key)
    if not hmac.compare_digest(str(data.get("key_id", "")), key_id):
        raise ValueError("signing_key_id_mismatch")
    return key, key_id


def _decode_signing_key_file_bytes(raw: bytes) -> tuple[bytes, str]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("signing_key_json_invalid") from exc
    if not isinstance(data, dict):
        raise ValueError("signing_key_schema_invalid")
    return _decode_signing_key(data)


def _load_signing_key_snapshot(
    path: Path,
    *,
    create: bool,
) -> tuple[bytes, str, bytes]:
    if not path.exists():
        if not create:
            raise ValueError("signing_key_missing")
        key = secrets.token_bytes(32)
        _atomic_write_json(path, _encode_signing_key(key))
        if os.name != "nt":
            path.chmod(0o600)
    raw = path.read_bytes()
    key, key_id = _decode_signing_key_file_bytes(raw)
    return key, key_id, raw


def _load_signing_key(path: Path, *, create: bool) -> tuple[bytes, str]:
    key, key_id, _raw = _load_signing_key_snapshot(path, create=create)
    return key, key_id


def _validate_host_metadata(seal_data: dict[str, Any]) -> None:
    status = seal_data.get("host_fingerprint_status")
    if status not in {"ok", "degraded"}:
        raise ValueError("seal_host_fingerprint_status_invalid")
    sources = seal_data.get("host_fingerprint_sources")
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(item, str) or item not in _ALLOWED_HOST_SOURCES for item in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("seal_host_fingerprint_sources_invalid")
    warnings = seal_data.get("host_fingerprint_warnings")
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", item)
        for item in warnings
    ):
        raise ValueError("seal_host_fingerprint_warnings_invalid")


def _signature_payload_v2(seal_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SEAL_SCHEMA_VERSION,
        "signature_version": 2,
        "key_id": str(seal_data.get("key_id", "")),
        "registry_hash": str(seal_data.get("registry_hash", "")),
        "host_fingerprint_hash": str(seal_data.get("host_fingerprint_hash", "")),
        "host_fingerprint_status": str(seal_data.get("host_fingerprint_status", "")),
        "host_fingerprint_sources": list(seal_data.get("host_fingerprint_sources", [])),
        "host_fingerprint_warnings": list(seal_data.get("host_fingerprint_warnings", [])),
        "updated_at": str(seal_data.get("updated_at", "")),
    }


def _signature_v2(payload: dict[str, Any], key: bytes) -> str:
    digest = hmac.new(key, canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()
    return "hmac-sha256:" + digest


def _legacy_signature_payload(registry_hash: str, host_hash: str, updated_at: str) -> dict[str, Any]:
    return {
        "schema_version": LEGACY_SEAL_SCHEMA_VERSION,
        "registry_hash": registry_hash,
        "host_fingerprint_hash": host_hash,
        "updated_at": updated_at,
    }


def _legacy_signature(payload: dict[str, Any], host_hash: str) -> str:
    key = hashlib.sha256(f"{host_hash}|{_LEGACY_SIGNING_SALT}".encode("utf-8")).digest()
    digest = hmac.new(key, canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()
    return "hmac-sha256:" + digest


def _build_trust_store_seal_for_host(
    data: dict[str, Any],
    host: HostFingerprint,
    *,
    signing_key: bytes,
    key_id: str,
) -> dict[str, Any]:
    seal: dict[str, Any] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "signature_version": 2,
        "key_id": key_id,
        "registry_hash": registry_data_hash(data),
        "host_fingerprint_hash": host.host_fingerprint_hash,
        "host_fingerprint_status": host.status,
        "host_fingerprint_sources": host.sources,
        "host_fingerprint_warnings": host.warnings,
        "updated_at": utc_now_iso(),
    }
    seal["signature"] = _signature_v2(_signature_payload_v2(seal), signing_key)
    return seal


def build_trust_store_seal(
    data: dict[str, Any],
    *,
    signing_key_path: str | Path,
    create_signing_key: bool = False,
) -> dict[str, Any]:
    key, key_id = _load_signing_key(
        Path(signing_key_path),
        create=create_signing_key,
    )
    return _build_trust_store_seal_for_host(
        data,
        current_host_fingerprint(),
        signing_key=key,
        key_id=key_id,
    )


def write_trust_store_seal(
    registry_path: str | Path,
    seal_path: str | Path,
    data: dict[str, Any],
) -> dict[str, Any]:
    registry = Path(registry_path).expanduser().resolve()
    seal = Path(seal_path).expanduser().resolve()
    signing_key_path = _seal_signing_key_path(seal)
    journal = _transition_journal_path(seal)
    lock = _transition_lock_path(seal)
    ensure_distinct_paths(
        {
            "registry": registry,
            "seal": seal,
            "signing_key": signing_key_path,
            "journal": journal,
            "lock": lock,
        }
    )
    with _transition_lock(lock):
        if journal.exists():
            raise RuntimeError("trust_store_transition_pending")
        existing: dict[str, Any] | None = None
        existing_raw: bytes | None = None
        create_signing_key = not seal.exists() and not signing_key_path.exists()
        if seal.exists():
            existing, existing_raw = _load_seal(seal)
            if int(existing.get("schema_version", 0)) != SEAL_SCHEMA_VERSION:
                raise ValueError("legacy_seal_requires_v2_migration")
        key, key_id, key_raw = _load_signing_key_snapshot(
            signing_key_path,
            create=create_signing_key,
        )
        if existing is not None:
            _validate_secure_seal_signature(
                existing,
                signing_key=key,
                key_id=key_id,
            )
        built = _build_trust_store_seal_for_host(
            data,
            current_host_fingerprint(),
            signing_key=key,
            key_id=key_id,
        )
        if not hmac.compare_digest(signing_key_path.read_bytes(), key_raw):
            raise RuntimeError("signing_key_changed_after_validation")
        if existing_raw is not None and not hmac.compare_digest(
            seal.read_bytes(),
            existing_raw,
        ):
            raise RuntimeError("seal_changed_after_validation")
        _atomic_write_json(seal, built)
        return built


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
    seal_schema_version: int | None = None
    signing_key_id: str | None = None
    signing_key_status = "not_created"
    signing_key_protection: str | None = None
    if seal_path.exists():
        try:
            seal_data = json.loads(seal_path.read_text(encoding="utf-8"))
            if isinstance(seal_data, dict):
                seal_schema_version = int(seal_data.get("schema_version", 0))
                signing_key_id = str(seal_data.get("key_id") or "") or None
        except Exception:
            seal_schema_version = None
    signing_key_path = _seal_signing_key_path(seal_path)
    if signing_key_path.exists():
        signing_key_status = "present"
        try:
            key_data = json.loads(signing_key_path.read_text(encoding="utf-8"))
            if isinstance(key_data, dict):
                signing_key_protection = str(key_data.get("protection") or "") or None
        except Exception:
            signing_key_status = "invalid"
    return TrustStoreIntegrity(
        ok=ok,
        status=status,
        reason=reason,
        registry_path=str(registry_path),
        registry_seal_path=str(seal_path),
        registry_hash=registry_hash,
        host_fingerprint_status=host.status,
        host_fingerprint_hash=host.host_fingerprint_hash,
        seal_schema_version=seal_schema_version,
        signing_key_status=signing_key_status,
        signing_key_id=signing_key_id,
        signing_key_protection=signing_key_protection,
        transition_pending=_transition_journal_path(seal_path).exists(),
    )


def _load_registry(registry: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = registry.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("registry_json_invalid") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError("registry_schema_invalid")
    if int(data.get("version", TRUST_STORE_SCHEMA_VERSION)) != TRUST_STORE_SCHEMA_VERSION:
        raise ValueError("registry_schema_version_unsupported")
    return data, raw


def _load_seal(seal: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = seal.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("seal_json_invalid") from exc
    if not isinstance(data, dict):
        raise ValueError("seal_schema_invalid")
    return data, raw


def _validate_secure_seal_signature(
    seal_data: dict[str, Any],
    *,
    signing_key: bytes,
    key_id: str,
) -> None:
    if set(seal_data) != _SEAL_V2_FIELDS:
        raise ValueError("seal_fields_invalid")
    if int(seal_data.get("schema_version", 0)) != SEAL_SCHEMA_VERSION:
        raise ValueError("seal_schema_invalid")
    if int(seal_data.get("signature_version", 0)) != 2:
        raise ValueError("seal_signature_version_invalid")
    if not hmac.compare_digest(str(seal_data.get("key_id", "")), key_id):
        raise ValueError("seal_signing_key_id_mismatch")
    _validate_host_metadata(seal_data)
    expected = _signature_v2(_signature_payload_v2(seal_data), signing_key)
    if not hmac.compare_digest(str(seal_data.get("signature", "")), expected):
        raise ValueError("seal_signature_mismatch")


def _validate_secure_seal(
    data: dict[str, Any],
    seal_data: dict[str, Any],
    *,
    signing_key: bytes,
    key_id: str,
) -> str:
    _validate_secure_seal_signature(
        seal_data,
        signing_key=signing_key,
        key_id=key_id,
    )
    reg_hash = registry_data_hash(data)
    if seal_data.get("registry_hash") != reg_hash:
        raise ValueError("registry_hash_mismatch")
    return reg_hash


def _verify_trust_store(
    registry_path: str | Path,
    seal_path: str | Path,
    *,
    ignore_transition_journal: bool,
) -> TrustStoreIntegrity:
    registry = Path(registry_path)
    seal = Path(seal_path)
    host = current_host_fingerprint()
    journal = _transition_journal_path(seal)

    if journal.exists() and not ignore_transition_journal:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason="trust_store_transition_pending",
            registry_path=registry,
            seal_path=seal,
            registry_hash=None,
            host=host,
        )

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
        data, _registry_raw = _load_registry(registry)
    except ValueError as exc:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason=str(exc),
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
        seal_data, _seal_raw = _load_seal(seal)
    except ValueError as exc:
        return _result(
            ok=False,
            status="trust_store_compromised",
            reason=str(exc),
            registry_path=registry,
            seal_path=seal,
            registry_hash=reg_hash,
            host=host,
        )

    schema_version = int(seal_data.get("schema_version", 0))
    if seal_data.get("registry_hash") != reg_hash:
        reason = "registry_hash_mismatch"
    elif schema_version == LEGACY_SEAL_SCHEMA_VERSION:
        reason = "legacy_seal_requires_v2_migration"
    elif schema_version != SEAL_SCHEMA_VERSION:
        reason = "seal_schema_invalid"
    else:
        signing_key_path = _seal_signing_key_path(seal)
        try:
            key, key_id = _load_signing_key(signing_key_path, create=False)
            _validate_secure_seal(data, seal_data, signing_key=key, key_id=key_id)
        except ValueError as exc:
            reason = str(exc)
        else:
            sealed_host_hash = str(seal_data.get("host_fingerprint_hash", ""))
            if sealed_host_hash != host.host_fingerprint_hash:
                reason = "host_fingerprint_mismatch"
            else:
                status = "ok" if host.status == "ok" else "host_fingerprint_degraded"
                verified_reason = "verified" if host.status == "ok" else "verified_with_degraded_host_fingerprint"
                return _result(
                    ok=True,
                    status=status,
                    reason=verified_reason,
                    registry_path=registry,
                    seal_path=seal,
                    registry_hash=reg_hash,
                    host=host,
                )

    return _result(
        ok=False,
        status="trust_store_compromised",
        reason=reason,
        registry_path=registry,
        seal_path=seal,
        registry_hash=reg_hash,
        host=host,
    )


def verify_trust_store(registry_path: str | Path, seal_path: str | Path) -> TrustStoreIntegrity:
    return _verify_trust_store(
        registry_path,
        seal_path,
        ignore_transition_journal=False,
    )


def _validate_operator_reason(operator_reason: str) -> str:
    reason = str(operator_reason or "").strip()
    if len(reason) < 16 or len(reason) > 512:
        raise ValueError("operator_reason_length_invalid")
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", reason):
        raise ValueError("operator_reason_text_required")
    meaningful = {char.casefold() for char in reason if char.isalnum()}
    if len(meaningful) < 4:
        raise ValueError("operator_reason_not_meaningful")
    return reason


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise ValueError(f"{field_name}_invalid")
    return normalized


def _journal_signature(record: dict[str, Any], key: bytes) -> str:
    payload = {name: value for name, value in record.items() if name != "journal_signature"}
    return _signature_v2(payload, key)


def _audit_signature(record: dict[str, Any], key: bytes) -> str:
    payload = {name: value for name, value in record.items() if name != "audit_signature"}
    return _signature_v2(payload, key)


def _append_signed_audit(path: Path, record: dict[str, Any], key: bytes) -> None:
    signed = dict(record)
    signed["audit_signature"] = _audit_signature(signed, key)
    _append_jsonl_durable(path, signed)


def _write_transition_journal(path: Path, record: dict[str, Any], key: bytes) -> None:
    signed = dict(record)
    signed["journal_signature"] = _journal_signature(signed, key)
    _atomic_write_json(path, signed)


def _read_transition_journal(path: Path, key: bytes) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("transition_journal_json_invalid") from exc
    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != TRANSITION_JOURNAL_SCHEMA_VERSION:
        raise ValueError("transition_journal_schema_invalid")
    expected = _journal_signature(data, key)
    if not hmac.compare_digest(str(data.get("journal_signature", "")), expected):
        raise ValueError("transition_journal_signature_mismatch")
    return data


def _audit_has_committed_operation(path: Path, operation_id: str, key: bytes) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        expected = _audit_signature(record, key)
        if not hmac.compare_digest(str(record.get("audit_signature", "")), expected):
            continue
        if record.get("operation_id") == operation_id and record.get("state") == "committed":
            return True
    return False


def _commit_seal_transition(
    *,
    registry: Path,
    seal: Path,
    data: dict[str, Any],
    validated_old_seal_bytes: bytes,
    previous_host_hash: str,
    current_host: HostFingerprint,
    signing_key: bytes,
    key_id: str,
    validated_signing_key_bytes: bytes,
    operator_reason: str,
    audit: Path,
    event: str,
    previous_seal_schema_version: int,
) -> TrustStoreRebind:
    operation_id = uuid.uuid4().hex
    journal = _transition_journal_path(seal)
    if journal.exists():
        raise RuntimeError("trust_store_transition_pending")
    if not hmac.compare_digest(seal.read_bytes(), validated_old_seal_bytes):
        raise RuntimeError("seal_changed_after_validation")

    timestamp = utc_now_iso().replace(":", "").replace("+", "_")
    backup = seal.with_name(
        f"{seal.stem}.transition-backup.{timestamp}.{operation_id[:8]}{seal.suffix}"
    )
    signing_key_path = _seal_signing_key_path(seal)
    ensure_distinct_paths(
        {
            "registry": registry,
            "seal": seal,
            "signing_key": signing_key_path,
            "journal": journal,
            "lock": _transition_lock_path(seal),
            "audit": audit,
            "backup": backup,
        }
    )
    if not hmac.compare_digest(
        signing_key_path.read_bytes(),
        validated_signing_key_bytes,
    ):
        raise RuntimeError("signing_key_changed_after_validation")

    new_seal = _build_trust_store_seal_for_host(
        data,
        current_host,
        signing_key=signing_key,
        key_id=key_id,
    )
    new_seal_bytes = json.dumps(
        new_seal,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    old_seal_sha256 = sha256_bytes(validated_old_seal_bytes)
    new_seal_sha256 = sha256_bytes(new_seal_bytes)

    _atomic_write_bytes(backup, validated_old_seal_bytes)
    if not hmac.compare_digest(sha256_bytes(backup.read_bytes()), old_seal_sha256):
        raise RuntimeError("transition_backup_verification_failed")

    journal_record = {
        "schema_version": TRANSITION_JOURNAL_SCHEMA_VERSION,
        "operation_id": operation_id,
        "state": "prepared",
        "event": event,
        "prepared_at": utc_now_iso(),
        "registry_path": str(registry),
        "registry_seal_path": str(seal),
        "registry_hash": registry_data_hash(data),
        "previous_host_fingerprint_hash": previous_host_hash,
        "current_host_fingerprint_hash": current_host.host_fingerprint_hash,
        "previous_seal_schema_version": previous_seal_schema_version,
        "new_seal_schema_version": SEAL_SCHEMA_VERSION,
        "signing_key_id": key_id,
        "old_seal_sha256": old_seal_sha256,
        "new_seal_sha256": new_seal_sha256,
        "backup_path": str(backup),
        "audit_log_path": str(audit),
    }
    _write_transition_journal(journal, journal_record, signing_key)

    committed = False
    original_error: Exception | None = None
    try:
        if not hmac.compare_digest(
            signing_key_path.read_bytes(),
            validated_signing_key_bytes,
        ):
            raise RuntimeError("signing_key_changed_before_seal_write")
        _atomic_write_bytes(seal, new_seal_bytes)
        checked = _verify_trust_store(
            registry,
            seal,
            ignore_transition_journal=True,
        )
        if not checked.ok:
            raise RuntimeError(f"post_transition_verification_failed:{checked.reason}")
        if not hmac.compare_digest(
            signing_key_path.read_bytes(),
            validated_signing_key_bytes,
        ):
            raise RuntimeError("signing_key_changed_before_audit_commit")
        audit_record = {
            "schema_version": 2,
            "event": event,
            "state": "committed",
            "operation_id": operation_id,
            "occurred_at": utc_now_iso(),
            "operator_reason": operator_reason,
            "registry_path": str(registry),
            "registry_seal_path": str(seal),
            "registry_hash": registry_data_hash(data),
            "previous_host_fingerprint_hash": previous_host_hash,
            "current_host_fingerprint_hash": current_host.host_fingerprint_hash,
            "current_host_fingerprint_status": current_host.status,
            "current_host_fingerprint_sources": current_host.sources,
            "previous_seal_schema_version": previous_seal_schema_version,
            "new_seal_schema_version": SEAL_SCHEMA_VERSION,
            "signing_key_id": key_id,
            "old_seal_sha256": old_seal_sha256,
            "new_seal_sha256": new_seal_sha256,
            "backup_path": str(backup),
        }
        _append_signed_audit(audit, audit_record, signing_key)
        committed = True
        journal.unlink()
        _fsync_parent(journal)
    except Exception as exc:
        original_error = exc

    if original_error is not None:
        rollback_error: Exception | None = None
        rollback_audit_error: Exception | None = None
        try:
            _atomic_write_bytes(seal, validated_old_seal_bytes)
        except Exception as exc:
            rollback_error = exc
        try:
            _append_signed_audit(
                audit,
                {
                    "schema_version": 2,
                    "event": event,
                    "state": "rolled_back" if rollback_error is None else "rollback_failed",
                    "operation_id": operation_id,
                    "occurred_at": utc_now_iso(),
                    "operator_reason": operator_reason,
                    "registry_hash": registry_data_hash(data),
                    "original_error": str(original_error),
                    "rollback_error": str(rollback_error) if rollback_error else "",
                    "backup_path": str(backup),
                },
                signing_key,
            )
        except Exception as exc:
            rollback_audit_error = exc
        if rollback_error is None and rollback_audit_error is None:
            try:
                journal.unlink()
                _fsync_parent(journal)
            except Exception as exc:
                rollback_audit_error = exc
        detail = (
            f"transition_failed:{original_error};"
            f"rollback_error:{rollback_error or 'none'};"
            f"rollback_audit_error:{rollback_audit_error or 'none'};"
            f"committed_before_failure:{committed}"
        )
        raise RuntimeError(detail) from original_error

    return TrustStoreRebind(
        ok=True,
        status="migrated" if event == "trust_store_seal_v2_migration" else "rebound",
        reason=(
            "legacy_seal_migrated_to_protected_v2"
            if event == "trust_store_seal_v2_migration"
            else "verified_v2_seal_and_bound_to_current_host"
        ),
        registry_path=str(registry),
        registry_seal_path=str(seal),
        registry_hash=registry_data_hash(data),
        previous_host_fingerprint_hash=previous_host_hash,
        current_host_fingerprint_hash=current_host.host_fingerprint_hash,
        current_host_fingerprint_status=current_host.status,
        current_host_fingerprint_sources=current_host.sources,
        signing_key_id=key_id,
        backup_path=str(backup),
        audit_log_path=str(audit),
        operation_id=operation_id,
    )


def rebind_trust_store(
    registry_path: str | Path,
    seal_path: str | Path,
    *,
    expected_previous_host_hash: str,
    operator_reason: str,
    audit_log_path: str | Path | None = None,
) -> TrustStoreRebind:
    """Rebind a valid v2 seal using a protected key not derivable from the seal."""

    registry = Path(registry_path).expanduser().resolve()
    seal = Path(seal_path).expanduser().resolve()
    audit = (
        Path(audit_log_path).expanduser().resolve()
        if audit_log_path is not None
        else _default_audit_path(seal)
    )
    signing_key_path = _seal_signing_key_path(seal)
    journal = _transition_journal_path(seal)
    lock = _transition_lock_path(seal)
    ensure_distinct_paths(
        {
            "registry": registry,
            "seal": seal,
            "signing_key": signing_key_path,
            "journal": journal,
            "lock": lock,
            "audit": audit,
        }
    )
    reason = _validate_operator_reason(operator_reason)
    expected_previous = _validate_sha256(
        expected_previous_host_hash,
        field_name="expected_previous_host_hash",
    )

    with _transition_lock(lock):
        if journal.exists():
            raise RuntimeError("trust_store_transition_pending")
        data, _registry_raw = _load_registry(registry)
        seal_data, seal_raw = _load_seal(seal)
        key, key_id, key_raw = _load_signing_key_snapshot(
            signing_key_path,
            create=False,
        )
        reg_hash = _validate_secure_seal(
            data,
            seal_data,
            signing_key=key,
            key_id=key_id,
        )
        del reg_hash
        previous_host_hash = _validate_sha256(
            str(seal_data.get("host_fingerprint_hash", "")),
            field_name="sealed_host_fingerprint_hash",
        )
        if not hmac.compare_digest(previous_host_hash, expected_previous):
            raise ValueError("expected_previous_host_hash_mismatch")
        current_host = current_host_fingerprint()
        if current_host.status != "ok":
            raise ValueError("current_host_fingerprint_not_ok")
        if hmac.compare_digest(previous_host_hash, current_host.host_fingerprint_hash):
            raise ValueError("host_fingerprint_unchanged")
        return _commit_seal_transition(
            registry=registry,
            seal=seal,
            data=data,
            validated_old_seal_bytes=seal_raw,
            previous_host_hash=previous_host_hash,
            current_host=current_host,
            signing_key=key,
            key_id=key_id,
            validated_signing_key_bytes=key_raw,
            operator_reason=reason,
            audit=audit,
            event="trust_store_host_rebind",
            previous_seal_schema_version=SEAL_SCHEMA_VERSION,
        )


def migrate_legacy_trust_store_seal(
    registry_path: str | Path,
    seal_path: str | Path,
    *,
    expected_host_hash: str,
    expected_registry_hash: str,
    operator_reason: str,
    audit_log_path: str | Path | None = None,
) -> TrustStoreRebind:
    """One-time explicit migration from the public-keyless v1 seal to v2."""

    registry = Path(registry_path).expanduser().resolve()
    seal = Path(seal_path).expanduser().resolve()
    audit = (
        Path(audit_log_path).expanduser().resolve()
        if audit_log_path is not None
        else _default_audit_path(seal)
    )
    signing_key_path = _seal_signing_key_path(seal)
    journal = _transition_journal_path(seal)
    lock = _transition_lock_path(seal)
    ensure_distinct_paths(
        {
            "registry": registry,
            "seal": seal,
            "signing_key": signing_key_path,
            "journal": journal,
            "lock": lock,
            "audit": audit,
        }
    )
    reason = _validate_operator_reason(operator_reason)
    expected_host = _validate_sha256(expected_host_hash, field_name="expected_host_hash")
    expected_registry = _validate_sha256(
        expected_registry_hash,
        field_name="expected_registry_hash",
    )

    with _transition_lock(lock):
        if journal.exists():
            raise RuntimeError("trust_store_transition_pending")
        data, _registry_raw = _load_registry(registry)
        seal_data, seal_raw = _load_seal(seal)
        if int(seal_data.get("schema_version", 0)) != LEGACY_SEAL_SCHEMA_VERSION:
            raise ValueError("legacy_seal_required")
        reg_hash = registry_data_hash(data)
        if not hmac.compare_digest(reg_hash, expected_registry):
            raise ValueError("expected_registry_hash_mismatch")
        if seal_data.get("registry_hash") != reg_hash:
            raise ValueError("registry_hash_mismatch")
        sealed_host_hash = _validate_sha256(
            str(seal_data.get("host_fingerprint_hash", "")),
            field_name="sealed_host_fingerprint_hash",
        )
        if not hmac.compare_digest(sealed_host_hash, expected_host):
            raise ValueError("expected_host_hash_mismatch")
        legacy_payload = _legacy_signature_payload(
            reg_hash,
            sealed_host_hash,
            str(seal_data.get("updated_at", "")),
        )
        legacy_expected = _legacy_signature(legacy_payload, sealed_host_hash)
        if not hmac.compare_digest(str(seal_data.get("signature", "")), legacy_expected):
            raise ValueError("legacy_seal_signature_mismatch")
        current_host = current_host_fingerprint()
        if current_host.status != "ok":
            raise ValueError("current_host_fingerprint_not_ok")
        if not hmac.compare_digest(sealed_host_hash, current_host.host_fingerprint_hash):
            raise ValueError("legacy_seal_host_mismatch_requires_manual_recovery")
        if signing_key_path.exists():
            raise ValueError("legacy_migration_signing_key_already_exists")
        key, key_id, key_raw = _load_signing_key_snapshot(
            signing_key_path,
            create=True,
        )
        return _commit_seal_transition(
            registry=registry,
            seal=seal,
            data=data,
            validated_old_seal_bytes=seal_raw,
            previous_host_hash=sealed_host_hash,
            current_host=current_host,
            signing_key=key,
            key_id=key_id,
            validated_signing_key_bytes=key_raw,
            operator_reason=reason,
            audit=audit,
            event="trust_store_seal_v2_migration",
            previous_seal_schema_version=LEGACY_SEAL_SCHEMA_VERSION,
        )


def recover_pending_trust_store_transition(
    registry_path: str | Path,
    seal_path: str | Path,
    *,
    expected_operation_id: str,
    operator_reason: str,
) -> TrustStoreRecovery:
    """Resolve a durable transition journal after process interruption."""

    registry = Path(registry_path).expanduser().resolve()
    seal = Path(seal_path).expanduser().resolve()
    journal_path = _transition_journal_path(seal)
    signing_key_path = _seal_signing_key_path(seal)
    lock = _transition_lock_path(seal)
    reason = _validate_operator_reason(operator_reason)
    operation_id = str(expected_operation_id or "").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", operation_id):
        raise ValueError("expected_operation_id_invalid")

    ensure_distinct_paths(
        {
            "registry": registry,
            "seal": seal,
            "signing_key": signing_key_path,
            "journal": journal_path,
            "lock": lock,
        }
    )
    with _transition_lock(lock):
        if not journal_path.exists():
            raise ValueError("transition_journal_missing")
        key, key_id, key_bytes = _load_signing_key_snapshot(
            signing_key_path,
            create=False,
        )
        journal = _read_transition_journal(journal_path, key)
        if not hmac.compare_digest(str(journal.get("operation_id", "")), operation_id):
            raise ValueError("expected_operation_id_mismatch")
        if Path(str(journal.get("registry_path", ""))).expanduser().resolve() != registry:
            raise ValueError("transition_journal_registry_path_mismatch")
        if Path(str(journal.get("registry_seal_path", ""))).expanduser().resolve() != seal:
            raise ValueError("transition_journal_seal_path_mismatch")
        if not hmac.compare_digest(
            str(journal.get("signing_key_id", "")),
            key_id,
        ):
            raise ValueError("transition_journal_signing_key_id_mismatch")
        registry_data, registry_bytes = _load_registry(registry)
        if not hmac.compare_digest(
            registry_data_hash(registry_data),
            str(journal.get("registry_hash", "")),
        ):
            raise ValueError("transition_journal_registry_hash_mismatch")
        audit = Path(str(journal.get("audit_log_path", ""))).expanduser().resolve()
        backup = Path(str(journal.get("backup_path", ""))).expanduser().resolve()
        ensure_distinct_paths(
            {
                "registry": registry,
                "seal": seal,
                "signing_key": signing_key_path,
                "journal": journal_path,
                "lock": lock,
                "audit": audit,
                "backup": backup,
            }
        )

        seal_hash = sha256_bytes(seal.read_bytes()) if seal.exists() else ""
        committed = _audit_has_committed_operation(audit, operation_id, key)
        if committed and hmac.compare_digest(seal_hash, str(journal.get("new_seal_sha256", ""))):
            if not hmac.compare_digest(signing_key_path.read_bytes(), key_bytes):
                raise RuntimeError("signing_key_changed_after_validation")
            checked = _verify_trust_store(
                registry,
                seal,
                ignore_transition_journal=True,
            )
            if not checked.ok:
                raise RuntimeError(f"pending_committed_seal_invalid:{checked.reason}")
            journal_path.unlink()
            _fsync_parent(journal_path)
            return TrustStoreRecovery(
                ok=True,
                status="committed_finalized",
                reason="committed_transition_journal_removed",
                operation_id=operation_id,
                registry_path=str(registry),
                registry_seal_path=str(seal),
                audit_log_path=str(audit),
            )

        backup_bytes = backup.read_bytes()
        if not hmac.compare_digest(
            sha256_bytes(backup_bytes),
            str(journal.get("old_seal_sha256", "")),
        ):
            raise RuntimeError("pending_transition_backup_hash_mismatch")
        try:
            backup_data = json.loads(backup_bytes.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("pending_transition_backup_json_invalid") from exc
        previous_schema = int(journal.get("previous_seal_schema_version", 0))
        if previous_schema == SEAL_SCHEMA_VERSION:
            if not isinstance(backup_data, dict):
                raise RuntimeError("pending_transition_backup_schema_invalid")
            _validate_secure_seal(
                registry_data,
                backup_data,
                signing_key=key,
                key_id=key_id,
            )
        elif previous_schema == LEGACY_SEAL_SCHEMA_VERSION:
            if not isinstance(backup_data, dict):
                raise RuntimeError("pending_transition_backup_schema_invalid")
            sealed_host_hash = str(backup_data.get("host_fingerprint_hash", ""))
            legacy_payload = _legacy_signature_payload(
                str(backup_data.get("registry_hash", "")),
                sealed_host_hash,
                str(backup_data.get("updated_at", "")),
            )
            if backup_data.get("registry_hash") != registry_data_hash(registry_data):
                raise RuntimeError("pending_transition_backup_registry_hash_mismatch")
            if not hmac.compare_digest(
                str(backup_data.get("signature", "")),
                _legacy_signature(legacy_payload, sealed_host_hash),
            ):
                raise RuntimeError("pending_transition_backup_signature_mismatch")
        else:
            raise RuntimeError("pending_transition_previous_schema_invalid")
        if not hmac.compare_digest(registry.read_bytes(), registry_bytes):
            raise RuntimeError("registry_changed_after_validation")
        if not hmac.compare_digest(signing_key_path.read_bytes(), key_bytes):
            raise RuntimeError("signing_key_changed_after_validation")
        _atomic_write_bytes(seal, backup_bytes)
        if not hmac.compare_digest(seal.read_bytes(), backup_bytes):
            raise RuntimeError("pending_transition_restore_verification_failed")
        _append_signed_audit(
            audit,
            {
                "schema_version": 2,
                "event": str(journal.get("event", "trust_store_transition")),
                "state": "recovery_rolled_back",
                "operation_id": operation_id,
                "occurred_at": utc_now_iso(),
                "operator_reason": reason,
                "backup_path": str(backup),
            },
            key,
        )
        if not hmac.compare_digest(registry.read_bytes(), registry_bytes):
            raise RuntimeError("registry_changed_before_recovery_commit")
        if not hmac.compare_digest(signing_key_path.read_bytes(), key_bytes):
            raise RuntimeError("signing_key_changed_before_recovery_commit")
        journal_path.unlink()
        _fsync_parent(journal_path)
        return TrustStoreRecovery(
            ok=True,
            status="rolled_back",
            reason="pending_transition_restored_verified_backup",
            operation_id=operation_id,
            registry_path=str(registry),
            registry_seal_path=str(seal),
            audit_log_path=str(audit),
        )


def attest_current_trust_store(
    registry_path: str | Path,
    seal_path: str | Path,
    *,
    expected_registry_hash: str,
    operator_reason: str,
    audit_log_path: str | Path | None = None,
) -> TrustStoreAttestation:
    """Append a signed audit attestation for the current verified v2 state."""

    registry = Path(registry_path).expanduser().resolve()
    seal = Path(seal_path).expanduser().resolve()
    audit = (
        Path(audit_log_path).expanduser().resolve()
        if audit_log_path is not None
        else _default_audit_path(seal)
    )
    signing_key_path = _seal_signing_key_path(seal)
    journal = _transition_journal_path(seal)
    lock = _transition_lock_path(seal)
    ensure_distinct_paths(
        {
            "registry": registry,
            "seal": seal,
            "signing_key": signing_key_path,
            "journal": journal,
            "lock": lock,
            "audit": audit,
        }
    )
    reason = _validate_operator_reason(operator_reason)
    expected_registry = _validate_sha256(
        expected_registry_hash,
        field_name="expected_registry_hash",
    )
    operation_id = uuid.uuid4().hex

    with _transition_lock(lock):
        if journal.exists():
            raise RuntimeError("trust_store_transition_pending")
        data, registry_bytes = _load_registry(registry)
        seal_data, seal_bytes = _load_seal(seal)
        key, key_id, key_bytes = _load_signing_key_snapshot(
            signing_key_path,
            create=False,
        )
        registry_hash = _validate_secure_seal(
            data,
            seal_data,
            signing_key=key,
            key_id=key_id,
        )
        current_host = current_host_fingerprint()
        if not hmac.compare_digest(
            str(seal_data.get("host_fingerprint_hash", "")),
            current_host.host_fingerprint_hash,
        ):
            raise ValueError("trust_store_compromised:host_fingerprint_mismatch")
        if not hmac.compare_digest(registry_hash, expected_registry):
            raise ValueError("expected_registry_hash_mismatch")
        if not hmac.compare_digest(registry.read_bytes(), registry_bytes):
            raise RuntimeError("registry_changed_after_validation")
        if not hmac.compare_digest(seal.read_bytes(), seal_bytes):
            raise RuntimeError("seal_changed_after_validation")
        if not hmac.compare_digest(signing_key_path.read_bytes(), key_bytes):
            raise RuntimeError("signing_key_changed_after_validation")
        _append_signed_audit(
            audit,
            {
                "schema_version": 2,
                "event": "trust_store_current_state_attestation",
                "state": "attested",
                "operation_id": operation_id,
                "occurred_at": utc_now_iso(),
                "operator_reason": reason,
                "registry_path": str(registry),
                "registry_seal_path": str(seal),
                "registry_hash": registry_hash,
                "seal_sha256": sha256_bytes(seal_bytes),
                "seal_schema_version": int(seal_data.get("schema_version", 0)),
                "signing_key_id": key_id,
                "host_fingerprint_hash": current_host.host_fingerprint_hash,
            },
            key,
        )
    return TrustStoreAttestation(
        ok=True,
        status="attested",
        reason="current_v2_trust_store_state_signed_in_audit",
        registry_path=str(registry),
        registry_seal_path=str(seal),
        registry_hash=expected_registry,
        seal_sha256=sha256_bytes(seal_bytes),
        signing_key_id=key_id,
        host_fingerprint_hash=current_host.host_fingerprint_hash,
        audit_log_path=str(audit),
        operation_id=operation_id,
    )
