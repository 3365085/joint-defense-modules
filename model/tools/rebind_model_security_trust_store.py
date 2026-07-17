from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate, rebind, or recover the Model Security trust-store seal "
            "using an independently protected signing key and durable journal."
        )
    )
    parser.add_argument("--project-root", default=".", help="Main model project root.")
    parser.add_argument(
        "--registry",
        default="runtime/model_security/trusted_registry.json",
        help="Registry path, relative to --project-root unless absolute.",
    )
    parser.add_argument(
        "--seal",
        default="runtime/model_security/trusted_registry.seal.json",
        help="Seal path, relative to --project-root unless absolute.",
    )
    parser.add_argument(
        "--audit-log",
        default="runtime/model_security/trust_store_rebind_audit.jsonl",
        help="Append-only audit path, relative to --project-root unless absolute.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--migrate-legacy-v1",
        action="store_true",
        help="Explicitly migrate an intact current-host v1 seal to protected v2.",
    )
    mode.add_argument(
        "--recover-operation-id",
        default="",
        help="Resolve a pending durable transition journal for this operation id.",
    )
    mode.add_argument(
        "--attest-current",
        action="store_true",
        help="Append a signed audit attestation for the current verified v2 state.",
    )
    parser.add_argument(
        "--expected-previous-host-hash",
        default="",
        help="Exact sha256:... hash recorded in the existing seal.",
    )
    parser.add_argument(
        "--expected-registry-hash",
        default="",
        help="Required for --migrate-legacy-v1; exact canonical registry sha256:... hash.",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Auditable operator reason (16-512 chars with meaningful text).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional structured result path, relative to --project-root unless absolute.",
    )
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root / "src"))

    from defense.model_security.integrity import (
        attest_current_trust_store,
        ensure_distinct_paths,
        migrate_legacy_trust_store_seal,
        rebind_trust_store,
        recover_pending_trust_store_transition,
    )

    registry = _resolve(root, args.registry)
    seal = _resolve(root, args.seal)
    audit = _resolve(root, args.audit_log)
    json_out = _resolve(root, args.json_out) if str(args.json_out).strip() else None
    signing_key = seal.with_name(f"{seal.stem}.signing-key.json")
    journal = seal.with_name(f"{seal.stem}.transition-journal.json")
    lock = seal.with_name(f"{seal.stem}.transition.lock")

    try:
        ensure_distinct_paths(
            {
                "registry": registry,
                "seal": seal,
                "audit": audit,
                "signing_key": signing_key,
                "journal": journal,
                "lock": lock,
                "json_out": json_out,
            }
        )
        if args.recover_operation_id:
            result = recover_pending_trust_store_transition(
                registry,
                seal,
                expected_operation_id=args.recover_operation_id,
                operator_reason=args.reason,
            )
        elif args.attest_current:
            if not str(args.expected_registry_hash).strip():
                raise ValueError("expected_registry_hash_required")
            result = attest_current_trust_store(
                registry,
                seal,
                expected_registry_hash=args.expected_registry_hash,
                operator_reason=args.reason,
                audit_log_path=audit,
            )
        elif args.migrate_legacy_v1:
            if not str(args.expected_previous_host_hash).strip():
                raise ValueError("expected_previous_host_hash_required")
            if not str(args.expected_registry_hash).strip():
                raise ValueError("expected_registry_hash_required")
            result = migrate_legacy_trust_store_seal(
                registry,
                seal,
                expected_host_hash=args.expected_previous_host_hash,
                expected_registry_hash=args.expected_registry_hash,
                operator_reason=args.reason,
                audit_log_path=audit,
            )
        else:
            if not str(args.expected_previous_host_hash).strip():
                raise ValueError("expected_previous_host_hash_required")
            result = rebind_trust_store(
                registry,
                seal,
                expected_previous_host_hash=args.expected_previous_host_hash,
                operator_reason=args.reason,
                audit_log_path=audit,
            )
        payload = result.to_dict()
        exit_code = 0
    except Exception as exc:
        payload = {
            "ok": False,
            "status": "trust_store_operation_rejected",
            "reason": str(exc),
            "registry_path": str(registry),
            "registry_seal_path": str(seal),
            "audit_log_path": str(audit),
        }
        exit_code = 2

    if json_out is not None:
        try:
            ensure_distinct_paths(
                {
                    "registry": registry,
                    "seal": seal,
                    "audit": audit,
                    "signing_key": signing_key,
                    "journal": journal,
                    "lock": lock,
                    "json_out": json_out,
                }
            )
            _atomic_write_json(json_out, payload)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "status": "json_output_rejected",
                        "reason": str(exc),
                        "operation_result": payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 3
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
