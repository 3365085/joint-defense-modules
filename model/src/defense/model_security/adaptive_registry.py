from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .fingerprint import sha256_file


SCHEMA_VERSION = "joint-defense-adaptive-purification/v1"
OFFICIAL_REGISTRY_SHA256 = "622e7d23b2a4890fd300956c74b913f66a16fa19d4969e93d198145f5277022f"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PARTS = {"incoming_deliveries", "b模块新算法", "rebuilt_demo"}
_OFFICIAL_EVIDENCE_SHA256 = {
    "oga_anchor_calibration_manifest.json": "40b34b304042b2e86419bad5e53ff252f105bd9d2650265ed7d9dee51a3e872e",
    "oda_corrective_delivery_manifest.json": "a5a86a756a2a32507198643c34fe4edcb293c1cc04930d9e1866cdd57cddc495",
    "multi_trigger_oda_manifest.json": "555ef4353338c6ee0d9f9b396053ce3810b94d9ebc3419b6ef880df2ad13b403",
    "semantic_oga_weights_manifest.json": "c7af137ec796644d2f86d5d85bda2200a9c95312e0008c9a7c1c15a92fed5c6d",
    "semantic_oda_result.json": "fd93aa2f089679bb8710fb07a5dd3dc547d3e0b152d5634cc2e51ce36a8c7487",
}


def _detox_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_security = config.get("model_security") if isinstance(config, Mapping) else None
    if not isinstance(model_security, Mapping):
        return {}
    detox = model_security.get("detox")
    return detox if isinstance(detox, Mapping) else {}


def _normalized_hash(value: Any) -> str:
    return str(value or "").strip().lower().removeprefix("sha256:")


def _required_hash(value: Any, *, field: str) -> str:
    normalized = _normalized_hash(value)
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise RuntimeError(f"自适应净化路由字段不是有效 SHA-256: {field}")
    return normalized


def _ensure_project_local(path: Path, root: Path, *, field: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"自适应净化路径必须位于主项目内: {field}: {resolved}") from exc
    forbidden = _FORBIDDEN_PARTS.intersection(part.casefold() for part in resolved.parts)
    if forbidden:
        raise RuntimeError(f"自适应净化路径命中禁止目录: {field}: {resolved}")
    return resolved


def _configured_path(value: Any, root: Path, *, field: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return _ensure_project_local(path, root, field=field)


def adaptive_registry_path(config: Mapping[str, Any], root: str | Path) -> Path | None:
    root_path = Path(root).resolve()
    detox = _detox_config(config)
    if not bool(detox.get("adaptive_routes_enabled", True)):
        return None
    configured = _configured_path(detox.get("adaptive_registry_path"), root_path, field="adaptive_registry_path")
    candidates = [configured] if configured is not None else [
        root_path / "configs" / "adaptive_purification_registry.json",
        root_path / "runtime" / "model_security" / "adaptive_assets" / "adaptive_purification_registry.json",
    ]
    for candidate in candidates:
        local = _ensure_project_local(candidate, root_path, field="adaptive_registry_path")
        if local.is_file():
            return local
    return None


def adaptive_workspace_root(config: Mapping[str, Any], root: str | Path) -> Path:
    root_path = Path(root).resolve()
    configured = _configured_path(
        _detox_config(config).get("adaptive_workspace_root"),
        root_path,
        field="adaptive_workspace_root",
    )
    return configured or root_path


def adaptive_assets_root(config: Mapping[str, Any], root: str | Path) -> Path:
    root_path = Path(root).resolve()
    configured = _configured_path(
        _detox_config(config).get("adaptive_assets_root"),
        root_path,
        field="adaptive_assets_root",
    )
    return configured or root_path / "runtime" / "model_security" / "adaptive_assets"


def _validated_entries(entries: Any, *, registry_path: Path) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        raise RuntimeError(f"自适应净化路由表缺少 entries: {registry_path}")
    validated: list[dict[str, Any]] = []
    model_ids: set[str] = set()
    source_hashes: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"自适应净化路由不是对象: entries[{index}]")
        record = dict(entry)
        model_id = str(record.get("model_id") or "").strip()
        if not model_id or model_id in model_ids:
            raise RuntimeError(f"自适应净化 model_id 缺失或重复: entries[{index}]")
        source_hash = _required_hash(record.get("source_sha256"), field=f"entries[{index}].source_sha256")
        candidate_hash = _required_hash(record.get("candidate_sha256"), field=f"entries[{index}].candidate_sha256")
        if source_hash in source_hashes:
            raise RuntimeError(f"源模型 SHA-256 匹配到多个自适应净化路由: {source_hash}")
        if not str(record.get("candidate_path") or "").strip():
            raise RuntimeError(f"自适应净化路由缺少 candidate_path: {model_id}")
        if not str(record.get("evidence_path") or "").strip():
            raise RuntimeError(f"自适应净化路由缺少 evidence_path: {model_id}")
        record["source_sha256"] = source_hash
        record["candidate_sha256"] = candidate_hash
        validated.append(record)
        model_ids.add(model_id)
        source_hashes.add(source_hash)
    return validated


def load_adaptive_registry(config: Mapping[str, Any], root: str | Path) -> dict[str, Any] | None:
    path = adaptive_registry_path(config, root)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"自适应净化路由表无法读取: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"自适应净化路由表版本无效: {path}")
    actual_registry_hash = sha256_file(path)
    expected_registry_hash = _required_hash(
        _detox_config(config).get("adaptive_registry_sha256", OFFICIAL_REGISTRY_SHA256),
        field="adaptive_registry_sha256",
    )
    if actual_registry_hash != expected_registry_hash:
        raise RuntimeError(
            f"自适应净化路由表哈希不匹配: {path}; expected={expected_registry_hash}; actual={actual_registry_hash}"
        )
    payload["entries"] = _validated_entries(payload.get("entries"), registry_path=path)
    payload["registry_path"] = str(path)
    payload["registry_hash"] = "sha256:" + actual_registry_hash
    return payload


def _select_evidence_record(payload: Any, model_id: str) -> Mapping[str, Any] | None:
    if isinstance(payload, list):
        matches = [item for item in payload if isinstance(item, Mapping) and item.get("model_id") == model_id]
        return matches[0] if len(matches) == 1 else None
    if isinstance(payload, Mapping) and payload.get("model_id") in {None, model_id}:
        return payload
    return None


def _evidence_candidate_hash(record: Mapping[str, Any]) -> str:
    for key in ("purified_sha256", "delivered_sha256", "candidate_sha256", "sha256"):
        value = _normalized_hash(record.get(key))
        if value:
            return value
    model = record.get("model")
    if isinstance(model, Mapping):
        return _normalized_hash(model.get("sha256"))
    return ""


def _evidence_acceptance(record: Mapping[str, Any]) -> tuple[bool, str]:
    acceptance = record.get("acceptance")
    if isinstance(acceptance, Mapping) and acceptance.get("accepted") is True:
        return True, "acceptance.accepted"
    if record.get("accepted") is True:
        return True, "accepted"
    if record.get("final_pass") is True and str(record.get("status") or "").lower() == "ok":
        return True, "final_pass"
    status = str(record.get("status") or "")
    if status == "user_accepted_calibration_video_candidate_not_strict_absolute_release":
        return True, "user_accepted_calibration_video_candidate"
    return False, "not_accepted"


def _evidence_risk_score(record: Mapping[str, Any]) -> float:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    calibration = record.get("calibration") if isinstance(record.get("calibration"), Mapping) else {}
    for container, keys in (
        (metrics, ("hard_box_success_rate", "witness_asr", "asr", "clean_asr")),
        (record, ("witness_asr", "metric")),
        (calibration, ("trigger_asr",)),
    ):
        for key in keys:
            value = container.get(key)
            if value is None:
                continue
            try:
                return max(0.0, min(1.0, abs(float(value))))
            except (TypeError, ValueError):
                continue
    return 0.0


def load_adaptive_evidence(
    evidence_path: str | Path,
    *,
    model_id: str,
    candidate_hash: str,
    expected_evidence_hash: str | None = None,
) -> dict[str, Any]:
    path = Path(evidence_path)
    actual_evidence_hash = sha256_file(path)
    expected_hash = _normalized_hash(expected_evidence_hash) or _OFFICIAL_EVIDENCE_SHA256.get(path.name, "")
    if not expected_hash or not _SHA256_PATTERN.fullmatch(expected_hash):
        raise RuntimeError(f"自适应净化证据缺少可信 SHA-256: {model_id}: {path}")
    if actual_evidence_hash != expected_hash:
        raise RuntimeError(
            f"自适应净化证据文件哈希不匹配: {model_id}; expected={expected_hash}; actual={actual_evidence_hash}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"自适应净化证据无法读取: {path}: {exc}") from exc
    record = _select_evidence_record(payload, model_id)
    if record is None:
        raise RuntimeError(f"自适应净化证据未唯一匹配模型: {model_id}: {path}")
    accepted, acceptance_source = _evidence_acceptance(record)
    if not accepted:
        raise RuntimeError(f"自适应净化证据未通过验收: {model_id}: {path}")
    evidence_candidate_hash = _evidence_candidate_hash(record)
    normalized_candidate_hash = _required_hash(candidate_hash, field=f"{model_id}.candidate_sha256")
    if evidence_candidate_hash != normalized_candidate_hash:
        raise RuntimeError(
            "自适应净化证据候选哈希不匹配: "
            f"{model_id}; evidence={evidence_candidate_hash}; candidate={normalized_candidate_hash}"
        )
    return {
        "accepted": True,
        "acceptance_source": acceptance_source,
        "status": record.get("status"),
        "evidence_model_id": record.get("model_id") or model_id,
        "evidence_candidate_hash": "sha256:" + evidence_candidate_hash,
        "evidence_file_hash": "sha256:" + actual_evidence_hash,
        "risk_score": _evidence_risk_score(record),
        "metrics": dict(record.get("metrics") or record.get("calibration") or {}),
    }


def _resolve_asset(
    value: Any,
    *,
    kind: str,
    config: Mapping[str, Any],
    root: Path,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"自适应净化路由缺少 {kind}_path")
    raw = Path(text)
    workspace = adaptive_workspace_root(config, root)
    assets = adaptive_assets_root(config, root)
    candidates = [raw] if raw.is_absolute() else [
        workspace / raw,
        assets / raw,
        assets / kind / raw.name,
    ]
    for candidate in candidates:
        local = _ensure_project_local(candidate, root, field=f"{kind}_path")
        if local.is_file():
            return local
    checked = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"自适应净化{kind}不存在于主项目资产目录: {checked}")


def adaptive_candidate_for_source(
    source_model_path: str | Path,
    *,
    config: Mapping[str, Any],
    root: str | Path,
) -> dict[str, Any] | None:
    source = Path(source_model_path)
    registry = load_adaptive_registry(config, root)
    if registry is None or not source.is_file():
        return None
    source_hash = sha256_file(source)
    matches = [entry for entry in registry["entries"] if entry["source_sha256"] == source_hash]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"源模型匹配到多个自适应净化路由: {source}")

    record = dict(matches[0])
    root_path = Path(root).resolve()
    candidate = _resolve_asset(record.get("candidate_path"), kind="weights", config=config, root=root_path)
    expected_candidate_hash = record["candidate_sha256"]
    actual_candidate_hash = sha256_file(candidate)
    if actual_candidate_hash != expected_candidate_hash:
        raise RuntimeError(
            f"自适应净化候选哈希不匹配: {candidate}; expected={expected_candidate_hash}; actual={actual_candidate_hash}"
        )

    evidence = _resolve_asset(record.get("evidence_path"), kind="evidence", config=config, root=root_path)
    evidence_summary = load_adaptive_evidence(
        evidence,
        model_id=str(record["model_id"]),
        candidate_hash=actual_candidate_hash,
        expected_evidence_hash=record.get("evidence_sha256"),
    )
    return {
        **record,
        "source_model_path": str(source),
        "source_model_hash": "sha256:" + source_hash,
        "candidate_path": str(candidate),
        "candidate_hash": "sha256:" + actual_candidate_hash,
        "evidence_path": str(evidence),
        "evidence_summary": evidence_summary,
        "registry_path": str(registry["registry_path"]),
        "registry_hash": str(registry["registry_hash"]),
        "workspace_root": str(adaptive_workspace_root(config, root_path)),
        "assets_root": str(adaptive_assets_root(config, root_path)),
    }
