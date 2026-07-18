from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from defense.runtime_paths import runtime_data_root


DEFAULT_RUNTIME_ROOT = Path(__file__).resolve().parents[3] / "runtime"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_runtime_root() -> Path:
    return runtime_data_root()


def default_catalog_path(root: str | Path | None = None) -> Path:
    base = Path(root) if root else default_runtime_root()
    return base / "db" / "runtime_catalog.sqlite3"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_domain TEXT NOT NULL,
            category TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            sha256 TEXT,
            fingerprint TEXT,
            source_path TEXT,
            source_hash TEXT,
            status TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_domain ON runtime_artifacts(business_domain, category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_fingerprint ON runtime_artifacts(fingerprint)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_updated ON runtime_artifacts(updated_at DESC)")


def register_artifact(
    *,
    path: str | Path,
    business_domain: str,
    category: str,
    artifact_type: str,
    catalog_root: str | Path | None = None,
    fingerprint: str | None = None,
    source_path: str | Path | None = None,
    source_hash: str | None = None,
    status: str | None = None,
    metadata: dict[str, Any] | None = None,
    compute_hash: bool = True,
) -> dict[str, Any]:
    target = Path(path)
    now = now_iso()
    digest = None
    if compute_hash and target.exists() and target.is_file():
        try:
            digest = sha256_file(target)
        except Exception:
            digest = None
    db_path = default_catalog_path(catalog_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(target),
        "business_domain": str(business_domain),
        "category": str(category),
        "artifact_type": str(artifact_type),
        "sha256": digest,
        "fingerprint": str(fingerprint or ""),
        "source_path": str(source_path or ""),
        "source_hash": str(source_hash or ""),
        "status": str(status or ""),
        "metadata": metadata or {},
        "updated_at": now,
    }
    with sqlite3.connect(str(db_path), timeout=5.0) as conn:
        _ensure(conn)
        existing = conn.execute("SELECT created_at FROM runtime_artifacts WHERE path = ?", (str(target),)).fetchone()
        created_at = str(existing[0]) if existing else now
        conn.execute(
            """
            INSERT INTO runtime_artifacts (
                business_domain, category, artifact_type, path, sha256, fingerprint,
                source_path, source_hash, status, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                business_domain=excluded.business_domain,
                category=excluded.category,
                artifact_type=excluded.artifact_type,
                sha256=excluded.sha256,
                fingerprint=excluded.fingerprint,
                source_path=excluded.source_path,
                source_hash=excluded.source_hash,
                status=excluded.status,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                payload["business_domain"],
                payload["category"],
                payload["artifact_type"],
                payload["path"],
                payload["sha256"],
                payload["fingerprint"],
                payload["source_path"],
                payload["source_hash"],
                payload["status"],
                json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                created_at,
                now,
            ),
        )
    payload["catalog_path"] = str(db_path)
    return payload


def list_artifacts(
    *,
    catalog_root: str | Path | None = None,
    business_domain: str | None = None,
    category: str | None = None,
    artifact_type: str | None = None,
    status: str | None = None,
    fingerprint: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    db_path = default_catalog_path(catalog_root)
    if not db_path.exists():
        return {"catalog_path": str(db_path), "count": 0, "artifacts": []}
    clauses: list[str] = []
    params: list[Any] = []
    if business_domain:
        clauses.append("business_domain = ?")
        params.append(str(business_domain))
    if category:
        clauses.append("category = ?")
        params.append(str(category))
    if artifact_type:
        clauses.append("artifact_type = ?")
        params.append(str(artifact_type))
    if status:
        clauses.append("status = ?")
        params.append(str(status))
    if fingerprint:
        clauses.append("fingerprint = ?")
        params.append(str(fingerprint))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    max_items = max(1, min(int(limit or 100), 500))
    with sqlite3.connect(str(db_path), timeout=5.0) as conn:
        _ensure(conn)
        rows = conn.execute(
            """
            SELECT business_domain, category, artifact_type, path, sha256, fingerprint,
                   source_path, source_hash, status, metadata_json, created_at, updated_at
            FROM runtime_artifacts
            """
            + where
            + " ORDER BY updated_at DESC LIMIT ?",
            (*params, max_items),
        ).fetchall()
    out = []
    for row in rows:
        metadata = {}
        try:
            metadata = json.loads(row[9] or "{}")
        except Exception:
            metadata = {}
        out.append(
            {
                "business_domain": row[0],
                "category": row[1],
                "artifact_type": row[2],
                "path": row[3],
                "sha256": row[4],
                "fingerprint": row[5],
                "source_path": row[6],
                "source_hash": row[7],
                "status": row[8],
                "metadata": metadata,
                "created_at": row[10],
                "updated_at": row[11],
            }
        )
    return {"catalog_path": str(db_path), "count": len(out), "artifacts": out}
