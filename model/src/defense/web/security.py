from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, WebSocket

from defense.runtime import resolve_source_path

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class SecurityPolicy:
    bind_host: str = "127.0.0.1"
    token: str | None = None

    @property
    def local_only(self) -> bool:
        return str(self.bind_host or "127.0.0.1").lower() in LOCAL_HOSTS

    @property
    def enabled(self) -> bool:
        return bool(self.token) or not self.local_only

    @classmethod
    def from_env(cls, bind_host: str = "127.0.0.1") -> "SecurityPolicy":
        token = os.environ.get("MODULE_A_WEB_TOKEN") or None
        return cls(bind_host=str(bind_host or "127.0.0.1"), token=token)

    def _matches(self, supplied: str | None) -> bool:
        if not self.enabled:
            return True
        return bool(self.token) and supplied == self.token

    def token_from_headers(self, headers: Any) -> str | None:
        direct = headers.get("x-module-a-token") if headers else None
        if direct:
            return str(direct)
        auth = headers.get("authorization") if headers else None
        if auth and str(auth).lower().startswith("bearer "):
            return str(auth)[7:].strip()
        return None

    def token_from_query(self, query_params: Any) -> str | None:
        token = query_params.get("token") if query_params else None
        return str(token) if token else None

    def require_http(self, request: Request) -> None:
        if self._matches(self.token_from_headers(request.headers)):
            return
        raise HTTPException(status_code=401, detail="module_a_token_required")

    async def require_ws(self, websocket: WebSocket) -> bool:
        if self._matches(self.token_from_query(websocket.query_params) or self.token_from_headers(websocket.headers)):
            return True
        await websocket.close(code=1008, reason="module_a_token_required")
        return False


def get_policy(app: Any) -> SecurityPolicy:
    policy = getattr(app.state, "security_policy", None)
    if isinstance(policy, SecurityPolicy):
        return policy
    policy = SecurityPolicy.from_env(getattr(app.state, "bind_host", "127.0.0.1"))
    app.state.security_policy = policy
    return policy


def require_http_access(request: Request) -> None:
    get_policy(request.app).require_http(request)


async def require_ws_access(websocket: WebSocket) -> bool:
    return await get_policy(websocket.app).require_ws(websocket)


def safe_current_media_path(status_payload: dict[str, Any]) -> Path | None:
    if str(status_payload.get("source_type") or "").lower() != "file":
        return None
    source = str(status_payload.get("source") or "")
    if not source:
        return None
    resolved = resolve_source_path(source).resolve()
    if not resolved.exists() or not resolved.is_file():
        return None
    # Current-source-only policy: the API never accepts an arbitrary requested file path.
    raw = Path(source).expanduser()
    if raw.is_absolute() and raw.resolve() != resolved:
        return None
    roots_raw = os.environ.get("MODULE_A_MEDIA_ROOTS", "")
    if roots_raw.strip():
        allowed = [Path(part).expanduser().resolve() for part in roots_raw.split(os.pathsep) if part.strip()]
        if not any(root == resolved or root in resolved.parents for root in allowed):
            return None
    return resolved
