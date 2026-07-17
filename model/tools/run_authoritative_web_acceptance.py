from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.authoritative_manifest import (  # noqa: E402
    validate_authoritative_manifest,
)
from defense.diagnostics.web_acceptance_report import (  # noqa: E402
    run_authoritative_web_acceptance,
    run_web_preflight,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "acceptance"
    / "module_a_authoritative_manifest_v1.json"
)


@dataclass(frozen=True)
class JsonResponse:
    status_code: int
    payload: Any

    def json(self) -> Any:
        return self.payload


class UrllibJsonClient:
    """Minimal HTTP JSON client matching the FastAPI TestClient surface."""

    def __init__(self, base_url: str, *, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = float(timeout_s)

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> JsonResponse:
        return self._request("GET", path, params=params)

    def post(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> JsonResponse:
        return self._request("POST", path, params=params, json_body=json)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> JsonResponse:
        url = urljoin(self.base_url, path.lstrip("/"))
        if params:
            url = f"{url}?{urlencode(dict(params))}"
        body = None
        headers = {"Accept": "application/json"}
        if json_body is not None:
            body = json.dumps(dict(json_body), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, method=method, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                return JsonResponse(int(response.status), payload)
        except HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeError, json.JSONDecodeError):
                payload = {"ok": False, "error": raw.decode("utf-8", errors="replace")}
            return JsonResponse(int(exc.code), payload)
        except URLError as exc:
            raise RuntimeError(f"HTTP request failed for {url}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _execute(args: argparse.Namespace, client: Any) -> tuple[dict[str, Any], int]:
    validation = validate_authoritative_manifest(
        args.manifest,
        verify_files=True,
        strict_counts=True,
    )
    validation_payload = validation.to_dict(include_records=False)
    if not validation.valid or validation.manifest is None:
        payload = {
            "ok": False,
            "mode": args.mode,
            "manifest_validation": validation_payload,
            "blockers": [
                {
                    "code": "manifest_invalid",
                    "message": (
                        "authoritative manifest failed strict schema/count/path/"
                        "size/hash validation"
                    ),
                }
            ],
        }
        return payload, 1
    manifest = validation.manifest

    if args.mode == "preflight":
        preflight = run_web_preflight(
            client,
            manifest,
            verify_status_artifact_files=(
                not args.skip_status_artifact_file_check
            ),
        )
        payload = {
            "ok": preflight["passed"],
            "mode": "preflight",
            "manifest_validation": validation_payload,
            "preflight": preflight,
            "results_generated": False,
            "message": (
                "Preflight only: no authoritative video acceptance result "
                "was generated."
            ),
        }
        return payload, 0 if preflight["passed"] else 1

    report = run_authoritative_web_acceptance(
        client,
        manifest,
        selected_asset_ids=args.asset_id or None,
        profile=args.profile,
        ready_timeout_s=args.ready_timeout_s,
        asset_timeout_s=args.asset_timeout_s,
        poll_interval_s=args.poll_interval_s,
        evidence_limit=args.evidence_limit,
        verify_status_artifact_files=(
            not args.skip_status_artifact_file_check
        ),
        require_preflight_pass=True,
    )
    report["manifest_validation"] = validation_payload
    return report, 0 if report.get("summary", {}).get("passed") is True else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the live production FastAPI/latest-only acceptance "
            "contract. Full mode requires source_ended=true for every selected "
            "asset; timeout is always failure."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "full"),
        default="preflight",
        help="Default is non-mutating preflight; full explicitly runs videos.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Use FastAPI TestClient/create_app instead of an HTTP server.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", default="default")
    parser.add_argument(
        "--asset-id",
        action="append",
        default=[],
        help=(
            "Full mode only. Select an asset for diagnostics; a partial "
            "selection is never final-acceptance eligible."
        ),
    )
    parser.add_argument("--ready-timeout-s", type=float, default=45.0)
    parser.add_argument("--asset-timeout-s", type=float, default=600.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.25)
    parser.add_argument("--http-timeout-s", type=float, default=60.0)
    parser.add_argument("--evidence-limit", type=int, default=5000)
    parser.add_argument(
        "--skip-status-artifact-file-check",
        action="store_true",
        help=(
            "Trust status hashes without reading source/engine files locally; "
            "intended only for a remote server."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if args.in_process:
            from fastapi.testclient import TestClient

            from defense.web.fastapi_app import create_app

            app = create_app(config_path=args.config, bind_host="127.0.0.1")
            with TestClient(app) as client:
                payload, exit_code = _execute(args, client)
        else:
            client = UrllibJsonClient(
                args.base_url,
                timeout_s=args.http_timeout_s,
            )
            payload, exit_code = _execute(args, client)
    except Exception as exc:
        payload = {
            "ok": False,
            "mode": args.mode,
            "error": "acceptance_tool_failed",
            "message": str(exc),
            "results_generated": False,
        }
        exit_code = 2

    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    print(rendered)
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
