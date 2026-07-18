"""Verified loader and observability bridge for ``module_a_native``.

The rebuilt production detector calls these operators through ``_native_call``.
This module provides a strict, inspectable loading boundary for that source-owned crate:

* reject any import/distribution evidence that points at ``rebuilt_demo``;
* require the A3b batch metadata/capability contract while preserving API v1;
* locate and hash the actual extension binary rather than a package wrapper;
* calculate a deterministic manifest/hash for the main-project Rust sources;
* require the binary's build-time source hash to match the current manifest;
* expose explicit unavailable and fallback reasons without silently importing
  an incompatible historical wheel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping
from urllib.parse import unquote


EXPECTED_API_VERSION = "1"
EXPECTED_A3B_BATCH_API_VERSION = "1"
EXPECTED_CRATE_VERSION = "0.2.0"
BUILD_SOURCE_SHA256_KEY = "source_manifest_sha256"
REQUIRED_CAPABILITIES = (
    "a1_lbp_features",
    "a2_change_features",
    "best_grid_value_f32",
    "a3b_boxes_stats",
    "a3b_one_box_stats",
    "blinding_laplacian_var",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CRATE_ROOT = _PROJECT_ROOT / "native" / "module_a_native"
_SOURCE_ROOT_FILES = (
    _CRATE_ROOT / "Cargo.toml",
    _CRATE_ROOT / "Cargo.lock",
    _CRATE_ROOT / "pyproject.toml",
    _CRATE_ROOT / "build.rs",
)


@dataclass(frozen=True, slots=True)
class NativeBridgeStatus:
    available: bool
    load_error: str | None
    fallback_reason: str | None
    api_version: str | None
    a3b_batch_api_version: str | None
    crate_version: str | None
    capabilities: tuple[str, ...]
    binary_path: str | None
    binary_sha256: str | None
    source_root: str
    source_sha256: str | None
    source_manifest: tuple[dict[str, Any], ...]
    build_info: dict[str, Any]
    distribution: dict[str, Any]


class _BridgeLoadFailure(RuntimeError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _source_files() -> tuple[Path, ...]:
    rust_sources = tuple(sorted((_CRATE_ROOT / "src").rglob("*.rs")))
    return tuple(path for path in (*_SOURCE_ROOT_FILES, *rust_sources) if path.is_file())


def _calculate_source_manifest() -> tuple[tuple[dict[str, Any], ...], str]:
    missing = [str(path) for path in _SOURCE_ROOT_FILES if not path.is_file()]
    if missing:
        raise _BridgeLoadFailure(
            "source_manifest_unavailable",
            f"main-project native source manifest is incomplete; missing={missing}",
        )
    paths = _source_files()
    if not paths:
        raise _BridgeLoadFailure(
            "source_manifest_unavailable",
            f"no native sources found below {_CRATE_ROOT}",
        )

    aggregate = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(_CRATE_ROOT).as_posix()
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest().upper()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(content)
        aggregate.update(b"\0")
        entries.append(
            {
                "path": relative,
                "size_bytes": len(content),
                "sha256": file_hash,
            }
        )
    return tuple(entries), aggregate.hexdigest().upper()


def _contains_rebuilt_demo(value: str | Path | None) -> bool:
    if value is None:
        return False
    normalized = unquote(str(value)).replace("\\", "/").casefold()
    return "rebuilt_demo" in normalized.split("/") or "/rebuilt_demo/" in normalized


def _distribution_evidence() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("module_a_native")
    except importlib.metadata.PackageNotFoundError:
        return {}

    metadata_path = getattr(distribution, "_path", None)
    direct_url = distribution.read_text("direct_url.json")
    return {
        "name": distribution.metadata.get("Name", "module_a_native"),
        "version": distribution.version,
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "direct_url": direct_url.strip() if direct_url else None,
    }


def _is_extension_path(path: Path) -> bool:
    lower_name = path.name.casefold()
    return any(
        lower_name.endswith(suffix.casefold())
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def _locate_extension_binary(module: ModuleType) -> Path:
    candidates: list[Path] = []
    for capability in REQUIRED_CAPABILITIES:
        function = getattr(module, capability, None)
        implementation_name = getattr(function, "__module__", None)
        if implementation_name:
            implementation = sys.modules.get(str(implementation_name))
            implementation_file = getattr(implementation, "__file__", None)
            if implementation_file:
                candidates.append(Path(implementation_file))

    module_file = getattr(module, "__file__", None)
    if module_file:
        candidates.append(Path(module_file))

    module_paths = getattr(module, "__path__", ())
    for package_path in module_paths:
        directory = Path(package_path)
        if directory.is_dir():
            for child in directory.iterdir():
                if child.is_file() and _is_extension_path(child):
                    candidates.append(child)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and _is_extension_path(resolved):
            return resolved
    raise _BridgeLoadFailure(
        "binary_not_found",
        "module_a_native imported, but the actual extension .pyd/.so could not be located",
    )


def _call_text_metadata(module: ModuleType, name: str) -> str:
    function = getattr(module, name, None)
    if not callable(function):
        raise _BridgeLoadFailure(
            "contract_mismatch",
            f"module_a_native is missing required metadata function {name}()",
        )
    try:
        return str(function())
    except Exception as exc:
        raise _BridgeLoadFailure(
            "contract_mismatch",
            f"module_a_native.{name}() failed: {type(exc).__name__}: {exc}",
        ) from exc


def _call_capabilities(module: ModuleType) -> tuple[str, ...]:
    function = getattr(module, "capabilities", None)
    if not callable(function):
        raise _BridgeLoadFailure(
            "contract_mismatch",
            "module_a_native is missing required metadata function capabilities()",
        )
    try:
        values = tuple(str(value) for value in function())
    except Exception as exc:
        raise _BridgeLoadFailure(
            "contract_mismatch",
            f"module_a_native.capabilities() failed: {type(exc).__name__}: {exc}",
        ) from exc
    missing = sorted(set(REQUIRED_CAPABILITIES) - set(values))
    if missing:
        raise _BridgeLoadFailure(
            "contract_mismatch",
            f"module_a_native capabilities are incomplete; missing={missing}",
        )
    for capability in REQUIRED_CAPABILITIES:
        if not callable(getattr(module, capability, None)):
            raise _BridgeLoadFailure(
                "contract_mismatch",
                f"module_a_native capability {capability} is not callable",
            )
    return values


def _call_build_info(module: ModuleType) -> dict[str, Any]:
    function = getattr(module, "build_info", None)
    if not callable(function):
        raise _BridgeLoadFailure(
            "contract_mismatch",
            "module_a_native is missing required metadata function build_info()",
        )
    try:
        value = function()
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
        return {str(key): item for key, item in value}
    except Exception as exc:
        raise _BridgeLoadFailure(
            "contract_mismatch",
            f"module_a_native.build_info() failed: {type(exc).__name__}: {exc}",
        ) from exc


def _build_source_sha256(build_info: Mapping[str, Any]) -> str:
    raw_value = build_info.get(BUILD_SOURCE_SHA256_KEY)
    if not isinstance(raw_value, str):
        raise _BridgeLoadFailure(
            "source_attestation_unavailable",
            "module_a_native build_info is missing the build-time source attestation "
            f"field {BUILD_SOURCE_SHA256_KEY!r}; rebuild the main-project native crate",
        )
    value = raw_value.strip().upper()
    if len(value) != 64 or any(character not in "0123456789ABCDEF" for character in value):
        raise _BridgeLoadFailure(
            "source_attestation_unavailable",
            "module_a_native build-time source attestation is not a canonical SHA-256: "
            f"{raw_value!r}; rebuild the main-project native crate",
        )
    return value


def _load_verified_native() -> tuple[ModuleType | None, NativeBridgeStatus]:
    source_manifest: tuple[dict[str, Any], ...] = ()
    source_sha256: str | None = None
    distribution = _distribution_evidence()
    api: str | None = None
    a3b_batch_api: str | None = None
    crate: str | None = None
    native_capabilities: tuple[str, ...] = ()
    binary: Path | None = None
    binary_hash: str | None = None
    observed_build_info: dict[str, Any] = {}
    try:
        source_manifest, source_sha256 = _calculate_source_manifest()

        if _contains_rebuilt_demo(distribution.get("metadata_path")) or _contains_rebuilt_demo(
            distribution.get("direct_url")
        ):
            raise _BridgeLoadFailure(
                "unsafe_rebuilt_demo_origin",
                "module_a_native distribution metadata points at rebuilt_demo: "
                f"{distribution}",
            )

        spec = importlib.util.find_spec("module_a_native")
        if spec is None:
            raise _BridgeLoadFailure(
                "module_not_found",
                "module_a_native is not installed in the active Python environment",
            )
        if _contains_rebuilt_demo(spec.origin):
            raise _BridgeLoadFailure(
                "unsafe_rebuilt_demo_origin",
                f"module_a_native import origin points at rebuilt_demo: {spec.origin}",
            )

        try:
            module = importlib.import_module("module_a_native")
        except Exception as exc:
            raise _BridgeLoadFailure(
                "import_failed",
                f"module_a_native import failed: {type(exc).__name__}: {exc}",
            ) from exc

        module_file = getattr(module, "__file__", None)
        if _contains_rebuilt_demo(module_file):
            raise _BridgeLoadFailure(
                "unsafe_rebuilt_demo_origin",
                f"module_a_native module file points at rebuilt_demo: {module_file}",
            )

        api = _call_text_metadata(module, "api_version")
        a3b_batch_api = _call_text_metadata(module, "a3b_batch_api_version")
        crate = _call_text_metadata(module, "crate_version")
        if api != EXPECTED_API_VERSION:
            raise _BridgeLoadFailure(
                "contract_mismatch",
                f"module_a_native API version {api!r} != expected {EXPECTED_API_VERSION!r}",
            )
        if a3b_batch_api != EXPECTED_A3B_BATCH_API_VERSION:
            raise _BridgeLoadFailure(
                "contract_mismatch",
                "module_a_native A3b batch API version "
                f"{a3b_batch_api!r} != expected {EXPECTED_A3B_BATCH_API_VERSION!r}",
            )
        if crate != EXPECTED_CRATE_VERSION:
            raise _BridgeLoadFailure(
                "contract_mismatch",
                f"module_a_native crate version {crate!r} != expected {EXPECTED_CRATE_VERSION!r}",
            )
        distribution_version = distribution.get("version")
        if distribution_version and str(distribution_version) != crate:
            raise _BridgeLoadFailure(
                "contract_mismatch",
                "module_a_native distribution/crate versions disagree: "
                f"distribution={distribution_version!r}, crate={crate!r}",
            )

        native_build_info = _call_build_info(module)
        expected_build_info = {
            "crate_name": "module_a_native",
            "crate_version": EXPECTED_CRATE_VERSION,
            "api_version": EXPECTED_API_VERSION,
            "a3b_batch_api_version": EXPECTED_A3B_BATCH_API_VERSION,
        }
        mismatched_build_info = {
            key: {
                "expected": expected,
                "actual": native_build_info.get(key),
            }
            for key, expected in expected_build_info.items()
            if str(native_build_info.get(key)) != expected
        }
        if mismatched_build_info:
            raise _BridgeLoadFailure(
                "contract_mismatch",
                "module_a_native build_info does not match the required source-owned "
                f"contract: {mismatched_build_info}",
            )
        binary = _locate_extension_binary(module)
        if _contains_rebuilt_demo(binary):
            raise _BridgeLoadFailure(
                "unsafe_rebuilt_demo_origin",
                f"module_a_native extension binary points at rebuilt_demo: {binary}",
            )
        binary_hash = _sha256_file(binary)

        observed_build_info = {
            **native_build_info,
            "binary_path": str(binary),
            "binary_sha256": binary_hash,
            "source_root": str(_CRATE_ROOT),
            "source_sha256": source_sha256,
            "source_manifest": [dict(entry) for entry in source_manifest],
            "source_attestation_match": None,
        }
        build_source_sha256 = _build_source_sha256(native_build_info)
        source_attestation_match = build_source_sha256 == source_sha256
        observed_build_info[BUILD_SOURCE_SHA256_KEY] = build_source_sha256
        observed_build_info["source_attestation_match"] = source_attestation_match
        if not source_attestation_match:
            raise _BridgeLoadFailure(
                "source_attestation_mismatch",
                "module_a_native was built from different Rust sources; "
                f"build_source_sha256={build_source_sha256}, "
                f"current_source_sha256={source_sha256}; rebuild and reinstall the "
                "main-project native crate before enabling it",
            )

        native_capabilities = _call_capabilities(module)
        return module, NativeBridgeStatus(
            available=True,
            load_error=None,
            fallback_reason=None,
            api_version=api,
            a3b_batch_api_version=a3b_batch_api,
            crate_version=crate,
            capabilities=native_capabilities,
            binary_path=str(binary),
            binary_sha256=binary_hash,
            source_root=str(_CRATE_ROOT),
            source_sha256=source_sha256,
            source_manifest=source_manifest,
            build_info=observed_build_info,
            distribution=distribution,
        )
    except _BridgeLoadFailure as exc:
        return None, NativeBridgeStatus(
            available=False,
            load_error=exc.detail,
            fallback_reason=exc.reason,
            api_version=api,
            a3b_batch_api_version=a3b_batch_api,
            crate_version=crate,
            capabilities=native_capabilities,
            binary_path=str(binary) if binary is not None else None,
            binary_sha256=binary_hash,
            source_root=str(_CRATE_ROOT),
            source_sha256=source_sha256,
            source_manifest=source_manifest,
            build_info=observed_build_info,
            distribution=distribution,
        )
    except Exception as exc:  # Defensive boundary: bridge import must remain observable.
        return None, NativeBridgeStatus(
            available=False,
            load_error=f"unexpected bridge load failure: {type(exc).__name__}: {exc}",
            fallback_reason="bridge_internal_error",
            api_version=api,
            a3b_batch_api_version=a3b_batch_api,
            crate_version=crate,
            capabilities=native_capabilities,
            binary_path=str(binary) if binary is not None else None,
            binary_sha256=binary_hash,
            source_root=str(_CRATE_ROOT),
            source_sha256=source_sha256,
            source_manifest=source_manifest,
            build_info=observed_build_info,
            distribution=distribution,
        )


_native_module, _status = _load_verified_native()

available = _status.available
load_error = _status.load_error
fallback_reason = _status.fallback_reason
api_version = _status.api_version
a3b_batch_api_version = _status.a3b_batch_api_version
crate_version = _status.crate_version
capabilities = _status.capabilities
binary_path = _status.binary_path
binary_sha256 = _status.binary_sha256
source_root = _status.source_root
source_sha256 = _status.source_sha256
source_manifest = _status.source_manifest
build_info = _status.build_info
distribution = _status.distribution


def require_native() -> ModuleType:
    """Return the verified extension module or raise with the recorded reason."""

    if _native_module is None:
        raise RuntimeError(
            "module_a_native is unavailable: "
            f"fallback_reason={fallback_reason!r}; load_error={load_error!r}"
        )
    return _native_module


def status() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the verified bridge state."""

    value = asdict(_status)
    value["capabilities"] = list(_status.capabilities)
    value["source_manifest"] = [dict(entry) for entry in _status.source_manifest]
    return value


__all__ = [
    "EXPECTED_API_VERSION",
    "EXPECTED_A3B_BATCH_API_VERSION",
    "EXPECTED_CRATE_VERSION",
    "BUILD_SOURCE_SHA256_KEY",
    "REQUIRED_CAPABILITIES",
    "api_version",
    "a3b_batch_api_version",
    "available",
    "binary_path",
    "binary_sha256",
    "build_info",
    "capabilities",
    "crate_version",
    "distribution",
    "fallback_reason",
    "load_error",
    "require_native",
    "source_manifest",
    "source_root",
    "source_sha256",
    "status",
]
