from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from defense.module_a import native_bridge


_SOURCE_ROOT_NAMES = (
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "build.rs",
)


@pytest.fixture
def installed_native():
    if not native_bridge.available:
        pytest.skip(
            "module_a_native unavailable: "
            f"fallback_reason={native_bridge.fallback_reason!r}; "
            f"load_error={native_bridge.load_error!r}"
        )
    return native_bridge.require_native()


def _copy_manifest_sources(destination: Path) -> Path:
    source_root = Path(native_bridge.source_root)
    for source in native_bridge._source_files():
        relative = source.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return destination


def _point_bridge_at_source_root(
    monkeypatch: pytest.MonkeyPatch,
    source_root: Path,
) -> None:
    monkeypatch.setattr(native_bridge, "_CRATE_ROOT", source_root)
    monkeypatch.setattr(
        native_bridge,
        "_SOURCE_ROOT_FILES",
        tuple(source_root / name for name in _SOURCE_ROOT_NAMES),
    )


def test_installed_binary_attests_current_source_manifest(installed_native) -> None:
    status = native_bridge.status()
    build_hash = status["build_info"][native_bridge.BUILD_SOURCE_SHA256_KEY]

    assert len(build_hash) == 64
    assert build_hash == status["source_sha256"]
    assert status["build_info"]["source_attestation_match"] is True
    assert dict(installed_native.build_info())[
        native_bridge.BUILD_SOURCE_SHA256_KEY
    ] == build_hash


def test_missing_build_time_source_hash_is_rejected() -> None:
    with pytest.raises(native_bridge._BridgeLoadFailure) as failure:
        native_bridge._build_source_sha256({})

    assert failure.value.reason == "source_attestation_unavailable"
    assert "rebuild" in failure.value.detail


def test_source_change_without_rebuild_rejects_installed_binary(
    installed_native,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_status = native_bridge.status()
    build_hash = original_status["build_info"][
        native_bridge.BUILD_SOURCE_SHA256_KEY
    ]
    copied_root = _copy_manifest_sources(tmp_path / "module_a_native")
    changed_source = copied_root / "src" / "lib.rs"
    changed_source.write_bytes(
        changed_source.read_bytes()
        + b"\n// source-attestation mismatch integration test\n"
    )
    _point_bridge_at_source_root(monkeypatch, copied_root)

    changed_manifest, changed_hash = native_bridge._calculate_source_manifest()
    module, status = native_bridge._load_verified_native()

    assert changed_manifest
    assert changed_hash != build_hash
    assert module is None
    assert status.available is False
    assert status.fallback_reason == "source_attestation_mismatch"
    assert status.source_sha256 == changed_hash
    assert status.binary_path == original_status["binary_path"]
    assert status.binary_sha256 == original_status["binary_sha256"]
    assert (
        status.build_info[native_bridge.BUILD_SOURCE_SHA256_KEY]
        == build_hash
    )
    assert status.build_info["source_sha256"] == changed_hash
    assert status.build_info["source_attestation_match"] is False
    assert build_hash in (status.load_error or "")
    assert changed_hash in (status.load_error or "")
