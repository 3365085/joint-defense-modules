from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from defense.pipelines._pynv_file_decoder import (
    PyNvFileDecoder,
    _AsciiSourceAlias,
)


def _authoritative_a3b(pkg_root: Path) -> Path:
    manifest_path = (
        pkg_root
        / "configs"
        / "acceptance"
        / "module_a_authoritative_manifest_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(
        next(
            item["canonical_path"]
            for item in manifest["videos"]
            if item["asset_id"] == "a3b.authoritative_target"
        )
    )
    if not source.is_file():
        pytest.skip(f"authoritative A3b video is unavailable: {source}")
    return source


def _require_nvdec() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("PyNvVideoCodec")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def test_ascii_alias_for_chinese_source_is_identity_bound_and_cleanable(
    tmp_path: Path,
) -> None:
    chinese_dir = tmp_path / "中文目录"
    chinese_dir.mkdir()
    source = chinese_dir / "视频.bin"
    source.write_bytes(b"identity-bound-alias")
    alias_root = tmp_path / "ascii_alias"

    alias = _AsciiSourceAlias.acquire(
        source,
        alias_root=alias_root,
        cleanup=True,
    )
    assert str(alias.decoder_path).isascii()
    assert alias.decoder_path.is_file()
    assert os.path.samefile(source, alias.decoder_path)
    assert alias.identity_metadata["source_size_bytes"] == source.stat().st_size
    assert "hardlink" in alias.mode
    assert alias.cleanup() is True
    assert alias.cleaned is True
    assert not alias.decoder_path.exists()
    assert source.read_bytes() == b"identity-bound-alias"


def test_pynv_read_returns_owned_surface_not_decoder_reuse(
    pkg_root: Path,
) -> None:
    _require_nvdec()
    import torch

    source = _authoritative_a3b(pkg_root)
    decoder = PyNvFileDecoder(source)
    leases = []
    try:
        first = decoder.read()
        assert first is not None
        assert first.cuda_tensor.is_cuda
        assert first.pixel_format == "rgbp"
        assert first.metadata["surface_cloned"] is True
        first_snapshot = first.cuda_tensor.clone()
        leases.append(first)

        for _ in range(10):
            lease = decoder.read()
            assert lease is not None
            leases.append(lease)

        torch.cuda.synchronize(first.cuda_tensor.device)
        assert torch.equal(first.cuda_tensor, first_snapshot)
        assert first.frame_idx == 0
        assert first.pts_s >= 0.0
        assert all(lease.pts_s >= 0.0 for lease in leases)
    finally:
        for lease in leases:
            lease.release()
        decoder.close()


def test_pynv_runtime_alias_is_ascii_and_cleanup_state_is_visible(
    pkg_root: Path,
) -> None:
    _require_nvdec()
    source = _authoritative_a3b(pkg_root)
    decoder = PyNvFileDecoder(source)
    status = decoder.status_snapshot()
    alias_path = Path(status["source_alias_path"])
    storage_paths = [Path(value) for value in status["source_alias_storage_paths"]]
    assert status["source_alias_created"] is True
    assert status["source_alias_is_ascii"] is True
    assert str(alias_path).isascii()
    assert alias_path.is_file()
    assert os.path.samefile(source, alias_path)
    assert any(
        "runtime" in path.parts and "video_decode_alias" in path.parts
        for path in storage_paths
    )
    assert status["source_alias_samefile_verified"] is True

    decoder.close()
    closed = decoder.status_snapshot()
    assert closed["closed"] is True
    # PyNv 2.1.0 can retain the demuxer file handle until module teardown on
    # Windows. Either immediate cleanup or an explicit deferred reason is valid;
    # silent leakage is not.
    assert closed["source_alias_cleaned"] or (
        closed["source_alias_cleanup_deferred"]
        and closed["source_alias_cleanup_error"]
        and closed["close_error"]
    )
