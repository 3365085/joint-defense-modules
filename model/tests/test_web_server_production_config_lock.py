from __future__ import annotations

from pathlib import Path

import pytest

from defense.runtime.config import DEFAULT_CONFIG_PATH
from defense.web.server import ALLOW_TEST_CONFIG_ENV, resolve_web_config


def test_web_server_uses_authoritative_config_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ALLOW_TEST_CONFIG_ENV, raising=False)
    assert resolve_web_config(DEFAULT_CONFIG_PATH) == DEFAULT_CONFIG_PATH.resolve()


def test_web_server_rejects_alternate_config_without_explicit_test_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ALLOW_TEST_CONFIG_ENV, raising=False)
    with pytest.raises(RuntimeError, match="production Web is locked"):
        resolve_web_config(tmp_path / "other.yaml")


def test_web_server_allows_alternate_config_only_in_explicit_test_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ALLOW_TEST_CONFIG_ENV, "1")
    alternate = tmp_path / "other.yaml"
    assert resolve_web_config(alternate) == alternate.resolve()

