from __future__ import annotations

from tools.verify_module_a_auxiliary_artifacts import DEFAULT_MANIFEST, verify


def test_main_project_auxiliary_artifact_manifest_is_valid() -> None:
    result = verify(DEFAULT_MANIFEST)
    assert result["ok"] is True
    assert result["artifacts"]["raft"]["valid"] is True
    assert result["artifacts"]["raft"]["required"] is True
    assert result["artifacts"]["a4"]["required"] is True
    assert result["artifacts"]["a4"]["valid"] is True
