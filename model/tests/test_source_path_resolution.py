from __future__ import annotations

from pathlib import Path

from defense.runtime.runner import normalize_source_text, resolve_source_path


def test_resolve_source_path_accepts_quoted_legacy_material_root(monkeypatch, tmp_path: Path) -> None:
    material = tmp_path / "素材" / "物理扰动攻击视频" / "case.mp4"
    material.parent.mkdir(parents=True)
    material.write_bytes(b"")
    monkeypatch.setenv("SECURITY_PROJECT_ROOT", str(tmp_path))

    resolved = resolve_source_path('"D:\\联合防御模块训练素材\\物理扰动攻击视频\\case.mp4"')

    assert resolved == material


def test_normalize_source_text_accepts_file_uri(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")

    assert Path(normalize_source_text(media.as_uri())) == media


def test_resolve_source_path_uses_tail_fallback_for_old_windows_paths(monkeypatch, tmp_path: Path) -> None:
    material = tmp_path / "素材" / "手机随意录制的视频" / "case.mp4"
    material.parent.mkdir(parents=True)
    material.write_bytes(b"")
    monkeypatch.setenv("SECURITY_PROJECT_ROOT", str(tmp_path))

    resolved = resolve_source_path(r"D:\旧工程\联合防御模块训练素材备份\手机随意录制的视频\case.mp4")

    assert resolved == material
