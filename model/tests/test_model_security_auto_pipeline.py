from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from defense.model_security.fingerprint import sha256_file
from defense.model_security.reports import ModelPurificationReport, ModelSecurityReport
from defense.model_security.service import ModelSecurityService
from defense.model_security import service as model_security_service


def _write_config(root: Path, source_pt: Path, *, device: str = "cpu") -> Path:
    config_path = root / "runtime.yaml"
    config_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: DEVICE
  image_size: 640
  confidence: 0.3
  names: [helmet, head]
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: DEVICE
  frame_size: 640
  keyframe_interval: 3
  light_flow_interval: 3
  static_image_interval: 4
runtime:
  production_unique_model: false
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
  max_missed_frames: 2
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  device: DEVICE
  auto_export: true
  auto_export_formats: [onnx, engine]
""".replace("SOURCE_PT", source_pt.as_posix()).replace("DEVICE", device),
        encoding="utf-8",
    )
    return config_path


def _scan_report(
    fingerprint: Any,
    source_model_path: str | Path,
    *,
    status: str,
) -> ModelSecurityReport:
    source_path = Path(source_model_path)
    return ModelSecurityReport(
        fingerprint=fingerprint.to_dict(),
        scan_type="full",
        status=status,
        risk_score=0.0 if status == "clean" else 0.95,
        diagnostics={
            "external_eval_policy": {
                "version": "ppe_three_class_target_v3",
            }
        },
        source_model_path=str(source_path),
        source_model_hash="sha256:" + sha256_file(source_path),
        runtime_artifact_path=fingerprint.model_path,
    )


def _fake_exporter(calls: list[str], *, fail_engine: bool = False):
    def export(*, source_pt: Path, target_path: Path, export_format: str) -> Path:
        calls.append(export_format)
        if export_format == "engine" and fail_engine:
            raise RuntimeError("unit TensorRT export failure")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(source_pt.read_bytes() + f"-{export_format}".encode())
        return target_path

    return export


def test_clean_pt_is_trusted_exported_and_second_start_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pt = tmp_path / "models" / "clean.pt"
    source_pt.parent.mkdir(parents=True)
    source_pt.write_bytes(b"clean-model" * 128)
    service = ModelSecurityService(
        config_path=_write_config(tmp_path, source_pt, device="cpu"),
        root=tmp_path,
    )
    scan_calls: list[Path] = []
    export_calls: list[str] = []

    def clean_scan(fingerprint: Any, **kwargs: Any) -> ModelSecurityReport:
        path = Path(kwargs["source_model_path"])
        scan_calls.append(path)
        return _scan_report(fingerprint, path, status="clean")

    monkeypatch.setattr(model_security_service, "full_scan", clean_scan)
    monkeypatch.setattr(service, "_run_export_tool", _fake_exporter(export_calls))

    first = service.prepare_runtime_for_start()
    second = service.prepare_runtime_for_start()

    assert first["allowed"] is True
    assert first["scan"]["status"] == "clean"
    assert first["custom_model"]["backend"] == "onnx"
    assert Path(first["custom_model"]["path"]).is_file()
    assert second["allowed"] is True
    assert second["custom_model"]["path"] == first["custom_model"]["path"]
    assert scan_calls == [source_pt]
    assert export_calls == ["onnx"]

    records = service.registry.list_records()
    assert sum(record.approval_source == "full_scan" for record in records) == 1
    exports = [record for record in records if record.approval_source == "trusted_source_pt_export"]
    assert len(exports) == 1
    assert exports[0].backend == "onnx"
    assert exports[0].source_model_hash == "sha256:" + sha256_file(source_pt)


def test_accelerated_runtime_without_bound_metadata_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pt = tmp_path / "models" / "clean.pt"
    source_pt.parent.mkdir(parents=True)
    source_pt.write_bytes(b"clean-model" * 128)
    service = ModelSecurityService(
        config_path=_write_config(tmp_path, source_pt, device="cpu"),
        root=tmp_path,
    )
    monkeypatch.setattr(
        model_security_service,
        "full_scan",
        lambda fingerprint, **kwargs: _scan_report(
            fingerprint,
            kwargs["source_model_path"],
            status="clean",
        ),
    )
    monkeypatch.setattr(service, "_run_export_tool", _fake_exporter([]))

    prepared = service.prepare_runtime_for_start()
    accelerated = prepared["custom_model"]
    record = service.registry.get(prepared["model_security"]["fingerprint"])
    assert record is not None
    metadata_path = Path(record.security_metrics["accelerated_artifact_metadata_path"])
    metadata_path.unlink()

    status = service.status(custom_model=accelerated)

    assert status["allowed"] is False
    assert status["whitelist_hit"] is False


def test_existing_accelerated_runtime_scans_its_explicit_source_pt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pt = tmp_path / "models" / "clean.pt"
    existing_onnx = tmp_path / "models" / "legacy.onnx"
    source_pt.parent.mkdir(parents=True)
    source_pt.write_bytes(b"clean-model" * 128)
    existing_onnx.write_bytes(b"legacy-unbound-onnx" * 128)
    config_path = _write_config(tmp_path, source_pt, device="cpu")
    config_text = config_path.read_text(encoding="utf-8")
    config_text = config_text.replace("backend: pytorch", "backend: onnx", 1)
    config_text = config_text.replace(
        "  artifacts:\n    pytorch:",
        f"  artifacts:\n    onnx:\n      - {existing_onnx.as_posix()}\n    pytorch:",
        1,
    )
    config_path.write_text(config_text, encoding="utf-8")
    service = ModelSecurityService(config_path=config_path, root=tmp_path)
    scanned_paths: list[Path] = []
    export_calls: list[str] = []

    def clean_scan(fingerprint: Any, **kwargs: Any) -> ModelSecurityReport:
        scanned_path = Path(kwargs["source_model_path"])
        scanned_paths.append(scanned_path)
        assert Path(fingerprint.model_path) == source_pt
        return _scan_report(fingerprint, scanned_path, status="clean")

    monkeypatch.setattr(model_security_service, "full_scan", clean_scan)
    monkeypatch.setattr(service, "_run_export_tool", _fake_exporter(export_calls))

    result = service.prepare_runtime_for_start()

    assert result["allowed"] is True, result
    assert scanned_paths == [source_pt]
    assert export_calls == ["onnx"]
    assert result["custom_model"]["backend"] == "onnx"
    assert Path(result["custom_model"]["path"]) != existing_onnx
    trusted_pt = [record for record in service.registry.list_records() if record.backend == "pytorch"]
    assert len(trusted_pt) == 1
    assert trusted_pt[0].runtime_model_path == str(source_pt)


def test_export_tool_stages_source_in_ascii_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pt = tmp_path / "中文模型" / "clean.pt"
    target = tmp_path / "runtime" / "clean.engine"
    source_pt.parent.mkdir(parents=True)
    source_pt.write_bytes(b"trusted-source")
    staged_paths: list[Path] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            self.path = Path(path)
            staged_paths.append(self.path)

        def export(self, **kwargs: Any) -> str:
            assert kwargs["format"] == "engine"
            exported = self.path.with_suffix(".engine")
            exported.write_bytes(self.path.read_bytes() + b"-engine")
            return str(exported)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    service = ModelSecurityService(root=tmp_path)

    exported = service._run_export_tool(
        source_pt=source_pt,
        target_path=target,
        export_format="engine",
    )

    assert exported == target
    assert target.read_bytes() == b"trusted-source-engine"
    assert staged_paths[0].parent != source_pt.parent
    assert all(ord(character) < 128 for character in str(staged_paths[0]))


def test_suspicious_pt_is_purified_rescanned_trusted_and_exported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pt = tmp_path / "models" / "suspicious.pt"
    candidate = tmp_path / "runtime" / "model_security" / "models" / "purified_pt" / "candidate.pt"
    source_pt.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    source_pt.write_bytes(b"suspicious-model" * 128)
    candidate.write_bytes(b"purified-model" * 128)
    final_purified = source_pt.parent / "suspicious_净化完毕.pt"
    final_onnx = source_pt.parent / "suspicious_净化完毕_onnx_加速.onnx"
    service = ModelSecurityService(
        config_path=_write_config(tmp_path, source_pt, device="cpu"),
        root=tmp_path,
    )
    scan_calls: list[Path] = []
    export_calls: list[str] = []

    def staged_scan(fingerprint: Any, **kwargs: Any) -> ModelSecurityReport:
        path = Path(kwargs["source_model_path"])
        scan_calls.append(path)
        return _scan_report(
            fingerprint,
            path,
            status="clean" if path == final_purified else "suspicious",
        )

    def purify(**kwargs: Any) -> ModelPurificationReport:
        fingerprint = kwargs["fp"]
        return ModelPurificationReport(
            fingerprint=fingerprint.to_dict(),
            status="candidate_ready",
            strategy="adaptive_delivery",
            source_model_path=str(source_pt),
            source_model_hash="sha256:" + sha256_file(source_pt),
            purified_model_path=str(candidate),
            purified_model_hash="sha256:" + sha256_file(candidate),
            candidates=[
                {
                    "output_model": str(candidate),
                    "candidate_source": "adaptive_delivery",
                    "source_sha256": sha256_file(source_pt),
                    "candidate_sha256": sha256_file(candidate),
                }
            ],
        )

    monkeypatch.setattr(model_security_service, "full_scan", staged_scan)
    monkeypatch.setattr(model_security_service, "run_new_purification", purify)
    monkeypatch.setattr(service, "_run_export_tool", _fake_exporter(export_calls))

    result = service.prepare_runtime_for_start()

    assert result["allowed"] is False, result
    assert result["custom_model"] is None
    assert result["scan"]["status"] == "suspicious"
    assert result["purification"]["status"] == "scan_clean_trusted"
    assert Path(result["purification"]["purified_model_path"]) == final_purified
    assert final_purified.is_file()
    assert result["model_security"]["allowed"] is False
    assert result["model_security"]["admission_status"] == "suspicious"
    assert scan_calls == [source_pt, final_purified]
    assert export_calls == []

    original_status = service.status()
    assert original_status["allowed"] is False
    assert original_status["admission_status"] not in {
        "trusted",
        "purified_alternative_available",
    }
    assert service.trusted_purified_runtime_model() is None

    records = service.registry.list_records()
    purified_records = [record for record in records if record.approval_source == "purified_full_scan"]
    assert len(purified_records) == 1
    original_hash = "sha256:" + sha256_file(source_pt)
    purified_hash = "sha256:" + sha256_file(final_purified)
    assert purified_records[0].runtime_model_path == str(final_purified)
    assert purified_records[0].source_model_hash == purified_hash
    assert purified_records[0].source_model_path == str(final_purified)
    assert purified_records[0].original_source_model_hash == original_hash
    assert purified_records[0].original_source_model_path == str(source_pt)
    assert not any(record.approval_source == "trusted_source_pt_export" for record in records)

    purified_custom_model = {
        "enabled": True,
        "path": str(final_purified),
        "backend": "pytorch",
        "model_family": "yolov5",
        "source_pt_path": str(final_purified),
    }
    purified_status = service.status(custom_model=purified_custom_model)
    assert purified_status["allowed"] is True
    assert purified_status["admission_status"] == "trusted"

    prepared_purified = service.prepare_runtime_for_start(
        custom_model=purified_custom_model,
        auto_remediate=False,
    )
    assert prepared_purified["allowed"] is True
    assert prepared_purified["custom_model"] == purified_custom_model

    export_result = service.export_accelerated_model(
        export_format="onnx",
        custom_model=purified_custom_model,
    )
    assert export_result["state"] == "completed"
    assert Path(export_result["exported_model_path"]) == final_onnx
    assert final_onnx.is_file()
    assert export_calls == ["onnx"]

    accelerated_status = service.status(
        custom_model={
            "enabled": True,
            "path": str(final_onnx),
            "backend": "onnx",
            "model_family": "yolov5",
        }
    )
    assert accelerated_status["allowed"] is True
    assert accelerated_status["admission_status"] == "trusted"
    assert Path(accelerated_status["source_pt_path"]) == final_purified

    records = service.registry.list_records()
    exported = [record for record in records if record.approval_source == "trusted_source_pt_export"]
    assert len(exported) == 1
    assert exported[0].runtime_model_path == str(final_onnx)
    assert exported[0].source_model_hash == purified_hash
    assert exported[0].source_model_path == str(final_purified)
    assert exported[0].original_source_model_hash == original_hash
    assert exported[0].original_source_model_path == str(source_pt)


def test_tensorrt_export_failure_keeps_trusted_onnx_runtime_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    source_pt = tmp_path / "models" / "clean.pt"
    source_pt.parent.mkdir(parents=True)
    source_pt.write_bytes(b"clean-model" * 128)
    service = ModelSecurityService(
        config_path=_write_config(tmp_path, source_pt, device="auto"),
        root=tmp_path,
    )
    export_calls: list[str] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        model_security_service,
        "full_scan",
        lambda fingerprint, **kwargs: _scan_report(
            fingerprint,
            kwargs["source_model_path"],
            status="clean",
        ),
    )
    monkeypatch.setattr(
        service,
        "_run_export_tool",
        _fake_exporter(export_calls, fail_engine=True),
    )

    result = service.prepare_runtime_for_start()

    assert result["allowed"] is True
    assert result["custom_model"]["backend"] == "onnx"
    assert Path(result["custom_model"]["path"]).is_file()
    assert export_calls == ["onnx", "engine"]
    assert any(entry["event"] == "export_failed" for entry in service.recent_logs(limit=20)["entries"])
    records = service.registry.list_records()
    assert any(record.backend == "onnx" and record.approved_for_runtime for record in records)
    assert not any(record.backend == "tensorrt" for record in records)
