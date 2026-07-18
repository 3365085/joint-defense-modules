from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
from fastapi.testclient import TestClient

from defense.model_security import ModelSecurityService
from defense.model_security import purifier
from defense.model_security import scanner as model_security_scanner
from defense.model_security.fingerprint import build_model_fingerprint, sha256_file
from defense.model_security.registry import ModelTrustRegistry
from defense.model_security.reports import ModelPurificationReport, ModelSecurityReport, ScanBudget
from defense.model_security.service import _read_torch_zip_class_names
from defense.model_security.scanner import _filter_external_contract_noise
from defense.module_a.backends.detector_backend import UltralyticsDetectorBackend
from defense.module_a.postprocess.ppe_tracking import canonical_label
from defense.module_a.ppe_postprocess import PPEPostprocessConfig, summarize_ppe_from_detections
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState
from defense.runtime.overlay_records import build_overlay_record
from defense.runtime import pipeline_factory
from defense.runtime.config import (
    DEFAULT_CONFIG_PATH,
    apply_custom_model,
    apply_feature_options,
    load_runtime_config,
    normalize_custom_model_options,
)
from defense.runtime.ppe_business import evaluate_ppe_business
from defense.runtime.ppe_state import SafetyHelmetState
from defense.runtime.runner import MonitorEngine
from defense.web.fastapi_app import create_app


def _offline_model_security_config(tmp_path: Path) -> Path:
    config_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    production_lock = "  production_unique_model: true"
    assert config_text.count(production_lock) == 1
    config_path = tmp_path / "module_a_runtime.offline-test.yaml"
    config_path.write_text(
        config_text.replace(production_lock, "  production_unique_model: false"),
        encoding="utf-8",
    )
    return config_path


def _offline_model_security_service(tmp_path: Path) -> ModelSecurityService:
    return ModelSecurityService(
        config_path=_offline_model_security_config(tmp_path),
        root=tmp_path,
    )


class _Detections:
    def __init__(self, *, boxes, classes, confidences, names):
        self.boxes = boxes
        self.classes = classes
        self.confidences = confidences
        self.names = names


def _write_seven_archive(tmp_path: Path) -> dict[str, Path | str]:
    archive_root = tmp_path / "purification_lab" / "seven_experiment_archive"
    exp_dir = archive_root / "oga_visible_patch"
    exp_dir.mkdir(parents=True)
    poisoned = exp_dir / "poisoned_best.pt"
    purified = exp_dir / "purified_best.pt"
    video = exp_dir / "clean_attack_purif.mp4"
    poisoned.write_bytes(b"poisoned-model")
    purified.write_bytes(b"purified-model")
    video.write_bytes(b"comparison-video")
    poisoned_sha = sha256_file(poisoned)
    purified_sha = sha256_file(purified)
    video_sha = sha256_file(video)
    summary_path = exp_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": "oga_visible_patch",
                "attack_algorithm": {"name": "oga_visible_patch", "attack_goal": "oga"},
                "purification_algorithm": "universal_sandwich_detox",
                "archive_files": {
                    "poisoned_checkpoint": {"path": str(poisoned), "sha256": poisoned_sha},
                    "purified_checkpoint": {"path": str(purified), "sha256": purified_sha},
                    "comparison_video": {"path": str(video), "sha256": video_sha},
                },
                "source_paths": {"poisoned_checkpoint": "source-poisoned.pt"},
                "reproduction_data": {"dataset": "oga_visible_patch_helmet_v4redxvideo"},
            }
        ),
        encoding="utf-8",
    )
    (archive_root / "manifest.json").write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment": "oga_visible_patch", "summary_json": str(summary_path)}
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "archive_root": archive_root,
        "summary_path": summary_path,
        "poisoned": poisoned,
        "purified": purified,
        "video": video,
        "poisoned_sha": poisoned_sha,
        "purified_sha": purified_sha,
        "video_sha": video_sha,
    }


def test_registry_preserves_purified_origin_and_security_metrics(tmp_path: Path):
    reg = ModelTrustRegistry(tmp_path / "trusted_registry.json")
    reg.mark_trusted(
        "sha256:purified",
        risk_score=0.0044,
        runtime_model_hash="sha256:purified-model",
        runtime_model_path="purified.pt",
        source_model_hash="sha256:purified-model",
        source_model_path="purified.pt",
        original_source_model_hash="sha256:poisoned-model",
        original_source_model_path="poisoned.pt",
        security_metrics={
            "original_asr": 1.0,
            "purified_asr": 0.0,
            "purified_wilson_upper": 0.0044,
        },
        approval_source="purified_full_scan",
    )

    rec = reg.get("sha256:purified")
    assert rec is not None
    assert rec.source_model_path == "purified.pt"
    assert rec.original_source_model_path == "poisoned.pt"
    assert rec.security_metrics["original_asr"] == 1.0
    assert rec.security_metrics["purified_wilson_upper"] == 0.0044


def test_trust_records_backfills_old_purified_metrics_from_reports(tmp_path: Path):
    svc = ModelSecurityService(root=tmp_path)
    original_fp = {"fingerprint": "sha256:original", "model_hash": "sha256:original-model"}
    purified_fp = {"fingerprint": "sha256:purified", "model_hash": "sha256:purified-model"}
    original_report = ModelSecurityReport(
        fingerprint=original_fp,
        scan_type="full",
        status="suspicious",
        risk_score=1.0,
        diagnostics={
            "new_algorithm_poisoned_evidence": {
                "family_tag": "b2",
                "original_attack_metrics": {
                    "max_asr": 1.0,
                    "attack": "sig_multiperiod_oda",
                    "source": "unit",
                },
            }
        },
        source_model_path="poisoned.pt",
        source_model_hash="sha256:poisoned-model",
    )
    svc._write_report(original_report)
    purified_report = ModelSecurityReport(
        fingerprint=purified_fp,
        scan_type="full",
        status="clean",
        risk_score=0.0044,
        diagnostics={
            "new_algorithm_strict_audit": {
                "family_tag": "b2",
                "k": 0,
                "N": 869,
                "wilson_upper": 0.004401095733793813,
                "mAP_drop_pp": -0.111,
                "defense": "Backbone-Soup",
            }
        },
        source_model_path="purified.pt",
        source_model_hash="sha256:purified-model",
    )
    svc._write_report(purified_report)
    purification = ModelPurificationReport(
        fingerprint=original_fp,
        status="scan_clean_trusted",
        strategy="autodetox_backbone_soup",
        source_model_path="poisoned.pt",
        source_model_hash="sha256:poisoned-model",
        purified_model_path="purified.pt",
        purified_model_hash="sha256:purified-model",
        scan_report_path=purified_report.report_path,
        scan_status="clean",
    )
    purification.write(svc.storage.reports_dir / "sha256_original_purification.json")
    svc.registry.mark_trusted(
        "sha256:purified",
        risk_score=0.0044,
        report_path=purified_report.report_path,
        runtime_model_hash="sha256:purified-model",
        runtime_model_path="purified.pt",
        source_model_hash="sha256:purified-model",
        source_model_path="purified.pt",
        purification_report_path=str(svc.storage.reports_dir / "sha256_original_purification.json"),
        approval_source="purified_full_scan",
    )

    records = svc.trust_records()["records"]
    assert records[0]["original_source_model_path"] == "poisoned.pt"
    assert records[0]["security_metrics"]["original_asr"] == 1.0
    assert records[0]["security_metrics"]["purified_asr"] == 0.0
    assert records[0]["security_metrics"]["purified_wilson_upper"] == 0.004401095733793813


def test_trusted_purified_status_prefers_registry_bound_purification_report(tmp_path: Path):
    svc = _offline_model_security_service(tmp_path)
    poisoned = tmp_path / "poisoned.pt"
    purified = tmp_path / "poisoned_净化完毕.pt"
    poisoned.write_bytes(b"poisoned-model" * 128)
    purified.write_bytes(b"purified-model" * 128)
    custom_model = {
        "enabled": True,
        "path": str(purified),
        "backend": "pytorch",
        "model_family": "yolov5",
        "source_pt_path": str(purified),
    }
    purified_fp = svc.current_fingerprint(custom_model=custom_model)
    purified_report = ModelSecurityReport(
        fingerprint={
            "fingerprint": purified_fp.fingerprint,
            "model_hash": purified_fp.model_hash,
        },
        scan_type="full",
        status="clean",
        risk_score=0.0044,
        source_model_path=str(purified),
        source_model_hash=purified_fp.model_hash,
        diagnostics={
            "new_algorithm_strict_audit": {
                "family_tag": "b4",
                "k": 0,
                "N": 869,
                "wilson_upper": 0.004401095733793813,
            }
        },
    )
    svc._write_report(purified_report)
    original_report = ModelSecurityReport(
        fingerprint={"fingerprint": "sha256:poisoned", "model_hash": "sha256:poisoned-model"},
        scan_type="full",
        status="suspicious",
        risk_score=1.0,
        source_model_path=str(poisoned),
        source_model_hash="sha256:poisoned-model",
        diagnostics={
            "new_algorithm_poisoned_evidence": {
                "family_tag": "b4",
                "original_attack_metrics": {"max_asr": 1.0, "attack": "sig_lowfreq_hi_oda", "source": "unit"},
            }
        },
    )
    svc._write_report(original_report)
    trusted_purification = ModelPurificationReport(
        fingerprint=original_report.fingerprint,
        status="scan_clean_trusted",
        strategy="autodetox_backbone_soup",
        source_model_path=str(poisoned),
        source_model_hash="sha256:poisoned-model",
        purified_model_path=str(purified),
        purified_model_hash=purified_fp.model_hash,
        scan_report_path=purified_report.report_path,
        scan_status="clean",
    )
    trusted_path = svc.storage.reports_dir / "sha256_poisoned_purification.json"
    trusted_purification.write(trusted_path)
    stale_purification = ModelPurificationReport(
        fingerprint={"fingerprint": purified_fp.fingerprint, "model_hash": purified_fp.model_hash},
        status="planned",
        strategy="autodetox_backbone_soup",
        source_model_path=str(purified),
        source_model_hash=purified_fp.model_hash,
        error="clean_anchor_required_for_backbone_soup",
    )
    stale_purification.write(svc.storage.reports_dir / f"{purified_fp.fingerprint.replace(':','_')}_purification.json")
    svc.registry.mark_trusted(
        purified_fp.fingerprint,
        risk_score=0.0044,
        report_path=purified_report.report_path,
        runtime_model_hash=purified_fp.model_hash,
        runtime_model_path=str(purified),
        source_model_hash=purified_fp.model_hash,
        source_model_path=str(purified),
        original_source_model_hash="sha256:poisoned-model",
        original_source_model_path=str(poisoned),
        scanner_version=purified_fp.scanner_version,
        class_names_hash=purified_fp.class_names_hash,
        ppe_mapping_hash=purified_fp.ppe_mapping_hash,
        purification_report_path=str(trusted_path),
        security_metrics={"original_asr": 1.0, "purified_asr": 0.0},
        approval_source="purified_full_scan",
    )

    status = svc.status(custom_model=custom_model)

    assert status["admission_status"] == "trusted"
    assert status["purification_status"] == "scan_clean_trusted"
    assert status["purification_report_path"] == str(trusted_path)
    assert status["purified_model_path"] == str(purified)
    assert status["security_metrics"]["original_asr"] == 1.0
    assert status["security_metrics"]["purified_asr"] == 0.0


def test_purification_report_requires_matching_source_hash_for_admission(tmp_path: Path):
    svc = _offline_model_security_service(tmp_path)
    engine = tmp_path / "runtime.engine"
    old_source = tmp_path / "old_source.pt"
    new_source = tmp_path / "new_source.pt"
    purified = tmp_path / "old_source_purified.pt"
    engine.write_bytes(b"same-runtime-engine" * 128)
    old_source.write_bytes(b"old-source-model" * 128)
    new_source.write_bytes(b"new-source-model" * 128)
    purified.write_bytes(b"purified-old-source" * 128)
    custom_model = {
        "enabled": True,
        "path": str(engine),
        "backend": "tensorrt",
        "model_family": "yolov5",
        "source_pt_path": str(new_source),
    }
    fp = svc.current_fingerprint(custom_model=custom_model)
    stale_report = ModelPurificationReport(
        fingerprint={"fingerprint": fp.fingerprint, "model_hash": fp.model_hash},
        status="scan_clean_trusted",
        strategy="autodetox_backbone_soup",
        source_model_path=str(old_source),
        source_model_hash="sha256:" + sha256_file(old_source),
        purified_model_path=str(purified),
        purified_model_hash="sha256:" + sha256_file(purified),
        scan_status="clean",
    )
    stale_report.write(svc.storage.reports_dir / f"{fp.fingerprint.replace(':','_')}_purification.json")

    status = svc.status(custom_model=custom_model)

    assert status["source_pt_hash"] == "sha256:" + sha256_file(new_source)
    assert status["admission_status"] == "blocked_scan_required"
    assert status["purification_status"] == "idle"
    assert status["purified_model_path"] is None


def test_purification_report_without_source_hash_is_not_source_bound(tmp_path: Path):
    svc = _offline_model_security_service(tmp_path)
    engine = tmp_path / "runtime.engine"
    source = tmp_path / "source.pt"
    purified = tmp_path / "legacy_purified.pt"
    engine.write_bytes(b"same-runtime-engine" * 128)
    source.write_bytes(b"current-source-model" * 128)
    purified.write_bytes(b"legacy-purified-model" * 128)
    custom_model = {
        "enabled": True,
        "path": str(engine),
        "backend": "tensorrt",
        "model_family": "yolov5",
        "source_pt_path": str(source),
    }
    fp = svc.current_fingerprint(custom_model=custom_model)
    legacy_report = ModelPurificationReport(
        fingerprint={"fingerprint": fp.fingerprint, "model_hash": fp.model_hash},
        status="scan_clean_trusted",
        strategy="legacy_report_without_source_hash",
        source_model_path=str(source),
        source_model_hash=None,
        purified_model_path=str(purified),
        purified_model_hash="sha256:" + sha256_file(purified),
        scan_status="clean",
    )
    legacy_report.write(svc.storage.reports_dir / f"{fp.fingerprint.replace(':','_')}_purification.json")

    status = svc.status(custom_model=custom_model)

    assert status["source_pt_hash"] == "sha256:" + sha256_file(source)
    assert status["admission_status"] == "blocked_scan_required"
    assert status["purification_status"] == "idle"
    assert status["purified_model_path"] is None


def test_full_scan_report_without_source_hash_is_not_source_bound(tmp_path: Path):
    svc = _offline_model_security_service(tmp_path)
    engine = tmp_path / "runtime.engine"
    source = tmp_path / "source.pt"
    engine.write_bytes(b"same-runtime-engine" * 128)
    source.write_bytes(b"current-source-model" * 128)
    custom_model = {
        "enabled": True,
        "path": str(engine),
        "backend": "tensorrt",
        "model_family": "yolov5",
        "source_pt_path": str(source),
    }
    fp = svc.current_fingerprint(custom_model=custom_model)
    legacy_report = ModelSecurityReport(
        fingerprint={"fingerprint": fp.fingerprint, "model_hash": fp.model_hash},
        scan_type="full",
        status="suspicious",
        risk_score=0.95,
        source_model_path=str(source),
        source_model_hash=None,
        runtime_artifact_path=str(engine),
    )
    legacy_report.write(svc.storage.reports_dir / f"{fp.fingerprint.replace(':','_')}_full.json")

    status = svc.status(custom_model=custom_model)

    assert status["source_pt_hash"] == "sha256:" + sha256_file(source)
    assert status["admission_status"] == "blocked_scan_required"
    assert status["report_path"] is None


def test_packaged_purification_candidates_follow_hash_when_model_is_renamed(tmp_path: Path, monkeypatch):
    package = tmp_path / "new_algorithm"
    poisoned_dir = package / "models" / "poisoned"
    purified_dir = package / "models" / "purified"
    clean_dir = package / "models" / "clean_baseline"
    audit_dir = package / "audit"
    poisoned_dir.mkdir(parents=True)
    purified_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    catalog_poisoned = poisoned_dir / "v2_mask_bd_v2_poisoned.pt"
    renamed_poisoned = tmp_path / "oga_visible_patch_poisoned.pt"
    packaged_purified = purified_dir / "v2_mask_bd_v2_visible_purified_strict.pt"
    clean_anchor = clean_dir / "v2_mask_bd_v2_clean_baseline.pt"
    catalog_poisoned.write_bytes(b"same-poisoned-v2")
    renamed_poisoned.write_bytes(b"same-poisoned-v2")
    packaged_purified.write_bytes(b"strict-purified-v2")
    clean_anchor.write_bytes(b"clean-v2")
    (audit_dir / purifier.STRICT_AUDIT_NAME).write_text(
        """
        {
          "rows": [
            {
              "tag": "v2",
              "status": "ok",
              "strict_pass": true,
              "certified": true,
              "N": 462,
              "wilson_upper": 0.0475,
              "mAP_drop_pp": 3.54
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(purifier, "new_algorithm_package_root", lambda _root: package)

    assert purifier.find_clean_anchor(renamed_poisoned, config={}, root=tmp_path) == clean_anchor
    candidates = purifier._packaged_purified_candidates(renamed_poisoned, root=tmp_path)
    assert candidates == [packaged_purified]
    staged = purifier._stage_packaged_candidates(renamed_poisoned, root=tmp_path, out_dir=tmp_path / "out")
    assert staged[0]["family_tag"] == "v2"
    assert staged[0]["new_algorithm_strict_audit"]["wilson_upper"] == 0.0475


def test_seven_experiment_archive_full_scan_uses_archive_scope(tmp_path: Path):
    archive_root = tmp_path / "purification_lab" / "seven_experiment_archive"
    exp_dir = archive_root / "oga_visible_patch"
    exp_dir.mkdir(parents=True)
    poisoned = exp_dir / "poisoned_best.pt"
    purified = exp_dir / "purified_best.pt"
    video = exp_dir / "clean_attack_purif.mp4"
    poisoned.write_bytes(b"poisoned-model")
    purified.write_bytes(b"purified-model")
    video.write_bytes(b"comparison-video")
    poisoned_sha = sha256_file(poisoned)
    purified_sha = sha256_file(purified)
    video_sha = sha256_file(video)
    summary_path = exp_dir / "summary.json"
    summary_path.write_text(
        f"""{{
          "experiment": "oga_visible_patch",
          "attack_algorithm": {{"name": "oga_visible_patch", "attack_goal": "oga"}},
          "purification_algorithm": "universal_sandwich_detox",
          "archive_files": {{
            "poisoned_checkpoint": {{"path": "{str(poisoned).replace(chr(92), chr(92) + chr(92))}", "sha256": "{poisoned_sha}"}},
            "purified_checkpoint": {{"path": "{str(purified).replace(chr(92), chr(92) + chr(92))}", "sha256": "{purified_sha}"}},
            "comparison_video": {{"path": "{str(video).replace(chr(92), chr(92) + chr(92))}", "sha256": "{video_sha}"}}
          }},
          "source_paths": {{"poisoned_checkpoint": "source-poisoned.pt"}},
          "reproduction_data": {{"dataset": "oga_visible_patch_helmet_v4redxvideo"}}
        }}""",
        encoding="utf-8",
    )
    (archive_root / "manifest.json").write_text(
        f"""{{
          "experiments": [
            {{"experiment": "oga_visible_patch", "summary_json": "{str(summary_path).replace(chr(92), chr(92) + chr(92))}"}}
          ]
        }}""",
        encoding="utf-8",
    )
    poisoned_cfg = {
        "inference": {
            "backend": "pytorch",
            "model_family": "yolov5",
            "artifacts": {"pytorch": [str(poisoned)]},
            "class_names": ["helmet", "head", "person"],
        }
    }
    purified_cfg = {
        "inference": {
            "backend": "pytorch",
            "model_family": "yolov5",
            "artifacts": {"pytorch": [str(purified)]},
            "class_names": ["helmet", "head", "person"],
        }
    }

    poisoned_report = model_security_scanner.full_scan(
        build_model_fingerprint(poisoned_cfg, root=tmp_path),
        source_model_path=poisoned,
        project_root=tmp_path,
    )
    purified_report = model_security_scanner.full_scan(
        build_model_fingerprint(purified_cfg, root=tmp_path),
        source_model_path=purified,
        project_root=tmp_path,
    )

    assert poisoned_report.status == "suspicious"
    assert poisoned_report.diagnostics["validation_scope"] == "seven_experiment_known_poisoned_archive"
    assert poisoned_report.diagnostics["new_algorithm_poisoned_evidence"]["purified_candidates"][0]["hash"] == "sha256:" + purified_sha
    assert purified_report.status == "clean"
    assert purified_report.diagnostics["validation_scope"] == "seven_experiment_purified_archive"
    assert purified_report.diagnostics["new_algorithm_strict_audit"]["metric_source"] == "archive_hash_verification_only"
    assert purified_report.diagnostics["new_algorithm_strict_audit"]["wilson_upper"] is None
    assert purified_report.diagnostics["new_algorithm_strict_audit"]["comparison_video_hash"] == "sha256:" + video_sha


def test_seven_archive_candidate_scope_survives_staging_and_fast_report(tmp_path: Path, monkeypatch):
    archive = _write_seven_archive(tmp_path)
    package = tmp_path / "new_algorithm"
    package.mkdir()
    monkeypatch.setattr(purifier, "new_algorithm_package_root", lambda _root: package)

    poisoned = Path(archive["poisoned"])
    purified = Path(archive["purified"])
    candidates = purifier._packaged_purified_candidates(poisoned, root=tmp_path)
    staged = purifier._stage_packaged_candidates(poisoned, root=tmp_path, out_dir=tmp_path / "out")

    assert candidates == [purified]
    assert staged[0]["validation_scope"] == "seven_experiment_purified_archive"
    assert staged[0]["new_algorithm_strict_audit"]["metric_source"] == "archive_hash_verification_only"
    assert staged[0]["new_algorithm_strict_audit"]["wilson_upper"] is None

    fp = build_model_fingerprint(
        {
            "inference": {
                "backend": "pytorch",
                "model_family": "yolov5",
                "artifacts": {"pytorch": [staged[0]["output_model"]]},
                "class_names": ["helmet", "head", "person"],
            }
        },
        root=tmp_path,
    )
    report = ModelSecurityService(root=tmp_path)._strict_candidate_report(
        fp=fp,
        candidate_path=Path(staged[0]["output_model"]),
        candidate=staged[0],
        budget=ScanBudget(),
    )

    assert report is not None
    assert report.status == "clean"
    assert report.risk_score == 0.0
    assert report.diagnostics["validation_scope"] == "seven_experiment_purified_archive"
    assert "archive" in report.reasons[0]


def test_default_runtime_config_scans_head_and_helmet_without_person_context():
    # 生产配置已切换为同学的 2 类模型(class_names=[helmet, head], 无 person)。
    # 扫描目标仍是 helmet/head; person 上下文类天然为空; preserve_classes 独立配置项仍保留。
    cfg = load_runtime_config(profile="default")

    resolution = model_security_scanner._external_target_resolution(cfg)

    assert resolution["target_classes"] == ["helmet", "head"]
    assert resolution["context_classes"] == []
    assert resolution["preserve_classes"] == ["person"]


def test_custom_model_class_names_are_applied_to_runtime_config(tmp_path: Path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"not a torch checkpoint")
    config = {"inference": {"backend": "tensorrt", "model_family": "yolov5", "artifacts": {}}}
    options = normalize_custom_model_options(
        {
            "enabled": True,
            "path": str(model_path),
            "backend": "auto",
            "model_family": "yolov5",
            "class_names": "person / head / helmet",
        }
    )

    resolved = apply_custom_model(config, options)

    assert resolved["class_names"] == ["person", "head", "helmet"]
    assert config["inference"]["class_names"] == ["person", "head", "helmet"]


def test_model_security_reports_class_name_override_mismatch(tmp_path: Path):
    svc = ModelSecurityService(root=tmp_path)
    source = tmp_path / "best.pt"
    source.write_bytes(b"not a torch checkpoint")

    def embedded_names(_path: Path) -> dict:
        return {
            "available": True,
            "class_names": {0: "helmet", 1: "head", 2: "person"},
            "source": str(source),
            "error": "",
        }

    svc._embedded_class_names = embedded_names  # type: ignore[method-assign]
    diagnostics = svc._class_names_diagnostics(
        {
            "inference": {"class_names": ["person", "head", "helmet"]},
            "runtime": {"custom_model": {"enabled": True}},
        },
        source,
    )

    assert diagnostics["configured_class_names"] == {0: "person", 1: "head", 2: "helmet"}
    assert diagnostics["embedded_class_names"] == {0: "helmet", 1: "head", 2: "person"}
    assert diagnostics["class_names_mismatch"] is True
    assert "differ" in diagnostics["class_names_warning"]


def test_safe_torch_zip_class_name_reader_extracts_names(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr(
            "archive/data.pkl",
            pickle.dumps({"names": {0: "helmet", 1: "head", 2: "person"}}),
        )

    assert _read_torch_zip_class_names(checkpoint) == {0: "helmet", 1: "head", 2: "person"}


def test_model_security_skips_embedded_class_names_for_default_runtime(tmp_path: Path):
    svc = ModelSecurityService(root=tmp_path)
    source = tmp_path / "best.pt"
    source.write_bytes(b"not a torch checkpoint")

    def fail_if_called(_path: Path) -> dict:
        raise AssertionError("default runtime should not load checkpoint class names")

    svc._embedded_class_names = fail_if_called  # type: ignore[method-assign]
    diagnostics = svc._class_names_diagnostics(
        {"inference": {"class_names": ["helmet", "head", "person"]}},
        source,
    )

    assert diagnostics["configured_class_names"] == {0: "helmet", 1: "head", 2: "person"}
    assert diagnostics["embedded_class_names"] == {}
    assert diagnostics["class_names_mismatch"] is False
    assert diagnostics["class_names_warning"] == ""


def test_person_aliases_are_consistent_across_ppe_layers():
    backend = UltralyticsDetectorBackend.__new__(UltralyticsDetectorBackend)
    backend.confidence = 0.25
    backend.candidate_confidence = 0.18

    for alias in ("person", "worker", "human", "pedestrian"):
        backend.names = {0: "helmet", 1: "head", 2: alias}
        detections = _Detections(
            boxes=[(80, 80, 220, 330)],
            classes=[2],
            confidences=[0.88],
            names={2: alias},
        )
        summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

        assert backend._prediction_confidence() == 0.18
        assert canonical_label(alias) == "person"
        assert summary["has_person_class"] is True
        assert summary["effective_person_count"] == 1


def test_ppe_summary_suppresses_head_when_kept_helmet_overlaps_display_mutex():
    config = PPEPostprocessConfig(prefer_helmet_on_head_overlap=True)
    detections = _Detections(
        boxes=[
            (100, 100, 180, 180),
            (120, 120, 220, 220),
            (80, 80, 260, 360),
        ],
        classes=[0, 1, 2],
        confidences=[0.78, 0.92, 0.90],
        names={0: "helmet", 1: "head", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, config=config, frame_shape=(640, 640))

    suppression = summary["helmet_fp_suppression"]
    assert summary["raw_head_count"] == 1
    assert summary["head_count"] == 0
    assert summary["helmet_count"] == 1
    assert summary["person_count"] == 1
    assert summary["candidate"] is False
    assert summary["reason"] == "helmet_evidence_present"
    assert suppression["covered_head_indices"] == [1]
    assert suppression["suppressed_head_indices"] == [1]
    assert suppression["suppressed_heads"][-1]["reason"] == "head_helmet_mutex"


def test_ppe_summary_prefers_overlapping_helmet_when_head_confidence_is_higher():
    config = PPEPostprocessConfig(prefer_helmet_on_head_overlap=True)
    detections = _Detections(
        boxes=[
            (100, 100, 180, 210),
            (100, 100, 180, 210),
            (80, 80, 260, 360),
        ],
        classes=[0, 1, 2],
        confidences=[0.36, 0.71, 0.90],
        names={0: "helmet", 1: "head", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, config=config, frame_shape=(640, 640))

    suppression = summary["helmet_fp_suppression"]
    assert summary["raw_head_count"] == 1
    assert summary["head_count"] == 0
    assert summary["helmet_count"] == 1
    assert suppression["suppressed_helmet_indices"] == []
    assert suppression["covered_head_indices"] == [1]
    assert suppression["suppressed_heads"][-1]["reason"] == "head_helmet_mutex"


def test_ppe_summary_keeps_head_when_helmet_does_not_overlap_mutex():
    config = PPEPostprocessConfig(prefer_helmet_on_head_overlap=True)
    detections = _Detections(
        boxes=[
            (20, 20, 80, 80),
            (210, 150, 290, 230),
            (180, 120, 330, 410),
        ],
        classes=[0, 1, 2],
        confidences=[0.82, 0.88, 0.91],
        names={0: "helmet", 1: "head", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, config=config, frame_shape=(640, 640))

    assert summary["head_count"] == 1
    assert summary["helmet_count"] == 1
    assert summary["candidate"] is True
    assert summary["helmet_fp_suppression"]["covered_head_indices"] == []
    assert summary["helmet_fp_suppression"]["suppressed_head_indices"] == []


def test_ppe_business_mutex_hides_existing_head_track_after_helmet_arrives():
    from defense.module_a.postprocess import PPEDisplayTracker

    config = PPEPostprocessConfig(prefer_helmet_on_head_overlap=True)
    state = SafetyHelmetState(window=2, trigger_count=1, fast_window=1, fast_trigger_count=1)
    tracker = PPEDisplayTracker(hold_frames=4, small_hold_frames=4)
    first = _Detections(
        boxes=[
            (120, 120, 220, 220),
            (80, 80, 260, 360),
        ],
        classes=[1, 2],
        confidences=[0.92, 0.90],
        names={0: "helmet", 1: "head", 2: "person"},
    )
    second = _Detections(
        boxes=[
            (100, 100, 180, 180),
            (120, 120, 220, 220),
            (80, 80, 260, 360),
        ],
        classes=[0, 1, 2],
        confidences=[0.78, 0.95, 0.90],
        names={0: "helmet", 1: "head", 2: "person"},
    )

    first_result = evaluate_ppe_business(
        first,
        frame_shape=(640, 640),
        ppe_state=state,
        ppe_tracker=tracker,
        tracking_enabled=True,
        postprocess_config=config,
    )
    second_result = evaluate_ppe_business(
        second,
        frame_shape=(640, 640),
        ppe_state=state,
        ppe_tracker=tracker,
        tracking_enabled=True,
        postprocess_config=config,
    )

    assert {track["label"] for track in first_result.tracks} == {"head", "person"}
    labels = {track["label"] for track in second_result.tracks}
    assert "head" not in labels
    assert {"helmet", "person"} <= labels
    assert second_result.ppe["raw_head_count"] == 1
    assert second_result.ppe["head_count"] == 0
    assert second_result.ppe["helmet_count"] == 1
    assert second_result.ppe["person_count"] == 1


def test_ppe_business_temporal_helmet_track_suppresses_raw_head_count():
    from defense.module_a.postprocess import PPEDisplayTracker

    config = PPEPostprocessConfig(prefer_helmet_on_head_overlap=True)
    state = SafetyHelmetState(window=2, trigger_count=1, fast_window=1, fast_trigger_count=1)
    tracker = PPEDisplayTracker(hold_frames=4, small_hold_frames=4)
    first = _Detections(
        boxes=[
            (272, 114, 334, 229),
            (224, 121, 387, 639),
        ],
        classes=[0, 2],
        confidences=[0.82, 0.90],
        names={0: "helmet", 1: "head", 2: "person"},
    )
    second = _Detections(
        boxes=[
            (272, 115, 334, 229),
            (224, 122, 388, 640),
        ],
        classes=[1, 2],
        confidences=[0.76, 0.88],
        names={0: "helmet", 1: "head", 2: "person"},
    )

    first_result = evaluate_ppe_business(
        first,
        frame_shape=(640, 640),
        ppe_state=state,
        ppe_tracker=tracker,
        tracking_enabled=True,
        postprocess_config=config,
    )
    second_result = evaluate_ppe_business(
        second,
        frame_shape=(640, 640),
        ppe_state=state,
        ppe_tracker=tracker,
        tracking_enabled=True,
        postprocess_config=config,
    )

    assert {track["label"] for track in first_result.tracks} == {"helmet", "person"}
    labels = {track["label"] for track in second_result.tracks}
    assert "head" not in labels
    assert {"helmet", "person"} <= labels
    assert second_result.ppe["raw_head_count"] == 1
    assert second_result.ppe["head_count"] == 0
    assert second_result.ppe["missing_helmet_count"] == 0
    assert second_result.ppe["candidate"] is False
    suppression = second_result.ppe["helmet_fp_suppression"]
    assert suppression["temporal_helmet_mutex_heads"][0]["reason"] == "temporal_helmet_mutex"


def test_source_auth_media_roi_suppresses_ppe_business_evidence():
    from defense.module_a.postprocess import PPEDisplayTracker

    detections = _Detections(
        boxes=[(140, 140, 230, 230), (120, 100, 260, 360)],
        classes=[1, 2],
        confidences=[0.92, 0.88],
        names={0: "helmet", 1: "head", 2: "person"},
    )
    state = SafetyHelmetState(window=2, trigger_count=1, fast_window=1, fast_trigger_count=1)
    tracker = PPEDisplayTracker(hold_frames=2, small_hold_frames=2)

    for _ in range(3):
        result = evaluate_ppe_business(
            detections,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
            source_auth_media_bbox=(100, 80, 300, 380),
            source_auth_suppression_active=True,
        )

    ppe = result.ppe
    assert ppe["candidate"] is False
    assert ppe["warning"] is False
    assert ppe["head_count"] == 0
    assert ppe["person_count"] == 0
    assert ppe["source_auth_media_suppression"]["suppressed_count"] == 2
    assert ppe["source_auth_temporal_reset"] is True
    assert result.tracks == []


def test_source_auth_media_roi_does_not_hide_external_head_violation():
    from defense.module_a.postprocess import PPEDisplayTracker

    detections = _Detections(
        boxes=[
            (140, 140, 230, 230),
            (120, 100, 260, 360),
            (420, 140, 500, 220),
            (390, 120, 540, 390),
        ],
        classes=[0, 2, 1, 2],
        confidences=[0.93, 0.86, 0.91, 0.88],
        names={0: "helmet", 1: "head", 2: "person"},
    )

    result = evaluate_ppe_business(
        detections,
        frame_shape=(640, 640),
        ppe_state=SafetyHelmetState(window=2, trigger_count=1, fast_window=1, fast_trigger_count=1),
        ppe_tracker=PPEDisplayTracker(hold_frames=2, small_hold_frames=2),
        tracking_enabled=True,
        source_auth_media_bbox=(100, 80, 300, 380),
        source_auth_suppression_active=True,
    )

    ppe = result.ppe
    assert ppe["candidate"] is True
    assert ppe["head_count"] == 1
    assert ppe["helmet_count"] == 0
    assert ppe["person_count"] == 1
    assert ppe["source_auth_media_suppression"]["suppressed_labels"] == ["helmet", "person"]
    assert {track["label"] for track in result.tracks} == {"head", "person"}


def test_overlay_record_exposes_a3b_bbox_and_source_auth_ppe_suppression():
    status = {
        "running": True,
        "a3b_p_media": 0.73,
        "a3b_bbox": [100, 80, 300, 380],
        "a3b_state": "suspect",
        "ppe_source_auth_media_suppressed": True,
        "ppe_source_auth_temporal_reset": True,
        "ppe_source_auth_media_bbox": [100, 80, 300, 380],
        "ppe_source_auth_media_suppressed_count": 2,
        "ppe_source_auth_media_suppressed_head_count": 1,
        "ppe_source_auth_media_suppressed_person_count": 1,
        "ppe_source_auth_media_suppression_reason": "a3b_media_roi",
    }

    record = build_overlay_record(
        status=status,
        ppe_tracks=[],
        run_id=7,
        display_options={},
    )

    assert record["a3b_p_media"] == 0.73
    assert record["a3b_bbox"] == [100, 80, 300, 380]
    assert record["ppe_source_auth_media_suppressed"] is True
    assert record["ppe_source_auth_temporal_reset"] is True
    assert record["ppe_source_auth_media_suppressed_count"] == 2
    assert record["ppe_source_auth_media_suppressed_head_count"] == 1
    assert record["ppe_source_auth_media_suppressed_person_count"] == 1


def test_empty_backend_preserves_configured_class_names():
    names = {0: "person", 1: "head", 2: "helmet"}
    backend = pipeline_factory.EmptyDetectorBackend(names)

    assert backend.names == names


def test_pipeline_cache_key_includes_custom_class_names(monkeypatch, tmp_path: Path):
    class WarmupPipeline:
        warmup_frames = 0

        def __init__(self, backend, *, config):
            self.backend = backend
            self.config = config

        def warmup(self, frames: int) -> None:
            return None

        def reset(self) -> None:
            return None

    def load_config(**kwargs):
        custom = kwargs.get("custom_model") or {}
        return {
            "runtime": {"allow_empty_backend": True},
            "inference": {
                "backend": "pytorch",
                "model_family": "yolov5",
                "class_names": custom.get("class_names"),
            },
        }

    monkeypatch.setattr(pipeline_factory, "VideoDefensePipeline", WarmupPipeline)
    monkeypatch.setattr(pipeline_factory, "load_runtime_config", load_config)

    cache = pipeline_factory.PipelineCache(root=tmp_path)
    first = cache.get(
        profile="empty_smoke",
        custom_model={
            "enabled": True,
            "path": "same.pt",
            "backend": "pytorch",
            "model_family": "yolov5",
            "class_names": ["person", "head", "helmet"],
        },
    )
    second = cache.get(
        profile="empty_smoke",
        custom_model={
            "enabled": True,
            "path": "same.pt",
            "backend": "pytorch",
            "model_family": "yolov5",
            "class_names": ["helmet", "head", "person"],
        },
    )

    assert first is not second
    assert second.cache_hit is False
    assert second.pipeline.backend.names == {0: "helmet", 1: "head", 2: "person"}


def test_empty_runtime_status_exposes_three_class_person_fields():
    status = MonitorEngine._empty_status()

    assert status["ppe_person_count"] == 0
    assert status["ppe_raw_person_count"] == 0
    assert status["ppe_inferred_person_count"] == 0
    assert status["ppe_person_context_count"] == 0
    assert status["ppe_weak_person_count"] == 0
    assert status["ppe_promoted_person_count"] == 0
    assert status["ppe_effective_person_count"] == 0


def test_external_contract_noise_does_not_drive_full_scan_asr():
    external = {
        "summary": {"n_rows": 3, "max_asr": 1.0, "mean_asr": 0.5, "asr_matrix": {}, "top_attacks": []},
        "rows": [
            {
                "suite": "mask_bd_external_eval",
                "attack": "badnet_oga_mask_bd_v2_visible_A_audited_clean_source",
                "goal": "oga",
                "success": True,
                "success_reason": "target_false_positive_on_negative",
            },
            {
                "suite": "mask_bd_external_eval",
                "attack": "badnet_oga_mask_bd_v2_visible_A_clean_negative_expanded_medium",
                "goal": "oga",
                "success": True,
                "success_reason": "target_false_positive_on_negative",
            },
            {
                "suite": "poison_benchmark_cuda_tuned_remap_v2",
                "attack": "semantic_green_cleanlabel",
                "goal": "semantic",
                "success": False,
                "success_reason": "semantic_gt_target_recalled",
            },
        ],
    }

    filtered = _filter_external_contract_noise(external, {"model_security": {"external_eval_strict_contract": False}})

    assert filtered["ignored_contract_rows"] == 2
    assert filtered["summary"]["n_rows"] == 1
    assert filtered["summary"]["max_asr"] == 0.0


@pytest.mark.parametrize(
    ("export_format", "expected_suffix", "expected_backend"),
    [("onnx", ".onnx", "onnx"), ("engine", ".engine", "tensorrt")],
)
def test_accelerated_export_writes_next_to_trusted_source_and_registers_lineage(
    tmp_path: Path,
    monkeypatch,
    export_format: str,
    expected_suffix: str,
    expected_backend: str,
):
    source = tmp_path / "clean.pt"
    source.write_bytes(b"trusted-source" * 128)
    svc = _offline_model_security_service(tmp_path)
    custom_model = {
        "enabled": True,
        "path": str(source),
        "backend": "pytorch",
        "model_family": "ultralytics",
        "source_pt_path": str(source),
    }
    fp = svc.current_fingerprint(custom_model=custom_model)
    report = ModelSecurityReport(
        fingerprint=fp.to_dict(),
        scan_type="full",
        status="clean",
        risk_score=0.0,
        source_model_path=str(source),
        source_model_hash=fp.model_hash,
    )
    svc._write_report(report)
    svc.registry.mark_trusted(
        fp.fingerprint,
        risk_score=0.0,
        report_path=report.report_path,
        runtime_model_hash=fp.model_hash,
        runtime_model_path=str(source),
        source_model_hash=fp.model_hash,
        source_model_path=str(source),
        scanner_version=fp.scanner_version,
        class_names_hash=fp.class_names_hash,
        ppe_mapping_hash=fp.ppe_mapping_hash,
        approval_source="full_scan",
    )

    def fake_export(*, source_pt: Path, target_path: Path, export_format: str) -> Path:
        assert export_format == expected_suffix.removeprefix(".")
        assert source_pt == source
        target_path.write_bytes(b"exported-onnx" * 128)
        return target_path

    monkeypatch.setattr(svc, "_run_export_tool", fake_export)

    result = svc.export_accelerated_model(export_format=export_format, custom_model=custom_model)

    assert result["state"] == "completed"
    assert result["backend"] == expected_backend
    exported_path = Path(result["exported_model_path"])
    assert exported_path.exists()
    assert exported_path.parent == source.parent
    assert exported_path.suffix == expected_suffix
    assert exported_path.stem.startswith(f"{source.stem}_")
    record = svc.registry.get(result["exported_fingerprint"])
    assert record is not None
    assert record.source_model_hash == fp.model_hash
    assert record.source_model_path == str(source)
    accelerated_status = svc.status(
        custom_model={
            "enabled": True,
            "path": str(exported_path),
            "backend": expected_backend,
            "model_family": "ultralytics",
        }
    )
    assert accelerated_status["allowed"] is True
    assert accelerated_status["source_pt_path"] == str(source)
    assert accelerated_status["source_pt_hash"] == fp.model_hash
    catalog = svc.output_catalog(category="accelerated_model")
    assert catalog["count"] == 1
    assert catalog["artifacts"][0]["status"] == "trusted"
    all_catalog = svc.runtime_catalog()
    assert all_catalog["count"] >= 2


@pytest.mark.parametrize(
    ("export_format", "expected_suffix", "expected_backend"),
    [("onnx", ".onnx", "onnx"), ("engine", ".engine", "tensorrt")],
)
def test_accelerated_export_from_purified_pt_binds_purified_and_original_lineage(
    tmp_path: Path,
    monkeypatch,
    export_format: str,
    expected_suffix: str,
    expected_backend: str,
):
    poisoned = tmp_path / "poisoned.pt"
    purified = tmp_path / "poisoned_净化完毕.pt"
    poisoned.write_bytes(b"poisoned-source" * 128)
    purified.write_bytes(b"trusted-purified" * 128)
    svc = _offline_model_security_service(tmp_path)
    custom_model = {
        "enabled": True,
        "path": str(purified),
        "backend": "pytorch",
        "model_family": "ultralytics",
        "source_pt_path": str(purified),
    }
    fp = svc.current_fingerprint(custom_model=custom_model)
    report = ModelSecurityReport(
        fingerprint=fp.to_dict(),
        scan_type="full",
        status="clean",
        risk_score=0.0,
        source_model_path=str(purified),
        source_model_hash=fp.model_hash,
    )
    svc._write_report(report)
    original_hash = "sha256:" + sha256_file(poisoned)
    svc.registry.mark_trusted(
        fp.fingerprint,
        risk_score=0.0,
        report_path=report.report_path,
        runtime_model_hash=fp.model_hash,
        runtime_model_path=str(purified),
        source_model_hash=fp.model_hash,
        source_model_path=str(purified),
        original_source_model_hash=original_hash,
        original_source_model_path=str(poisoned),
        scanner_version=fp.scanner_version,
        class_names_hash=fp.class_names_hash,
        ppe_mapping_hash=fp.ppe_mapping_hash,
        approval_source="purified_full_scan",
    )

    purified_status = svc.status(custom_model=custom_model)
    assert purified_status["allowed"] is True
    assert purified_status["admission_status"] == "trusted"

    poisoned_custom_model = {
        "enabled": True,
        "path": str(poisoned),
        "backend": "pytorch",
        "model_family": "ultralytics",
        "source_pt_path": str(poisoned),
    }
    poisoned_status = svc.status(custom_model=poisoned_custom_model)
    assert poisoned_status["allowed"] is False
    assert poisoned_status["admission_status"] not in {
        "trusted",
        "purified_alternative_available",
    }
    assert svc.trusted_purified_runtime_model(custom_model=poisoned_custom_model) is None

    def fake_export(*, source_pt: Path, target_path: Path, export_format: str) -> Path:
        assert source_pt == purified
        assert export_format == expected_suffix.removeprefix(".")
        target_path.write_bytes(b"purified-accelerated" * 128)
        return target_path

    monkeypatch.setattr(svc, "_run_export_tool", fake_export)

    result = svc.export_accelerated_model(export_format=export_format, custom_model=custom_model)

    exported_path = Path(result["exported_model_path"])
    assert exported_path.parent == purified.parent
    assert exported_path.stem.startswith(f"{purified.stem}_")
    assert exported_path.suffix == expected_suffix
    assert result["backend"] == expected_backend
    record = svc.registry.get(result["exported_fingerprint"])
    assert record is not None
    assert record.source_model_hash == fp.model_hash
    assert record.source_model_path == str(purified)
    assert record.original_source_model_hash == original_hash
    assert record.original_source_model_path == str(poisoned)

    accelerated_status = svc.status(
        custom_model={
            "enabled": True,
            "path": str(exported_path),
            "backend": expected_backend,
            "model_family": "ultralytics",
        }
    )
    assert accelerated_status["allowed"] is True
    assert accelerated_status["admission_status"] == "trusted"
    assert accelerated_status["source_pt_path"] == str(purified)


def test_legacy_purified_lineage_outside_source_directory_is_not_reused(tmp_path: Path):
    poisoned = tmp_path / "models" / "poisoned.pt"
    expected_purified = poisoned.with_name("poisoned_净化完毕.pt")
    legacy_purified = tmp_path / "runtime" / "model_security" / "purified" / "legacy.pt"
    poisoned.parent.mkdir(parents=True)
    legacy_purified.parent.mkdir(parents=True)
    poisoned.write_bytes(b"poisoned" * 128)
    legacy_purified.write_bytes(b"purified" * 128)
    service = _offline_model_security_service(tmp_path)

    legacy_record = SimpleNamespace(
        source_model_hash="sha256:purified",
        source_model_path=str(legacy_purified),
        original_source_model_hash="sha256:poisoned",
        original_source_model_path=str(poisoned),
        runtime_model_path=str(legacy_purified),
        backend="pytorch",
    )
    current_record = SimpleNamespace(
        source_model_hash="sha256:purified",
        source_model_path=str(expected_purified),
        original_source_model_hash="sha256:poisoned",
        original_source_model_path=str(poisoned),
        runtime_model_path=str(expected_purified),
        backend="pytorch",
    )

    assert service._trusted_lineage_path_current(legacy_record) is False
    assert service._trusted_lineage_path_current(current_record) is True


def test_clear_model_security_logs_removes_all_entries(tmp_path: Path):
    svc = ModelSecurityService(root=tmp_path)
    svc._log_event("scan_started", status="running", message="扫描开始")

    result = svc.clear_logs()

    assert result["cleared"] is True
    assert result["entries"] == []
    assert svc.recent_logs()["entries"] == []


def test_fastapi_model_security_status_exposes_class_name_warning():
    class FakeModelSecurity:
        def __init__(self) -> None:
            self.received_custom_model = None

        def status(self, **kwargs) -> dict:
            self.received_custom_model = kwargs.get("custom_model")
            return {
                "enabled": True,
                "allowed": False,
                "admission_status": "blocked_scan_required",
                "class_names": {0: "person", 1: "head", 2: "helmet"},
                "class_names_mismatch": True,
                "class_names_warning": "configured class_names differ from checkpoint embedded names",
                "class_names_diagnostics": {
                    "configured_class_names": {0: "person", 1: "head", 2: "helmet"},
                    "embedded_class_names": {0: "helmet", 1: "head", 2: "person"},
                },
            }

    security = FakeModelSecurity()
    app = create_app(engine=object(), model_security=security, bind_host="127.0.0.1")
    client = TestClient(app)
    custom_model = {
        "enabled": True,
        "path": "D:/tmp/best.pt",
        "class_names": ["person", "head", "helmet"],
    }

    res = client.post("/api/model-security/status", json={"custom_model": custom_model})

    assert res.status_code == 200
    payload = res.json()["model_security"]
    assert security.received_custom_model == custom_model
    assert payload["class_names_mismatch"] is True
    assert "differ" in payload["class_names_warning"]
    assert payload["class_names_diagnostics"]["embedded_class_names"]["2"] == "person"


def test_model_security_page_displays_class_name_warning():
    html_path = Path(__file__).resolve().parents[1] / "src" / "defense" / "web" / "static" / "model_security.html"
    html = html_path.read_text(encoding="utf-8")

    assert 'id="customClassNames"' in html
    assert 'id="classNames"' in html
    assert "class_names_warning" in html
    assert "WARNING:" in html
    assert "helmet,head,person" in html


def test_fastapi_start_can_bypass_model_security_for_test_only(tmp_path: Path):
    class FakeEngine:
        def __init__(self) -> None:
            self.run_id = 0
            self.started_with = None

        def get_status(self) -> dict:
            return {"run_id": self.run_id, "running": False}

        def start(self, **kwargs) -> int:
            self.run_id = 7
            self.started_with = kwargs
            return self.run_id

        def wait_ready_for_preview(self, run_id: int, *, timeout: float) -> dict:
            return {"run_id": run_id, "running": True, "ready_for_preview": True, "timeout": timeout}

    class FakeModelSecurity:
        def __init__(self) -> None:
            self.events = []
            self.purification_starts = []

        def status(self, **_kwargs) -> dict:
            return {
                "enabled": True,
                "allowed": False,
                "admission_status": "suspicious",
                "blocking_reason": "last_full_scan_suspicious",
                "can_purify": True,
            }

        def prepare_runtime_for_start(
            self,
            *,
            profile: str,
            custom_model: dict,
            auto_remediate: bool,
        ) -> dict:
            assert profile == "default"
            assert custom_model["path"] == "D:/tmp/poisoned.pt"
            assert auto_remediate is False
            return {
                "allowed": False,
                "custom_model": None,
                "model_security": self.status(),
                "scan": None,
                "purification": None,
                "runtime_replacement": None,
            }

        def start_background_purification(self, **kwargs) -> dict:
            self.purification_starts.append(kwargs)
            return {"started": True, "fingerprint": "sha256:poisoned", "scan_after": True}

        def _log_event(self, event: str, **fields) -> None:
            self.events.append((event, fields))

    engine = FakeEngine()
    security = FakeModelSecurity()
    app = create_app(
        config_path=_offline_model_security_config(tmp_path),
        engine=engine,
        model_security=security,
        bind_host="127.0.0.1",
    )
    client = TestClient(app)

    custom_model = {
        "enabled": True,
        "path": "D:/tmp/poisoned.pt",
        "backend": "pytorch",
        "model_family": "yolov5",
    }
    blocked = client.post(
        "/api/start",
        json={
            "source_type": "file",
            "source": "D:/tmp/source.mp4",
            "profile": "default",
            "custom_model": custom_model,
        },
    )

    assert blocked.status_code == 409
    blocked_payload = blocked.json()
    assert blocked_payload["error"] == "model_security_blocked"
    assert blocked_payload["model_security"]["admission_status"] == "suspicious"
    assert engine.started_with is None
    assert len(security.purification_starts) == 1

    res = client.post(
        "/api/test/start",
        json={
            "source_type": "file",
            "source": "D:/tmp/source.mp4",
            "profile": "default",
            "custom_model": custom_model,
            "test_bypass_model_security": True,
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert payload["model_security"]["admission_status"] == "bypassed_for_test"
    assert payload["model_security"]["test_bypass_model_security"] is True
    assert engine.started_with["custom_model"] == custom_model
    assert security.events[0][0] == "model_security_bypass_start"

    disabled_custom_model = {
        "enabled": False,
        "path": "D:/tmp/stale-old-model.pt",
        "backend": "auto",
        "model_family": "auto",
    }
    res = client.post(
        "/api/start",
        json={
            "source_type": "file",
            "source": "D:/tmp/source.mp4",
            "profile": "desktop_rtx",
            "custom_model": disabled_custom_model,
            "test_bypass_model_security": True,
        },
    )

    assert res.status_code == 403
    assert res.json()["error"] == "test_security_bypass_endpoint_required"
    assert len(security.events) == 1


def test_file_realtime_preview_drops_unmatched_interpolated_tracks():
    engine = MonitorEngine(object())
    engine.status.update(
        {
            "source_type": "file",
            "realtime": True,
            "detector_process_fps_cap": 15.0,
            "overlay_match_window_ms": 180.0,
            "overlay_hold_ms": 550.0,
            "overlay_interpolate_ms": 400.0,
            "overlay_max_age_ms": 950.0,
        }
    )
    engine.overlay_timeline.append(
        {
            "overlay_seq": 1,
            "source_epoch": 0,
            "video_time_s": 1.0,
            "ppe_tracks": [
                {
                    "track_id": 7,
                    "label": "person",
                    "box": [300, 180, 520, 360],
                    "source": "detected",
                    "hold_eligible": True,
                }
            ],
        }
    )
    engine.overlay_timeline.append(
        {
            "overlay_seq": 2,
            "source_epoch": 0,
            "video_time_s": 1.1,
            "ppe_tracks": [],
            "ppe_source_auth_media_suppressed": True,
        }
    )

    selected = engine._select_preview_overlay(1.05, 0)

    assert selected is not None
    assert selected.get("interpolated") is True
    assert selected["ppe_tracks"] == []
    assert engine._preview_last_overlay is not None
    assert engine._preview_last_overlay["ppe_tracks"] == []


def test_file_realtime_preview_bridge_window_is_configurable():
    engine = MonitorEngine(object())
    engine.status.update(
        {
            "source_type": "file",
            "realtime": True,
            "detector_process_fps_cap": 15.0,
            "overlay_match_window_ms": 180.0,
            "overlay_hold_ms": 550.0,
            "overlay_interpolate_ms": 400.0,
            "overlay_max_age_ms": 950.0,
            "file_realtime_overlay_bridge_frames": 3.8,
            "file_realtime_overlay_bridge_min_s": 0.24,
            "file_realtime_overlay_bridge_max_s": 0.40,
        }
    )
    engine.overlay_timeline.append(
        {
            "overlay_seq": 1,
            "source_epoch": 0,
            "video_time_s": 1.0,
            "ppe_tracks": [
                {
                    "track_id": 7,
                    "label": "head",
                    "box": [300, 180, 340, 225],
                    "source": "detected",
                    "hold_eligible": True,
                }
            ],
        }
    )

    first = engine._select_preview_overlay(1.0, 0)
    selected = engine._select_preview_overlay(1.23, 0)

    assert first is not None
    assert selected is not None
    assert selected.get("held") is True
    assert selected["ppe_tracks"][0]["source"] == "held"


def test_high_a3b_sensitivity_warns_after_two_observed_only_hits():
    config = {
        "module_a": {"static_image_enabled": True, "static_image_interval": 4},
        "a3b": {
            "observed_threshold": 0.42,
            "trigger_threshold": 0.62,
            "strong_single_frame_threshold": 0.78,
            "observed_only_warning_threshold": 0.50,
            "observed_only_track_threshold": 0.50,
            "observed_only_min_window_hits": 3,
        },
    }
    apply_feature_options(config, {"a3b_sensitivity": "high"})
    state = A3BSoftTriggerState(config["a3b"])

    result = None
    for score in [0.43, 0.44]:
        result = state.update(
            {
                "live_score": score,
                "score": score,
                "p_media": score,
                "p_media_scores": {"track": 0.43},
                "p_media_border_state": {"suppressed": False},
                "p_media_camera_motion_state": {"suppressed": False},
                "p_media_physical_motion_state": {"suppressed": False},
                "source_path": r"D:\security_project_d\素材\视频中出现干扰视频\case.mp4",
            }
        )

    assert result is not None
    assert config["module_a"]["static_image_interval"] == 2
    assert config["a3b"]["observed_only_min_window_hits"] == 2
    assert result["triggered"] is True
    assert result["triggered_source"] == "observed_window"
    assert result["state"] == "suspect"
    assert result["debug"]["observed_only_window_hits"] == 2


def test_fastapi_start_returns_json_when_source_file_is_missing(tmp_path: Path):
    class FakeEngine:
        run_id = 0

        def get_status(self) -> dict:
            return {"run_id": self.run_id, "running": False}

        def start(self, **_kwargs) -> int:
            raise FileNotFoundError("视频文件不存在或不可访问: D:/missing.mp4")

    class FakeModelSecurity:
        def prepare_runtime_for_start(
            self,
            *,
            profile: str,
            custom_model: dict,
            auto_remediate: bool,
        ) -> dict:
            assert profile == "default"
            assert auto_remediate is False
            return {
                "allowed": True,
                "custom_model": custom_model,
                "model_security": {
                    "enabled": True,
                    "allowed": True,
                    "admission_status": "trusted",
                    "blocking_reason": "",
                },
                "scan": None,
                "purification": None,
                "runtime_replacement": None,
            }

        def status(self, **_kwargs) -> dict:
            return {
                "enabled": True,
                "allowed": True,
                "admission_status": "trusted",
                "blocking_reason": "",
            }

    app = create_app(
        config_path=_offline_model_security_config(tmp_path),
        engine=FakeEngine(),
        model_security=FakeModelSecurity(),
        bind_host="127.0.0.1",
    )
    client = TestClient(app)

    res = client.post(
        "/api/start",
        json={
            "source_type": "file",
            "source": "D:/missing.mp4",
            "profile": "default",
        },
    )

    assert res.status_code == 400
    payload = res.json()
    assert payload["ok"] is False
    assert payload["error"] == "source_unavailable"
    assert "视频文件不存在" in payload["message"]
