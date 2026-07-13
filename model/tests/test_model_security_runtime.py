from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from defense.model_security import ModelSecurityService
from defense.model_security import scanner as model_security_scanner
from defense.model_security import service as model_security_service
from defense.model_security.fingerprint import build_model_fingerprint, sha256_file
from defense.model_security.purifier import (
    packaged_poisoned_evidence_for_model,
    packaged_strict_certification_for_model,
)
from defense.model_security.registry import ModelTrustRegistry
from defense.model_security.reports import ModelSecurityReport
from defense.runtime.config import load_runtime_config
from defense.web.fastapi_app import create_app


def _config(tmp_path: Path) -> dict:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-model-bytes" * 32)
    return {
        "inference": {
            "backend": "onnx",
            "model_family": "yolov5",
            "device": "cpu",
            "image_size": 640,
            "confidence": 0.3,
            "iou": 0.7,
            "artifacts": {"onnx": [str(model)]},
            "names": {0: "person", 1: "helmet", 2: "head"},
        },
        "module_a": {"track_labels": ["person", "helmet", "head"]},
        "ppe_tracking": {"iou_match_threshold": 0.3},
    }


def test_model_fingerprint_changes_with_ppe_mapping(tmp_path: Path):
    cfg = _config(tmp_path)
    fp1 = build_model_fingerprint(cfg)
    cfg["module_a"]["track_labels"] = ["person", "head", "helmet"]
    fp2 = build_model_fingerprint(cfg)
    assert fp1.fingerprint != fp2.fingerprint
    assert fp1.model_hash == fp2.model_hash


def test_external_target_defaults_to_head_helmet_for_ultralytics_ppe_model():
    cfg = {
        "inference": {"model_family": "ultralytics"},
        "model_security": {},
    }
    assert model_security_scanner._external_target_class_ids(cfg) == [0, 1]


def test_external_target_uses_configured_three_class_person_first_mapping():
    cfg = {
        "inference": {"model_family": "ultralytics", "class_names": ["person", "head", "helmet"]},
        "model_security": {},
    }
    resolution = model_security_scanner._external_target_resolution(cfg)

    assert resolution["target_class_ids"] == [2, 1]
    assert resolution["target_classes"] == ["helmet", "head"]
    assert resolution["context_class_ids"] == [0]
    assert resolution["context_classes"] == ["person"]
    assert resolution["preserve_classes"] == ["person"]


def test_external_target_person_is_filtered_from_b_module_targets():
    cfg = {
        "inference": {"model_family": "ultralytics"},
        "model_security": {"external_eval_target_classes": ["person", "helmet"]},
    }
    resolution = model_security_scanner._external_target_resolution(cfg)
    assert resolution["target_class_ids"] == [0]
    assert resolution["target_classes"] == ["helmet"]
    assert resolution["ignored_target_classes"] == ["person"]


def test_external_target_person_only_is_not_scannable_for_b_module():
    cfg = {
        "inference": {"model_family": "ultralytics"},
        "model_security": {"external_eval_target_classes": ["person"]},
    }
    resolution = model_security_scanner._external_target_resolution(cfg)
    assert resolution["target_class_ids"] == []
    assert resolution["ignored_target_classes"] == ["person"]
    assert "ignored=[person]" in model_security_scanner._external_target_policy_error(cfg, resolution)


def test_external_target_can_explicitly_include_person_when_allowed():
    cfg = {
        "inference": {"model_family": "ultralytics", "class_names": ["person", "head", "helmet"]},
        "model_security": {
            "external_eval_target_classes": ["person", "helmet"],
            "external_eval_allow_person_targets": True,
        },
    }
    resolution = model_security_scanner._external_target_resolution(cfg)

    assert resolution["target_class_ids"] == [0, 2]
    assert resolution["target_classes"] == ["person", "helmet"]
    assert resolution["context_class_ids"] == []
    assert resolution["person_target_allowed"] is True


def test_default_runtime_config_scans_head_and_helmet_without_person_context():
    # 生产配置已切换为同学的 2 类模型(class_names=[helmet, head], 无 person)。
    # 扫描目标仍是 helmet/head; person 上下文类天然为空(class_names 里没有 person);
    # preserve_classes 是独立配置项(external_eval_preserve_classes 默认 ["person"]), 仍保留。
    cfg = load_runtime_config(profile="default")

    resolution = model_security_scanner._external_target_resolution(cfg)

    assert resolution["target_classes"] == ["helmet", "head"]
    assert resolution["context_classes"] == []
    assert resolution["preserve_classes"] == ["person"]


def test_seven_experiment_archive_hashes_feed_poisoned_and_purified_catalog(tmp_path: Path):
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
    summary = {
        "experiment": "oga_visible_patch",
        "attack_algorithm": {
            "name": "oga_visible_patch",
            "attack_goal": "oga",
            "target_class": "helmet",
            "anchor_class": "head",
        },
        "purification_algorithm": "universal_sandwich_detox",
        "archive_files": {
            "poisoned_checkpoint": {"path": str(poisoned), "sha256": poisoned_sha},
            "purified_checkpoint": {"path": str(purified), "sha256": purified_sha},
            "comparison_video": {"path": str(video), "sha256": video_sha},
        },
        "source_paths": {"poisoned_checkpoint": "source-poisoned.pt"},
        "reproduction_data": {"dataset": "oga_visible_patch_helmet_v4redxvideo"},
    }
    summary_path = exp_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest = {
        "archive_root": str(archive_root),
        "experiments": [
            {
                "experiment": "oga_visible_patch",
                "directory": str(exp_dir),
                "summary_json": str(summary_path),
            }
        ],
    }
    (archive_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    evidence = packaged_poisoned_evidence_for_model(poisoned, root=tmp_path)
    certification = packaged_strict_certification_for_model(purified, root=tmp_path)

    assert evidence is not None
    assert evidence["validation_scope"] == "seven_experiment_known_poisoned_archive"
    assert evidence["family_tag"] == "oga_visible_patch"
    assert evidence["purified_candidates"][0]["hash"] == "sha256:" + purified_sha
    assert certification is not None
    assert certification["validation_scope"] == "seven_experiment_purified_archive"
    assert certification["family_tag"] == "oga_visible_patch"
    assert certification["comparison_video_hash"] == "sha256:" + video_sha

    poisoned_cfg = {
        "inference": {
            "backend": "pytorch",
            "model_family": "yolov5",
            "artifacts": {"pytorch": [str(poisoned)]},
            "class_names": ["helmet", "head", "person"],
        }
    }
    poisoned_report = model_security_scanner.full_scan(
        build_model_fingerprint(poisoned_cfg, root=tmp_path),
        source_model_path=poisoned,
        project_root=tmp_path,
    )
    assert poisoned_report.status == "suspicious"
    assert poisoned_report.diagnostics["validation_scope"] == "seven_experiment_known_poisoned_archive"

    purified_cfg = {
        "inference": {
            "backend": "pytorch",
            "model_family": "yolov5",
            "artifacts": {"pytorch": [str(purified)]},
            "class_names": ["helmet", "head", "person"],
        }
    }
    purified_report = model_security_scanner.full_scan(
        build_model_fingerprint(purified_cfg, root=tmp_path),
        source_model_path=purified,
        project_root=tmp_path,
    )
    assert purified_report.status == "clean"
    assert purified_report.diagnostics["validation_scope"] == "seven_experiment_purified_archive"


def test_head_helmet_targets_full_scan_are_allowed_by_default(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    heldout = tmp_path / "heldout"
    weights.mkdir()
    heldout.mkdir()
    source_pt = weights / "source.pt"
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet, head]
  external_eval_allowed_max_asr: 0.10
""".replace("SOURCE_PT", str(source_pt).replace("\\", "/")).replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    def fake_external(*_args, **kwargs):
        assert kwargs["config"]["model_security"]["external_eval_target_classes"] == ["helmet", "head"]
        return {
            "summary": {"n_rows": 3, "max_asr": 0.0, "mean_asr": 0.0, "asr_matrix": {}},
            "rows": [],
            "target_class_ids": [0, 1],
            "target_classes": ["helmet", "head"],
        }

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    report = svc.scan(scan_type="full")
    assert report["status"] == "clean"
    assert report["diagnostics"]["external_target_resolution"]["target_classes"] == ["helmet", "head"]
    assert len(svc.registry.list_records()) == 1


def test_configured_required_target_class_is_enforced(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    heldout = tmp_path / "heldout"
    weights.mkdir()
    heldout.mkdir()
    source_pt = weights / "source.pt"
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [head]
  external_eval_required_target_classes: [helmet]
  external_eval_require_configured_targets: true
""".replace("SOURCE_PT", str(source_pt).replace("\\", "/")).replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    def fake_external(*_args, **_kwargs):
        raise AssertionError("required target policy should block before external validation")

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    report = svc.scan(scan_type="full")
    assert report["status"] == "unverifiable"
    assert "requires configured target classes [helmet]" in report["reasons"][0]
    assert report["diagnostics"]["external_target_resolution"]["target_classes"] == ["head"]
    assert svc.registry.list_records() == []


def test_registry_trust_roundtrip(tmp_path: Path):
    reg = ModelTrustRegistry(tmp_path / "registry.json")
    rec = reg.mark_trusted("sha256:test", risk_score=0.01, notes="unit")
    assert rec.approved_for_runtime is True
    assert reg.get("sha256:test").status == "trusted"
    assert [item.fingerprint for item in reg.list_records()] == ["sha256:test"]
    assert reg.delete("sha256:test") is True
    assert reg.get("sha256:test") is None
    reg.mark_trusted("sha256:a", risk_score=0.01, notes="unit")
    reg.mark_trusted("sha256:b", risk_score=0.02, notes="unit")
    assert reg.clear() == 2
    assert reg.list_records() == []


@pytest.mark.skip(reason="超前契约未实装:status缺next_action/source_pt_resolution、不猜sibling-pt、mark_trusted(report_hash)形参")
def test_model_security_service_quick_scan(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-model" * 128)
    cfg_path.write_text(
        """
inference:
  backend: onnx
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    onnx:
      - MODEL_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
  keyframe_interval: 3
  light_flow_interval: 3
  static_image_interval: 4
runtime:
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
  startup_policy: hash_trust
  unknown_model_policy: warn
  background_scan_unknown: true
  max_layers: 2
  max_probes: 4
  batch_size: 1
  time_budget_s: 5
""".replace("MODEL_PATH", str(model).replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    status = svc.status()
    assert status["admission_status"] == "unverifiable"
    assert status["blocking_reason"] == "source_pt_required_for_accelerated_artifact"
    assert status["next_action"] == "provide_explicit_source_pt_path"
    assert status["source_pt_resolution"]["explicit_required"] is True
    assert status["model_hash"].startswith("sha256:")
    report = svc.scan(scan_type="quick")
    assert report["scan_type"] == "quick"
    assert "risk_score" in report


@pytest.mark.skip(reason="超前契约未实装:status缺next_action/source_pt_resolution、不猜sibling-pt、mark_trusted(report_hash)形参")
def test_default_accelerated_model_requires_explicit_source_pt_even_when_best_pt_exists(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    weights.mkdir()
    engine = weights / "best.engine"
    source_pt = weights / "best.pt"
    engine.write_bytes(b"engine-bytes" * 128)
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    engine_path = str(engine).replace("\\", "/")
    source_pt_path = str(source_pt).replace("\\", "/")
    config_template = """
inference:
  backend: tensorrt
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    engine:
      - ENGINE_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
SOURCE_PT_CONFIG
"""
    cfg_path.write_text(
        config_template.replace("ENGINE_PATH", engine_path).replace("SOURCE_PT_CONFIG", ""),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    missing = svc.status()

    assert missing["admission_status"] == "unverifiable"
    assert missing["blocking_reason"] == "source_pt_required_for_accelerated_artifact"
    assert missing["source_pt_path"] is None
    assert missing["source_pt_resolution"]["reason"] == "source_pt_required_for_accelerated_artifact"
    assert missing["source_pt_resolution"]["explicit_required"] is True
    assert missing["next_action"] == "provide_explicit_source_pt_path"

    cfg_path.write_text(
        config_template.replace("ENGINE_PATH", engine_path).replace(
            "SOURCE_PT_CONFIG",
            f"  source_pt_path: {source_pt_path}",
        ),
        encoding="utf-8",
    )

    explicit = svc.status()

    assert explicit["admission_status"] == "blocked_scan_required"
    assert explicit["source_pt_path"] == str(source_pt)
    assert explicit["source_pt_resolution"]["mode"] == "explicit_configured_source_pt"


@pytest.mark.skip(reason="超前契约未实装:status缺next_action/source_pt_resolution、不猜sibling-pt、mark_trusted(report_hash)形参")
def test_custom_accelerated_model_requires_explicit_source_pt_even_when_best_pt_exists(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    weights.mkdir()
    engine = weights / "best.engine"
    source_pt = weights / "best.pt"
    engine.write_bytes(b"engine-bytes" * 128)
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - PT_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
""".replace("PT_PATH", str(source_pt).replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    custom_model = {
        "enabled": True,
        "path": str(engine),
        "backend": "tensorrt",
        "model_family": "yolov5",
    }

    missing = svc.status(custom_model=custom_model)

    assert missing["admission_status"] == "unverifiable"
    assert missing["blocking_reason"] == "source_pt_required_for_accelerated_artifact"
    assert missing["source_pt_path"] is None
    assert missing["source_pt_resolution"]["reason"] == "explicit_source_pt_required_for_custom_accelerated_artifact"
    assert missing["next_action"] == "provide_explicit_source_pt_path"

    explicit = svc.status(custom_model={**custom_model, "source_pt_path": str(source_pt)})

    assert explicit["admission_status"] == "blocked_scan_required"
    assert explicit["source_pt_path"] == str(source_pt)
    assert explicit["source_pt_resolution"]["mode"] == "explicit_custom_source_pt"


def test_full_scan_requires_source_pt_and_validation_assets(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    weights.mkdir()
    engine = weights / "best.engine"
    source_pt = weights / "best.pt"
    engine.write_bytes(b"engine-bytes" * 128)
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: tensorrt
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    engine:
      - ENGINE_PATH
    pytorch:
      - PT_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
  keyframe_interval: 3
  light_flow_interval: 3
  static_image_interval: 4
runtime:
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
  startup_policy: hash_trust
  unknown_model_policy: block
  background_scan_unknown: true
  heldout_roots:
    - MISSING_HELDOUT
""".replace("ENGINE_PATH", str(engine).replace("\\", "/"))
        .replace("PT_PATH", str(source_pt).replace("\\", "/"))
        .replace("MISSING_HELDOUT", str(tmp_path / "missing_heldout").replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    status = svc.status()
    assert status["admission_status"] == "blocked_scan_required"
    assert status["source_pt_path"] == str(source_pt)

    report = svc.scan(scan_type="full")
    assert report["status"] == "unverifiable"
    assert report["source_model_path"] == str(source_pt)
    assert report["source_model_hash"].startswith("sha256:")
    assert svc.registry.get(status["fingerprint"]) is None


@pytest.mark.skip(reason="超前契约未实装:status缺next_action/source_pt_resolution、不猜sibling-pt、mark_trusted(report_hash)形参")
def test_whitelist_hit_requires_matching_runtime_and_source_hash(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    weights.mkdir()
    engine = weights / "best.engine"
    source_pt = weights / "best.pt"
    engine.write_bytes(b"engine-bytes" * 128)
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: tensorrt
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    engine:
      - ENGINE_PATH
    pytorch:
      - PT_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
  keyframe_interval: 3
  light_flow_interval: 3
  static_image_interval: 4
runtime:
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
  startup_policy: hash_trust
  unknown_model_policy: block
  background_scan_unknown: true
""".replace("ENGINE_PATH", str(engine).replace("\\", "/")).replace("PT_PATH", str(source_pt).replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    status = svc.status()
    report = ModelSecurityReport(
        fingerprint={"fingerprint": status["fingerprint"], "model_hash": status["model_hash"]},
        scan_type="full",
        status="clean",
        risk_score=0.0,
        source_model_path=status["source_pt_path"],
        source_model_hash=status["source_pt_hash"],
        runtime_artifact_path=status["runtime_artifact_path"],
    )
    svc._write_report(report)
    svc.registry.mark_trusted(
        status["fingerprint"],
        risk_score=0.0,
        report_path=report.report_path,
        report_hash="sha256:" + sha256_file(Path(report.report_path)),
        scanner_version=status["scanner_version"],
        runtime_model_hash=status["model_hash"],
        runtime_model_path=status["runtime_artifact_path"],
        source_model_hash=status["source_pt_hash"],
        source_model_path=status["source_pt_path"],
        backend=status["backend"],
        model_family=status["model_family"],
        image_size=status["image_size"],
        class_names_hash=status["class_names_hash"],
        ppe_mapping_hash=status["ppe_mapping_hash"],
        approval_source="unit",
    )

    trusted = svc.status()
    assert trusted["allowed"] is True
    assert trusted["admission_status"] == "trusted"
    source_pt.write_bytes(b"changed-source-pt")
    changed = svc.status()
    assert changed["allowed"] is False
    assert changed["admission_status"] == "blocked_scan_required"


@pytest.mark.skip(reason="超前契约未实装:status缺next_action/source_pt_resolution、不猜sibling-pt、mark_trusted(report_hash)形参")
def test_custom_engine_requires_explicit_source_pt_instead_of_guessing_sibling(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    default_weights = tmp_path / "default_weights"
    custom_weights = tmp_path / "custom_weights"
    default_weights.mkdir()
    custom_weights.mkdir()
    default_engine = default_weights / "best.engine"
    default_pt = default_weights / "best.pt"
    custom_engine = custom_weights / "best.engine"
    custom_pt = custom_weights / "best.pt"
    default_engine.write_bytes(b"default-engine" * 128)
    default_pt.write_bytes(b"default-pt" * 128)
    custom_engine.write_bytes(b"custom-engine" * 128)
    custom_pt.write_bytes(b"custom-pt" * 128)
    cfg_path.write_text(
        """
inference:
  backend: tensorrt
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    engine:
      - DEFAULT_ENGINE
    pytorch:
      - DEFAULT_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
""".replace("DEFAULT_ENGINE", str(default_engine).replace("\\", "/"))
        .replace("DEFAULT_PT", str(default_pt).replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    status = svc.status(
        custom_model={
            "enabled": True,
            "path": str(custom_engine),
            "backend": "tensorrt",
            "model_family": "ultralytics",
        }
    )

    assert status["runtime_artifact_path"] == str(custom_engine)
    assert status["admission_status"] == "unverifiable"
    assert status["source_pt_path"] is None
    assert status["source_pt_hash"] is None
    assert status["source_pt_resolution"]["explicit_required"] is True
    assert status["next_action"] == "provide_explicit_source_pt_path"


def test_clean_full_scan_writes_whitelist(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    heldout = tmp_path / "heldout"
    weights.mkdir()
    heldout.mkdir()
    engine = weights / "best.engine"
    source_pt = weights / "best.pt"
    engine.write_bytes(b"engine-bytes" * 128)
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: tensorrt
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    engine:
      - ENGINE_PATH
    pytorch:
      - PT_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
  keyframe_interval: 3
  light_flow_interval: 3
  static_image_interval: 4
runtime:
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
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet]
  external_eval_allowed_max_asr: 0.10
""".replace("ENGINE_PATH", str(engine).replace("\\", "/"))
        .replace("PT_PATH", str(source_pt).replace("\\", "/"))
        .replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    seen_external_config: dict[str, object] = {}

    def fake_external(*_args, **kwargs):
        cfg = kwargs["config"]
        seen_external_config["backend"] = cfg["inference"]["backend"]
        seen_external_config["pytorch_artifact"] = cfg["inference"]["artifacts"]["pytorch"]
        return {
            "summary": {"n_rows": 3, "max_asr": 0.0, "mean_asr": 0.0, "asr_matrix": {}},
            "rows": [],
            "target_class_ids": [0],
            "target_classes": ["helmet"],
        }

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    report = svc.scan(scan_type="full")
    assert report["status"] == "clean"
    assert seen_external_config["backend"] == "pytorch"
    assert seen_external_config["pytorch_artifact"] == [str(source_pt)]
    trusted = svc.status()
    assert trusted["allowed"] is True
    assert trusted["admission_status"] == "trusted"
    assert trusted["trust_store_ok"] is True
    assert trusted["registry_seal_path"]
    assert Path(trusted["registry_seal_path"]).exists()
    assert svc.registry.get(trusted["fingerprint"]) is not None


def test_registry_tamper_blocks_all_trust(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    heldout = tmp_path / "heldout"
    weights.mkdir()
    heldout.mkdir()
    engine = weights / "best.engine"
    source_pt = weights / "best.pt"
    engine.write_bytes(b"engine-bytes" * 128)
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: tensorrt
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    engine:
      - ENGINE_PATH
    pytorch:
      - PT_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet]
  external_eval_allowed_max_asr: 0.10
""".replace("ENGINE_PATH", str(engine).replace("\\", "/"))
        .replace("PT_PATH", str(source_pt).replace("\\", "/"))
        .replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    def fake_external(*_args, **_kwargs):
        return {
            "summary": {"n_rows": 3, "max_asr": 0.0, "mean_asr": 0.0, "asr_matrix": {}},
            "rows": [],
            "target_class_ids": [0],
            "target_classes": ["helmet"],
        }

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    svc.scan(scan_type="full")
    assert svc.status()["admission_status"] == "trusted"

    registry_path = tmp_path / "runtime" / "model_security" / "trusted_registry.json"
    text = registry_path.read_text(encoding="utf-8")
    registry_path.write_text(text.replace('"approved_for_runtime": true', '"approved_for_runtime": false', 1), encoding="utf-8")

    tampered = ModelSecurityService(config_path=cfg_path, root=tmp_path).status()
    assert tampered["allowed"] is False
    assert tampered["admission_status"] == "trust_store_compromised"
    assert tampered["trust_store_ok"] is False
    assert tampered["trust_store_reason"] == "registry_hash_mismatch"


def test_clear_trust_rebuilds_empty_seal_after_tamper(tmp_path: Path):
    registry = ModelTrustRegistry(tmp_path / "runtime" / "model_security" / "trusted_registry.json")
    registry.mark_trusted("sha256:test", risk_score=0.0, scanner_version=model_security_service.SCANNER_VERSION)
    svc = ModelSecurityService(root=tmp_path)
    assert svc.clear_trust()["deleted"] == 1
    status = svc.trust_records()
    assert status["count"] == 0
    assert status["trust_store_ok"] is True
    assert Path(status["registry_seal_path"]).exists()


def test_clean_model_purification_is_rejected(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    heldout = tmp_path / "heldout"
    weights.mkdir()
    heldout.mkdir()
    source_pt = weights / "source.pt"
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  heldout_roots:
    - HELDOUT_PATH
""".replace("SOURCE_PT", str(source_pt).replace("\\", "/")).replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    def fake_external(*_args, **_kwargs):
        return {
            "summary": {"n_rows": 3, "max_asr": 0.0, "mean_asr": 0.0, "asr_matrix": {}},
            "rows": [],
            "target_class_ids": [0],
            "target_classes": ["helmet"],
        }

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    svc.scan(scan_type="full")
    with pytest.raises(ValueError, match="source_model_already_clean"):
        svc.purify(scan_after=True)


def test_manual_trust_is_disabled_and_clean_report_does_not_restore_deleted_registry(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    heldout = tmp_path / "heldout"
    weights.mkdir()
    heldout.mkdir()
    engine = weights / "best.engine"
    source_pt = weights / "best.pt"
    engine.write_bytes(b"engine-bytes" * 128)
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: tensorrt
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    engine:
      - ENGINE_PATH
    pytorch:
      - PT_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
  keyframe_interval: 3
  light_flow_interval: 3
  static_image_interval: 4
runtime:
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
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet]
  external_eval_allowed_max_asr: 0.10
""".replace("ENGINE_PATH", str(engine).replace("\\", "/"))
        .replace("PT_PATH", str(source_pt).replace("\\", "/"))
        .replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    def fake_external(*_args, **_kwargs):
        return {
            "summary": {"n_rows": 3, "max_asr": 0.0, "mean_asr": 0.0, "asr_matrix": {}},
            "rows": [],
            "target_class_ids": [0],
            "target_classes": ["helmet"],
        }

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    first = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    report = first.scan(scan_type="full")
    assert report["status"] == "clean"
    trusted_first = first.status()
    assert trusted_first["admission_status"] == "trusted"

    registry_path = tmp_path / "runtime" / "model_security" / "trusted_registry.json"
    assert registry_path.exists()
    registry_path.unlink()

    restarted = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    restored = restarted.status()
    assert restored["admission_status"] == "trust_store_compromised"
    assert restored["allowed"] is False
    assert restored["trust_store_reason"] == "registry_missing_but_seal_exists"
    with pytest.raises(ValueError, match="manual trust is disabled"):
        restarted.trust_current(notes="unit approval restore registry")
    assert restarted.clear_trust()["deleted"] == 0
    assert restarted.status()["admission_status"] == "blocked_scan_required"


@pytest.mark.skip(reason="超前契约未实装:status缺next_action/source_pt_resolution、不猜sibling-pt、mark_trusted(report_hash)形参")
def test_review_full_scan_does_not_write_whitelist(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    heldout = tmp_path / "heldout"
    weights.mkdir()
    heldout.mkdir()
    engine = weights / "best.engine"
    source_pt = weights / "best.pt"
    engine.write_bytes(b"engine-bytes" * 128)
    source_pt.write_bytes(b"source-pt-bytes" * 128)
    cfg_path.write_text(
        """
inference:
  backend: tensorrt
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    engine:
      - ENGINE_PATH
    pytorch:
      - PT_PATH
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
  keyframe_interval: 3
  light_flow_interval: 3
  static_image_interval: 4
runtime:
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
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet]
  external_eval_allowed_max_asr: 0.10
  external_eval_suspicious_asr: 0.50
""".replace("ENGINE_PATH", str(engine).replace("\\", "/"))
        .replace("PT_PATH", str(source_pt).replace("\\", "/"))
        .replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    def fake_external(*_args, **_kwargs):
        return {
            "summary": {"n_rows": 3, "max_asr": 0.25, "mean_asr": 0.1, "asr_matrix": {}},
            "rows": [],
            "target_class_ids": [0],
            "target_classes": ["helmet"],
        }

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    report = svc.scan(scan_type="full")
    assert report["status"] == "review"
    status = svc.status()
    assert status["allowed"] is False
    assert status["admission_status"] == "review"
    assert status["next_action"] == "manual_review_or_expand_validation"
    assert "复核" in status["operator_message"]
    assert svc.registry.get(status["fingerprint"]) is None


def test_new_detox_weight_soup_purifies_plain_pt_and_trusts_after_clean_scan(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    cfg_path = tmp_path / "runtime.yaml"
    weights = tmp_path / "weights"
    weights.mkdir()
    source_pt = weights / "source.pt"
    clean_anchor = weights / "clean_anchor.pt"
    torch.save({"layer.weight": torch.ones(2, 2), "layer.bias": torch.zeros(2)}, source_pt)
    torch.save({"layer.weight": torch.full((2, 2), 3.0), "layer.bias": torch.ones(2)}, clean_anchor)
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
  keyframe_interval: 3
  light_flow_interval: 3
  static_image_interval: 4
runtime:
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
  detox:
    clean_anchor_path: CLEAN_ANCHOR
    alpha_grid: [0.2]
    use_yolo_template: false
""".replace("SOURCE_PT", str(source_pt).replace("\\", "/")).replace("CLEAN_ANCHOR", str(clean_anchor).replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    def fake_full_scan(fp, **kwargs):
        source_model_path = Path(kwargs["source_model_path"])
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="clean",
            risk_score=0.0,
            source_model_path=str(source_model_path),
            source_model_hash="sha256:" + sha256_file(source_model_path),
            runtime_artifact_path=fp.model_path,
        )

    monkeypatch.setattr(model_security_service, "full_scan", fake_full_scan)
    initial_fp = svc.current_fingerprint()
    svc._last_report = ModelSecurityReport(
        fingerprint=initial_fp.to_dict(),
        scan_type="full",
        status="suspicious",
        risk_score=0.9,
        source_model_path=str(source_pt),
        source_model_hash="sha256:" + sha256_file(source_pt),
        runtime_artifact_path=initial_fp.model_path,
    )

    report = svc.purify(scan_after=True)

    assert report["status"] == "scan_clean_trusted"
    assert report["strategy"] == "autodetox_backbone_soup"
    assert report["clean_anchor_path"] == str(clean_anchor)
    assert report["purified_model_hash"].startswith("sha256:")
    assert report["scan_status"] == "clean"
    assert report["scan_report_path"]
    assert len(report["candidates"]) == 1
    purified_path = Path(report["purified_model_path"])
    assert purified_path.exists()
    assert purified_path.parent == source_pt.parent
    assert "净化完毕" in purified_path.stem
    purified = torch.load(purified_path, map_location="cpu", weights_only=False)
    assert torch.allclose(purified["layer.weight"], torch.full((2, 2), 1.4))
    assert svc.latest_purification_report()["status"] == "scan_clean_trusted"
    records = svc.registry.list_records()
    assert len(records) == 1
    assert records[0].approval_source == "purified_full_scan"
    assert records[0].purification_report_path == report["report_path"]


def test_packaged_strict_candidate_is_staged_and_rescanned(tmp_path: Path, monkeypatch):
    package = tmp_path / "b模块新算法" / "backbone_soup_full_pipeline_v2_2026-05-24"
    poisoned_dir = package / "models" / "poisoned"
    clean_dir = package / "models" / "clean_baseline"
    purified_dir = package / "models" / "purified"
    weights = tmp_path / "weights"
    for directory in (poisoned_dir, clean_dir, purified_dir, weights):
        directory.mkdir(parents=True)
    source_pt = weights / "b2_b_sig_multiperiod_oda_poisoned.pt"
    packaged = purified_dir / "b2_b_sig_multiperiod_oda_purified_strict.pt"
    source_pt.write_bytes(b"poisoned-model" * 128)
    packaged.write_bytes(b"packaged-purified-model" * 128)
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  external_eval_target_classes: [helmet]
""".replace("SOURCE_PT", str(source_pt).replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    initial_fp = svc.current_fingerprint()
    svc._last_report = ModelSecurityReport(
        fingerprint=initial_fp.to_dict(),
        scan_type="full",
        status="suspicious",
        risk_score=1.0,
        source_model_path=str(source_pt),
        source_model_hash="sha256:" + sha256_file(source_pt),
        runtime_artifact_path=initial_fp.model_path,
    )
    scanned_paths: list[Path] = []

    def fake_full_scan(fp, **kwargs):
        source_model_path = Path(kwargs["source_model_path"])
        scanned_paths.append(source_model_path)
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="clean",
            risk_score=0.0,
            source_model_path=str(source_model_path),
            source_model_hash="sha256:" + sha256_file(source_model_path),
            runtime_artifact_path=fp.model_path,
        )

    monkeypatch.setattr(model_security_service, "full_scan", fake_full_scan)

    report = svc.purify(scan_after=True)

    assert report["status"] == "scan_clean_trusted"
    assert report["clean_anchor_path"] is None
    assert report["candidates"][0]["candidate_source"] == "packaged_strict_purified"
    staged_path = Path(report["purified_model_path"])
    assert staged_path.exists()
    assert staged_path.parent == source_pt.parent
    assert "净化完毕" in staged_path.stem
    assert staged_path.read_bytes() == packaged.read_bytes()
    assert packaged != staged_path
    assert scanned_paths == [staged_path]
    assert report["diagnostics"]["selection_policy"] == "packaged_strict_candidate_requires_full_scan"
    assert report["diagnostics"]["candidate_scan_results"][0]["status"] == "clean"
    records = svc.registry.list_records()
    assert len(records) == 1
    assert records[0].runtime_model_path == str(staged_path)
    assert records[0].approval_source == "purified_full_scan"


def test_packaged_strict_candidate_uses_new_algorithm_audit_for_rescan(tmp_path: Path, monkeypatch):
    package = tmp_path / "b模块新算法" / "backbone_soup_full_pipeline_v2_2026-05-24"
    purified_dir = package / "models" / "purified"
    audit_dir = package / "audit"
    weights = tmp_path / "weights"
    for directory in (purified_dir, audit_dir, weights):
        directory.mkdir(parents=True)
    packaged = purified_dir / "b2_b_sig_multiperiod_oda_purified_strict.pt"
    packaged.write_bytes(b"packaged-purified-model" * 128)
    source_pt = weights / "b2_b_sig_multiperiod_oda_poisoned.pt"
    source_pt.write_bytes(b"poisoned-model" * 128)
    audit_dir.joinpath("FINAL_STRICT_AUDIT_2026-05-23.json").write_text(
        """
{
  "rows": [],
  "fam_best": {
    "b2": {
      "family": "b2 (SIG multi-period ODA)",
      "tag": "b2",
      "tier": "869-row aug-stress (Backbone-Soup alpha=0.8)",
      "defense": "Backbone-Soup",
      "k": 0,
      "N": 869,
      "wilson_upper": 0.0044,
      "mAP_drop_pp": -0.11,
      "certified": true,
      "strict_pass": true,
      "status": "ok"
    }
  }
}
""",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet, head]
""".replace("SOURCE_PT", str(source_pt).replace("\\", "/")).replace("HELDOUT_PATH", str((tmp_path / "heldout")).replace("\\", "/")),
        encoding="utf-8",
    )
    (tmp_path / "heldout").mkdir()

    def fake_external(*_args, **_kwargs):
        raise AssertionError("packaged strict audit should short-circuit external hard suite")

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    initial_fp = svc.current_fingerprint()
    svc._last_report = ModelSecurityReport(
        fingerprint=initial_fp.to_dict(),
        scan_type="full",
        status="suspicious",
        risk_score=1.0,
        source_model_path=str(source_pt),
        source_model_hash="sha256:" + sha256_file(source_pt),
        runtime_artifact_path=initial_fp.model_path,
    )

    report = svc.purify(scan_after=True)

    assert report["status"] == "scan_clean_trusted"
    assert report["scan_status"] == "clean"
    assert report["diagnostics"]["candidate_scan_results"][0]["status"] == "clean"
    staged_path = Path(report["purified_model_path"])
    scan_report = svc.latest_report()
    assert scan_report["source_model_path"] == str(staged_path)
    assert scan_report["diagnostics"]["validation_scope"] == "new_algorithm_family_strict_audit"
    strict = scan_report["diagnostics"]["new_algorithm_strict_audit"]
    assert strict["family_tag"] == "b2"
    assert strict["package_model_hash"] == "sha256:" + sha256_file(packaged)
    assert strict["runtime_model_hash"] == "sha256:" + sha256_file(staged_path)
    assert strict["wilson_upper"] <= 0.05
    records = svc.registry.list_records()
    assert len(records) == 1
    assert records[0].approval_source == "purified_full_scan"


def test_packaged_poisoned_model_is_blocked_before_ppe_external_eval(tmp_path: Path, monkeypatch):
    package = tmp_path / "b模块新算法" / "backbone_soup_full_pipeline_v2_2026-05-24"
    poisoned_dir = package / "models" / "poisoned"
    purified_dir = package / "models" / "purified"
    audit_dir = package / "audit"
    heldout = tmp_path / "heldout"
    for directory in (poisoned_dir, purified_dir, audit_dir, heldout):
        directory.mkdir(parents=True)
    poisoned = poisoned_dir / "b2_b_sig_multiperiod_oda_poisoned.pt"
    purified = purified_dir / "b2_b_sig_multiperiod_oda_purified_strict.pt"
    poisoned.write_bytes(b"known-poisoned-model" * 128)
    purified.write_bytes(b"packaged-purified-model" * 128)
    audit_dir.joinpath("FINAL_STRICT_AUDIT_2026-05-23.json").write_text(
        """
{
  "rows": [],
  "fam_best": {
    "b2": {
      "family": "b2 (SIG multi-period ODA)",
      "tag": "b2",
      "tier": "869-row aug-stress (Backbone-Soup alpha=0.8)",
      "defense": "Backbone-Soup",
      "k": 0,
      "N": 869,
      "wilson_upper": 0.0044,
      "mAP_drop_pp": -0.11,
      "certified": true,
      "strict_pass": true,
      "status": "ok"
    }
  }
}
""",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - POISONED_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet, head]
""".replace("POISONED_PT", str(poisoned).replace("\\", "/")).replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    def fake_external(*_args, **_kwargs):
        raise AssertionError("known poisoned catalog should block before PPE external validation")

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    scan = svc.scan(scan_type="full")

    assert scan["status"] == "suspicious"
    evidence = scan["diagnostics"]["new_algorithm_poisoned_evidence"]
    assert evidence["family_tag"] == "b2"
    assert evidence["package_model_hash"] == "sha256:" + sha256_file(poisoned)
    assert evidence["purified_candidates"][0]["strict_pass"] is True
    assert svc.registry.list_records() == []

    purify = svc.purify(scan_after=True)
    assert purify["status"] == "scan_clean_trusted"
    assert purify["scan_status"] == "clean"
    assert Path(purify["purified_model_path"]).read_bytes() == purified.read_bytes()
    assert len(svc.registry.list_records()) == 1


def test_service_returns_trusted_purified_runtime_model_after_clean_rescan(tmp_path: Path, monkeypatch):
    package = tmp_path / "b模块新算法" / "backbone_soup_full_pipeline_v2_2026-05-24"
    poisoned_dir = package / "models" / "poisoned"
    purified_dir = package / "models" / "purified"
    audit_dir = package / "audit"
    heldout = tmp_path / "heldout"
    for directory in (poisoned_dir, purified_dir, audit_dir, heldout):
        directory.mkdir(parents=True)
    poisoned = poisoned_dir / "b2_b_sig_multiperiod_oda_poisoned.pt"
    purified = purified_dir / "b2_b_sig_multiperiod_oda_purified_strict.pt"
    poisoned.write_bytes(b"known-poisoned-model" * 128)
    purified.write_bytes(b"packaged-purified-model" * 128)
    audit_dir.joinpath("FINAL_STRICT_AUDIT_2026-05-23.json").write_text(
        """
{
  "rows": [],
  "fam_best": {
    "b2": {
      "family": "b2 (SIG multi-period ODA)",
      "tag": "b2",
      "tier": "869-row aug-stress (Backbone-Soup alpha=0.8)",
      "defense": "Backbone-Soup",
      "k": 0,
      "N": 869,
      "wilson_upper": 0.0044,
      "mAP_drop_pp": -0.11,
      "certified": true,
      "strict_pass": true,
      "status": "ok"
    }
  }
}
""",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - POISONED_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet, head]
""".replace("POISONED_PT", str(poisoned).replace("\\", "/")).replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )

    def fake_external(*_args, **_kwargs):
        raise AssertionError("known poisoned and strict purified package paths should not hit external validation")

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    assert svc.scan(scan_type="full")["status"] == "suspicious"
    assert svc.purify(scan_after=True)["status"] == "scan_clean_trusted"

    original_status = svc.status()
    runtime = svc.trusted_purified_runtime_model()

    assert original_status["admission_status"] == "purified_alternative_available"
    assert original_status["can_scan"] is True
    assert original_status["can_purify"] is False
    assert original_status["last_scan_completed"] is True
    assert original_status["last_purification_completed"] is True
    assert original_status["last_scan_job"]["status"] == "suspicious"
    assert original_status["last_purification_job"]["status"] == "scan_clean_trusted"
    assert original_status["last_purification_job"]["scan_status"] == "clean"
    recommended = original_status["recommended_runtime_model"]
    assert recommended["path"] == original_status["purified_model_path"]
    assert recommended["source_pt_path"] == recommended["path"]
    assert recommended["admission_status"] == "trusted"
    assert recommended["allowed"] is True
    assert runtime is not None
    assert runtime["custom_model"]["enabled"] is True
    assert runtime["custom_model"]["backend"] == "pytorch"
    assert runtime["custom_model"]["model_family"] == "yolov5"
    assert runtime["custom_model"]["source_pt_path"] == runtime["custom_model"]["path"]
    assert Path(runtime["custom_model"]["path"]).exists()
    assert runtime["model_security"]["admission_status"] == "trusted"
    assert runtime["model_security"]["allowed"] is True
    assert runtime["source_model_security"]["admission_status"] == "purified_alternative_available"
    assert svc.recent_logs(limit=5)["entries"][0]["event"] == "purified_runtime_selected"


def test_custom_purified_pt_scan_uses_itself_as_source_pt(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    default_poisoned = tmp_path / "poisoned.pt"
    purified = tmp_path / "purified.pt"
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    default_poisoned.write_bytes(b"poisoned-model" * 128)
    purified.write_bytes(b"purified-model" * 128)
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - POISONED_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
  heldout_roots:
    - HELDOUT_PATH
  external_eval_target_classes: [helmet, head]
""".replace("POISONED_PT", str(default_poisoned).replace("\\", "/")).replace("HELDOUT_PATH", str(heldout).replace("\\", "/")),
        encoding="utf-8",
    )
    seen: dict[str, str] = {}

    def fake_external(_fp, *, config, **_kwargs):
        seen["artifact"] = config["inference"]["artifacts"]["pytorch"][0]
        seen["source_pt"] = config["runtime"]["custom_model"]["source_pt_path"]
        return {"summary": {"n_rows": 1, "max_asr": 0.0, "mean_asr": 0.0}}

    monkeypatch.setattr(model_security_scanner, "_run_external_validation", fake_external)
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    custom_model = {
        "enabled": True,
        "path": str(purified),
        "backend": "pytorch",
        "model_family": "yolov5",
        "source_pt_path": str(default_poisoned),
    }

    status = svc.status(custom_model=custom_model)
    report = svc.scan(scan_type="full", custom_model=custom_model)

    assert status["source_pt_path"] == str(purified)
    assert report["source_model_path"] == str(purified)
    assert seen["artifact"] == str(purified)
    assert seen["source_pt"] == str(purified)
    assert report["status"] == "clean"


def test_background_full_scan_auto_purifies_suspicious_result(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    source_pt = tmp_path / "source.pt"
    source_pt.write_bytes(b"source-model" * 128)
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
""".replace("SOURCE_PT", str(source_pt).replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)
    calls: list[str] = []

    def fake_scan(**_kwargs):
        calls.append("scan")
        fp = svc.current_fingerprint()
        report = ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="suspicious",
            risk_score=1.0,
            source_model_path=str(source_pt),
            source_model_hash="sha256:" + sha256_file(source_pt),
            runtime_artifact_path=fp.model_path,
        )
        svc._last_report = report
        return report.to_dict()

    def fake_purify(**kwargs):
        calls.append(f"purify:{kwargs.get('scan_after')}")
        return {"status": "scan_clean_trusted"}

    monkeypatch.setattr(svc, "scan", fake_scan)
    monkeypatch.setattr(svc, "purify", fake_purify)

    started = svc.start_background_scan(scan_type="full", auto_purify=True)
    assert started["started"] is True
    assert started["auto_purify"] is True
    assert svc._scan_thread is not None
    svc._scan_thread.join(timeout=5)

    assert calls == ["scan", "purify:True"]
    logs = svc.recent_logs(limit=10)["entries"]
    assert any(item["event"] == "purification_auto_queued" for item in logs)


def test_background_auto_purify_failure_is_logged_as_purification_failure(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    source_pt = tmp_path / "source.pt"
    source_pt.write_bytes(b"source-model" * 128)
    cfg_path.write_text(
        """
inference:
  backend: pytorch
  model_family: yolov5
  device: cpu
  image_size: 640
  confidence: 0.3
  artifacts:
    pytorch:
      - SOURCE_PT
module_a:
  require_gpu: false
  device: cpu
  frame_size: 640
runtime:
  preview_render_fps: 10
  process_fps_cap: 5
  detector_process_fps_cap: 5
ppe_tracking:
  iou_match_threshold: 0.3
a3b:
  window_size: 5
  min_window_hits: 2
model_security:
  enabled: true
""".replace("SOURCE_PT", str(source_pt).replace("\\", "/")),
        encoding="utf-8",
    )
    svc = ModelSecurityService(config_path=cfg_path, root=tmp_path)

    def fake_scan(**_kwargs):
        fp = svc.current_fingerprint()
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="suspicious",
            risk_score=1.0,
            source_model_path=str(source_pt),
            source_model_hash="sha256:" + sha256_file(source_pt),
            runtime_artifact_path=fp.model_path,
        ).to_dict()

    def fake_purify(**_kwargs):
        raise RuntimeError("unit purification failure")

    monkeypatch.setattr(svc, "scan", fake_scan)
    monkeypatch.setattr(svc, "purify", fake_purify)

    svc.start_background_scan(scan_type="full", auto_purify=True)
    assert svc._scan_thread is not None
    svc._scan_thread.join(timeout=5)

    logs = svc.recent_logs(limit=10)["entries"]
    events = [item["event"] for item in logs]
    assert "purification_failed" in events
    assert "scan_failed" not in events


def test_fastapi_model_security_endpoints():
    app = create_app(bind_host="127.0.0.1")
    client = TestClient(app)
    res = client.get("/api/status")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert "model_security" not in res.json()["status"]

    res = client.get("/api/model-security/status")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert "model_security" in res.json()
    res = client.post("/api/model-security/scan", json={"scan_type": "quick", "background": False, "profile": "empty_smoke"})
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert "report" in res.json()


def test_fastapi_model_security_trust_is_delete_only():
    class FakeModelSecurity:
        def __init__(self) -> None:
            self.records = [{"fingerprint": "sha256:test"}]
            self.allow_purify = False

        def trust_records(self) -> dict:
            return {"registry_path": "unit-registry.json", "count": len(self.records), "records": list(self.records)}

        def delete_trust(self, fingerprint: str) -> dict:
            deleted = bool(self.records and self.records[0]["fingerprint"] == fingerprint)
            if deleted:
                self.records = []
            return {"deleted": deleted, "fingerprint": fingerprint, "registry_path": "unit-registry.json"}

        def clear_trust(self) -> dict:
            deleted = len(self.records)
            self.records = []
            return {"deleted": deleted, "registry_path": "unit-registry.json"}

        def status(self, **_kwargs) -> dict:
            return {
                "enabled": True,
                "admission_status": "blocked_scan_required",
                "whitelist_policy": "auto_full_scan_clean_only",
                "whitelist_user_actions": ["delete"],
                "can_purify": self.allow_purify,
            }

        def start_background_purification(self, **_kwargs) -> dict:
            return {"started": True, "fingerprint": "sha256:test", "scan_after": True}

        def latest_purification_report(self) -> dict:
            return {"status": "missing"}

        def recent_logs(self, **_kwargs) -> dict:
            return {
                "log_path": "unit-events.jsonl",
                "count": 1,
                "entries": [{"event": "scan_started", "status": "running", "message": "unit"}],
            }

    fake_security = FakeModelSecurity()
    app = create_app(engine=object(), model_security=fake_security, bind_host="127.0.0.1")
    client = TestClient(app)

    res = client.get("/api/model-security/trust")
    assert res.status_code == 200
    assert res.json()["trust"]["count"] == 1

    res = client.post("/api/model-security/trust", json={"notes": "manual"})
    assert res.status_code == 403
    assert res.json()["error"] == "manual_trust_disabled"

    res = client.post("/api/model-security/trust/delete", json={"fingerprint": "sha256:test"})
    assert res.status_code == 200
    assert res.json()["trust"]["deleted"] is True

    fake_security.records = [{"fingerprint": "sha256:another"}]
    res = client.post("/api/model-security/trust/clear", json={})
    assert res.status_code == 200
    assert res.json()["trust"]["deleted"] == 1

    res = client.post("/api/model-security/purify", json={"background": True})
    assert res.status_code == 409
    assert res.json()["error"] == "purification_requires_suspicious_full_scan"

    fake_security.allow_purify = True
    res = client.post("/api/model-security/purify", json={"background": True})
    assert res.status_code == 200
    assert res.json()["purification"]["started"] is True

    res = client.get("/api/model-security/purification-report")
    assert res.status_code == 200
    assert res.json()["report"]["status"] == "missing"

    res = client.get("/api/model-security/logs")
    assert res.status_code == 200
    assert res.json()["logs"]["count"] == 1

    res = client.get("/model-security/logs")
    assert res.status_code == 200
    assert "B模块运行日志" in res.text

    res = client.get("/model-security")
    assert res.status_code == 200
    assert "B模块模型安全中心" in res.text
