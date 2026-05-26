from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from defense.model_security import ModelSecurityService
from defense.model_security import purifier
from defense.model_security.registry import ModelTrustRegistry
from defense.model_security.reports import ModelPurificationReport, ModelSecurityReport
from defense.web.fastapi_app import create_app


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


def test_fastapi_start_can_bypass_model_security_for_test_only():
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

        def status(self, **_kwargs) -> dict:
            return {
                "enabled": True,
                "allowed": False,
                "admission_status": "suspicious",
                "blocking_reason": "last_full_scan_suspicious",
            }

        def _log_event(self, event: str, **fields) -> None:
            self.events.append((event, fields))

    engine = FakeEngine()
    security = FakeModelSecurity()
    app = create_app(engine=engine, model_security=security, bind_host="127.0.0.1")
    client = TestClient(app)

    custom_model = {
        "enabled": True,
        "path": "D:/tmp/poisoned.pt",
        "backend": "pytorch",
        "model_family": "yolov5",
    }
    res = client.post(
        "/api/start",
        json={
            "source_type": "file",
            "source": "D:/tmp/source.mp4",
            "profile": "empty_smoke",
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


def test_fastapi_start_returns_json_when_source_file_is_missing():
    class FakeEngine:
        run_id = 0

        def get_status(self) -> dict:
            return {"run_id": self.run_id, "running": False}

        def start(self, **_kwargs) -> int:
            raise FileNotFoundError("视频文件不存在或不可访问: D:/missing.mp4")

    class FakeModelSecurity:
        def status(self, **_kwargs) -> dict:
            return {
                "enabled": True,
                "allowed": True,
                "admission_status": "trusted",
                "blocking_reason": "",
            }

    app = create_app(engine=FakeEngine(), model_security=FakeModelSecurity(), bind_host="127.0.0.1")
    client = TestClient(app)

    res = client.post(
        "/api/start",
        json={
            "source_type": "file",
            "source": "D:/missing.mp4",
            "profile": "empty_smoke",
            "test_bypass_model_security": True,
        },
    )

    assert res.status_code == 400
    payload = res.json()
    assert payload["ok"] is False
    assert payload["error"] == "source_unavailable"
    assert "视频文件不存在" in payload["message"]
