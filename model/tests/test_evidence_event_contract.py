from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from defense.runtime.evidence import EvidenceSession, default_evidence_root, list_evidence_events, load_evidence_event
from defense.web.fastapi_app import create_app


def test_default_evidence_root_is_outside_source_tree(monkeypatch) -> None:
    monkeypatch.delenv("MODULE_A_EVIDENCE_ROOT", raising=False)

    root = default_evidence_root()

    assert root.parts[-3:] == ("runtime", "evidence", "monitor")
    assert "src" not in root.parts


def test_default_evidence_root_uses_environment_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MODULE_A_EVIDENCE_ROOT", str(tmp_path))

    assert default_evidence_root() == tmp_path


def test_a3b_evidence_finalize_exports_ui_compatibility_fields(tmp_path) -> None:
    session = EvidenceSession(
        source_type="file",
        source="case.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    session.update(
        frame_idx=12,
        frame=frame,
        info={},
        ppe={},
        status={
            "a3b_triggered": True,
            "a3b_event_score": 0.84,
            "a3b_triggered_source": "observed_window",
        },
    )

    events = session.close()

    assert len(events) == 1
    event = events[0]
    assert event["channel"] == "a3b"
    assert event["event_id"] == 1
    assert event["trigger_frame"] == 12
    assert event["last_warning_frame"] == 12
    assert event["peak_a3b_score"] == 0.84
    assert event["peak_score"] == 0.84
    assert event["reason"] == "observed_window"
    assert event["evidence_saved"] is True
    assert event["evidence_saved_frame_count"] == 1
    assert event["evidence_frames_dir"]
    assert event["evidence_representative_path"]
    assert event["evidence_event_key"]
    assert event["evidence_preview_url"].startswith("/evidence?event=")
    assert event["evidence_representative_url"].startswith("/api/evidence/file?token=")
    assert event["source"] == "case.mp4"
    assert event["profile"] == "desktop_rtx"


def test_ppe_evidence_uses_event_hold_after_current_warning_drops(tmp_path) -> None:
    session = EvidenceSession(
        source_type="file",
        source="ppe_case.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    session.update(
        frame_idx=10,
        frame=frame,
        info={},
        ppe={"warning": True, "reason": "bare_head_without_matched_helmet"},
        status={
            "ppe_warning": True,
            "ppe_confirmed": True,
            "ppe_event_active": True,
            "ppe_reason": "bare_head_without_matched_helmet",
            "ppe_window_positive": 3,
            "ppe_inferred_person_count": 1,
            "ppe_raw_person_count": 0,
            "ppe_head_count": 1,
            "ppe_helmet_count": 0,
            "ppe_missing_helmet_count": 1,
            "ppe_confirmed_source": "temporal_window",
        },
    )
    still_active = session.update(
        frame_idx=11,
        frame=frame,
        info={},
        ppe={"warning": False, "event_active": True, "event_last_reason": "bare_head_without_matched_helmet"},
        status={
            "ppe_warning": False,
            "ppe_confirmed": False,
            "ppe_event_active": True,
            "ppe_event_last_reason": "bare_head_without_matched_helmet",
        },
    )
    completed = session.update(
        frame_idx=12,
        frame=frame,
        info={},
        ppe={},
        status={"ppe_warning": False, "ppe_confirmed": False, "ppe_event_active": False},
    )

    assert still_active == []
    assert len(completed) == 1
    event = completed[0]
    assert event["channel"] == "ppe"
    assert event["started_frame"] == 10
    assert event["last_active_frame"] == 11
    assert event["reason"] == "bare_head_without_matched_helmet"
    assert event["ppe_inferred_person_count"] == 1
    assert event["ppe_raw_person_count"] == 0
    assert event["ppe_head_count"] == 1
    assert event["ppe_helmet_count"] == 0
    assert event["ppe_missing_helmet_count"] == 1
    assert event["ppe_confirmed_source"] == "temporal_window"
    assert event["evidence_saved_frame_count"] == 2
    assert event["evidence_clip_path"].endswith("clip.mp4")
    assert event["evidence_clip_url"].startswith("/api/evidence/file?token=")
    assert event["evidence_clip_codec"] == "h264"
    assert event["evidence_clip_browser_playable"] is True


def test_evidence_event_can_be_loaded_by_preview_key(tmp_path) -> None:
    session = EvidenceSession(
        source_type="file",
        source="ppe_case.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    session.update(
        frame_idx=7,
        frame=frame,
        info={},
        ppe={"warning": True},
        status={
            "ppe_warning": True,
            "ppe_event_active": True,
            "ppe_missing_helmet_count": 1,
            "ppe_window_positive": 1,
        },
    )
    event = session.close()[0]

    loaded = load_evidence_event(event["evidence_event_key"], root=tmp_path)

    assert loaded["event_id"] == event["event_id"]
    assert loaded["evidence_preview_url"] == event["evidence_preview_url"]
    assert len(loaded["frames"]) == 1
    assert loaded["frames"][0]["url"].startswith("/api/evidence/file?token=")


def test_evidence_events_are_indexed_for_management_and_replay(tmp_path) -> None:
    session = EvidenceSession(
        source_type="file",
        source="replay_case.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    session.update(
        frame_idx=3,
        frame=frame,
        info={},
        ppe={"warning": True},
        status={"ppe_warning": True, "ppe_event_active": True, "ppe_missing_helmet_count": 1},
    )
    event = session.close()[0]

    listed = list_evidence_events(root=tmp_path)

    assert listed["database"].endswith("evidence_index.sqlite3")
    assert listed["count"] == 1
    assert listed["events"][0]["evidence_event_key"] == event["evidence_event_key"]
    assert listed["events"][0]["source"] == "replay_case.mp4"


def test_evidence_artifacts_use_same_runtime_catalog_when_root_is_default_shape(tmp_path) -> None:
    runtime_root = tmp_path / "model" / "runtime"
    evidence_root = runtime_root / "evidence" / "monitor"
    session = EvidenceSession(
        source_type="file",
        source="catalog_case.mp4",
        profile="desktop_rtx",
        root=evidence_root,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    session.update(
        frame_idx=3,
        frame=frame,
        info={},
        ppe={"warning": True},
        status={"ppe_warning": True, "ppe_event_active": True, "ppe_missing_helmet_count": 1},
    )
    session.close()

    assert (runtime_root / "db" / "runtime_catalog.sqlite3").exists()
    assert not (evidence_root / "db" / "runtime_catalog.sqlite3").exists()


def test_evidence_file_download_can_force_attachment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODULE_A_EVIDENCE_ROOT", str(tmp_path))
    session = EvidenceSession(
        source_type="file",
        source="download_case.mp4",
        profile="desktop_rtx",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    session.update(
        frame_idx=3,
        frame=frame,
        info={},
        ppe={"warning": True},
        status={"ppe_warning": True, "ppe_event_active": True, "ppe_missing_helmet_count": 1},
    )
    event = session.close()[0]
    token = event["evidence_clip_url"].split("token=", 1)[1]

    app = create_app(bind_host="127.0.0.1")
    client = TestClient(app)
    response = client.get(f"/api/evidence/file?token={token}&inline=false")

    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment;")
