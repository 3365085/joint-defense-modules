from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import cv2

from defense.module_a.result_contract import adapt_a3b_result
from defense.runtime.frame_processor import FrameProcessor
from defense.runtime.pipeline_factory import PipelineCache

ProgressCallback = Callable[[int, int, dict[str, Any]], None]
A3B_HELDOUT_EVALUATOR_VERSION = "frameprocessor-module-a-a3b-heldout-v2"
_A3B_POSITIVE_ATTACK_TYPES = frozenset(
    {
        "a3b",
        "a3b_replay",
        "paper_photo",
        "replay",
        "screen_replay",
        "static_image_spoof",
        "static_media",
    }
)
_HEALTH_GATE_FIELDS = (
    "clips_with_errors",
    "clips_with_backend_errors",
    "clips_with_worker_timeouts",
    "clips_with_worker_rejections",
    "clips_with_schedule_blocked",
    "clips_with_stale_results",
    "clips_with_result_expiry",
    "clips_with_temporal_predecessor_gaps",
)
_BEHAVIOR_GATE_FIELDS = (
    "clean_module_a_alert_videos",
    "clean_a3b_fp_videos",
    "attack_wrong_channel_videos",
)
_SOURCE_IDENTITY_PATHS = (
    "model/src/defense/diagnostics/a3b_heldout.py",
    "model/tools/run_a3b_heldout.py",
    "model/src/defense/runtime/a3b_soft_trigger.py",
    "model/src/defense/runtime/frame_processor.py",
    "model/src/defense/runtime/pipeline_factory.py",
    "model/src/defense/runtime/runner.py",
    "model/src/defense/runtime/overlay_records.py",
    "model/src/defense/runtime/evidence.py",
    "model/src/defense/module_a/rebuilt/detector.py",
    "model/src/defense/module_a/result_contract.py",
    "model/src/defense/pipelines/video_defense_pipeline.py",
    "model/src/defense/visualization/overlay.py",
)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(repository_root: Path) -> dict[str, Any]:
    fingerprints = {
        relative: _sha256_file(repository_root / relative)
        for relative in _SOURCE_IDENTITY_PATHS
        if (repository_root / relative).is_file()
    }
    git_head = None
    git_dirty = None
    git_status_sha256 = None
    git_error = None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode == 0:
            git_head = head.stdout.strip() or None
        else:
            git_error = head.stderr.strip() or f"git rev-parse exited {head.returncode}"
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                *_SOURCE_IDENTITY_PATHS,
                "model/configs/module_a_runtime.yaml",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode == 0:
            status_text = status.stdout
            git_dirty = bool(status_text.strip())
            git_status_sha256 = hashlib.sha256(
                status_text.encode("utf-8")
            ).hexdigest()
        elif git_error is None:
            git_error = status.stderr.strip() or f"git status exited {status.returncode}"
    except Exception as exc:
        git_error = f"{type(exc).__name__}: {exc}"
    return {
        "evaluator_version": A3B_HELDOUT_EVALUATOR_VERSION,
        "source_sha256": fingerprints,
        "git_head": git_head,
        "git_dirty": git_dirty,
        "git_status_sha256": git_status_sha256,
        "git_error": git_error,
    }


def heldout_gate_failures(report: dict[str, Any]) -> list[str]:
    summary = (
        report.get("summary", {})
        if isinstance(report.get("summary"), dict)
        else {}
    )
    metadata = (
        report.get("metadata", {})
        if isinstance(report.get("metadata"), dict)
        else {}
    )
    failures = [
        f"{field}={_int(summary.get(field))}"
        for field in (
            *_HEALTH_GATE_FIELDS,
            *_BEHAVIOR_GATE_FIELDS,
        )
        if _int(summary.get(field)) > 0
    ]
    physical_attack_clips = _int(summary.get("physical_attack_clips"))
    physical_attack_hits = _int(
        summary.get("physical_attack_hit_videos")
    )
    if physical_attack_clips >= 21 and physical_attack_hits < 20:
        failures.append(
            "physical_attack_hit_videos="
            f"{physical_attack_hits}/{physical_attack_clips}"
        )
    if metadata.get("thread_warmup_error"):
        failures.append(
            f"thread_warmup_error={metadata['thread_warmup_error']}"
        )
    return failures


def load_a3b_heldout_manifest(
    manifest: str | Path,
    *,
    split: str = "heldout",
    repository_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    manifest_path = Path(manifest).expanduser().resolve()
    repo_root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else manifest_path.parents[2]
    )
    rows: list[dict[str, Any]] = []
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        videos = payload.get("videos", []) if isinstance(payload, dict) else []
        for raw in videos:
            if not isinstance(raw, dict) or str(raw.get("role") or "") != "a3b":
                continue
            video_path = Path(
                str(raw.get("canonical_path") or raw.get("relative_path") or "")
            ).expanduser()
            if not video_path.is_absolute():
                material_root = Path(str(payload.get("material_root") or repo_root))
                video_path = material_root / video_path
            rows.append(
                {
                    **raw,
                    "clip_id": str(raw.get("asset_id") or "").strip(),
                    "path": str(video_path.resolve(strict=False)),
                    "label": 1 if str(raw.get("label") or "") == "attack" else 0,
                    "attack_type": str(
                        raw.get("attack_type") or "a3b_static_media"
                    ).strip(),
                    "split": split,
                }
            )
        if not rows:
            raise RuntimeError(
                f"authoritative manifest contains no A3b row: {manifest_path}"
            )
        return rows
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("split") or "").strip().lower() != str(split).lower():
                continue
            raw_path = str(raw.get("path") or "").strip()
            video_path = Path(raw_path).expanduser()
            if not video_path.is_absolute():
                video_path = repo_root / video_path
            rows.append(
                {
                    **raw,
                    "clip_id": str(raw.get("clip_id") or "").strip(),
                    "path": str(video_path.resolve(strict=False)),
                    "label": _int(raw.get("label")),
                    "attack_type": str(raw.get("attack_type") or "clean").strip(),
                }
            )
    if not rows:
        raise RuntimeError(
            f"manifest contains no rows for split={split!r}: {manifest_path}"
        )
    return rows


def _clip_result_template(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": str(row.get("clip_id") or ""),
        "path": str(row.get("path") or ""),
        "label": _int(row.get("label")),
        "attack_type": str(row.get("attack_type") or "clean"),
        "frames": 0,
        "source_fps": 0.0,
        "module_a_single_frame_suspicious_frames": 0,
        "module_a_attack_detected_frames": 0,
        "module_a_alert_confirmed_frames": 0,
        "module_a_fresh_confirmed_frames": 0,
        "module_a_held_confirmed_frames": 0,
        "module_a_attack_state_frames": 0,
        "module_a_evidence_condition_frames": 0,
        "first_module_a_alert": None,
        "first_module_a_alert_time_s": None,
        "first_module_a_alert_channel": None,
        "first_module_a_alert_reason": None,
        "module_a_primary_channel_counts": {},
        "max_p_adv": 0.0,
        "max_p_blind": 0.0,
        "a3b_trigger_frames": 0,
        "first_a3b_trigger": None,
        "first_a3b_trigger_time_s": None,
        "first_a3b_trigger_source": None,
        "authoritative_media_confirmed_frames": 0,
        "max_a3b_score": 0.0,
        "max_a3b_observed_score": 0.0,
        "max_a3b_confirmed_score": 0.0,
        "max_background_error_count": 0,
        "max_timed_out_workers": 0,
        "max_worker_rejected_count": 0,
        "max_active_workers": 0,
        "max_retired_workers": 0,
        "max_local_live_workers": 0,
        "max_global_live_workers": 0,
        "max_result_expired_count": 0,
        "schedule_blocked_frames": 0,
        "stale_result_frames": 0,
        "strict_temporal_frames": 0,
        "temporal_predecessor_missing_frames": 0,
        "elapsed_s": 0.0,
        "error": None,
    }


def _update_clip_metrics(
    result: dict[str, Any],
    *,
    frame_idx: int,
    source_time_s: float,
    status: dict[str, Any],
    info: dict[str, Any],
) -> None:
    module_a_single_frame_suspicious = bool(
        status.get("single_frame_suspicious", False)
    )
    module_a_attack_detected = bool(
        status.get("attack_detected", False)
    )
    module_a_alert_confirmed = bool(
        status.get("alert_confirmed", False)
    )
    module_a_attack_state_active = bool(
        status.get("attack_state_active", False)
    )
    module_a_alert_held = bool(
        status.get("module_a_alert_held", False)
    )
    module_a_primary_channel = str(
        status.get("module_a_primary_channel") or "none"
    )
    if module_a_single_frame_suspicious:
        result["module_a_single_frame_suspicious_frames"] += 1
    if module_a_attack_detected:
        result["module_a_attack_detected_frames"] += 1
    if module_a_alert_confirmed:
        result["module_a_alert_confirmed_frames"] += 1
        if module_a_alert_held:
            result["module_a_held_confirmed_frames"] += 1
        else:
            result["module_a_fresh_confirmed_frames"] += 1
        channel_counts = result["module_a_primary_channel_counts"]
        channel_counts[module_a_primary_channel] = (
            _int(channel_counts.get(module_a_primary_channel)) + 1
        )
        if result["first_module_a_alert"] is None:
            result["first_module_a_alert"] = int(frame_idx)
            result["first_module_a_alert_time_s"] = float(source_time_s)
            result["first_module_a_alert_channel"] = (
                module_a_primary_channel
            )
            result["first_module_a_alert_reason"] = str(
                status.get("reason") or "unknown"
            )
    if module_a_attack_state_active:
        result["module_a_attack_state_frames"] += 1
    if module_a_alert_confirmed or module_a_attack_state_active:
        result["module_a_evidence_condition_frames"] += 1
    result["max_p_adv"] = max(
        _float(result.get("max_p_adv")),
        _float(status.get("p_adv")),
    )
    result["max_p_blind"] = max(
        _float(result.get("max_p_blind")),
        _float(status.get("p_blind")),
    )

    if bool(status.get("a3b_triggered", False)):
        result["a3b_trigger_frames"] += 1
        if result["first_a3b_trigger"] is None:
            result["first_a3b_trigger"] = int(frame_idx)
            result["first_a3b_trigger_time_s"] = float(source_time_s)
            result["first_a3b_trigger_source"] = str(
                status.get("a3b_triggered_source") or "unknown"
            )

    static_media = adapt_a3b_result(info)
    if bool(static_media.get("media_confirmed", False)):
        result["authoritative_media_confirmed_frames"] += 1

    for output_key, status_key in (
        ("max_a3b_score", "a3b_score"),
        ("max_a3b_observed_score", "a3b_observed_score"),
        ("max_a3b_confirmed_score", "a3b_confirmed_score"),
    ):
        result[output_key] = max(
            _float(result.get(output_key)),
            _float(status.get(status_key)),
        )
    for output_key, status_key in (
        ("max_background_error_count", "a3b_error_count"),
        ("max_timed_out_workers", "a3b_timed_out_worker_count"),
        ("max_worker_rejected_count", "a3b_worker_rejected_count"),
        ("max_active_workers", "a3b_active_worker_count"),
        ("max_retired_workers", "a3b_retired_worker_count"),
        ("max_local_live_workers", "a3b_live_worker_count"),
        ("max_global_live_workers", "a3b_global_live_worker_count"),
        ("max_result_expired_count", "a3b_result_expired_count"),
    ):
        result[output_key] = max(
            _int(result.get(output_key)),
            _int(status.get(status_key)),
        )
    if bool(status.get("a3b_schedule_blocked", False)):
        result["schedule_blocked_frames"] += 1
    if _int(status.get("a3b_result_seq")) > 0 and not bool(
        status.get("a3b_result_fresh", False)
    ):
        result["stale_result_frames"] += 1

    temporal = (
        status.get("temporal_input", {})
        if isinstance(status.get("temporal_input"), dict)
        else {}
    )
    if frame_idx > 0:
        if bool(temporal.get("strict_source_predecessor", False)):
            result["strict_temporal_frames"] += 1
        else:
            result["temporal_predecessor_missing_frames"] += 1


def _summary(
    *,
    profile: str,
    cap_frames: int,
    rows: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    clean = [row for row in rows if _int(row.get("label")) == 0]
    attacks = [row for row in rows if _int(row.get("label")) != 0]
    a3b_positives = [
        row
        for row in attacks
        if str(row.get("attack_type") or "").strip().lower()
        in _A3B_POSITIVE_ATTACK_TYPES
    ]
    wrong_channel_attacks = [
        row for row in attacks if row not in a3b_positives
    ]
    clean_module_a_channels: dict[str, int] = {}
    for row in clean:
        counts = (
            row.get("module_a_primary_channel_counts", {})
            if isinstance(
                row.get("module_a_primary_channel_counts"),
                dict,
            )
            else {}
        )
        for channel, count in counts.items():
            key = str(channel or "none")
            clean_module_a_channels[key] = (
                clean_module_a_channels.get(key, 0)
                + _int(count)
            )
    return {
        "profile": str(profile),
        "cap_frames": int(cap_frames),
        "clips": len(rows),
        "clean_clips": len(clean),
        "attack_clips": len(attacks),
        "clean_module_a_suspicious_videos": sum(
            1
            for row in clean
            if _int(
                row.get(
                    "module_a_single_frame_suspicious_frames"
                )
            )
            > 0
        ),
        "clean_module_a_suspicious_frames": sum(
            _int(
                row.get(
                    "module_a_single_frame_suspicious_frames"
                )
            )
            for row in clean
        ),
        "clean_module_a_attack_detected_videos": sum(
            1
            for row in clean
            if _int(row.get("module_a_attack_detected_frames")) > 0
        ),
        "clean_module_a_attack_detected_frames": sum(
            _int(row.get("module_a_attack_detected_frames"))
            for row in clean
        ),
        "clean_module_a_alert_videos": sum(
            1
            for row in clean
            if _int(row.get("module_a_alert_confirmed_frames")) > 0
        ),
        "clean_module_a_alert_frames": sum(
            _int(row.get("module_a_alert_confirmed_frames"))
            for row in clean
        ),
        "clean_module_a_fresh_confirmed_frames": sum(
            _int(row.get("module_a_fresh_confirmed_frames"))
            for row in clean
        ),
        "clean_module_a_held_confirmed_frames": sum(
            _int(row.get("module_a_held_confirmed_frames"))
            for row in clean
        ),
        "clean_module_a_evidence_condition_videos": sum(
            1
            for row in clean
            if _int(
                row.get("module_a_evidence_condition_frames")
            )
            > 0
        ),
        "clean_module_a_evidence_condition_frames": sum(
            _int(row.get("module_a_evidence_condition_frames"))
            for row in clean
        ),
        "clean_module_a_alert_channels": clean_module_a_channels,
        "physical_attack_clips": len(wrong_channel_attacks),
        "physical_attack_hit_videos": sum(
            1
            for row in wrong_channel_attacks
            if _int(row.get("module_a_alert_confirmed_frames")) > 0
        ),
        "physical_attack_alert_frames": sum(
            _int(row.get("module_a_alert_confirmed_frames"))
            for row in wrong_channel_attacks
        ),
        "physical_attack_missed_videos": sum(
            1
            for row in wrong_channel_attacks
            if _int(row.get("module_a_alert_confirmed_frames")) == 0
        ),
        "clean_a3b_fp_videos": sum(
            1 for row in clean if _int(row.get("a3b_trigger_frames")) > 0
        ),
        "clean_a3b_fp_frames": sum(
            _int(row.get("a3b_trigger_frames")) for row in clean
        ),
        "a3b_positive_clips": len(a3b_positives),
        "a3b_positive_hit_videos": sum(
            1
            for row in a3b_positives
            if _int(row.get("a3b_trigger_frames")) > 0
        ),
        "a3b_positive_trigger_frames": sum(
            _int(row.get("a3b_trigger_frames"))
            for row in a3b_positives
        ),
        "attack_wrong_channel_videos": sum(
            1
            for row in wrong_channel_attacks
            if _int(row.get("a3b_trigger_frames")) > 0
        ),
        "attack_wrong_channel_frames": sum(
            _int(row.get("a3b_trigger_frames"))
            for row in wrong_channel_attacks
        ),
        "authoritative_media_confirmed_videos": sum(
            1
            for row in rows
            if _int(row.get("authoritative_media_confirmed_frames")) > 0
        ),
        "clips_with_errors": sum(1 for row in rows if row.get("error")),
        "clips_with_backend_errors": sum(
            1 for row in rows if _int(row.get("max_background_error_count")) > 0
        ),
        "clips_with_worker_timeouts": sum(
            1 for row in rows if _int(row.get("max_timed_out_workers")) > 0
        ),
        "clips_with_worker_rejections": sum(
            1 for row in rows if _int(row.get("max_worker_rejected_count")) > 0
        ),
        "clips_with_schedule_blocked": sum(
            1 for row in rows if _int(row.get("schedule_blocked_frames")) > 0
        ),
        "clips_with_stale_results": sum(
            1 for row in rows if _int(row.get("stale_result_frames")) > 0
        ),
        "clips_with_result_expiry": sum(
            1 for row in rows if _int(row.get("max_result_expired_count")) > 0
        ),
        "clips_with_temporal_predecessor_gaps": sum(
            1
            for row in rows
            if _int(row.get("temporal_predecessor_missing_frames")) > 0
        ),
        "max_retired_workers_seen": max(
            (_int(row.get("max_retired_workers")) for row in rows),
            default=0,
        ),
        "max_global_live_workers_seen": max(
            (_int(row.get("max_global_live_workers")) for row in rows),
            default=0,
        ),
        "elapsed_s": float(elapsed_s),
    }


def evaluate_a3b_heldout(
    *,
    manifest: str | Path,
    output_json: str | Path,
    profile: str = "desktop_rtx",
    config: str | Path | None = None,
    cap_frames: int = 240,
    split: str = "heldout",
    repository_root: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if cap_frames <= 0:
        raise ValueError("cap_frames must be positive")
    repo_root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[4]
    )
    model_root = repo_root / "model"
    manifest_path = Path(manifest).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    config_path = (
        Path(config).expanduser().resolve()
        if config is not None
        else model_root / "configs" / "module_a_runtime.yaml"
    )
    clips = load_a3b_heldout_manifest(
        manifest_path,
        split=split,
        repository_root=repo_root,
    )

    started = time.perf_counter()
    cache = PipelineCache(config_path=config_path, root=model_root)
    results: list[dict[str, Any]] = []
    effective_config: dict[str, Any] | None = None
    thread_warmup_error = ""
    try:
        bundle = cache.get(
            profile=profile,
            feature_options={},
            custom_model={},
        )
    except BaseException:
        cache.clear()
        raise
    runtime_config = (
        bundle.config.get("runtime", {})
        if isinstance(bundle.config.get("runtime"), dict)
        else {}
    )
    detector_cap = _float(
        runtime_config.get(
            "detector_process_fps_cap",
            runtime_config.get("process_fps_cap", 30.0),
        ),
        30.0,
    )
    try:
        warmup_frames = _int(
            runtime_config.get(
                "detector_thread_warmup_frames",
                bundle.warmup_frames,
            )
        )
        try:
            bundle.pipeline.warmup(warmup_frames)
        except Exception as exc:
            thread_warmup_error = f"{type(exc).__name__}: {exc}"
        finally:
            bundle.pipeline.reset()
        processor = FrameProcessor(
            bundle,
            jpeg_quality=_int(runtime_config.get("jpeg_quality"), 82),
        )

        for index, row in enumerate(clips, start=1):
            clip_started = time.perf_counter()
            result = _clip_result_template(row)
            video_path = Path(str(row["path"]))
            cap = None
            try:
                if not video_path.is_file():
                    raise FileNotFoundError(f"video does not exist: {video_path}")
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"failed to open video: {video_path}")
                processor.reset()
                source_fps = _float(cap.get(cv2.CAP_PROP_FPS), 30.0)
                if source_fps <= 0.0:
                    source_fps = 30.0
                result["source_fps"] = source_fps
                target_budget_ms = 1000.0 / max(
                    1.0,
                    min(source_fps, max(1.0, detector_cap)),
                )
                previous_frame = None
                previous_frame_idx = None
                previous_source_time_s = None
                for frame_idx in range(cap_frames):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    source_time_s = _float(
                        cap.get(cv2.CAP_PROP_POS_MSEC),
                        0.0,
                    ) / 1000.0
                    if source_time_s <= 0.0 and frame_idx > 0:
                        source_time_s = frame_idx / source_fps
                    processed = processor.process(
                        frame,
                        frame_idx=frame_idx,
                        source_type="file",
                        source=str(video_path),
                        profile=profile,
                        realtime=False,
                        video_time_s=source_time_s,
                        source_fps=source_fps,
                        dropped_frames=0,
                        display_options={
                            "show_boxes": True,
                            "show_person_boxes": True,
                            "show_module_hud": True,
                            "show_ppe_hud": True,
                        },
                        feature_options={},
                        custom_model={},
                        target_frame_budget_ms=target_budget_ms,
                        temporal_previous_frame=previous_frame,
                        temporal_previous_frame_idx=previous_frame_idx,
                        temporal_previous_source_time_s=previous_source_time_s,
                    )
                    status = (
                        processed.status
                        if isinstance(processed.status, dict)
                        else {}
                    )
                    if effective_config is None and isinstance(
                        status.get("module_a_effective_config"),
                        dict,
                    ):
                        effective_config = dict(
                            status["module_a_effective_config"]
                        )
                    _update_clip_metrics(
                        result,
                        frame_idx=frame_idx,
                        source_time_s=source_time_s,
                        status=status,
                        info=(
                            processed.info
                            if isinstance(processed.info, dict)
                            else {}
                        ),
                    )
                    result["frames"] += 1
                    previous_frame = frame
                    previous_frame_idx = frame_idx
                    previous_source_time_s = source_time_s
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if cap is not None:
                    cap.release()
                result["elapsed_s"] = time.perf_counter() - clip_started
                results.append(result)
                if progress is not None:
                    progress(index, len(clips), result)
    finally:
        cache.clear()

    report = {
        "summary": _summary(
            profile=profile,
            cap_frames=cap_frames,
            rows=results,
            elapsed_s=time.perf_counter() - started,
        ),
        "metadata": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "config": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "split": str(split),
            "evaluation_path": (
                "FrameProcessor/ModuleAResult+A3BSoftTriggerState"
            ),
            "evaluation_scope": (
                "module_a_alerts_and_a3b_runtime_state"
            ),
            "temporal_policy": "strict_consecutive_predecessor",
            "queue_policy": "deterministic_full_frame_no_drop",
            "thread_warmup_error": thread_warmup_error or None,
            "backend": str(bundle.backend),
            "model_family": str(bundle.model_family),
            "artifact_path": str(bundle.artifact_path),
            "artifact_fingerprint": list(bundle.artifact_fingerprint),
            "auxiliary_artifact_fingerprint": list(
                bundle.auxiliary_artifact_fingerprint
            ),
            "module_a_effective_config": effective_config,
            "source_identity": _source_identity(repo_root),
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
